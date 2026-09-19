import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np
from collections import deque
import threading
import queue


class ServoFaceTracker:
    def __init__(
        self,
        serial_port="/dev/ttyUSB0",
        baud_rate=115200,
        timeout=1,
        frame_w=1080,
        frame_h=720,
        reverse_y=True,
        debug=True,
        default_horiz=100.0,   
        default_vert=90.0,     
    ):
        self.BAUD_RATE = baud_rate
        self.SERIAL_PORT = serial_port
        self.TIMEOUT = timeout
        self.SERIAL_SEND_INTERVAL = 0.08

        self.default_horiz = default_horiz
        self.default_vert = default_vert
        self.servo_horiz_cmd = default_horiz
        self.servo_vert_cmd = default_vert
        self.send_horiz = default_horiz
        self.send_vert = default_vert

        self.SERVO_STEP_ANGLE = 0.25
        self.QUANT_HYSTERESIS_H = 0.2
        self.QUANT_HYSTERESIS_V = 0.2
        
        self.P_X = 0.00025
        self.P_Y = 0.00025
        
        self.DEAD_ZONE_X = 150
        self.DEAD_ZONE_Y = 100
        self.STEP_X_PER_FRAME = 0.10
        self.STEP_Y_PER_FRAME = 0.15
        
        self.LOCK_STABLE_FRAMES = 3
        self.lock_counter = 0

        self.last_known_face_center = None
        self.face_lost_counter = 0
        self.FACE_LOST_BUFFER = 5

        self.frame_w = frame_w
        self.frame_h = frame_h
        self.center_x = frame_w // 2
        self.center_y = frame_h // 2

        self.ENABLE_PREDICT = False
        self.PREDICT_FRAMES = 2
        self.prev_positions = deque(maxlen=5)

        self.SMOOTH_ALPHA_X = 0.10
        self.SMOOTH_ALPHA_Y = 0.08
        self.smooth_cx = None
        self.smooth_cy = None
        self.last_raw_cx = None
        self.last_raw_cy = None

        self.JUMP_THRESHOLD_X = 250
        self.JUMP_THRESHOLD_Y = 200
        self.JUMP_HOLD_FRAMES = 4
        self.jump_hold_counter = 0

        self.Y_HYSTERESIS_FRAMES = 3
        self.y_exceed_counter = 0
        self.Y_MEDIAN_FILTER_SIZE = 5
        self.y_history = deque(maxlen=self.Y_MEDIAN_FILTER_SIZE)

        self.CENTER_LOCK_X = 100
        self.CENTER_LOCK_Y = 100
        self.CENTER_UNLOCK_X = 200
        self.CENTER_UNLOCK_Y = 200
        self.center_locked = False
        self.locked_h_angle = None
        self.locked_v_angle = None

        self.NEAR_CENTER_X = 380
        self.MID_ERROR_X = 460
        self.NEAR_CENTER_Y = 250
        self.MID_ERROR_Y = 320

        self.SEND_CONFIRM_FRAMES_H_FAR = 1
        self.SEND_CONFIRM_FRAMES_H_MID = 1
        self.SEND_CONFIRM_FRAMES_H_NEAR = 1

        self.SEND_CONFIRM_FRAMES_V_FAR = 1
        self.SEND_CONFIRM_FRAMES_V_MID = 1
        self.SEND_CONFIRM_FRAMES_V_NEAR = 1

        self.pending_h = None
        self.pending_v = None
        self.pending_h_count = 0
        self.pending_v_count = 0

        self.H_CROSS_STEP_MIN_ERR = 20
        self.H_SINGLE_STEP_EXTRA_CONFIRM = 2
        self.stable_h_target = None
        self.stable_h_target_count = 0

        self.DEBUG = debug
        self.CONSOLE_LOG_INTERVAL = 5
        self.frame_counter = 0

        self.reverse_y = reverse_y
        self.last_send_time = 0.0
        self.current_track_id = None

        self.ser = self.init_serial()
        self.stop_event = threading.Event()
        self.cmd_queue = queue.Queue(maxsize=1)
        self.serial_lock = threading.Lock()

        self.worker_thread = threading.Thread(
            target=self._servo_worker,
            name="ServoSerialWorker",
            daemon=True,
        )
        self.worker_thread.start()

    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def quantize_with_hysteresis(self, cmd_angle, last_sent, step=5.0, hysteresis=2.5,
                                 min_angle=0.0, max_angle=180.0):
        cmd_angle = self.clamp(cmd_angle, min_angle, max_angle)
        # 支持浮点级步长的高精度取整
        target = round(cmd_angle / step) * step

        if last_sent is None:
            return self.clamp(target, min_angle, max_angle)

        if abs(cmd_angle - last_sent) < hysteresis:
            return self.clamp(last_sent, min_angle, max_angle)

        return self.clamp(target, min_angle, max_angle)

    def move_towards(self, current, target, max_step):
        diff = target - current
        if abs(diff) <= max_step:
            return target, diff
        step = max(-max_step, min(max_step, diff))
        return current + step, step

    def get_axis_zone_x(self, err_x, locked):
        ax = abs(err_x)
        if locked:
            return "LOCK"
        if ax < self.NEAR_CENTER_X:
            return "NEAR"
        if ax < self.MID_ERROR_X:
            return "MID"
        return "FAR"

    def get_axis_zone_y(self, err_y, locked):
        ay = abs(err_y)
        if locked:
            return "LOCK"
        if ay < self.NEAR_CENTER_Y:
            return "NEAR"
        if ay < self.MID_ERROR_Y:
            return "MID"
        return "FAR"

    def get_confirm_frames_h(self, err_x, locked):
        zone_x = self.get_axis_zone_x(err_x, locked)
        if zone_x == "LOCK":
            return 999
        if zone_x == "NEAR":
            return self.SEND_CONFIRM_FRAMES_H_NEAR
        if zone_x == "MID":
            return self.SEND_CONFIRM_FRAMES_H_MID
        return self.SEND_CONFIRM_FRAMES_H_FAR

    def get_confirm_frames_v(self, err_y, locked):
        zone_y = self.get_axis_zone_y(err_y, locked)
        if zone_y == "LOCK":
            return 999
        if zone_y == "NEAR":
            return self.SEND_CONFIRM_FRAMES_V_NEAR
        if zone_y == "MID":
            return self.SEND_CONFIRM_FRAMES_V_MID
        return self.SEND_CONFIRM_FRAMES_V_FAR

    def predict_next_position(self, positions, predict_frames):
        if len(positions) < 2:
            return None
        vel_x = 0.0
        vel_y = 0.0
        for i in range(1, len(positions)):
            vel_x += positions[i][0] - positions[i - 1][0]
            vel_y += positions[i][1] - positions[i - 1][1]
        vel_x /= (len(positions) - 1)
        vel_y /= (len(positions) - 1)
        last = positions[-1]
        return (last[0] + vel_x * predict_frames, last[1] + vel_y * predict_frames)

    def find_serial_port(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "USB" in port.description or "ttyUSB" in port.device:
                return port.device
        return None

    def init_serial(self):
        ser = None
        try:
            ser = serial.Serial(
                port=self.SERIAL_PORT,
                baudrate=self.BAUD_RATE,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=self.TIMEOUT,
                write_timeout=0.2,
            )
            time.sleep(0.3)
            ser.write(f"1_{self.default_vert:.2f}\r\n".encode("ascii"))
            time.sleep(0.1)
            ser.write(f"2_{self.default_horiz:.2f}\r\n".encode("ascii"))
            time.sleep(0.03)
            print(f"✅ 串口 {self.SERIAL_PORT} 打开，云台已移至默认角度 (H={self.default_horiz:.1f}°, V={self.default_vert:.1f}°)")
            return ser
        except Exception:
            auto = self.find_serial_port()
            if auto:
                try:
                    ser = serial.Serial(auto, self.BAUD_RATE, timeout=self.TIMEOUT, write_timeout=0.2)
                    time.sleep(0.3)
                    ser.write(f"1_{self.default_vert:.2f}\r\n".encode("ascii"))
                    time.sleep(0.1)
                    ser.write(f"2_{self.default_horiz:.2f}\r\n".encode("ascii"))
                    time.sleep(0.03)
                    print(f"✅ 自动串口 {auto} 打开，云台已移至默认角度 (H={self.default_horiz:.1f}°, V={self.default_vert:.1f}°)")
                    return ser
                except Exception:
                    return None
            return None

    def _enqueue_latest_command(self, horiz, vert):
        cmd = (float(horiz), float(vert))
        try:
            while True:
                self.cmd_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            pass

    def _servo_worker(self):
        last_sent_h = self.send_horiz
        last_sent_v = self.send_vert
        last_send_ts = 0.0

        while not self.stop_event.is_set():
            try:
                cmd = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd is None:
                break

            horiz, vert = cmd
            now = time.time()
            interval = now - last_send_ts
            if interval < self.SERIAL_SEND_INTERVAL:
                time.sleep(max(0.0, self.SERIAL_SEND_INTERVAL - interval))

            if self.ser is None:
                continue

            try:
                with self.serial_lock:
                    if self.ser and self.ser.is_open:
                        if abs(vert - last_sent_v) > 1e-6:
                            self.ser.write(f"1_{vert:.2f}\r\n".encode("ascii"))
                            last_sent_v = vert

                        if abs(horiz - last_sent_h) > 1e-6:
                            self.ser.write(f"2_{horiz:.2f}\r\n".encode("ascii"))
                            last_sent_h = horiz

                        last_send_ts = time.time()
            except serial.SerialException as e:
                print(f"⚠️ 串口发送错误: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            except Exception as e:
                print(f"⚠️ 后台舵机线程异常: {e}")

    def safe_recenter_and_close(self):
        if self.ser and self.ser.is_open:
            try:
                while True:
                    self.cmd_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.cmd_queue.put_nowait((self.default_horiz, self.default_vert))
            except queue.Full:
                self.cmd_queue.put((self.default_horiz, self.default_vert))

        time.sleep(2.5)

        self.stop_event.set()
        try:
            self.cmd_queue.put_nowait(None)
        except Exception:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        if self.ser and self.ser.is_open:
            try:
                with self.serial_lock:
                    self.ser.close()
                    print("✅ 串口已关闭")
            except Exception as e:
                print(f"⚠️ 关闭串口失败: {e}")

    def reset_tracking_state(self):
        self.lock_counter = 0
        self.last_known_face_center = None
        self.face_lost_counter = 0
        self.prev_positions.clear()

        self.smooth_cx = None
        self.smooth_cy = None
        self.last_raw_cx = None
        self.last_raw_cy = None
        self.jump_hold_counter = 0

        self.y_exceed_counter = 0
        self.y_history.clear()

        self.center_locked = False
        self.locked_h_angle = None
        self.locked_v_angle = None

        self.pending_h = None
        self.pending_v = None
        self.pending_h_count = 0
        self.pending_v_count = 0

        self.stable_h_target = None
        self.stable_h_target_count = 0
        self.current_track_id = None

    def update_frame_size(self, frame):
        h, w = frame.shape[:2]
        if w > 0 and h > 0 and (w != self.frame_w or h != self.frame_h):
            self.frame_w = w
            self.frame_h = h
            self.center_x = w // 2
            self.center_y = h // 2

    def _score_item(self, item):
        x1, y1, x2, y2 = item["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h

        state = str(item.get("state", "unknown"))
        name = str(item.get("name", "Unknown"))
        sim = float(item.get("sim_score", item.get("score", 0.0)))
        tid = int(item.get("track_id", -1))

        score = area * 1.0

        if tid == self.current_track_id:
            score += 200000

        if state == "confirmed" and name != "Unknown":
            score += 120000
        elif state == "tentative":
            score += 60000
        elif state == "unknown":
            score += 20000

        score += sim * 10000
        return score

    def select_target(self, items):
        valid = []
        for item in items:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1
            if w < 20 or h < 20:
                continue
            valid.append(item)

        if not valid:
            return None

        valid.sort(key=self._score_item, reverse=True)
        best = valid[0]
        self.current_track_id = int(best.get("track_id", -1))
        return best

    def update(self, frame, items):
        self.frame_counter += 1
        self.update_frame_size(frame)

        target = self.select_target(items)
        current_has_face = False
        is_tracking = False

        err_x = 0
        err_y = 0
        zone_x = "NOFACE"
        zone_y = "NOFACE"
        confirm_h = self.SEND_CONFIRM_FRAMES_H_FAR
        confirm_v = self.SEND_CONFIRM_FRAMES_V_FAR
        delta_x = 0.0
        delta_y = 0.0
        target_horiz = self.servo_horiz_cmd
        target_vert = self.servo_vert_cmd

        if target is not None:
            x1, y1, x2, y2 = map(int, target["bbox"])
            raw_cx = (x1 + x2) // 2
            raw_cy = (y1 + y2) // 2

            current_has_face = True
            self.face_lost_counter = 0

            if self.ENABLE_PREDICT:
                self.prev_positions.append((raw_cx, raw_cy))
                pred = self.predict_next_position(self.prev_positions, self.PREDICT_FRAMES)
                if pred is not None:
                    raw_cx, raw_cy = int(pred[0]), int(pred[1])

            jump_detected = False
            if self.last_raw_cx is not None and self.last_raw_cy is not None:
                if abs(raw_cx - self.last_raw_cx) > self.JUMP_THRESHOLD_X or \
                   abs(raw_cy - self.last_raw_cy) > self.JUMP_THRESHOLD_Y:
                    jump_detected = True

            self.last_raw_cx = raw_cx
            self.last_raw_cy = raw_cy

            if jump_detected:
                self.jump_hold_counter = self.JUMP_HOLD_FRAMES
                self.servo_horiz_cmd = self.send_horiz
                self.servo_vert_cmd = self.send_vert
                self.smooth_cx = raw_cx
                self.smooth_cy = raw_cy
                self.stable_h_target = None
                self.stable_h_target_count = 0
            elif self.jump_hold_counter > 0:
                self.jump_hold_counter -= 1

            if self.smooth_cx is None:
                self.smooth_cx = raw_cx
                self.smooth_cy = raw_cy
            else:
                self.smooth_cx = (1 - self.SMOOTH_ALPHA_X) * self.smooth_cx + self.SMOOTH_ALPHA_X * raw_cx
                self.smooth_cy = (1 - self.SMOOTH_ALPHA_Y) * self.smooth_cy + self.SMOOTH_ALPHA_Y * raw_cy

            cx = int(self.smooth_cx)
            cy = int(self.smooth_cy)

            self.last_known_face_center = (cx, cy)
            is_tracking = True
        else:
            if self.face_lost_counter < self.FACE_LOST_BUFFER and self.last_known_face_center is not None:
                cx, cy = self.last_known_face_center
                self.face_lost_counter += 1
                current_has_face = True
                is_tracking = True
            else:
                self.reset_tracking_state()
                cx, cy = self.center_x, self.center_y

        if current_has_face:
            err_x = cx - self.center_x
            err_y = cy - self.center_y

            if self.center_locked:
                if abs(err_x) > self.CENTER_UNLOCK_X or abs(err_y) > self.CENTER_UNLOCK_Y:
                    self.center_locked = False
                    self.locked_h_angle = None
                    self.locked_v_angle = None
            else:
                if abs(err_x) < self.CENTER_LOCK_X and abs(err_y) < (self.CENTER_LOCK_Y * 2):
                    self.lock_counter += 1
                    if self.lock_counter >= self.LOCK_STABLE_FRAMES:
                        self.center_locked = True
                        self.locked_h_angle = self.servo_horiz_cmd
                        self.locked_v_angle = self.servo_vert_cmd
                        self.pending_h = None
                        self.pending_v = None
                        self.pending_h_count = 0
                        self.pending_v_count = 0
                        self.stable_h_target = None
                        self.stable_h_target_count = 0
                else:
                    self.lock_counter = 0

            if self.center_locked:
                target_horiz = self.locked_h_angle
                target_vert = self.locked_v_angle
                self.servo_horiz_cmd = self.locked_h_angle
                self.servo_vert_cmd = self.locked_v_angle
            elif self.jump_hold_counter > 0:
                zone_x = "HOLD"
                zone_y = "HOLD"
            else:
                # 水平
                if abs(err_x) > self.DEAD_ZONE_X:
                    zone_x = self.get_axis_zone_x(err_x, self.center_locked)
                    
                    # === 核心修改 4：移除引发顿挫的 0.0 死逻辑，恢复分段平滑限速 ===
                    if zone_x == "NEAR":
                        px_eff = self.P_X * 0.4
                        step_x_eff = self.STEP_X_PER_FRAME * 0.4
                    elif zone_x == "MID":
                        px_eff = self.P_X * 0.7
                        step_x_eff = self.STEP_X_PER_FRAME * 0.7
                    else:
                        px_eff = self.P_X
                        step_x_eff = self.STEP_X_PER_FRAME

                    target_horiz = self.servo_horiz_cmd + err_x * px_eff
                    target_horiz = self.clamp(target_horiz, 0, 180)
                    self.servo_horiz_cmd, delta_x = self.move_towards(
                        self.servo_horiz_cmd, target_horiz, step_x_eff
                    )
                else:
                    target_horiz = self.servo_horiz_cmd

                # 垂直
                if abs(err_y) < self.DEAD_ZONE_Y:
                    target_vert = self.servo_vert_cmd
                    self.y_exceed_counter = 0
                else:
                    self.y_exceed_counter += 1
                    zone_y = self.get_axis_zone_y(err_y, self.center_locked)

                    if self.y_exceed_counter >= self.Y_HYSTERESIS_FRAMES:
                        if zone_y == "NEAR":
                            py_eff = self.P_Y * 0.4
                        elif zone_y == "MID":
                            py_eff = self.P_Y * 0.7
                        else:
                            py_eff = self.P_Y

                        if abs(err_y) < self.DEAD_ZONE_Y * 2:
                            py_eff *= 0.6

                        if self.reverse_y:
                            target_vert = self.servo_vert_cmd - err_y * py_eff
                        else:
                            target_vert = self.servo_vert_cmd + err_y * py_eff

                        target_vert = self.clamp(target_vert, 0, 140)
                    else:
                        target_vert = self.servo_vert_cmd

                    self.servo_vert_cmd, delta_y = self.move_towards(
                        self.servo_vert_cmd, target_vert, self.STEP_Y_PER_FRAME
                    )

                self.servo_horiz_cmd = self.clamp(self.servo_horiz_cmd, 0, 180)
                self.servo_vert_cmd = self.clamp(self.servo_vert_cmd, 0, 140)

            if zone_x not in ("LOCK", "HOLD"):
                zone_x = self.get_axis_zone_x(err_x, self.center_locked)
            if zone_y not in ("LOCK", "HOLD"):
                zone_y = self.get_axis_zone_y(err_y, self.center_locked)

            confirm_h = self.get_confirm_frames_h(err_x, self.center_locked)
            confirm_v = self.get_confirm_frames_v(err_y, self.center_locked)

        # 串口发送
        if self.ser and is_tracking:
            if self.center_locked or self.jump_hold_counter > 0:
                quant_h = self.send_horiz
                quant_v = self.send_vert
                self.pending_h = None
                self.pending_v = None
                self.pending_h_count = 0
                self.pending_v_count = 0
                self.stable_h_target = None
                self.stable_h_target_count = 0
            else:
                quant_h = self.quantize_with_hysteresis(
                    self.servo_horiz_cmd, self.send_horiz,
                    step=self.SERVO_STEP_ANGLE,
                    hysteresis=self.QUANT_HYSTERESIS_H,
                    min_angle=0.0, max_angle=180.0
                )
                quant_v = self.quantize_with_hysteresis(
                    self.servo_vert_cmd, self.send_vert,
                    step=self.SERVO_STEP_ANGLE,
                    hysteresis=self.QUANT_HYSTERESIS_V,
                    min_angle=0.0, max_angle=140.0
                )

            if not self.center_locked and self.jump_hold_counter == 0:
                if quant_h != self.send_horiz:
                    if self.stable_h_target == quant_h:
                        self.stable_h_target_count += 1
                    else:
                        self.stable_h_target = quant_h
                        self.stable_h_target_count = 1

                    # 修复针对浮点数的比较逻辑
                    if abs(quant_h - self.send_horiz) <= self.SERVO_STEP_ANGLE * 1.1:
                        if abs(err_x) < self.H_CROSS_STEP_MIN_ERR:
                            quant_h = self.send_horiz
                        elif self.stable_h_target_count < (confirm_h * self.H_SINGLE_STEP_EXTRA_CONFIRM):
                            quant_h = self.send_horiz
                else:
                    self.stable_h_target = None
                    self.stable_h_target_count = 0

            if quant_h != self.send_horiz:
                if self.pending_h == quant_h:
                    self.pending_h_count += 1
                else:
                    self.pending_h = quant_h
                    self.pending_h_count = 1
            else:
                self.pending_h = None
                self.pending_h_count = 0

            if quant_v != self.send_vert:
                if self.pending_v == quant_v:
                    self.pending_v_count += 1
                else:
                    self.pending_v = quant_v
                    self.pending_v_count = 1
            else:
                self.pending_v = None
                self.pending_v_count = 0

            send_h_now = self.pending_h is not None and self.pending_h_count >= confirm_h
            send_v_now = self.pending_v is not None and self.pending_v_count >= confirm_v

            if send_h_now and send_v_now:
                if abs(err_x) >= abs(err_y):
                    send_v_now = False
                else:
                    send_h_now = False

            if send_h_now:
                self.send_horiz = self.pending_h
                self.pending_h = None
                self.pending_h_count = 0
                self.stable_h_target = None
                self.stable_h_target_count = 0

            if send_v_now:
                self.send_vert = self.pending_v
                self.pending_v = None
                self.pending_v_count = 0

            if send_h_now or send_v_now:
                self._enqueue_latest_command(self.send_horiz, self.send_vert)

        return {
            "tracking": is_tracking,
            "current_has_face": current_has_face,
            "err_x": err_x,
            "err_y": err_y,
            "send_horiz": self.send_horiz,
            "send_vert": self.send_vert,
            "servo_horiz_cmd": self.servo_horiz_cmd,
            "servo_vert_cmd": self.servo_vert_cmd,
            "center_locked": self.center_locked,
            "jump_hold_counter": self.jump_hold_counter,
            "track_id": self.current_track_id,
        }

    def draw_debug(self, frame, current_has_face, err_x, err_y, zone_x, zone_y, confirm_h, confirm_v):
        cv2.circle(frame, (self.center_x, self.center_y), 5, (0, 0, 255), -1)
        cv2.rectangle(
            frame,
            (self.center_x - self.DEAD_ZONE_X, self.center_y - self.DEAD_ZONE_Y),
            (self.center_x + self.DEAD_ZONE_X, self.center_y + self.DEAD_ZONE_Y),
            (255, 0, 0), 2
        )

        if self.center_locked:
            cv2.putText(frame, "LOCKED", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.jump_hold_counter > 0:
            cv2.putText(frame, "HOLD", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        elif current_has_face:
            cv2.putText(frame, "TRACKING", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "NO FACE", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Hcmd:{self.servo_horiz_cmd:5.1f} Hout:{self.send_horiz:5.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, f"Vcmd:{self.servo_vert_cmd:5.1f} Vout:{self.send_vert:5.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, f"ZoneX:{zone_x} Cx:{confirm_h}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(frame, f"ZoneY:{zone_y} Cy:{confirm_v}",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.putText(frame, f"JumpHold:{self.jump_hold_counter}",
                    (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 120), 2)
        cv2.putText(frame, f"Err:({err_x:+d},{err_y:+d})",
                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 255), 2)