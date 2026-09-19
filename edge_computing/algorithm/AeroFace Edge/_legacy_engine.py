# coding: utf-8
import os
import cv2
import re
import json
import time
import base64
import pickle
import signal
import queue
import asyncio
import logging
import socket
import threading
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from servo import ServoFaceTracker
from app.config import (
    CAMERA_DEVICES, CAMERA_FOURCC, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH,
    DETECT_EVERY_N, ENABLE_FFMPEG_STREAM, ENABLE_INFER_RESIZE,
    ENABLE_LOCAL_PREVIEW, ENABLE_SERVO, ENABLE_SMALL_FACE_X2,
    ENABLE_UNKNOWN_ALERT, FEATURE_DB_PATH, FFMPEG_BIN, FUSION_MAX_FEATURES,
    INFER_LONG_SIDE, LOW_CONF_REPORT_COOLDOWN_SEC, MAX_CONNECTION_ATTEMPTS,
    MAX_REPORT_QUEUE, MIN_DET_SCORE_FOR_RECOG, MIN_DET_SCORE_FOR_SHOW,
    Model_MODEL, MONITOR_TOKEN, RECOGNIZE_COOLDOWN, RECOGNIZE_GROWTH_RATIO,
    RECOGNIZE_MIN_FACE, RECOGNIZE_MIN_STABLE, RECOGNIZE_SHARP_DELTA,
    REPORT_COOLDOWN_SEC, SCRFD_MODEL, SERVER_IP, SMALL_FACE_MAX_SIZE,
    SMALL_FACE_MIN_SHARPNESS, SMALL_FACE_MIN_SIZE, SMALL_FACE_SCALE,
    STREAM_BITRATE, STREAM_BUFSIZE, STREAM_FPS, STREAM_GOP, STREAM_HEIGHT,
    STREAM_MAXRATE, STREAM_PRESET, STREAM_URL, STREAM_USE_ANNOTATED,
    STREAM_WIDTH, TRAIN_IMG_DIR, TRAIN_TOKEN, TRAIN_WS_PORT,
    UNKNOWN_ALERT_FRAMES, UNKNOWN_TRACK_TTL, WEBSOCKET_PORT,
)

import aidlite
import numpy as np
import websockets

try:
    import psutil
except Exception:
    psutil = None


# =========================
# 全局对象（多线程共享）
# =========================
ws_queue = queue.Queue(maxsize=MAX_REPORT_QUEUE)
jpeg_encode_queue = queue.Queue(maxsize=10)

is_connected = False
train_is_connected = False
connection_attempts = 0
train_connection_attempts = 0
is_recognition_paused = False
SERVO_TRACKER = None

# 线程锁
feature_db_lock = threading.Lock()
model_lock = threading.Lock()
recognizer_lock = threading.Lock()
streamer_lock = threading.Lock()

logger = logging.getLogger("drone_face_engine")

# =========================
# 工具函数
# =========================
def resize_keep_ratio(img, long_side):
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img
    scale = long_side / float(max(h, w))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def format_track_display_text(item: Dict[str, Any]) -> str:
    tid = int(item.get("track_id", -1))
    state = str(item.get("state", "unknown"))
    name = str(item.get("name", "Unknown"))
    candidate = str(item.get("candidate", "Unknown"))
    low_conf_reason = str(item.get("low_conf_reason", ""))
    sim = float(item.get("sim_score", item.get("score", 0.0)))

    if state == "confirmed" and name != "Unknown":
        return f"ID:{tid} {name}:{sim:.2f}"
    elif state == "tentative" and candidate != "Unknown":
        return f"ID:{tid} tentative:{candidate}:{sim:.2f}"
    elif state == "low_confidence":
        reason = low_conf_reason or "unreliable"
        return f"ID:{tid} low_conf:{reason}"
    else:
        return f"ID:{tid} unknown:{sim:.2f}"

def rotate_image(img, angle):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def bind_current_thread_to_core(core_id: int):
    return 

def signal_handler(sig, frame):
    global ENGINE
    logger.info("🛑 程序终止")
    if ENABLE_LOCAL_PREVIEW:
        cv2.destroyAllWindows()
    raise SystemExit(0)

def is_server_reachable(ip, port):
    try:
        with socket.create_connection((ip, port), timeout=2):
            return True
    except Exception:
        return False

def l2_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x) + 1e-12
    return x / n

def variance_of_laplacian(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clip_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    return x1, y1, x2, y2

def expand_square_bbox(x1, y1, x2, y2, img_w, img_h, scale=1.10):
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * scale
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))
    return clip_box(nx1, ny1, nx2, ny2, img_w, img_h)

def _crop_with_margin(frame: np.ndarray, bbox, margin=0.20):
    if frame is None or frame.size == 0:
        return None, (0, 0, 0, 0)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * float(margin)
    my = bh * float(margin)
    cx1 = int(max(0, np.floor(x1 - mx)))
    cy1 = int(max(0, np.floor(y1 - my)))
    cx2 = int(min(w, np.ceil(x2 + mx)))
    cy2 = int(min(h, np.ceil(y2 + my)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, (0, 0, 0, 0)
    roi = frame[cy1:cy2, cx1:cx2]
    return roi, (cx1, cy1, cx2, cy2)

def _kps_to_crop_coords(kps: np.ndarray, crop_box):
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    x1, y1, _x2, _y2 = crop_box
    out = kps.copy()
    out[:, 0] -= float(x1)
    out[:, 1] -= float(y1)
    return out

def enhance_small_face_roi(roi: Optional[np.ndarray], kps_roi: Optional[np.ndarray]):
    if not ENABLE_SMALL_FACE_X2 or roi is None or roi.size == 0 or kps_roi is None:
        return roi, kps_roi, False, 0.0, 0.0
    try:
        kps_roi = np.asarray(kps_roi, dtype=np.float32).reshape(5, 2)
    except Exception:
        return roi, kps_roi, False, 0.0, 0.0
    h, w = roi.shape[:2]
    face_size = float(min(w, h))
    sharpness = variance_of_laplacian(roi)
    should_enhance = (
        SMALL_FACE_MIN_SIZE <= face_size <= SMALL_FACE_MAX_SIZE and
        sharpness >= SMALL_FACE_MIN_SHARPNESS
    )
    if not should_enhance:
        return roi, kps_roi, False, face_size, sharpness
    scale = float(SMALL_FACE_SCALE)
    enhanced = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, enhanced, -0.5, 0)
    enhanced_kps = kps_roi * scale
    return enhanced, enhanced_kps, True, face_size, sharpness

def prepare_inference_frame(frame: np.ndarray):
    if frame is None or not ENABLE_INFER_RESIZE:
        return frame, 1.0, 1.0
    h, w = frame.shape[:2]
    long_side = max(h, w)
    target = int(max(64, INFER_LONG_SIDE))
    if long_side <= target:
        return frame, 1.0, 1.0
    scale = target / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    scale_x = w / float(nw)
    scale_y = h / float(nh)
    return resized, scale_x, scale_y

def remap_detections_to_original(dets, scale_x: float, scale_y: float, dst_w: int, dst_h: int):
    if not dets or (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6):
        return dets
    mapped = []
    for det in dets:
        new_det = dict(det)
        bbox = np.asarray(det["bbox"], dtype=np.float32).copy()
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
        bbox[0] = np.clip(bbox[0], 0, dst_w - 1)
        bbox[1] = np.clip(bbox[1], 0, dst_h - 1)
        bbox[2] = np.clip(bbox[2], 0, dst_w - 1)
        bbox[3] = np.clip(bbox[3], 0, dst_h - 1)
        new_det["bbox"] = bbox
        kps = det.get("kps")
        if kps is not None:
            kps = np.asarray(kps, dtype=np.float32).copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            kps[:, 0] = np.clip(kps[:, 0], 0, dst_w - 1)
            kps[:, 1] = np.clip(kps[:, 1], 0, dst_h - 1)
            new_det["kps"] = kps
        mapped.append(new_det)
    return mapped

def draw_annotated_frame(frame, items):
    vis = frame.copy()
    for item in items:
        x1, y1, x2, y2 = map(int, item["bbox"])
        state = str(item.get("state", "unknown"))
        name = str(item.get("name", "Unknown"))
        if state == "confirmed" and name != "Unknown":
            color = (0, 255, 0)
        elif state == "tentative":
            color = (0, 255, 255)
        elif state == "low_confidence":
            color = (255, 165, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return vis

def show_local_preview(frame, items):
    if not ENABLE_LOCAL_PREVIEW:
        return
    vis = draw_annotated_frame(frame, items)
    cv2.imshow("Drone Face Engine", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        logger.info("🛑 用户通过ESC退出")
        cv2.destroyAllWindows()
        os._exit(0)

def norm_crop_face(img, kps, image_size=112):
    src = np.array([
        [38.2946, 53.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    if image_size != 112:
        src = src * (image_size / 112.0)
    kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(kps, src, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return aligned

def _estimate_specular_ratio_from_roi(roi: np.ndarray, kps_roi: Optional[np.ndarray]) -> float:
    if roi is None or roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    bright_mask = (v >= 245)
    low_sat_mask = (s <= 40)
    spec_mask = bright_mask & low_sat_mask
    spec_ratio = float(np.mean(spec_mask))
    b, g, r = cv2.split(roi)
    near_white = (r >= 245) & (g >= 245) & (b >= 245)
    white_ratio = float(np.mean(near_white))
    out = 0.65 * spec_ratio + 0.35 * white_ratio
    return float(np.clip(out, 0.0, 1.0))

def _estimate_occlusion_ratio_from_roi(roi: np.ndarray, kps_roi: Optional[np.ndarray]) -> float:
    if roi is None or roi.size == 0 or kps_roi is None:
        return 1.0
    kps_roi = np.asarray(kps_roi, dtype=np.float32)
    if kps_roi.shape != (5, 2):
        return 1.0
    h, _w = roi.shape[:2]
    _le, _re, nose, lm, rm = kps_roi
    y_top = int(max(0, min(nose[1], h - 1)))
    y_bottom = int(min(h, max(lm[1], rm[1]) + 0.18 * h))
    if y_bottom <= y_top:
        return 0.0
    region = roi[y_top:y_bottom, :]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 120)
    edge_ratio = float(np.mean(edge > 0))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    smooth_score = 1.0 / (1.0 + blur_var / 40.0)
    occ_ratio = 0.55 * smooth_score + 0.45 * max(0.0, 0.12 - edge_ratio) / 0.12
    return float(np.clip(occ_ratio, 0.0, 1.0))

def _estimate_face_pose_and_visibility(tlbr: np.ndarray, kps_full: Optional[np.ndarray], roi: Optional[np.ndarray] = None, kps_roi: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ret = {
        "valid": False,
        "visible_points": 0,
        "yaw_score": 0.0,
        "pitch_score": 0.0,
        "roll_score": 0.0,
        "occlusion_score": 0.0,
        "specular_score": 0.0,
        "mask_like": False,
        "is_profile_like": False,
        "is_heavy_occlusion": False,
        "is_low_confidence": False,
        "reason": "unknown",
    }
    if kps_full is None:
        ret["reason"] = "no_kps"
        ret["is_low_confidence"] = True
        return ret
    kps_full = np.asarray(kps_full, dtype=np.float32)
    if kps_full.shape != (5, 2):
        ret["reason"] = "bad_kps_shape"
        ret["is_low_confidence"] = True
        return ret
    x1, y1, x2, y2 = map(float, tlbr)
    le, re, nose, lm, rm = kps_full
    visible = 0
    for p in [le, re, nose, lm, rm]:
        if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
            visible += 1
    ret["visible_points"] = visible
    eye_mid = 0.5 * (le + re)
    mouth_mid = 0.5 * (lm + rm)
    eye_dist = max(np.linalg.norm(re - le), 1e-6)
    mouth_dist = max(np.linalg.norm(rm - lm), 1e-6)
    nose_eye_offset_x = abs((nose[0] - eye_mid[0]) / eye_dist)
    nose_mouth_offset_x = abs((nose[0] - mouth_mid[0]) / mouth_dist)
    yaw_score = 0.5 * (nose_eye_offset_x + nose_mouth_offset_x)
    roll_score = abs(re[1] - le[1]) / eye_dist
    eye_to_mouth = max(mouth_mid[1] - eye_mid[1], 1e-6)
    nose_ratio = (nose[1] - eye_mid[1]) / eye_to_mouth
    pitch_score = abs(nose_ratio - 0.5)
    mouth_asym = abs(np.linalg.norm(nose - lm) - np.linalg.norm(nose - rm)) / max(mouth_dist, 1e-6)
    eye_asym = abs(np.linalg.norm(nose - le) - np.linalg.norm(nose - re)) / max(eye_dist, 1e-6)
    occ_proxy = 0.35 * (5 - visible) / 5.0 + 0.35 * mouth_asym + 0.30 * eye_asym
    roi_occ = 0.0
    if roi is not None and kps_roi is not None:
        roi_occ = _estimate_occlusion_ratio_from_roi(roi, kps_roi)
    specular_score = 0.0
    if roi is not None and roi.size > 0:
        specular_score = _estimate_specular_ratio_from_roi(roi, kps_roi)
    occlusion_score = 0.65 * occ_proxy + 0.35 * roi_occ
    if specular_score > 0.015:
        reduce_ratio = min(0.18, 0.55 * specular_score)
        occlusion_score = max(0.0, occlusion_score - reduce_ratio)
    ret["valid"] = True
    ret["yaw_score"] = float(yaw_score)
    ret["pitch_score"] = float(pitch_score)
    ret["roll_score"] = float(roll_score)
    ret["occlusion_score"] = float(np.clip(occlusion_score, 0.0, 1.0))
    ret["specular_score"] = float(np.clip(specular_score, 0.0, 1.0))
    ret["mask_like"] = bool((roi_occ > 0.36) and (specular_score < 0.03))
    ret["is_profile_like"] = bool(yaw_score > 0.38)
    ret["is_heavy_occlusion"] = bool((visible < 4) or (occlusion_score > 0.42))
    if visible < 4:
        ret["is_low_confidence"] = True
        ret["reason"] = "few_kps"
    elif yaw_score > 0.45:
        ret["is_low_confidence"] = True
        ret["reason"] = "profile_like"
    elif occlusion_score > 0.50:
        ret["is_low_confidence"] = True
        ret["reason"] = "heavy_occlusion"
    elif roll_score > 0.32:
        ret["is_low_confidence"] = True
        ret["reason"] = "large_roll"
    elif pitch_score > 0.36:
        ret["is_low_confidence"] = True
        ret["reason"] = "large_pitch"
    return ret

# =========================
# FFmpeg 推流器
# =========================
class FFmpegStreamer:
    def __init__(self, url: str, width: int, height: int, fps: int):
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.proc = None
        self.running = False
        self.thread = None
        self.monitor_thread = None
        self.last_push_ts = 0.0
        self.frame_interval = 1.0 / float(self.fps)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.restart_event = threading.Event()

    def _build_cmd(self):
        threads = max(1, min(os.cpu_count() or 1, 4))
        gop = max(1, int(STREAM_GOP))
        return [
            FFMPEG_BIN,
            "-loglevel", "error",
            "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", STREAM_PRESET,
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-bf", "0",
            "-refs", "1",
            "-threads", str(threads),
            "-x264-params", "rc-lookahead=0:sync-lookahead=0:sliced-threads=1:force-cfr=0",
            "-b:v", STREAM_BITRATE,
            "-maxrate", STREAM_MAXRATE,
            "-bufsize", STREAM_BUFSIZE,
            "-flush_packets", "1",
            "-f", "flv",
            self.url,
        ]

    def _start_process(self):
        try:
            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self.last_push_ts = 0.0
        except Exception as e:
            logger.error(f"❌ FFmpeg 进程启动失败: {e}")
            self.proc = None

    def start(self):
        if self.running:
            return
        self._start_process()
        if self.proc is None:
            return
        self.running = True
        self.restart_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        logger.info(f"✅ FFmpeg 推流已启动: {self.url}")

    def _monitor(self):
        while self.running:
            time.sleep(1.0)
            if self.proc is None:
                continue
            if self.proc.poll() is not None:
                logger.warning("⚠️ FFmpeg 进程已退出，尝试重启...")
                self.restart_event.set()
                time.sleep(0.2)
                with self.lock:
                    self.latest_frame = None
                self._start_process()
                if self.proc is not None:
                    logger.info("✅ FFmpeg 进程已重启")
                    self.restart_event.clear()
                    if self.thread is not None and not self.thread.is_alive():
                        self.thread = threading.Thread(target=self._worker, daemon=True)
                        self.thread.start()
                else:
                    logger.error("❌ FFmpeg 进程重启失败")

    def stop(self):
        self.running = False
        with self.lock:
            self.latest_frame = None
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
        logger.info("🛑 FFmpeg 推流已停止")

    def push_frame(self, frame: np.ndarray):
        if not self.running or self.proc is None or frame is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        with self.lock:
            self.latest_frame = frame

    def _worker(self):
        frame_interval = 1.0 / float(self.fps)
        while self.running and not self.restart_event.is_set():
            start_time = time.time()
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    
            if frame is not None and self.proc is not None and self.proc.stdin is not None:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.error("❌ FFmpeg 管道断开，触发重启")
                    self.restart_event.set()
                    break
                except Exception as e:
                    logger.error(f"❌ FFmpeg 写帧失败: {e}")
                    self.restart_event.set()
                    break

            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        logger.info("🛑 推流 _worker 线程退出")

# =========================
# AIDLite: SCRFD
# =========================
class SCRFD_AIDLite:
    def __init__(self, tflite_path, confThreshold=0.20, nmsThreshold=0.4):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = 640
        self.inpHeight = 640
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

        self.model = self._init_model(tflite_path)
        self.config = self._init_config()
        self.interpreter = self._build_interpreter()
        self._anchor_cache = {s: self._make_anchors(s) for s in self._feat_stride_fpn}

    def _init_model(self, tflite_path):
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"SCRFD 模型不存在: {tflite_path}")
        model = aidlite.Model.create_instance(tflite_path)
        if model is None:
            raise RuntimeError("SCRFD 模型实例创建失败")
        return model

    def _init_config(self):
        config = aidlite.Config.create_instance()
        config.framework_type = aidlite.FrameworkType.TYPE_TFLITE
        config.accelerate_type = aidlite.AccelerateType.TYPE_GPU
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.number_of_threads = 4
        config.is_quantify_model = 0
        config.fast_timeout = -1
        return config

    def _build_interpreter(self):
        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(self.model, self.config)
        if interpreter.init() != 0:
            raise RuntimeError("SCRFD 解释器初始化失败")
        self.model.set_model_properties(
            input_shapes=[[1, self.inpHeight, self.inpWidth, 3]],
            input_data_type=aidlite.DataType.TYPE_FLOAT32,
            output_shapes=[],
            output_data_type=aidlite.DataType.TYPE_FLOAT32,
        )
        if interpreter.load_model() != 0:
            raise RuntimeError("SCRFD 模型加载失败")
        return interpreter

    def _make_anchors(self, stride):
        height = self.inpHeight // stride
        width = self.inpWidth // stride
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = np.repeat(anchor_centers, self._num_anchors, axis=0)
        return anchor_centers.astype(np.float32)

    def _dedup_by_center(self, dets, center_thr_ratio=0.18, size_thr_ratio=0.45):
        if not dets:
            return []
        kept = []
        dets = sorted(dets, key=lambda x: float(x["score"]), reverse=True)
        for d in dets:
            box = np.asarray(d["bbox"], dtype=np.float32)
            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            duplicated = False
            for k in kept:
                kbox = np.asarray(k["bbox"], dtype=np.float32)
                kx1, ky1, kx2, ky2 = kbox
                kbw = max(1.0, kx2 - kx1)
                kbh = max(1.0, ky2 - ky1)
                kcx = 0.5 * (kx1 + kx2)
                kcy = 0.5 * (ky1 + ky2)
                center_dist = np.hypot(cx - kcx, cy - kcy)
                ref_size = max(1.0, 0.5 * (max(bw, bh) + max(kbw, kbh)))
                size_gap = abs(max(bw, bh) - max(kbw, kbh)) / ref_size
                if center_dist < ref_size * center_thr_ratio and size_gap < size_thr_ratio:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(d)
        return kept

    def preprocess_frame(self, frame):
        src_h, src_w = frame.shape[:2]
        scale = min(self.inpWidth / src_w, self.inpHeight / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.inpWidth - new_w) // 2
        pad_h = (self.inpHeight - new_h) // 2
        img_padded = np.zeros((self.inpHeight, self.inpWidth, 3), dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_float = (img_rgb - 127.5) / 128.0
        img_input = np.expand_dims(img_float, axis=0).astype(np.float32)
        return img_input, scale, pad_w, pad_h, src_w, src_h

    @staticmethod
    def distance2bbox(anchors, offsets):
        x1 = anchors[:, 0] - offsets[:, 0]
        y1 = anchors[:, 1] - offsets[:, 1]
        x2 = anchors[:, 0] + offsets[:, 2]
        y2 = anchors[:, 1] + offsets[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def distance2kps(anchors, offsets):
        kps = []
        for i in range(0, min(10, offsets.shape[1]), 2):
            px = anchors[:, 0] + offsets[:, i]
            py = anchors[:, 1] + offsets[:, i + 1]
            kps.append(px)
            kps.append(py)
        return np.stack(kps, axis=-1).reshape(-1, 5, 2)

    def postprocess(self, outputs, scale, pad_w, pad_h, src_w, src_h):
        all_scores, all_bboxes, all_kpss = [], [], []
        output_map = [
            (8, outputs[7], outputs[2], outputs[1]),
            (16, outputs[6], outputs[5], outputs[4]),
            (32, outputs[0], outputs[8], outputs[3]),
        ]
        for stride, score_out, box_out, kps_out in output_map:
            anchors = self._anchor_cache[stride]
            scores = score_out.reshape(-1, 1).astype(np.float32)
            bbox_preds = box_out.reshape(-1, 4).astype(np.float32) * stride
            kps_preds = kps_out.reshape(-1, 10).astype(np.float32) * stride
            if not (len(anchors) == len(scores) == len(bbox_preds) == len(kps_preds)):
                continue
            pos_inds = np.where(scores[:, 0] >= self.confThreshold)[0]
            if len(pos_inds) == 0:
                continue
            bboxes = self.distance2bbox(anchors, bbox_preds)
            kpss = self.distance2kps(anchors, kps_preds)
            all_scores.append(scores[pos_inds])
            all_bboxes.append(bboxes[pos_inds])
            all_kpss.append(kpss[pos_inds])
        if not all_scores:
            return None
        scores = np.vstack(all_scores).ravel()
        bboxes = np.vstack(all_bboxes)
        kpss = np.vstack(all_kpss)
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_w) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_h) / scale
        kpss[:, :, 0] = (kpss[:, :, 0] - pad_w) / scale
        kpss[:, :, 1] = (kpss[:, :, 1] - pad_h) / scale
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, src_w - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, src_h - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, src_w - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, src_h - 1)
        boxes_xywh = bboxes.copy()
        boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
        boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh[:, :4].tolist(),
            scores.tolist(),
            self.confThreshold,
            self.nmsThreshold
        )
        if len(indices) == 0:
            return None
        keep = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        return bboxes[keep], scores[keep], kpss[keep]

    def detect(self, frame, fast_mode=True):
        all_dets = []
        if fast_mode:
            plan = [(1280, 0.22)]
        else:
            plan = [(1280, 0.22), (768, 0.28), (640, 0.24)]
        for long_side, thr in plan:
            test_img, scale = self._resize_for_detect(frame, long_side)
            inv_scale = 1.0 / scale
            dets = self.detect_single_scale(test_img, conf_threshold=thr)
            if dets:
                dets = self._map_dets_back(dets, inv_scale)
                all_dets.extend(dets)
        filtered = []
        for d in all_dets:
            x1, y1, x2, y2 = d["bbox"]
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                filtered.append(d)
        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
        filtered = self._dedup_dets(filtered, iou_thr=0.35)
        filtered = self._dedup_by_center(filtered, center_thr_ratio=0.16, size_thr_ratio=0.38)
        return filtered

    def detect_with_threshold(self, frame, conf_threshold=None):
        old_thr = self.confThreshold
        if conf_threshold is not None:
            self.confThreshold = conf_threshold
        try:
            return self.detect(frame)
        finally:
            self.confThreshold = old_thr

    def detect_single_scale(self, frame, conf_threshold=None):
        old_thr = self.confThreshold
        if conf_threshold is not None:
            self.confThreshold = conf_threshold
        try:
            img_input, scale, pad_w, pad_h, src_w, src_h = self.preprocess_frame(frame)
            self.interpreter.set_input_tensor(0, img_input)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_output_tensor(i) for i in range(9)]
            result = self.postprocess(outputs, scale, pad_w, pad_h, src_w, src_h)
            if result is None:
                return []
            bboxes, scores, kpss = result
            dets = []
            for box, score, kps in zip(bboxes, scores, kpss):
                dets.append({
                    "bbox": box.astype(np.float32),
                    "score": float(score),
                    "kps": kps.astype(np.float32),
                })
            return dets
        finally:
            self.confThreshold = old_thr

    @staticmethod
    def _resize_for_detect(frame, long_side):
        h, w = frame.shape[:2]
        scale = long_side / float(max(h, w))
        if abs(scale - 1.0) < 1e-6:
            return frame.copy(), 1.0
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    @staticmethod
    def _map_dets_back(dets, inv_scale):
        mapped = []
        for d in dets:
            box = d["bbox"].copy() * inv_scale
            kps = d["kps"].copy() * inv_scale
            mapped.append({
                "bbox": box.astype(np.float32),
                "score": float(d["score"]),
                "kps": kps.astype(np.float32),
            })
        return mapped

    def _dedup_dets(self, dets, iou_thr=0.4):
        if not dets:
            return []
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        scores = np.array([d["score"] for d in dets], dtype=np.float32)
        xywh = boxes.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        idxs = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            score_threshold=0.01,
            nms_threshold=iou_thr
        )
        if len(idxs) == 0:
            return []
        keep = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in idxs]
        return [dets[i] for i in keep]

# =========================
# AIDLite: 
# =========================
class ModelAIDLite:
    def __init__(self, tflite_path, use_gpu=True):
        self.model_path = os.path.abspath(tflite_path)
        self.input_shape = [1, 3, 112, 112]
        self.output_shapes = [[1, 512], [1, 1]]
        self.face_size = 112
        self.model = self._create_model()
        self.config = self._create_config(use_gpu)
        self.interpreter = self._build_interpreter()

    def _create_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model 模型不存在: {self.model_path}")
        model = aidlite.Model.create_instance(self.model_path)
        if model is None:
            raise RuntimeError("Model Model.create_instance 失败")
        return model

    def _create_config(self, use_gpu):
        config = aidlite.Config.create_instance()
        config.framework_type = aidlite.FrameworkType.TYPE_TFLITE
        config.accelerate_type = (
            aidlite.AccelerateType.TYPE_GPU
            if use_gpu else aidlite.AccelerateType.TYPE_CPU
        )
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.number_of_threads = 4
        config.is_quantify_model = 0
        config.fast_timeout = -1
        return config

    def _build_interpreter(self):
        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(
            self.model, self.config
        )
        if interpreter.init() != 0:
            raise RuntimeError("Model interpreter.init() 失败")
        self.model.set_model_properties(
            input_shapes=[self.input_shape],
            input_data_type=aidlite.DataType.TYPE_FLOAT32,
            output_shapes=self.output_shapes,
            output_data_type=aidlite.DataType.TYPE_FLOAT32,
        )
        if interpreter.load_model() != 0:
            raise RuntimeError("Model interpreter.load_model() 失败")
        return interpreter

    def _prepare_face_input(self, face_bgr: np.ndarray):
        if face_bgr is None or face_bgr.size == 0:
            return None
        face = cv2.resize(face_bgr, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = face.transpose(2, 0, 1)
        face = np.expand_dims(face, axis=0).astype(np.float32)
        return face

    def _crop_face_by_bbox(self, frame: np.ndarray, bbox, scale=1.10):
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1, x2, y2 = expand_square_bbox(x1, y1, x2, y2, w, h, scale=scale)
        face = frame[y1:y2, x1:x2]
        if face is None or face.size == 0:
            return None
        return face

    def infer_feature(self, face_input: np.ndarray):
        self.interpreter.set_input_tensor(0, face_input)
        self.interpreter.invoke()
        out0 = np.array(self.interpreter.get_output_tensor(0))
        out1 = np.array(self.interpreter.get_output_tensor(1))
        feature = l2_norm(out0)
        aux_score = float(out1.reshape(-1)[0])
        return aux_score, feature

    def extract(self, frame: np.ndarray, bbox):
        face_bgr = self._crop_face_by_bbox(frame, bbox, scale=1.10)
        if face_bgr is None:
            return None, None
        face_input = self._prepare_face_input(face_bgr)
        if face_input is None:
            return None, None
        return self.infer_feature(face_input)

    def extract_from_aligned_crop(self, face_bgr: np.ndarray):
        face_input = self._prepare_face_input(face_bgr)
        if face_input is None:
            return None, None
        return self.infer_feature(face_input)

# =========================
# 跟踪器
# =========================
try:
    from tracker.byte_tracker import BYTETracker as _RealBYTETracker
    HAS_REAL_BYTETRACK = True
    logger.info("✅ 成功导入ByteTrack")
except Exception as e:
    _RealBYTETracker = None
    HAS_REAL_BYTETRACK = False
    logger.exception(f"❌ 导入 ByteTrack 失败: {e}")

@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray
    score: float
    kps: Optional[np.ndarray] = None
    kps_frame_idx: int = -1
    age: int = 1
    hits: int = 1
    lost: int = 0
    recognized_name: str = "Unknown"
    best_bbox_size: float = 0.0
    best_sharpness: float = 0.0
    last_recognize_frame: int = -9999
    first_seen_frame: int = 0
    stable_frames: int = 1
    feature_history: List[np.ndarray] = field(default_factory=list)
    feature_bank: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    last_similarity: float = 0.0
    aux_score: float = 0.0
    cv_tracker: object = None
    first_recognized_done: bool = False
    unknown_retry_count: int = 0
    last_trigger_reason: str = ""
    last_seen_frame: int = 0
    is_confirmed_visible: bool = True
    low_confidence: bool = False
    low_conf_reason: str = ""
    low_conf_hold_until: int = -999999

    def fused_feature(self):
        if self.feature_bank:
            feats = [x[0] for x in self.feature_bank]
            return l2_norm(np.mean(np.stack(feats, axis=0), axis=0))
        if self.feature_history:
            feats = np.stack(self.feature_history, axis=0)
            return l2_norm(np.mean(feats, axis=0))
        return None

class _ArgsLike:
    def __init__(self, track_thresh=0.25, track_buffer=15, match_thresh=0.7, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20

class RealByteTrackAdapter:
    def __init__(self, fps=30, track_thresh=0.25, track_buffer=15, match_thresh=0.7):
        if not HAS_REAL_BYTETRACK:
            raise RuntimeError("真实 ByteTrack 不可用")
        args = _ArgsLike(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )
        self.tracker = _RealBYTETracker(args, frame_rate=fps)
        self.track_states: Dict[int, TrackState] = {}

    @staticmethod
    def _iou(box1, box2):
        x11, y11, x12, y12 = map(float, box1)
        x21, y21, x22, y22 = map(float, box2)
        xa = max(x11, x21)
        ya = max(y11, y21)
        xb = min(x12, x22)
        yb = min(y12, y22)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
        a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
        union = a1 + a2 - inter + 1e-6
        return inter / union

    def _assign_dets_to_tracks(self, online_targets, detections):
        assigned = {}
        used_det_idx = set()
        target_boxes = []
        for t in online_targets:
            if hasattr(t, "tlwh"):
                x, y, w, h = t.tlwh
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            else:
                bbox = np.array(t[:4], dtype=np.float32)
            target_boxes.append(bbox)
        for ti, tb in enumerate(target_boxes):
            tx1, ty1, tx2, ty2 = tb
            tcx = 0.5 * (tx1 + tx2)
            tcy = 0.5 * (ty1 + ty2)
            tsize = max(1.0, max(tx2 - tx1, ty2 - ty1))
            best_di = -1
            best_score = -1e9
            for di, det in enumerate(detections):
                if di in used_det_idx:
                    continue
                db = np.asarray(det["bbox"], dtype=np.float32)
                dx1, dy1, dx2, dy2 = db
                dcx = 0.5 * (dx1 + dx2)
                dcy = 0.5 * (dy1 + dy2)
                iou = self._iou(tb, db)
                center_dist = np.hypot(tcx - dcx, tcy - dcy) / tsize
                score = 1.0 * iou - 0.35 * center_dist
                if score > best_score:
                    best_score = score
                    best_di = di
            if best_di >= 0 and best_score > -0.10:
                assigned[ti] = detections[best_di]
                used_det_idx.add(best_di)
        return assigned

    def update(self, detections: List[dict], frame_idx: int, frame_shape=None) -> List[TrackState]:
        if detections:
            det_arr = np.array([
                [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["score"]]
                for d in detections
            ], dtype=np.float32)
        else:
            det_arr = np.zeros((0, 5), dtype=np.float32)
        online_targets = self.tracker.update(det_arr, frame_shape, frame_shape)
        det_assignment = self._assign_dets_to_tracks(online_targets, detections)
        active_ids = set()
        results = []
        for ti, t in enumerate(online_targets):
            if hasattr(t, "tlwh"):
                x, y, w, h = t.tlwh
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            else:
                bbox = np.array(t[:4], dtype=np.float32)
            tid = int(t.track_id)
            score = float(getattr(t, "score", 0.0))
            active_ids.add(tid)
            matched_det = det_assignment.get(ti, None)
            matched_kps = matched_det.get("kps") if matched_det is not None else None
            matched_score = float(matched_det["score"]) if matched_det is not None else score
            if tid not in self.track_states:
                self.track_states[tid] = TrackState(
                    track_id=tid,
                    bbox=bbox.copy(),
                    score=matched_score,
                    kps=matched_kps.copy() if matched_kps is not None else None,
                    kps_frame_idx=frame_idx if matched_kps is not None else -1,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    is_confirmed_visible=True,
                    best_bbox_size=max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
            else:
                trk = self.track_states[tid]
                trk.bbox = bbox.copy()
                trk.score = matched_score
                trk.age += 1
                trk.hits += 1
                trk.lost = 0
                trk.stable_frames += 1
                trk.last_seen_frame = frame_idx
                trk.is_confirmed_visible = True
                if matched_kps is not None:
                    trk.kps = matched_kps.copy()
                    trk.kps_frame_idx = frame_idx
            results.append(self.track_states[tid])
        stale_ids = []
        for tid, st in self.track_states.items():
            if tid not in active_ids:
                st.age += 1
                st.lost += 1
                st.stable_frames = max(0, st.stable_frames - 1)
                st.is_confirmed_visible = False
                if st.lost > 1:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del self.track_states[tid]
        return results

class IOUFallbackTracker:
    def __init__(self, iou_threshold=0.25, max_lost=4):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def iou(box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        xa = max(x11, x21)
        ya = max(y11, y21)
        xb = min(x12, x22)
        yb = min(y12, y22)
        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = max(0, x12 - x11) * max(0, y12 - y11)
        a2 = max(0, x22 - x21) * max(0, y22 - y21)
        union = a1 + a2 - inter + 1e-6
        return inter / union

    def update(self, detections: List[dict], frame_idx: int, frame_shape=None) -> List[TrackState]:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        matches = []
        pairs = []
        for tid, trk in self.tracks.items():
            for di, det in enumerate(detections):
                i = self.iou(trk.bbox, det["bbox"])
                pairs.append((i, tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for iou_val, tid, di in pairs:
            if iou_val < self.iou_threshold:
                break
            if tid in unmatched_tracks and di in unmatched_dets:
                matches.append((tid, di))
                unmatched_tracks.remove(tid)
                unmatched_dets.remove(di)
        for tid, di in matches:
            det = detections[di]
            trk = self.tracks[tid]
            trk.bbox = det["bbox"].copy()
            trk.kps = det.get("kps")
            trk.kps_frame_idx = frame_idx if det.get("kps") is not None else trk.kps_frame_idx
            trk.score = float(det["score"])
            trk.age += 1
            trk.hits += 1
            trk.lost = 0
            trk.stable_frames += 1
            trk.last_seen_frame = frame_idx
            trk.is_confirmed_visible = True
        to_delete = []
        for tid in unmatched_tracks:
            trk = self.tracks[tid]
            trk.age += 1
            trk.lost += 1
            trk.stable_frames = max(0, trk.stable_frames - 1)
            trk.is_confirmed_visible = False
            if trk.lost > self.max_lost:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
        for di in unmatched_dets:
            det = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                bbox=det["bbox"].copy(),
                score=float(det["score"]),
                kps=det.get("kps"),
                kps_frame_idx=frame_idx if det.get("kps") is not None else -1,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                is_confirmed_visible=True,
                best_bbox_size=max(det["bbox"][2] - det["bbox"][0], det["bbox"][3] - det["bbox"][1]),
            )
        return list(self.tracks.values())

class TrackerCompensator:
    def __init__(self):
        self.enabled = hasattr(cv2, "legacy") or hasattr(cv2, "TrackerCSRT_create")

    def _create_tracker(self):
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
        except Exception:
            return None
        return None

    def init_for_track(self, track: TrackState, frame: np.ndarray):
        if not self.enabled:
            return
        trk = self._create_tracker()
        if trk is None:
            return
        x1, y1, x2, y2 = map(int, track.bbox)
        ok = trk.init(frame, (x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
        if ok:
            track.cv_tracker = trk

    def update_for_track(self, track: TrackState, frame: np.ndarray):
        if track.cv_tracker is None:
            return False
        ok, box = track.cv_tracker.update(frame)
        if not ok:
            track.cv_tracker = None
            return False
        x, y, w, h = box
        track.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        return True

# =========================
# 特征数据库
# =========================
class FeatureDatabase:
    def __init__(self, db_path=FEATURE_DB_PATH):
        self.db_path = db_path
        self.db = {}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.db = pickle.load(f)
                logger.info(f"✅ 加载特征库: {len(self.db)} 人")
            except Exception as e:
                logger.warning(f"⚠️ 加载特征库失败，使用空库: {e}")
                self.db = {}
        else:
            self.db = {}

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def add_feature(self, name: str, feature: np.ndarray):
        with feature_db_lock:
            name = re.sub(r"[^\w_.-]", "", name)
            if name not in self.db:
                self.db[name] = {"features": [], "updated_at": time.time()}
            self.db[name]["features"].append(l2_norm(feature).tolist())
            self.db[name]["features"] = self.db[name]["features"][-30:]
            self.db[name]["updated_at"] = time.time()
            self.save()
            logger.info(f"[DB] add_feature name={name} total={len(self.db[name]['features'])}")

    def delete_names(self, names: List[str]) -> int:
        deleted = 0
        with feature_db_lock:
            for name in names:
                if name in self.db:
                    del self.db[name]
                    deleted += 1
            self.save()
        return deleted

    def query_names(self):
        with feature_db_lock:
            return sorted(list(self.db.keys()))

    def compare(self, feature: np.ndarray, threshold=0.38) -> Tuple[str, float]:
        feature = l2_norm(feature)
        best_name = "Unknown"
        best_sim = -1.0
        sims_log = []
        with feature_db_lock:
            for name, item in self.db.items():
                feats = item.get("features", [])
                if not feats:
                    continue
                arr = np.array(feats, dtype=np.float32)
                sims = arr @ feature.reshape(-1, 1)
                sim = float(np.max(sims))
                sims_log.append((name, sim))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        sims_log.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[DB] compare top={sims_log[:5]} threshold={threshold}")
        if best_sim < threshold:
            return "Unknown", best_sim
        return best_name, best_sim

# =========================
# 训练管理
# =========================
class TrainingManager:
    def __init__(self, detector: SCRFD_AIDLite, recognizer: ModelAIDLite, db: FeatureDatabase):
        self.detector = detector
        self.recognizer = recognizer
        self.db = db

    def _is_valid_train_face(self, det: dict, img_shape) -> Tuple[bool, str, float]:
        h, w = img_shape[:2]
        x1, y1, x2, y2 = det["bbox"].astype(np.float32)
        bw = x2 - x1
        bh = y2 - y1
        if bw < 60 or bh < 60:
            return False, "too_small", 0.0
        ratio = bw / float(bh + 1e-6)
        if ratio < 0.65 or ratio > 1.45:
            return False, "bad_ratio", 0.0
        margin_x = 0.01 * w
        margin_y = 0.01 * h
        if x1 <= margin_x or y1 <= margin_y or x2 >= (w - 1 - margin_x) or y2 >= (h - 1 - margin_y):
            edge_penalty = 0.15
        else:
            edge_penalty = 0.0
        kps = det.get("kps", None)
        if kps is None or len(kps) != 5:
            return False, "no_kps", 0.0
        kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        tolx = bw * 0.10
        toly = bh * 0.10
        if np.any(kps[:, 0] < x1 - tolx) or np.any(kps[:, 0] > x2 + tolx) or \
           np.any(kps[:, 1] < y1 - toly) or np.any(kps[:, 1] > y2 + toly):
            return False, "kps_outside_bbox", 0.0
        left_eye, right_eye, nose, left_mouth, right_mouth = kps
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        mouth_dist = float(np.linalg.norm(left_mouth - right_mouth))
        eye_center_y = float((left_eye[1] + right_eye[1]) * 0.5)
        mouth_center_y = float((left_mouth[1] + right_mouth[1]) * 0.5)
        if eye_dist < bw * 0.16:
            return False, "eye_dist_too_small", 0.0
        if mouth_dist < bw * 0.10:
            return False, "mouth_dist_too_small", 0.0
        if not (eye_center_y < nose[1] < mouth_center_y):
            return False, "nose_not_between_eyes_mouth", 0.0
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        nose_center_offset = (
            abs(float(nose[0] - cx)) / (bw + 1e-6) +
            abs(float(nose[1] - cy)) / (bh + 1e-6)
        )
        area_ratio = (bw * bh) / float(w * h + 1e-6)
        score = float(det["score"]) * 2.0 + area_ratio * 1.2 - edge_penalty - nose_center_offset * 0.5
        return True, "ok", score

    def _pick_best_train_det(self, dets: List[dict], img_shape):
        candidates = []
        debug_lines = []
        for idx, det in enumerate(dets):
            ok, reason, rank_score = self._is_valid_train_face(det, img_shape)
            x1, y1, x2, y2 = det["bbox"].astype(int)
            bw = x2 - x1
            bh = y2 - y1
            debug_lines.append(
                f"det#{idx} score={det['score']:.4f} bbox={[x1,y1,x2,y2]} "
                f"size=({bw},{bh}) valid={ok} reason={reason} rank={rank_score:.4f}"
            )
            if ok:
                candidates.append((rank_score, det))
        if not candidates:
            return None, debug_lines
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], debug_lines

    def detect_train_face(self, img: np.ndarray):
        debug_info = []
        best_result = None
        best_score = -1e9
        angle_plan = [0, 90, 270, 180]
        long_side_plan = [1280, 960, 768]
        thr_plan = [0.35, 0.30, 0.25, 0.20]
        for angle in angle_plan:
            rot = rotate_image(img, angle)
            for long_side in long_side_plan:
                test_img = resize_keep_ratio(rot, long_side)
                for thr in thr_plan:
                    with model_lock:
                        dets = self.detector.detect_single_scale(test_img, conf_threshold=thr)
                    debug_info.append(
                        f"angle={angle}, long_side={long_side}, thr={thr}, raw_dets={len(dets)}"
                    )
                    best_det, det_debug = self._pick_best_train_det(dets, test_img.shape)
                    debug_info.extend([f"[TRAIN_DET] {x}" for x in det_debug])
                    if best_det is None:
                        continue
                    ok, reason, rank_score = self._is_valid_train_face(best_det, test_img.shape)
                    if ok and rank_score > best_score:
                        best_score = rank_score
                        best_result = {
                            "angle": angle,
                            "detect_img": test_img,
                            "det": best_det,
                            "debug_info": list(debug_info),
                        }
                if angle == 0 and best_result is not None:
                    break
            if angle == 0 and best_result is not None:
                break
        if best_result is None:
            return {
                "detect_img": img,
                "det": None,
                "debug_info": debug_info,
            }
        return best_result

    def process_train_image(self, name: str, image_b64: str):
        try:
            if "," in image_b64 and image_b64.startswith("data:image"):
                image_b64 = image_b64.split(",", 1)[1]
            raw = base64.b64decode(image_b64)
            nparr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, "图像解码失败"
            logger.info(f"[TRAIN] img shape={img.shape}")
            result = self.detect_train_face(img)
            for line in result["debug_info"]:
                logger.info(f"[TRAIN][TRY] {line}")
            best = result["det"]
            detect_img = result["detect_img"]
            if best is None:
                return False, "训练图未检测到有效完整人脸，请换一张不过曝、不要过近的正脸图"
            bbox = best["bbox"]
            kps = best.get("kps", None)
            x1, y1, x2, y2 = bbox.astype(int)
            debug_img = detect_img.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if kps is not None and len(kps) == 5:
                kps_draw = np.asarray(kps, dtype=np.int32).reshape(5, 2)
                for (px, py) in kps_draw:
                    cv2.circle(debug_img, (int(px), int(py)), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(TRAIN_IMG_DIR, f"{name}_debug_detected.jpg"), debug_img)
            aligned = None
            if kps is not None and len(kps) == 5:
                aligned = norm_crop_face(detect_img, kps, image_size=112)
            if aligned is not None and aligned.size > 0:
                cv2.imwrite(os.path.join(TRAIN_IMG_DIR, f"{name}_aligned.jpg"), aligned)
                with recognizer_lock:
                    aux, feat = self.recognizer.extract_from_aligned_crop(aligned)
            else:
                with recognizer_lock:
                    aux, feat = self.recognizer.extract(detect_img, bbox)
            if feat is None:
                return False, "Model 特征提取失败"
            self.db.add_feature(name, feat)
            check_name, check_sim = self.db.compare(feat, threshold=0.0)
            logger.info(f"[TRAIN_CHECK] name={name} top1={check_name} sim={check_sim:.4f}")
            cv2.imwrite(os.path.join(TRAIN_IMG_DIR, f"{name}.jpg"), img)
            return True, f"训练成功 aux={aux:.4f} self_sim={check_sim:.4f}"
        except Exception as e:
            logger.exception("process_train_image 异常")
            return False, f"训练异常: {e}"

# =========================
# 业务引擎
# =========================
class DroneFaceEngine:
    def __init__(self):
        logger.info("🚀 初始化业务引擎")
        self.detector = SCRFD_AIDLite(SCRFD_MODEL, confThreshold=0.20, nmsThreshold=0.4)
        self.recognizer = ModelAIDLite(Model_MODEL, use_gpu=True)
        self.db = FeatureDatabase(FEATURE_DB_PATH)
        self.training = TrainingManager(self.detector, self.recognizer, self.db)
        if HAS_REAL_BYTETRACK:
            logger.info("✅ 使用真实 ByteTrack")
            self.tracker = RealByteTrackAdapter(
                fps=CAMERA_FPS,
                track_thresh=0.22,
                track_buffer=12,
                match_thresh=0.65
            )
        else:
            logger.warning("⚠️ 未检测到真实 ByteTrack，回退到 IoU 跟踪器")
            self.tracker = IOUFallbackTracker(iou_threshold=0.30, max_lost=1)
        self.compensator = TrackerCompensator() if DETECT_EVERY_N > 1 else None
        self.frame_idx = 0
        self.last_items = []
        self.last_encoded_frame = None
        self.last_report_name = None
        self.last_report_ts = 0

        # 注意：此处彻底移除推流器的初始化和控制，推流逻辑已剥离至单独的进程

        self.low_conf_streak: Dict[int, int] = defaultdict(int)
        self.last_report_ts_by_track: Dict[int, float] = {}
        self.id2state: Dict[int, str] = {}
        self.id2candidate: Dict[int, str] = {}
        self.track_identity_state: Dict[int, Dict[str, Any]] = defaultdict(self._new_identity_state)
        self.unknown_streak: Dict[int, int] = defaultdict(int)
        self.unknown_alerted: Dict[int, bool] = defaultdict(bool)
        self.track_last_seen: Dict[int, int] = {}
        self.strong_lock_state: Dict[int, Dict[str, Any]] = {}
        self.confirm_hold_until: Dict[int, int] = {}
        self.min_unknown_report_sim = -1.0

        self.strong_match_thr = 0.4
        self.weak_match_thr = 0.35
        self.reject_match_thr = 0.38
        self.min_margin_thr = 0.02
        self.confirm_min_votes = 2
        self.confirm_min_strong = 1
        self.confirm_avg_thr = 0.35
        self.release_streak = 5
        self.candidate_ttl = 16
        self.instant_confirm_thr = 0.45

        self.enable_dynamic_threshold = False
        self.dynamic_far_face_thr = 55
        self.dynamic_mid_face_thr = 70
        self.dynamic_strong_match_thr_far = 0.36
        self.dynamic_confirm_avg_thr_far = 0.33
        self.dynamic_weak_match_thr_far = 0.32
        self.dynamic_reject_match_thr_far = 0.29
        self.dynamic_strong_match_thr_mid = 0.48
        self.dynamic_confirm_avg_thr_mid = 0.45
        self.dynamic_weak_match_thr_mid = 0.41
        self.dynamic_reject_match_thr_mid = 0.37

        self.enable_strong_lock = True
        self.strong_lock_sim_thr = 0.60
        self.strong_lock_ttl = 18
        self.enable_confirm_hold = True
        self.confirm_hold_frames_near = 6
        self.confirm_hold_frames_far = 12

        self.enable_face_reliability_gate = True
        self.profile_yaw_thr = 0.38
        self.profile_yaw_soft_thr = 0.30
        self.roll_reject_thr = 0.28
        self.pitch_reject_thr = 0.32
        self.mask_occlusion_thr = 0.42
        self.low_conf_reid_penalty = 0.025
        self.low_confidence_hold = 6
        self.low_conf_direct_reject = False
        self.low_conf_unknown_penalty = 0.045
        self.specular_relief_max = 0.020
        self.low_conf_reid_min_q = 0.03
        self.low_conf_grace_frames = 4

    def reload_feature_db(self):
        try:
            self.db.load()
            logger.info(f"✅ 特征库已重载: {len(self.db.db)} 人")
        except Exception as e:
            logger.warning(f"⚠️ 特征库重载失败: {e}")

    def reset_runtime_state(self):
        if hasattr(self.tracker, "track_states"):
            self.tracker.track_states.clear()
        if hasattr(self.tracker, "tracks"):
            self.tracker.tracks.clear()
        if hasattr(self.tracker, "next_id"):
            self.tracker.next_id = 1
        self.frame_idx = 0
        self.last_items = []
        self.last_encoded_frame = None
        self.last_report_name = None
        self.last_report_ts = 0
        self.low_conf_streak.clear()
        self.last_report_ts_by_track.clear()
        self.id2state.clear()
        self.id2candidate.clear()
        self.track_identity_state.clear()
        self.unknown_streak.clear()
        self.unknown_alerted.clear()
        self.track_last_seen.clear()
        self.strong_lock_state.clear()
        self.confirm_hold_until.clear()

    def _new_identity_state(self) -> Dict[str, Any]:
        return {
            "state": "unknown",
            "confirmed_name": "Unknown",
            "confirmed_sim": -1.0,
            "candidate_name": "Unknown",
            "candidate_votes": 0,
            "candidate_strong_votes": 0,
            "candidate_score_sum": 0.0,
            "candidate_last_frame": -999999,
            "negative_streak": 0,
            "last_update_frame": -999999,
            "low_conf_reason": "",
        }

    def _iou(self, box1, box2):
        x11, y11, x12, y12 = map(float, box1)
        x21, y21, x22, y22 = map(float, box2)
        xa = max(x11, x21)
        ya = max(y11, y21)
        xb = min(x12, x22)
        yb = min(y12, y22)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
        a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
        return inter / (a1 + a2 - inter + 1e-6)

    def _item_rank_score(self, item):
        state = str(item.get("state", "unknown"))
        name = str(item.get("name", "Unknown"))
        det_score = float(item.get("det_score", item.get("score", 0.0)))
        sim_score = float(item.get("sim_score", item.get("score", 0.0)))
        state_bonus = 0.0
        if state == "confirmed" and name != "Unknown":
            state_bonus = 3.0
        elif state == "tentative":
            state_bonus = 1.5
        elif state == "low_confidence":
            state_bonus = 0.5
        known_bonus = 0.6 if name != "Unknown" else 0.0
        return state_bonus + known_bonus + det_score + 0.35 * sim_score

    def _suppress_duplicate_items(self, items):
        if not items:
            return []
        items = sorted(items, key=self._item_rank_score, reverse=True)
        kept = []
        for item in items:
            box = item["bbox"]
            x1, y1, x2, y2 = map(float, box)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            duplicated = False
            for k in kept:
                kbox = k["bbox"]
                kx1, ky1, kx2, ky2 = map(float, kbox)
                kbw = max(1.0, kx2 - kx1)
                kbh = max(1.0, ky2 - ky1)
                kcx = 0.5 * (kx1 + kx2)
                kcy = 0.5 * (ky1 + ky2)
                iou = self._iou(box, kbox)
                center_dist = np.hypot(cx - kcx, cy - kcy)
                ref_size = max(1.0, 0.5 * (max(bw, bh) + max(kbw, kbh)))
                if iou > 0.38 or center_dist < ref_size * 0.22:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(item)
        return kept

    def _reset_candidate_state(self, st: Dict[str, Any]):
        st["candidate_name"] = "Unknown"
        st["candidate_votes"] = 0
        st["candidate_strong_votes"] = 0
        st["candidate_score_sum"] = 0.0
        st["candidate_last_frame"] = -999999

    def _get_dynamic_thresholds(self, face_short_side: float) -> Dict[str, float]:
        thr = {
            "strong": self.strong_match_thr,
            "weak": self.weak_match_thr,
            "reject": self.reject_match_thr,
            "confirm_avg": self.confirm_avg_thr,
        }
        if not self.enable_dynamic_threshold:
            return thr
        ss = float(face_short_side)
        if ss < self.dynamic_far_face_thr:
            thr["strong"] = self.dynamic_strong_match_thr_far
            thr["weak"] = self.dynamic_weak_match_thr_far
            thr["reject"] = self.dynamic_reject_match_thr_far
            thr["confirm_avg"] = self.dynamic_confirm_avg_thr_far
        elif ss < self.dynamic_mid_face_thr:
            thr["strong"] = self.dynamic_strong_match_thr_mid
            thr["weak"] = self.dynamic_weak_match_thr_mid
            thr["reject"] = self.dynamic_reject_match_thr_mid
            thr["confirm_avg"] = self.dynamic_confirm_avg_thr_mid
        return thr

    def _get_confirm_hold_frames(self, face_short_side: float) -> int:
        return self.confirm_hold_frames_far if float(face_short_side) < self.dynamic_far_face_thr else self.confirm_hold_frames_near

    def _update_strong_lock(self, tid: int, raw_name: str, sim: float, frame_id: int):
        if not self.enable_strong_lock:
            return
        if raw_name != "Unknown" and sim >= self.strong_lock_sim_thr:
            self.strong_lock_state[int(tid)] = {
                "name": str(raw_name),
                "sim": float(sim),
                "last_frame": int(frame_id),
            }

    def _get_valid_strong_lock(self, tid: int, frame_id: int) -> Optional[Dict[str, Any]]:
        if not self.enable_strong_lock:
            return None
        st = self.strong_lock_state.get(int(tid))
        if not st:
            return None
        if (int(frame_id) - int(st.get("last_frame", -999999))) > self.strong_lock_ttl:
            self.strong_lock_state.pop(int(tid), None)
            return None
        return st

    def _update_track_identity_state_machine(self, tid: int, raw_name: str, sim: float, frame_id: int, margin: float, face_short_side: float) -> Tuple[str, float, str]:
        st = self.track_identity_state[int(tid)]
        st["last_update_frame"] = int(frame_id)
        dyn_thr = self._get_dynamic_thresholds(face_short_side)
        hold_frames = self._get_confirm_hold_frames(face_short_side)
        self._update_strong_lock(tid, raw_name, sim, frame_id)
        strong_lock = self._get_valid_strong_lock(tid, frame_id)
        positive = (raw_name != "Unknown" and sim >= dyn_thr["weak"] and margin >= self.min_margin_thr)
        strong = positive and sim >= dyn_thr["strong"]
        locked_positive = False
        if strong_lock is not None:
            lock_name = str(strong_lock.get("name", "Unknown"))
            if raw_name == lock_name and sim >= max(dyn_thr["weak"] - 0.02, 0.0):
                locked_positive = True
        if st["state"] == "confirmed" and st["confirmed_name"] != "Unknown":
            confirmed_name = st["confirmed_name"]
            if positive and raw_name == confirmed_name:
                st["negative_streak"] = 0
                st["confirmed_sim"] = max(float(sim), float(st.get("confirmed_sim", -1.0)))
                self.confirm_hold_until[int(tid)] = max(int(self.confirm_hold_until.get(int(tid), -999999)), int(frame_id + hold_frames))
                self._reset_candidate_state(st)
            elif (
                strong_lock is not None
                and str(strong_lock.get("name", "Unknown")) == confirmed_name
                and frame_id <= int(self.confirm_hold_until.get(int(tid), -999999))
                and sim >= max(dyn_thr["reject"], dyn_thr["weak"] - 0.03)
            ):
                st["negative_streak"] = 0
                st["confirmed_sim"] = max(float(sim), float(st.get("confirmed_sim", -1.0)))
                self._reset_candidate_state(st)
            elif self.enable_confirm_hold and frame_id <= int(self.confirm_hold_until.get(int(tid), -999999)) and sim >= max(dyn_thr["reject"] - 0.01, 0.0):
                st["negative_streak"] = 0
            elif (not positive) or sim < dyn_thr["reject"]:
                st["negative_streak"] += 1
                if st["negative_streak"] >= self.release_streak:
                    st["state"] = "unknown"
                    st["confirmed_name"] = "Unknown"
                    st["confirmed_sim"] = -1.0
                    st["negative_streak"] = 0
                    self._reset_candidate_state(st)
        else:
            st["state"] = "unknown"
            st["confirmed_name"] = "Unknown"
            st["confirmed_sim"] = -1.0
            if positive or locked_positive:
                effective_name = raw_name
                effective_strong = strong
                if locked_positive and strong_lock is not None:
                    effective_name = str(strong_lock.get("name", raw_name))
                    if sim >= max(dyn_thr["weak"], dyn_thr["strong"] - 0.03):
                        effective_strong = True
                ttl_expired = (frame_id - int(st["candidate_last_frame"])) > self.candidate_ttl
                if ttl_expired or st["candidate_name"] != effective_name:
                    st["candidate_name"] = effective_name
                    st["candidate_votes"] = 1
                    st["candidate_strong_votes"] = 1 if effective_strong else 0
                    st["candidate_score_sum"] = float(sim)
                else:
                    st["candidate_votes"] += 1
                    if effective_strong:
                        st["candidate_strong_votes"] += 1
                    st["candidate_score_sum"] += float(sim)
                st["candidate_last_frame"] = int(frame_id)
                avg_sim = st["candidate_score_sum"] / max(st["candidate_votes"], 1)
                needed_votes = self.confirm_min_votes
                needed_strong = self.confirm_min_strong
                confirm_avg_thr = dyn_thr["confirm_avg"]
                if face_short_side < self.dynamic_mid_face_thr:
                    needed_votes = 2
                    needed_strong = 1
                if locked_positive:
                    confirm_avg_thr = max(confirm_avg_thr - 0.015, 0.0)
                if strong and sim >= self.instant_confirm_thr:
                    st["state"] = "confirmed"
                    st["confirmed_name"] = st["candidate_name"]
                    st["confirmed_sim"] = float(sim)
                    st["negative_streak"] = 0
                    self.confirm_hold_until[int(tid)] = int(frame_id + hold_frames)
                    self._reset_candidate_state(st)
                elif (
                    strong_lock is not None
                    and st["candidate_name"] == str(strong_lock.get("name", "Unknown"))
                    and sim >= max(dyn_thr["strong"], 0.47)
                    and st["candidate_votes"] >= 2
                ):
                    st["state"] = "confirmed"
                    st["confirmed_name"] = st["candidate_name"]
                    st["confirmed_sim"] = max(float(sim), float(strong_lock.get("sim", sim)))
                    st["negative_streak"] = 0
                    self.confirm_hold_until[int(tid)] = int(frame_id + hold_frames)
                    self._reset_candidate_state(st)
                elif (
                    st["candidate_votes"] >= needed_votes
                    and st["candidate_strong_votes"] >= needed_strong
                    and avg_sim >= confirm_avg_thr
                ):
                    st["state"] = "confirmed"
                    st["confirmed_name"] = st["candidate_name"]
                    st["confirmed_sim"] = float(sim)
                    st["negative_streak"] = 0
                    self.confirm_hold_until[int(tid)] = int(frame_id + hold_frames)
                    self._reset_candidate_state(st)
                else:
                    st["state"] = "tentative"
            else:
                if (frame_id - int(st["candidate_last_frame"])) > self.candidate_ttl:
                    self._reset_candidate_state(st)
                st["state"] = "unknown"
        final_name = st["confirmed_name"] if st["state"] == "confirmed" else "Unknown"
        final_sim = float(st["confirmed_sim"]) if st["state"] == "confirmed" else float(sim)
        self.id2state[int(tid)] = st["state"]
        self.id2candidate[int(tid)] = st["candidate_name"] if st["state"] == "tentative" else "Unknown"
        return final_name, final_sim, st["state"]

    def should_report_track(self, tid: int, now_ts: float, cooldown_sec: float = REPORT_COOLDOWN_SEC) -> bool:
        last_ts = self.last_report_ts_by_track.get(int(tid), 0.0)
        return (now_ts - last_ts) >= cooldown_sec

    def mark_track_reported(self, tid: int, now_ts: float):
        self.last_report_ts_by_track[int(tid)] = float(now_ts)

    def _set_low_confidence(self, track: TrackState, reason: str):
        track.low_confidence = True
        track.low_conf_reason = str(reason or "unreliable")
        track.low_conf_hold_until = int(self.frame_idx + self.low_confidence_hold)
        track.recognized_name = "Unknown"
        self.id2state[int(track.track_id)] = "low_confidence"
        self.id2candidate[int(track.track_id)] = track.low_conf_reason

    def _clear_low_confidence(self, track: TrackState):
        if self.frame_idx > int(getattr(track, "low_conf_hold_until", -999999)):
            track.low_confidence = False
            track.low_conf_reason = ""
        if not track.low_confidence and self.id2state.get(int(track.track_id)) == "low_confidence":
            self.id2state[int(track.track_id)] = "unknown"
            self.id2candidate[int(track.track_id)] = "Unknown"

    def should_recognize(self, track: TrackState, frame: np.ndarray) -> Tuple[bool, str, float]:
        x1, y1, x2, y2 = map(int, track.bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return False, "empty", 0.0
        bw = x2 - x1
        bh = y2 - y1
        cur_size = max(bw, bh)
        sharpness = variance_of_laplacian(face)
        if cur_size < RECOGNIZE_MIN_FACE:
            return False, "small_face", sharpness
        if track.stable_frames < RECOGNIZE_MIN_STABLE:
            return False, "not_stable", sharpness
        if not track.first_recognized_done:
            return True, "first_recognition", sharpness
        if self.frame_idx - track.last_recognize_frame < RECOGNIZE_COOLDOWN:
            return False, "cooldown", sharpness
        if track.recognized_name != "Unknown":
            if cur_size > track.best_bbox_size * RECOGNIZE_GROWTH_RATIO:
                return True, "size_grow", sharpness
            if sharpness > track.best_sharpness + RECOGNIZE_SHARP_DELTA:
                return True, "sharpness_up", sharpness
            return False, "recognized_skip", sharpness
        if cur_size > track.best_bbox_size * RECOGNIZE_GROWTH_RATIO:
            return True, "unknown_size_grow", sharpness
        if sharpness > track.best_sharpness + RECOGNIZE_SHARP_DELTA:
            return True, "unknown_sharpness_up", sharpness
        if self.frame_idx - track.last_recognize_frame >= RECOGNIZE_COOLDOWN * 2:
            return True, "unknown_periodic_retry", sharpness
        return False, "unknown_skip", sharpness

    def update_unknown_state(self, items: List[Dict[str, Any]], frame_id: int):
        alert_tids = []
        active_tids = set()
        for item in items:
            tid = int(item.get("track_id", -1))
            if tid < 0:
                continue
            active_tids.add(tid)
            self.track_last_seen[tid] = int(frame_id)
            state = str(item.get("state", "unknown"))
            name = str(item.get("name", "Unknown"))
            if state == "confirmed" and name != "Unknown":
                self.unknown_streak[tid] = 0
                self.unknown_alerted[tid] = False
                continue
            if state == "low_confidence":
                continue
            self.unknown_streak[tid] += 1
            if ENABLE_UNKNOWN_ALERT and self.unknown_streak[tid] >= UNKNOWN_ALERT_FRAMES and not self.unknown_alerted[tid]:
                self.unknown_alerted[tid] = True
                alert_tids.append(tid)
        expired = []
        for tid, last_seen in list(self.track_last_seen.items()):
            if int(frame_id) - int(last_seen) > UNKNOWN_TRACK_TTL:
                expired.append(tid)
        for tid in expired:
            self.track_last_seen.pop(tid, None)
            self.unknown_streak.pop(tid, None)
            self.unknown_alerted.pop(tid, None)
            self.last_report_ts_by_track.pop(tid, None)
            self.id2state.pop(tid, None)
            self.id2candidate.pop(tid, None)
            self.track_identity_state.pop(tid, None)
            self.strong_lock_state.pop(tid, None)
            self.confirm_hold_until.pop(tid, None)
            self.low_conf_streak.pop(tid, None)
        return alert_tids

    def update_track_identity(self, track: TrackState, feature: np.ndarray, aux_score: float, sharpness: float, reason: str, face_eval: Optional[Dict[str, Any]] = None):
        cur_size = max(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1])
        face_short_side = float(cur_size)
        quality = float(aux_score) * 0.6 + float(sharpness) * 0.02 + float(cur_size) * 0.01
        track.feature_history.append(feature)
        track.feature_history = track.feature_history[-FUSION_MAX_FEATURES:]
        track.feature_bank.append((feature.copy(), quality))
        track.feature_bank = sorted(track.feature_bank, key=lambda x: x[1], reverse=True)[:FUSION_MAX_FEATURES]
        fused = track.fused_feature()
        if hasattr(self.db, 'compare_top2'):
            raw_name, raw_sim, second_sim, _suffix = self.db.compare_top2(fused, face_short_side=face_short_side)
        else:
            raw_name, raw_sim = self.db.compare(fused, threshold=0.0)
            second_sim = -1.0
        face_eval = face_eval or {}
        sim_penalty = 0.0
        if float(face_eval.get("yaw_score", 0.0)) > self.profile_yaw_soft_thr:
            sim_penalty += self.low_conf_reid_penalty
        if float(face_eval.get("occlusion_score", 0.0)) > self.mask_occlusion_thr:
            sim_penalty += self.low_conf_reid_penalty
        if bool(face_eval.get("is_low_confidence", False)):
            sim_penalty += self.low_conf_unknown_penalty
        specular_score = float(face_eval.get("specular_score", 0.0))
        if specular_score > 0.015:
            sim_penalty -= min(self.specular_relief_max, 0.6 * specular_score)
        raw_sim = float(raw_sim - max(0.0, sim_penalty))
        second_sim = float(second_sim - max(0.0, sim_penalty)) if second_sim >= 0.0 else -1.0
        margin = raw_sim - second_sim if second_sim >= 0.0 else raw_sim + 1.0
        track.aux_score = aux_score
        track.best_sharpness = max(track.best_sharpness, sharpness)
        track.best_bbox_size = max(track.best_bbox_size, cur_size)
        track.last_recognize_frame = self.frame_idx
        track.first_recognized_done = True
        track.last_trigger_reason = reason
        low_conf_reason = str(face_eval.get("reason", ""))
        if bool(face_eval.get("is_low_confidence", False)):
            self.low_conf_streak[int(track.track_id)] += 1
        else:
            self.low_conf_streak[int(track.track_id)] = 0
        if bool(face_eval.get("is_low_confidence", False)) and self.low_conf_streak[int(track.track_id)] >= self.low_conf_grace_frames:
            self._set_low_confidence(track, low_conf_reason)
        else:
            self._clear_low_confidence(track)
        if track.low_confidence:
            st = self.track_identity_state[int(track.track_id)]
            st["state"] = "low_confidence"
            st["low_conf_reason"] = track.low_conf_reason
            st["negative_streak"] = 0
            st["last_update_frame"] = int(self.frame_idx)
            self._reset_candidate_state(st)
            track.recognized_name = "Unknown"
            track.last_similarity = raw_sim
            self.id2state[int(track.track_id)] = "low_confidence"
            self.id2candidate[int(track.track_id)] = track.low_conf_reason
        else:
            final_name, final_sim, final_state = self._update_track_identity_state_machine(
                tid=int(track.track_id),
                raw_name=str(raw_name),
                sim=float(raw_sim),
                frame_id=int(self.frame_idx),
                margin=float(margin),
                face_short_side=float(face_short_side),
            )
            track.recognized_name = final_name
            track.last_similarity = final_sim
            if final_state != "low_confidence":
                track.low_conf_reason = ""
        if track.recognized_name == "Unknown":
            track.unknown_retry_count += 1
        else:
            track.unknown_retry_count = 0
        logger.info(
            f"[REC] track={track.track_id} reason={reason} state={self.id2state.get(int(track.track_id), 'unknown')} "
            f"cand={self.id2candidate.get(int(track.track_id), 'Unknown')} name={track.recognized_name} "
            f"sim={track.last_similarity:.4f} raw={raw_name}:{raw_sim:.4f} margin={margin:.4f} "
            f"aux={track.aux_score:.4f} quality={quality:.4f} low_conf={track.low_confidence} "
            f"low_conf_reason={track.low_conf_reason}"
        )

    def process_frame(self, frame: np.ndarray):
        self.frame_idx += 1
        run_det = (self.frame_idx % DETECT_EVERY_N == 0)
        if not run_det:
            if hasattr(self.tracker, "track_states"):
                track_iter = list(self.tracker.track_states.values())
            else:
                track_iter = list(self.tracker.tracks.values())
            for tr in track_iter:
                tr.age += 1
                if self.compensator is None:
                    tr.is_confirmed_visible = False
                    tr.stable_frames = max(0, tr.stable_frames - 1)
                    continue
                ok = self.compensator.update_for_track(tr, frame)
                if ok:
                    tr.is_confirmed_visible = True
                    tr.stable_frames += 1
                else:
                    tr.is_confirmed_visible = False
                    tr.stable_frames = max(0, tr.stable_frames - 1)
                    tr.cv_tracker = None
        if run_det:
            infer_frame, infer_scale_x, infer_scale_y = prepare_inference_frame(frame)
            with model_lock:
                dets = self.detector.detect(infer_frame)
            dets = remap_detections_to_original(dets, infer_scale_x, infer_scale_y, frame.shape[1], frame.shape[0])
            for i, det in enumerate(dets[:10]):
                x1, y1, x2, y2 = det["bbox"].astype(int)
                bw = x2 - x1
                bh = y2 - y1
            tracks = self.tracker.update(dets, self.frame_idx, frame.shape[:2])
            if self.compensator is not None:
                for track in tracks:
                    if getattr(track, "cv_tracker", None) is None:
                        self.compensator.init_for_track(track, frame)
        else:
            if hasattr(self.tracker, "track_states"):
                tracks = list(self.tracker.track_states.values())
            else:
                tracks = list(self.tracker.tracks.values())
        h, w = frame.shape[:2]
        items = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
            bw = x2 - x1
            bh = y2 - y1
            ratio = bw / float(bh + 1e-6)
            det_score = float(track.score)
            if det_score < MIN_DET_SCORE_FOR_SHOW:
                # logger.info(f"[SKIP] track={track.track_id} reason=low_det_score_show score={det_score:.4f}")
                continue
            if run_det and getattr(track, "last_seen_frame", -1) != self.frame_idx:
                # logger.info(f"[SKIP] track={track.track_id} reason=not_seen_this_det_frame")
                continue
            if not getattr(track, "is_confirmed_visible", False):
                # logger.info(f"[SKIP] track={track.track_id} reason=not_confirmed_visible")
                continue
            if getattr(track, "lost", 0) > 0:
                # logger.info(f"[SKIP] track={track.track_id} reason=lost hidden lost={track.lost}")
                continue
            if bw <= 0 or bh <= 0:
                # logger.info(f"[SKIP] track={track.track_id} reason=bad_size bbox={[x1, y1, x2, y2]}")
                continue
            if bw < RECOGNIZE_MIN_FACE or bh < RECOGNIZE_MIN_FACE:
                # logger.info(f"[SKIP] track={track.track_id} reason=too_small bbox={[x1, y1, x2, y2]}")
                continue
            if ratio < 0.30 or ratio > 2.00:
                # logger.info(f"[SKIP] track={track.track_id} reason=bad_ratio ratio={ratio:.2f}")
                continue
            if bw > w * 0.92 or bh > h * 0.98:
                # logger.info(f"[SKIP] track={track.track_id} reason=too_large bbox={[x1, y1, x2, y2]}")
                continue
            track.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            if run_det:
                should, reason, sharpness = self.should_recognize(track, frame)
                # logger.info(
                #     f"[GATE] track={track.track_id} stable={track.stable_frames} lost={track.lost} "
                #     f"visible={getattr(track, 'is_confirmed_visible', True)} last_seen={getattr(track, 'last_seen_frame', -1)} "
                #     f"last_rec_frame={track.last_recognize_frame} name={track.recognized_name} reason={reason} bbox={[x1, y1, x2, y2]}"
                # )
                if should and det_score >= MIN_DET_SCORE_FOR_RECOG:
                    aux_score, feature = None, None
                    used_kps_align = False
                    used_bbox_crop = False
                    use_kps_align = (
                        track.kps is not None and
                        len(track.kps) == 5 and
                        getattr(track, "kps_frame_idx", -1) >= self.frame_idx - 1
                    )
                    if use_kps_align:
                        try:
                            kps = np.asarray(track.kps, dtype=np.float32).reshape(5, 2)
                            kps[:, 0] = np.clip(kps[:, 0], 0, w - 1)
                            kps[:, 1] = np.clip(kps[:, 1], 0, h - 1)
                            roi, crop_box = _crop_with_margin(frame, track.bbox, margin=0.20)
                            kp_roi = _kps_to_crop_coords(kps, crop_box) if roi is not None else None
                            aligned_face = None
                            sr_used = False
                            sr_face_size = 0.0
                            sr_sharpness = 0.0
                            if roi is not None and kp_roi is not None:
                                roi_for_align, kp_roi_for_align, sr_used, sr_face_size, sr_sharpness = enhance_small_face_roi(roi, kp_roi)
                                aligned_face = norm_crop_face(roi_for_align, kp_roi_for_align, image_size=112)
                            if aligned_face is None or aligned_face.size == 0:
                                aligned_face = norm_crop_face(frame, kps, image_size=112)
                            if aligned_face is not None and aligned_face.size > 0:
                                used_kps_align = True
                                if sr_used:
                                    logger.info(f"[SR] track={track.track_id} used=1 face_size={sr_face_size:.1f} sharpness={sr_sharpness:.1f}")
                                with recognizer_lock:
                                    aux_score, feature = self.recognizer.extract_from_aligned_crop(aligned_face)
                        except Exception as e:
                            logger.warning(f"[ALIGN] track={track.track_id} kps_align_failed: {e}")
                    if feature is None:
                        try:
                            used_bbox_crop = True
                            with recognizer_lock:
                                aux_score, feature = self.recognizer.extract(frame, track.bbox)
                        except Exception as e:
                            logger.warning(f"[ALIGN] track={track.track_id} bbox_extract_failed: {e}")
                    # logger.info(
                    #     f"[ALIGN] track={track.track_id} use_kps={used_kps_align} use_bbox={used_bbox_crop} "
                    #     f"kps_valid={track.kps is not None} kps_frame_idx={getattr(track, 'kps_frame_idx', -1)} "
                    #     f"cur_frame={self.frame_idx} feature_none={feature is None} aux={aux_score}"
                    # )
                    if feature is not None:
                        face_eval = {"is_low_confidence": False, "reason": ""}
                        if self.enable_face_reliability_gate:
                            roi, crop_box = _crop_with_margin(frame, track.bbox, margin=0.20)
                            kp_roi = _kps_to_crop_coords(
                                np.asarray(track.kps, dtype=np.float32).reshape(5, 2),
                                crop_box
                            ) if (track.kps is not None and len(track.kps) == 5 and roi is not None) else None
                            face_eval = _estimate_face_pose_and_visibility(
                                track.bbox,
                                track.kps,
                                roi=roi,
                                kps_roi=kp_roi
                            )
                            if roi is not None and face_eval.get("is_low_confidence", False) and variance_of_laplacian(roi) < 8.0:
                                continue
                        self.update_track_identity(
                            track,
                            feature,
                            aux_score,
                            sharpness,
                            reason,
                            face_eval=face_eval
                        )
            items.append({
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2],
                "name": track.recognized_name,
                "score": track.last_similarity if track.recognized_name != "Unknown" else track.score,
                "aux_score": track.aux_score,
                "det_score": float(track.score),
                "sim_score": float(track.last_similarity),
                "state": self.id2state.get(int(track.track_id), "unknown"),
                "candidate": self.id2candidate.get(int(track.track_id), "Unknown"),
                "low_conf_reason": track.low_conf_reason,
            })
        items = self._suppress_duplicate_items(items)
        self.last_items = items
        return items

    def build_report(self, frame, items):
        if not items:
            self.update_unknown_state([], self.frame_idx)
            return
        h, w = frame.shape[:2]
        now_ts = time.time()
        
        # 【新增】1. 全局上报冷却阀：防止人脸闪烁引发大量新ID，导致瞬间生成海量上报塞满队列
        # 强制限制系统的最大上报频率（例如 0.5 秒最多发一次全图结果）
        if now_ts - getattr(self, 'last_report_ts', 0) < 0.5:
            # 但仍需更新底层状态，避免逻辑断层
            self.update_unknown_state(items, self.frame_idx)
            return

        alert_tids = set(self.update_unknown_state(items, self.frame_idx))

        # ================= 步骤 1：判断当前画面是否需要触发上报 =================
        trigger_report = False
        for item in items:
            tid = int(item["track_id"])
            state = str(item.get("state", "unknown"))
            name = str(item.get("name", "Unknown"))
            sim = float(item.get("sim_score", item.get("score", -1.0)))

            if state == "low_confidence":
                if self.should_report_track(tid, now_ts, cooldown_sec=LOW_CONF_REPORT_COOLDOWN_SEC):
                    trigger_report = True
                    self.mark_track_reported(tid, now_ts)
            elif state == "confirmed" and name != "Unknown":
                if self.should_report_track(tid, now_ts, cooldown_sec=REPORT_COOLDOWN_SEC):
                    trigger_report = True
                    self.mark_track_reported(tid, now_ts)
            else:
                # 【修改】2. 移除 `if tid in alert_tids:` 的严苛限制
                # 只要人脸在画面中（哪怕还在收集特征的 tentative 状态），只要过了常规冷却期就上报
                # 这样前端就能立即且持续地看到未确认的识别框，不会出现几秒的“静默假死”
                if len(self.db.db) > 0 and sim < self.min_unknown_report_sim:
                    continue
                
                if self.should_report_track(tid, now_ts, cooldown_sec=REPORT_COOLDOWN_SEC):
                    trigger_report = True
                    self.mark_track_reported(tid, now_ts)

        # 如果画面里所有人脸都处于冷却期（都不需要上报），则直接返回
        if not trigger_report:
            return
            
        # 记录本次成功触发上报的全局时间
        self.last_report_ts = now_ts

        # ================= 步骤 2：打包画面中所有有效人脸，一并发送 =================
        report_names = []
        report_states = []
        report_candidates = []
        report_low_conf_reasons = []
        report_track_ids = []
        report_confidences = []
        report_bboxes = []
        valid_items = []

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            # 过滤掉尺寸异常的人脸
            if not (bw > 0 and bh > 0 and bw < w * 0.95 and bh < h * 0.98):
                continue

            tid = int(item["track_id"])
            state = str(item.get("state", "unknown"))
            name = str(item.get("name", "Unknown"))
            sim = float(item.get("sim_score", item.get("score", -1.0)))
            candidate = str(item.get("candidate", "Unknown"))
            low_conf_reason = str(item.get("low_conf_reason", ""))

            # 按照原逻辑重构 report_name
            report_name = "unknown"
            if state == "low_confidence":
                report_name = "low_confidence"
            elif state == "confirmed" and name != "Unknown":
                report_name = name

            valid_items.append(item)
            report_names.append(report_name)
            report_states.append(state)
            report_candidates.append(candidate)
            report_low_conf_reasons.append(low_conf_reason)
            report_track_ids.append(tid)
            report_confidences.append(round(sim, 3))
            report_bboxes.append(item["bbox"])
            
        if not valid_items:
            return 
            
        annotated = draw_annotated_frame(frame, valid_items)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        submit_jpeg_encode_task(
            annotated,
            report_names,
            timestamp,
            states=report_states,
            candidates=report_candidates,
            low_conf_reasons=report_low_conf_reasons,
            track_ids=report_track_ids,
            confidences=report_confidences,
            bboxes=report_bboxes
        )

# =========================
# 异步编码任务函数
# =========================
def submit_jpeg_encode_task(frame, names, timestamp, states=None, candidates=None,
                            low_conf_reasons=None, track_ids=None, confidences=None, bboxes=None):
    task = {
        "frame": frame.copy(),
        "names": list(names),
        "timestamp": timestamp,
        "states": list(states) if states is not None else [],
        "candidates": list(candidates) if candidates is not None else [],
        "low_conf_reasons": list(low_conf_reasons) if low_conf_reasons is not None else [],
        "track_ids": list(track_ids) if track_ids is not None else [],
        "confidences": list(confidences) if confidences is not None else [],
        "bboxes": list(bboxes) if bboxes is not None else []
    }
    try:
        jpeg_encode_queue.put_nowait(task)
    except queue.Full:
        try:
            _ = jpeg_encode_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            jpeg_encode_queue.put_nowait(task)
        except queue.Full:
            logger.warning("⚠️ JPEG编码队列持续满，丢弃当前帧")

# ========== 修改点 1: 优化 ws_queue 的“挤压”策略 ==========
def jpeg_encode_worker():
    bind_current_thread_to_core(2)
    while True:
        try:
            task = jpeg_encode_queue.get()
            if task is None:
                logger.info("🛑 JPEG编码线程退出")
                break
            frame = task["frame"]
            names = task["names"]
            timestamp = task["timestamp"]
            bboxes = task.get("bboxes", [])
            
            face_images_base64 = []
            h_img, w_img = frame.shape[:2]

            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)
                
                face_roi = frame[cy1:cy2, cx1:cx2]
                
                if face_roi.size > 0:
                    face_roi = cv2.resize(face_roi, (112, 112))
                    ok, buffer = cv2.imencode(".jpg", face_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        encoded = base64.b64encode(buffer).decode("utf-8")
                        face_images_base64.append(f"data:image/jpeg;base64,{encoded}")
                    else:
                        face_images_base64.append("")
                else:
                    face_images_base64.append("")

            # === 核心修改点：对全图进行缩放，大幅降低 Base64 传输体积 ===
            scale_ratio = 800.0 / float(w_img)
            if scale_ratio < 1.0:
                report_frame = cv2.resize(frame, (800, int(h_img * scale_ratio)), interpolation=cv2.INTER_AREA)
            else:
                report_frame = frame

            ok_full, buffer_full = cv2.imencode(".jpg", report_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            encoded_full_image = ""
            if ok_full:
                encoded_full_image = base64.b64encode(buffer_full).decode("utf-8")

            payload = {
                "type": "recognition",
                "names": names,
                "states": task.get("states", []),
                "candidates": task.get("candidates", []),
                "low_conf_reasons": task.get("low_conf_reasons", []),
                "track_ids": task.get("track_ids", []),
                "confidences": task.get("confidences", []),
                "bboxes": bboxes,
                "timestamp": timestamp,
                "image": f"data:image/jpeg;base64,{encoded_full_image}" if encoded_full_image else "",
                "face_images": face_images_base64 
            }
            try:
                ws_queue.put_nowait(payload)
            except queue.Full:
                # 出现拥堵时，主动丢弃最老的一条数据，为新数据腾出空间
                try:
                    _ = ws_queue.get_nowait()
                except queue.Empty:
                    pass
                # 再次尝试推入最新数据
                try:
                    ws_queue.put_nowait(payload)
                except queue.Full:
                    logger.warning("⚠️ 上报队列持续满，丢弃当前编码结果")
        except Exception as e:
            logger.error(f"❌ JPEG异步编码失败: {e}", exc_info=True)


# ========== 修改点 2: 消除发送循环的无谓休眠 ==========
async def monitor_server_loop():
    global is_connected
    ws = None
    while True:
        try:
            if not is_connected or ws is None:
                ws = await connect_monitor_server()
                if ws is None:
                    await asyncio.sleep(2)
                    continue
                
                # === 核心修改点：重连成功后，如果队列里堆积了大量数据，主动丢弃过时数据 ===
                discard_count = 0
                while ws_queue.qsize() > 1:
                    try:
                        ws_queue.get_nowait()
                        discard_count += 1
                    except queue.Empty:
                        break
                if discard_count > 0:
                    logger.warning(f"⚠️ 网络恢复，主动丢弃了 {discard_count} 条过时的积压数据")

            # 使用 while 循环，只要队列有数据就一直发，不中途休眠
            while not ws_queue.empty():
                msg = ws_queue.get()
                await ws.send(json.dumps(msg))
            
            # 队列发送完毕后，才进行休眠，避免死循环占满 CPU
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"上报异常: {e}")
            is_connected = False
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
            ws = None
            await asyncio.sleep(1)

# 独立进程：摄像头读取与纯净推流
def camera_stream_process(devices, width, height, fps, fourcc, frame_queue, stop_event, shared_push_fps):
    """
    独立摄像头进程，带自动重连和持续重试机制
    """
    bind_current_thread_to_core(0)
    logger.info("📷 独立摄像头读取进程已启动")

    cap = None
    streamer = None
    retry_interval = 2.0  # 重试间隔（秒）

    # 循环直到成功打开摄像头或被要求停止
    while not stop_event.is_set():
        cap = try_open_camera(devices, width, height, fps, fourcc)
        if cap is not None:
            logger.info("✅ 摄像头打开成功")
            break
        logger.warning(f"⚠️ 摄像头打开失败，{retry_interval}秒后重试...")
        # 等待期间检查是否需要退出
        for _ in range(int(retry_interval / 0.1)):
            if stop_event.is_set():
                return
            time.sleep(0.1)

    if cap is None:
        logger.error("❌ 摄像头始终无法打开，进程退出")
        return

    # 启动推流器（仅当全局开关开启）
    if ENABLE_FFMPEG_STREAM:
        streamer = FFmpegStreamer(
            url=STREAM_URL,
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            fps=STREAM_FPS,
        )
        streamer.start()

    frame_count = 0
    last_time = time.time()
    fail_count = 0
    slow_read_timestamps = []
    SLOW_READ_WINDOW = 5.0
    SLOW_READ_THRESHOLD = 3
    SLOW_READ_MS = 500

    while not stop_event.is_set():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        t_read_ms = (time.perf_counter() - t_start) * 1000

        # 检测慢读
        if t_read_ms > SLOW_READ_MS:
            now = time.time()
            slow_read_timestamps = [ts for ts in slow_read_timestamps if now - ts <= SLOW_READ_WINDOW]
            slow_read_timestamps.append(now)
            logger.warning(f"⚠️ cap.read() slow {t_read_ms:.1f} ms, count={len(slow_read_timestamps)}")
            if len(slow_read_timestamps) >= SLOW_READ_THRESHOLD:
                logger.warning("⚠️ 频繁慢读，重新打开摄像头")
                cap.release()
                cap = try_open_camera(devices, width, height, fps, fourcc)
                slow_read_timestamps.clear()
                fail_count = 0
                if cap is None:
                    logger.error("❌ 摄像头重新打开失败，进入重试循环")
                    # 回到外层重试
                    while not stop_event.is_set():
                        cap = try_open_camera(devices, width, height, fps, fourcc)
                        if cap is not None:
                            logger.info("✅ 摄像头重新打开成功")
                            break
                        time.sleep(retry_interval)
                    continue
                continue

        # 读取失败处理
        if not ret or frame is None:
            fail_count += 1
            if fail_count >= 5:
                logger.warning("⚠️ 连续读取失败，重新打开摄像头")
                cap.release()
                cap = try_open_camera(devices, width, height, fps, fourcc)
                fail_count = 0
                slow_read_timestamps.clear()
                if cap is None:
                    logger.error("❌ 摄像头重新打开失败，进入重试循环")
                    while not stop_event.is_set():
                        cap = try_open_camera(devices, width, height, fps, fourcc)
                        if cap is not None:
                            logger.info("✅ 摄像头重新打开成功")
                            break
                        time.sleep(retry_interval)
                    continue
            else:
                time.sleep(0.01)
            continue

        fail_count = 0
        now = time.time()

        # 推流纯净帧
        if streamer is not None:
            streamer.push_frame(frame)

        # 放入队列（挤压策略）
        try:
            if frame_queue.full():
                frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait((now, frame))
        except queue.Full:
            pass

        # 统计FPS
        frame_count += 1
        if now - last_time >= 1.0:
            current_fps = frame_count / (now - last_time)
            shared_push_fps.value = current_fps
            frame_count = 0
            last_time = now

    # 清理资源
    if streamer is not None:
        streamer.stop()
    if cap is not None:
        cap.release()
    logger.info("🛑 摄像头进程已退出")

def try_open_camera(devices, width, height, fps, fourcc):
    for dev in devices:
        cap = None
        try:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                logger.warning(f"❌ 无法打开摄像头 {dev}")
                if cap is not None:
                    cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 关闭自动曝光 (0 = 手动模式, 1 = 自动模式, 3 = 快门优先等，通常用 0)
            #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)  

            time.sleep(0.15)
            ok = False
            for _ in range(5):
                grabbed = cap.grab()
                if not grabbed:
                    time.sleep(0.03)
                    continue
                ret, frame = cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    break
                time.sleep(0.03)
            if ok:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"✅ 摄像头 {dev} 初始化成功 ({actual_w}x{actual_h}, fps={actual_fps:.2f})")
                return cap
            logger.warning(f"⚠️ 摄像头 {dev} 打开成功但取帧失败，尝试下一个")
            cap.release()
            cap = None
        except Exception as e:
            logger.error(f"❌ 摄像头 {dev} 初始化异常: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return None

# =========================
# WebSocket 上报与训练
# =========================
async def connect_monitor_server():
    global is_connected, connection_attempts
    while connection_attempts < MAX_CONNECTION_ATTEMPTS:
        try:
            if not is_server_reachable(SERVER_IP, WEBSOCKET_PORT):
                connection_attempts += 1
                await asyncio.sleep(3)
                continue
            ws = await websockets.connect(
                f"ws://{SERVER_IP}:{WEBSOCKET_PORT}",
                ping_interval=30,          # 30秒发一次 Ping
                ping_timeout=60,           # 允许 60 秒等待 Pong 回复
                close_timeout=10,          # 关闭等待时间
                max_size=10 * 1024 * 1024  # 10MB 消息大小限制
            )
            await ws.send(json.dumps({"token": MONITOR_TOKEN}))
            logger.info("已发送监控端认证")
            try:
                auth_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp_data = json.loads(auth_resp)
                if resp_data.get("type") == "auth_success":
                    logger.info("监控通道认证成功")
                    is_connected = True
                    connection_attempts = 0
                    return ws
                else:
                    logger.error(f"监控认证失败: {resp_data}")
                    await ws.close()
                    await asyncio.sleep(2)
                    continue
            except asyncio.TimeoutError:
                logger.error("监控认证超时")
                await ws.close()
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"监控认证解析失败: {e}")
                await ws.close()
                await asyncio.sleep(2)
                continue
        except Exception as e:
            connection_attempts += 1
            logger.warning(f"上报连接失败 ({connection_attempts}/{MAX_CONNECTION_ATTEMPTS}): {e}")
            await asyncio.sleep(3)
    return None

def start_monitor_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_server_loop())

async def connect_train_server():
    global train_connection_attempts
    while train_connection_attempts < MAX_CONNECTION_ATTEMPTS:
        try:
            if not is_server_reachable(SERVER_IP, TRAIN_WS_PORT):
                train_connection_attempts += 1
                await asyncio.sleep(3)
                continue
            ws = await websockets.connect(
                    f"ws://{SERVER_IP}:{TRAIN_WS_PORT}",
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024
                )
            train_connection_attempts = 0
            return ws
        except Exception as e:
            train_connection_attempts += 1
            logger.warning(f"⚠️ 训练连接失败 ({train_connection_attempts}/{MAX_CONNECTION_ATTEMPTS}): {e}")
            await asyncio.sleep(3)
    return None

async def train_server_loop():
    global train_is_connected, is_recognition_paused, ENGINE
    ws = None
    while True:
        try:
            ws = await connect_train_server()
            if ws is None:
                await asyncio.sleep(2)
                continue
            await ws.send(json.dumps({"token": TRAIN_TOKEN}))
            logger.info("已发送训练端认证")
            try:
                auth_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp_data = json.loads(auth_resp)
                if resp_data.get("type") == "auth_success":
                    logger.info("训练通道认证成功")
                    train_is_connected = True
                else:
                    logger.error(f"训练认证失败: {resp_data}")
                    await ws.close()
                    await asyncio.sleep(2)
                    continue
            except asyncio.TimeoutError:
                logger.error("训练认证超时")
                await ws.close()
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"训练认证解析失败: {e}")
                await ws.close()
                await asyncio.sleep(2)
                continue
            logger.info("开始监听训练请求...")
            async for message in ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    func = data.get("function")
                    if msg_type == "QFaceInform" and func == "query":
                        names = ENGINE.db.query_names()
                        resp = {"type": "FaceInform", "names": [], "images": []}
                        for name in names:
                            found = False
                            for ext in [".jpg", ".jpeg", ".png"]:
                                p = os.path.join(TRAIN_IMG_DIR, f"{name}{ext}")
                                if os.path.exists(p):
                                    with open(p, "rb") as f:
                                        resp["names"].append(name)
                                        resp["images"].append(base64.b64encode(f.read()).decode("utf-8"))
                                        found = True
                                        break
                            if not found:
                                resp["names"].append(name)
                                resp["images"].append("")
                        await ws.send(json.dumps(resp))
                        continue
                    if msg_type == "QFaceInform" and func == "delete":
                        names = data.get("deletenames", [])
                        deleted = ENGINE.db.delete_names(names)
                        for n in names:
                            for ext in [".jpg", ".jpeg", ".png"]:
                                p = os.path.join(TRAIN_IMG_DIR, f"{n}{ext}")
                                if os.path.exists(p):
                                    try:
                                        os.remove(p)
                                    except Exception:
                                        pass
                            extra_files = [
                                os.path.join(TRAIN_IMG_DIR, f"{n}_debug_detected.jpg"),
                                os.path.join(TRAIN_IMG_DIR, f"{n}_aligned.jpg"),
                            ]
                            for p in extra_files:
                                if os.path.exists(p):
                                    try:
                                        os.remove(p)
                                    except Exception:
                                        pass
                        if hasattr(ENGINE, "reload_feature_db"):
                            ENGINE.reload_feature_db()
                        if hasattr(ENGINE, "reset_runtime_state"):
                            ENGINE.reset_runtime_state()
                        await ws.send(json.dumps({
                            "status": "success",
                            "type": "delete_result",
                            "deleted": deleted,
                            "names": names,
                            "message": f"已删除 {deleted} 个身份，并完成特征库重载"
                        }))
                        continue
                    if msg_type == "train_data":
                        name = re.sub(r"[^\w_.-]", "", data.get("name", ""))
                        image_b64 = data.get("image")
                        if not name or not image_b64:
                            await ws.send(json.dumps({
                                "status": "error",
                                "message": "缺少 name 或 image"
                            }))
                            continue
                        is_recognition_paused = True
                        try:
                            ok, info = ENGINE.training.process_train_image(name, image_b64)
                            if ok:
                                if hasattr(ENGINE, "reload_feature_db"):
                                    ENGINE.reload_feature_db()
                                if hasattr(ENGINE, "reset_runtime_state"):
                                    ENGINE.reset_runtime_state()
                            await ws.send(json.dumps({
                                "status": "success" if ok else "error",
                                "name": name,
                                "message": info,
                            }))
                        finally:
                            is_recognition_paused = False
                        continue
                except Exception as e:
                    logger.error(f"训练消息处理失败: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("训练连接被关闭")
        except Exception as e:
            logger.error(f"训练通道异常: {e}")
        finally:
            train_is_connected = False
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
            ws = None
            await asyncio.sleep(2)

def start_train_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_server_loop())

# =========================
# 主识别循环
# =========================
def main_recognition_loop():
    global ENGINE, SERVO_TRACKER

    ENGINE = DroneFaceEngine()
    if ENABLE_SERVO:
        try:
            SERVO_TRACKER = ServoFaceTracker(
                serial_port="/dev/ttyUSB0",
                baud_rate=115200,
                timeout=1,
                frame_w=CAMERA_WIDTH,
                frame_h=CAMERA_HEIGHT,
                reverse_y=True,
                debug=False,
            )
        except Exception as e:
            SERVO_TRACKER = None
            logger.warning(f"⚠️ 舵机初始化失败: {e}")
    else:
        SERVO_TRACKER = None
        logger.info("⏸️ 舵机追踪功能已禁用")

    # ================= 初始化多进程资源 =================
    frame_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()
    shared_push_fps = mp.Value('d', 0.0)

    # 启动独立的摄像头子进程（内部集成了纯净视频推流）
    cam_process = mp.Process(
        target=camera_stream_process,
        args=(
            CAMERA_DEVICES, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FOURCC, 
            frame_queue, stop_event, shared_push_fps
        ),
        daemon=True
    )
    cam_process.start()
    time.sleep(0.5)
    # ========================================================

    threading.Thread(target=jpeg_encode_worker, daemon=True).start()

    last_report = time.time()
    infer_frames = 0
    infer_sum = 0.0
    last_frame_warning_time = 0.0

    try:
        while True:
            try:
                frame_ts, frame = frame_queue.get(timeout=0.01)
            except queue.Empty:
                time.sleep(0.005)
                # 心跳指标上报
                now = time.time()
                if now - last_report >= 3.0:
                    push_fps_val = shared_push_fps.value
                    metrics = {
                        "type": "metrics",
                        "infer_fps": 0.0,
                        "push_fps": round(push_fps_val, 2),
                        "avg_latency_ms": 0.0,
                        "fps": round(push_fps_val, 2),
                    }
                    if psutil is not None:
                        metrics["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
                        metrics["memory_percent"] = round(psutil.virtual_memory().percent, 1)
                    try:
                        ws_queue.put_nowait(metrics)
                    except queue.Full:
                        pass
                    logger.info(f"📊 metric: {metrics}")
                    last_report = now
                    infer_frames = 0
                    infer_sum = 0.0
                continue

            now = time.time()
            if now - frame_ts > 2.0:
                if now - last_frame_warning_time > 5.0:
                    logger.warning("⚠️ 队列获取的帧已过时 (>2s)，跳过本次推理")
                    last_frame_warning_time = now
                continue

            if is_recognition_paused:
                show_local_preview(frame, getattr(ENGINE, 'last_items', []))
                continue

            t1 = time.perf_counter()

            # 推理业务处理
            items = ENGINE.process_frame(frame)

            if SERVO_TRACKER is not None:
                SERVO_TRACKER.update(frame, items)

            ENGINE.build_report(frame, items)
            show_local_preview(frame, items)

            infer_ms = time.perf_counter() - t1
            infer_frames += 1
            infer_sum += infer_ms

            now2 = time.time()
            if now2 - last_report >= 3.0:
                window = max(now2 - last_report, 1e-6)
                push_fps_val = shared_push_fps.value 
                
                metrics = {
                    "type": "metrics",
                    "infer_fps": round(infer_frames / window, 2),
                    "push_fps": round(push_fps_val, 2),
                    "avg_latency_ms": round(infer_sum / max(infer_frames, 1) * 1000, 2),
                    "fps": round(push_fps_val, 2),
                }
                if psutil is not None:
                    metrics["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
                    metrics["memory_percent"] = round(psutil.virtual_memory().percent, 1)
                try:
                    ws_queue.put_nowait(metrics)
                except queue.Full:
                    pass
                logger.info(f"📊 metric: {metrics}")
                last_report = now2
                infer_frames = 0
                infer_sum = 0.0

    except KeyboardInterrupt:
        logger.info("🛑 收到键盘中断，正在退出...")
    finally:
        stop_event.set()
        if cam_process.is_alive():
            cam_process.join(timeout=2.0)
            if cam_process.is_alive():
                cam_process.terminate()
        
        if SERVO_TRACKER is not None:
            try:
                SERVO_TRACKER.safe_recenter_and_close()
            except Exception as e:
                logger.warning(f"⚠️ 舵机关闭失败: {e}")
        if ENABLE_LOCAL_PREVIEW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    cv2.setLogLevel(0)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("🚀 启动无人机远距离人脸识别引擎")
    threading.Thread(target=start_monitor_client, daemon=True).start()
    threading.Thread(target=start_train_client, daemon=True).start()
    try:
        main_recognition_loop()
    except Exception as e:
        logger.exception("❌ 主程序异常退出")
        raise
