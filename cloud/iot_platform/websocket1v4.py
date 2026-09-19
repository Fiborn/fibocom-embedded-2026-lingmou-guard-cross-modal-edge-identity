import asyncio
import websockets
import json
import os
import socket
import logging
from logging.handlers import RotatingFileHandler
import http.server
import socketserver
import threading
import smtplib
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ========== 配置 ==========
SERVER_IP = "0.0.0.0"
WEBSOCKET_PORT =          # 人脸识别结果上报端口
TRAIN_WEBSOCKET_PORT =   # 训练数据接收端口
HTTP_PORT =
LOG_FILE = "server.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# 🔑 两个独立的认证密钥
MONITOR_TOKEN = ""   # 人脸识别监控端密钥
TRAIN_TOKEN = ""       # 训练数据客户端密钥

# ========== 🔒 固定的 SMTP 配置（不允许通过 API 或文件修改） ==========
FIXED_SMTP = {
    "host": "",
    "port": '',
    "user": "",
    "password": "",
    "sender": ""
}

# 报警配置文件（只保存非 SMTP 部分）
ALARM_CONFIG_FILE = "alarm_config.json"

# ========== 日志配置 ==========
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ========== 全局变量 ==========
clients = {}          # 所有连接的客户端（调试用）
monitor_clients = {}  # {client_id: websocket} —— 人脸识别监控端
train_clients = {}    # {client_id: websocket} —— 训练数据客户端
appearance_tracker = {}
monitor_counter = 0
train_counter = 0
active_connections = 0
http_server = None

# 报警配置（不含 smtp，smtp 固定使用 FIXED_SMTP）
alarm_config = {
    "enabled": False,
    "mode": "unknown",      # unknown / selected / both
    "target_names": [],
    "email": "",
    "cooldown": 30,
    "duration": 3,
    # smtp 不存储在此，始终使用 FIXED_SMTP
}
last_alert_times = {}
alarm_lock = threading.Lock()


# ========== 安全文件名处理 ==========
def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip() or "Unknown"


# ========== 广播给监控端 ==========
# ========== 修改点 3: 并发广播，防止单点阻塞全局 ==========
async def broadcast_to_monitors(message, clients_dir):
    """并发向指定客户端集合发送消息，防止单个慢节点阻塞全局"""
    async def send_single(cid, ws):
        try:
            # 增加超时限制，最多等待 1.5 秒
            await asyncio.wait_for(ws.send(json.dumps(message)), timeout=1.5)
            if clients_dir is monitor_clients:
                logger.info(f"客户端 {cid} 发送数据成功: {message.get('type', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ 向客户端 {cid} 发送失败，准备清理: {e}")
            return cid
        return None

    # 并发创建所有发送任务
    tasks = [send_single(cid, ws) for cid, ws in clients_dir.items()]
    
    if tasks:
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 收集并清理发送失败（超时或断开）的死连接
        dead_clients = [res for res in results if res and isinstance(res, str)]
        for cid in dead_clients:
            clients_dir.pop(cid, None)

async def send_alert_email_async(targets, email_to, timestamp, image_b64=""):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        send_alert_email,
        targets,
        email_to,
        timestamp,
        image_b64
    )

# ========== 报警配置与邮件 ==========
def load_alarm_config():
    global alarm_config
    try:
        if os.path.exists(ALARM_CONFIG_FILE):
            with open(ALARM_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                with alarm_lock:
                    alarm_config.update({
                        "enabled": bool(data.get("enabled", False)),
                        "mode": str(data.get("mode", "unknown")),
                        "target_names": [
                            str(x).strip()
                            for x in data.get("target_names", [])
                            if str(x).strip()
                        ],
                        "email": str(data.get("email", "")).strip(),
                        "cooldown": max(1, int(data.get("cooldown", 30))),
                        "duration": max(1, int(data.get("duration", 3))),
                        # 忽略文件中的 smtp，使用固定值
                    })
        # 确保 smtp 始终固定（即使文件不存在）
        # 不存储到 alarm_config 中，而是直接使用 FIXED_SMTP
        logger.info(f"✅ 已加载报警配置（SMTP 固定为 {FIXED_SMTP['host']}）: {alarm_config}")
    except Exception as e:
        logger.error(f"❌ 加载报警配置失败: {e}")


def save_alarm_config():
    """保存配置时只保存非 SMTP 字段，避免文件中混入无用 SMTP"""
    try:
        with alarm_lock:
            # 构建仅包含非 SMTP 的字典
            data = {
                "enabled": alarm_config.get("enabled", False),
                "mode": alarm_config.get("mode", "unknown"),
                "target_names": alarm_config.get("target_names", []),
                "email": alarm_config.get("email", ""),
                "cooldown": alarm_config.get("cooldown", 30),
                "duration": alarm_config.get("duration", 3),
            }
        with open(ALARM_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("✅ 报警配置已保存（不含 SMTP）")
    except Exception as e:
        logger.error(f"❌ 保存报警配置失败: {e}")

def check_duration_trigger(targets, duration):
    now = time.time()
    valid_targets = []

    for t in targets:
        key = str(t).lower()

        if key not in appearance_tracker:
            appearance_tracker[key] = {
                "first_seen": now,
                "last_seen": now
            }
        else:
            appearance_tracker[key]["last_seen"] = now
            if now - appearance_tracker[key]["first_seen"] >= duration:
                valid_targets.append(t)

    return valid_targets

def get_alarm_config():
    """返回配置，并自动附加固定的 SMTP 供前端显示"""
    with alarm_lock:
        cfg = {
            "enabled": bool(alarm_config.get("enabled", False)),
            "mode": str(alarm_config.get("mode", "unknown")),
            "target_names": list(alarm_config.get("target_names", [])),
            "email": str(alarm_config.get("email", "")),
            "cooldown": max(1, int(alarm_config.get("cooldown", 30))),
            "duration": max(1, int(alarm_config.get("duration", 3))),
            "smtp": FIXED_SMTP.copy(),  # 始终返回固定 SMTP
        }
        return cfg


def update_alarm_config(new_cfg):
    """更新配置，忽略任何 smtp 字段"""
    with alarm_lock:
        alarm_config["enabled"] = bool(new_cfg.get("enabled", alarm_config["enabled"]))
        alarm_config["mode"] = str(new_cfg.get("mode", alarm_config["mode"]))
        alarm_config["target_names"] = [
            str(x).strip()
            for x in new_cfg.get("target_names", alarm_config["target_names"])
            if str(x).strip()
        ]
        alarm_config["email"] = str(new_cfg.get("email", alarm_config["email"])).strip()
        alarm_config["cooldown"] = max(1, int(new_cfg.get("cooldown", alarm_config["cooldown"])))
        alarm_config["duration"] = max(1, int(new_cfg.get("duration", alarm_config["duration"])))
        # 禁止修改 SMTP，忽略新配置中的 smtp 字段
        # （即使前端传入，也不会改动）
    save_alarm_config()


def should_trigger_alert(names, states=None):
    cfg = get_alarm_config()
    if not cfg["enabled"]:
        return []

    if states is None:
        states = ["unknown"] * len(names)

    target_names = {str(x).strip().lower() for x in cfg.get("target_names", [])}
    hits = []

    for name, state in zip(names, states):
        lowered = str(name).strip().lower()
        st = str(state).strip().lower()

        # 低可信状态不触发报警
        if st == "low_confidence":
            continue

        if not lowered:
            continue

        if cfg["mode"] in ("unknown", "both") and lowered == "unknown":
            hits.append(name)
        elif cfg["mode"] in ("selected", "both") and lowered in target_names:
            hits.append(name)

    seen = set()
    result = []
    for item in hits:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def in_alert_cooldown(targets, cooldown):
    now = time.time()
    active = False

    for target in targets:
        key = str(target).lower()
        last_ts = last_alert_times.get(key, 0)
        if now - last_ts < cooldown:
            active = True

    if not active:
        for target in targets:
            last_alert_times[str(target).lower()] = now

    return active


def send_alert_email(targets, email_to, timestamp, image_b64=""):
    """使用固定 SMTP 发送报警邮件"""
    smtp_cfg = FIXED_SMTP  # 直接使用固定配置
    smtp_host = smtp_cfg["host"]
    smtp_port = smtp_cfg["port"]
    smtp_user = smtp_cfg["user"]
    smtp_password = smtp_cfg["password"]
    smtp_sender = smtp_cfg["sender"]

    if not email_to:
        return False, "未设置报警邮箱"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_sender
        msg["To"] = email_to
        msg["Subject"] = f"检测到报警目标：{', '.join(targets)}"

        body = f"""人脸识别系统检测到报警目标。

报警时间：{timestamp}
报警对象：{', '.join(targets)}

请及时查看网页端告警信息。"""
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if image_b64:
            try:
                from email.mime.base import MIMEBase
                from email import encoders
                import base64

                raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(base64.b64decode(raw))
                encoders.encode_base64(attachment)
                attachment.add_header("Content-Disposition", 'attachment; filename="alert.jpg"')
                msg.attach(attachment)
            except Exception as e:
                logger.warning(f"⚠️ 报警图片附件添加失败: {e}")

        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_sender, [email_to], msg.as_string())

        return True, "邮件发送成功"
    except Exception as e:
        logger.error(f"❌ 报警邮件发送失败: {e}")
        return False, str(e)


# ========== WebSocket 处理（人脸识别结果） ==========
async def websocket_handler(websocket):
    global monitor_counter, active_connections
    client_id = f"monitor_client_{monitor_counter}"
    monitor_counter += 1
    clients[client_id] = websocket
    active_connections += 1
    is_monitor = False

    logger.info(f"✅ 新客户端连接: {client_id} (当前连接数: {active_connections})")

    try:
        first_msg = await websocket.recv()
        try:
            data = json.loads(first_msg)
            if data.get("token") == MONITOR_TOKEN:
                is_monitor = True
                monitor_clients[client_id] = websocket
                logger.info(f"🛡️ 客户端 {client_id} 通过认证（人脸识别监控端）")
                await websocket.send(json.dumps({
                    "type": "auth_success",
                    "message": "认证成功，您将接收人脸检测结果。"
                }))
            else:
                logger.info(f"📷 客户端 {client_id} 作为数据发送端连接（未认证为监控端）")
        except json.JSONDecodeError:
            logger.warning(f"⚠️ 客户端 {client_id} 发送了无效 JSON，视为数据发送端")

        # 主消息循环：持续接收人脸检测数据
        while True:
            message = await websocket.recv()
            try:
                data = json.loads(message)

                if data.get("type") == "recognition" and data.get("names") != "None":
                    names = data.get("names", [])

                    if isinstance(names, list):
                        name_str = ", ".join(names)
                        safe_name = sanitize_filename(name_str)
                        if not safe_name.strip():
                            safe_name = "Unknown"
                    else:
                        safe_name = sanitize_filename(names) or "Unknown"
                        names = [safe_name]
                    # =====维护当前帧目标集合 =====
                    current_set = set(str(n).lower() for n in names)
                    now_ts = time.time()

                    for key, state in list(appearance_tracker.items()):
                        if key in current_set:
                            continue
                        if now_ts - state.get("last_seen", now_ts) > 1.5:
                            appearance_tracker.pop(key, None)
                    logger.info(f"📥 收到人脸数据：safe_name={safe_name} 来自 {client_id} (监控端: {is_monitor})")

                    event_timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    image_data = data.get("image", "")
                    states = data.get("states", [])
                    low_conf_reasons = data.get("low_conf_reasons", [])
                    track_ids = data.get("track_ids", [])
                    confidences = data.get("confidences", [])

                    broadcast_data = {
                        "type": "face_detection",
                        "name": safe_name,
                        "names": names,
                        "states": states,
                        "low_conf_reasons": low_conf_reasons,
                        "track_ids": track_ids,
                        "confidences": confidences,
                        "timestamp": event_timestamp,
                        "image": image_data
                    }
                    await broadcast_to_monitors(broadcast_data, monitor_clients)

                    alert_hits = should_trigger_alert(names, states)
                    cfg = get_alarm_config()
                    duration_hits = check_duration_trigger(alert_hits, cfg.get("duration", 3))
                    if duration_hits and not in_alert_cooldown(duration_hits, cfg["cooldown"]):
                        track_id_for_alert = "-"
                        confidence_for_alert = None

                        if duration_hits:
                            first_target = str(duration_hits[0]).strip().lower()
                            for idx, n in enumerate(names):
                                if str(n).strip().lower() == first_target:
                                    if idx < len(track_ids):
                                        track_id_for_alert = track_ids[idx]
                                    if idx < len(confidences):
                                        confidence_for_alert = confidences[idx]
                                    break
                        # 先立即通知前端，不等邮件
                        alert_message = {
                            "type": "alert",
                            "names": duration_hits,
                            "track_id": track_id_for_alert,
                            "confidence": confidence_for_alert,
                            "timestamp": event_timestamp,
                            "email": cfg.get("email", ""),
                            "email_sent": None,
                            "email_message": "邮件发送中",
                            "image": image_data
                        }
                        await broadcast_to_monitors(alert_message, monitor_clients)

                        # 再异步发邮件
                        targets_for_email = list(duration_hits)
                        email_to = cfg.get("email", "")
                        timestamp_for_email = event_timestamp
                        image_for_email = image_data

                        async def do_send_email(
                            targets=targets_for_email,
                            email_addr=email_to,
                            ts=timestamp_for_email,
                            img=image_for_email
                        ):
                            email_ok, email_msg = await send_alert_email_async(
                                targets,
                                email_addr,
                                ts,
                                img
                            )

                            email_status_message = {
                                "type": "alert_email_status",
                                "names": targets,
                                "track_id": track_id_for_alert,
                                "confidence": confidence_for_alert,
                                "timestamp": ts,
                                "email": email_addr,
                                "email_sent": email_ok,
                                "email_message": email_msg
                            }
                            await broadcast_to_monitors(email_status_message, monitor_clients)

                            logger.warning(
                                f"🚨 报警触发: {targets}, 邮件={email_ok}, 目标邮箱={email_addr}"
                            )

                        asyncio.ensure_future(do_send_email())


                elif data.get("type") == "metrics":
                    logger.info(f"📊 收到程序运行信息：来自 {client_id}")
                    broadcast_data = {
                        "type": "metrics",
                        "fps": data.get("fps", 0),
                        "avg_latency_ms": data.get("avg_latency_ms", 0),
                        "cpu_percent": data.get("cpu_percent", 0),
                        "memory_percent": data.get("memory_percent", 0),
                    }
                    await broadcast_to_monitors(broadcast_data, monitor_clients)

                else:
                    logger.debug(f"📥 收到其他消息 来自 {client_id}: {list(data.keys())}")

            except json.JSONDecodeError:
                logger.warning(f"⚠️ 无效 JSON: {message[:50]}... (客户端: {client_id})")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"✅ 客户端断开: {client_id}")
    except Exception as e:
        logger.error(f"❌ 客户端异常 {client_id}: {e}")
    finally:
        clients.pop(client_id, None)
        monitor_clients.pop(client_id, None)
        active_connections -= 1
        logger.debug(f"🧹 清理客户端: {client_id}")
# ========== WebSocket 处理（训练数据）端口8766 ==========
async def train_websocket_handler(websocket):
    global train_counter, active_connections
    client_id = f"train_client_{train_counter}"
    train_counter += 1
    clients[client_id] = websocket
    active_connections += 1
    logger.info(f"✅ 训练客户端连接: {client_id} (当前连接数: {active_connections})")
    try:
        auth_msg = await websocket.recv()
        try:
            auth_data = json.loads(auth_msg)
            if auth_data.get("token") == TRAIN_TOKEN:
                train_clients[client_id] = websocket
                logger.info(f"🔑 训练客户端 {client_id} 通过认证")
                await websocket.send(json.dumps({
                    "type": "auth_success",
                    "message": "认证成功，可发送训练数据。"
                }))
            else:
                logger.warning(f"❌ 训练客户端 {client_id} 认证失败 (密钥: {auth_data.get('token', 'N/A')})")
                await websocket.send(json.dumps({
                    "type": "auth_failure",
                    "message": "认证失败，请使用正确的训练密钥。"
                }))
                return
        except json.JSONDecodeError:
            logger.warning(f"⚠️ 训练客户端 {client_id} 发送了无效认证消息")
            await websocket.send(json.dumps({
                "type": "auth_failure",
                "message": "认证消息格式错误。"
            }))
            return

        while True:
            train_msg = await websocket.recv()
            try:
                data = json.loads(train_msg)

                if data.get("type") != "train" or "name" not in data or "image" not in data:
                    if data.get("type") == "QFaceInform" and data.get("function") == "query":
                        broadcast_data = {
                            "type": "QFaceInform",
                            "function": "query"
                        }
                        await broadcast_to_monitors(broadcast_data, train_clients)
                        logger.info("📥 收到 QFaceInform 数据，已广播")

                    elif data.get("type") == "QFaceInform" and data.get("function") == "delete":
                        deletenames = data.get("deletenames", [])
                        broadcast_data = {
                            "type": "QFaceInform",
                            "function": "delete",
                            "deletenames": deletenames
                        }
                        await broadcast_to_monitors(broadcast_data, train_clients)
                        logger.info("📥 收到 deleteNames 数据，已广播")

                    elif data.get("type") == "alarm_config":
                        function_name = data.get("function")
                        if function_name == "set":
                            update_alarm_config(data.get("config", {}) or {})
                            # 返回更新后的配置（含固定 SMTP）
                            await websocket.send(json.dumps({
                                "type": "alarm_config",
                                "config": get_alarm_config()
                            }))
                            logger.info("✅ 已更新报警配置")
                        elif function_name == "get":
                            # 每次读取前重新从文件加载，确保本地文件改动也能同步
                            load_alarm_config()
                            await websocket.send(json.dumps({
                                "type": "alarm_config",
                                "config": get_alarm_config()
                            }))
                            logger.info("📤 已发送报警配置")
                        else:
                            logger.warning(f"⚠️ 未知报警配置请求: {data}")

                    elif data.get("type") == "FaceInform":
                        names = data.get("names", [])
                        images = data.get("images", [])
                        broadcast_data = {
                            "type": "FaceInform",
                            "names": names,
                            "images": images
                        }
                        await broadcast_to_monitors(broadcast_data, train_clients)
                        logger.info(f"📥 收到 FaceInform 数据，已广播 (names: {len(names)}, images: {len(images)})")

                    else:
                        logger.warning(f"⚠️ 无效的训练数据格式: {data}")

                    continue

                safe_name = sanitize_filename(data["name"])
                broadcast_data = {
                    "type": "train_data",
                    "name": safe_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": data["image"]
                }
                await broadcast_to_monitors(broadcast_data, train_clients)
                logger.info(f"📤 已广播训练数据: {safe_name} (监控端: {len(train_clients)})")
                await websocket.send(json.dumps({
                    "status": "success",
                    "message": "训练数据已广播"
                }))

            except json.JSONDecodeError:
                logger.warning(f"⚠️ 无效的训练数据 JSON: {train_msg[:50]}...")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "训练数据格式错误"
                }))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"✅ 训练客户端断开: {client_id}")
    except Exception as e:
        logger.error(f"❌ 训练客户端异常 {client_id}: {e}")
    finally:
        clients.pop(client_id, None)
        train_clients.pop(client_id, None)
        active_connections -= 1
        logger.debug(f"🧹 清理训练客户端: {client_id}")


# ========== WebSocket 服务器启动 ==========
async def start_websocket_server():
    try:
        server = await websockets.serve(
            websocket_handler,
            SERVER_IP,start_train_websocket_server
            WEBSOCKET_PORT,
            ping_interval=30,          # 保持与客户端一致
            ping_timeout=60,           # 保持与客户端一致
            close_timeout=10,
            max_size=10 * 1024 * 1024  # 提升服务端接收大小上限至 10MB
        )
        logger.info(f"🚀 人脸识别WebSocket服务器启动成功 - ws://{SERVER_IP}:{WEBSOCKET_PORT}")
        return server
    except OSError as e:
        if e.errno == 98:
            logger.error(f"❌ 人脸识别WebSocket端口 {WEBSOCKET_PORT} 已被占用！")
        else:
            logger.error(f"❌ 人脸识别WebSocket启动失败: {e}")
        sys.exit(1)


async def start_train_websocket_server():
    try:
        server = await websockets.serve(
            train_websocket_handler,
            SERVER_IP,
            TRAIN_WEBSOCKET_PORT,
            ping_interval=30,          # 保持与客户端一致
            ping_timeout=60,           # 保持与客户端一致
            close_timeout=10,
            max_size=10 * 1024 * 1024  # 提升服务端接收大小上限至 10MB
        )
        logger.info(f"🚀 训练数据WebSocket服务器启动成功 - ws://{SERVER_IP}:{TRAIN_WEBSOCKET_PORT}")
        return server
    except OSError as e:
        if e.errno == 98:
            logger.error(f"❌ 训练数据WebSocket端口 {TRAIN_WEBSOCKET_PORT} 已被占用！")
        else:
            logger.error(f"❌ 训练数据WebSocket启动失败: {e}")
        sys.exit(1)


# ========== HTTP 服务器 ==========
def start_http_server():
    global http_server
    try:
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", HTTP_PORT), handler)
        logger.info(f"🌐 HTTP 服务器启动成功 - http://localhost:{HTTP_PORT}/")
        http_server = httpd
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:
            logger.error(f"❌ HTTP 端口 {HTTP_PORT} 已被占用！")
        else:
            logger.error(f"❌ HTTP 启动失败: {e}")
        sys.exit(1)


# ========== 工具函数 ==========
def create_directories():
    try:
        os.makedirs("images", exist_ok=True)
        test_file = "images/test_write.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        logger.info("📁 images/ 目录可写")
    except Exception as e:
        logger.error(f"❌ 目录创建失败: {e}")
        sys.exit(1)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def cleanup():
    global http_server
    if http_server:
        http_server.shutdown()
        logger.info("🧹 HTTP 服务器已关闭")


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def wait_for_port(port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        if check_port(port):
            return True
        time.sleep(0.5)
    return False


# ========== 主程序 ==========
async def main():
    create_directories()
    load_alarm_config()

    for port, name in [
        (WEBSOCKET_PORT, "人脸识别"),
        (TRAIN_WEBSOCKET_PORT, "训练数据"),
        (HTTP_PORT, "HTTP")
    ]:
        if not check_port(port):
            logger.warning(f"⚠️ {name} 端口 {port} 被占用，等待 10 秒...")
            if not wait_for_port(port, 10):
                logger.error(f"❌ {name} 端口 {port} 仍被占用，退出")
                sys.exit(1)

    hostname = socket.gethostname()
    ip_address = get_local_ip()
    logger.info(f"💻 主机名: {hostname}")
    logger.info(f"🏠 本地 IP: {ip_address}")
    logger.info(f"📍 访问前端: http://{ip_address}:{HTTP_PORT}/")
    logger.info(f"📍 人脸识别地址: ws://{ip_address}:{WEBSOCKET_PORT}/")
    logger.info(f"📍 训练数据地址: ws://{ip_address}:{TRAIN_WEBSOCKET_PORT}/")
    logger.info(f"🔑 人脸识别监控密钥: {MONITOR_TOKEN}")
    logger.info(f"🔑 训练数据客户端密钥: {TRAIN_TOKEN}")

    logger.info("⏳ 启动服务器...")
    try:
        threading.Thread(target=start_http_server, daemon=True).start()
        ws_server = await start_websocket_server()
        train_ws_server = await start_train_websocket_server()
        await asyncio.gather(ws_server.wait_closed(), train_ws_server.wait_closed())
    except KeyboardInterrupt:
        logger.info("🛑 用户中断")
    except Exception as e:
        logger.error(f"❌ 服务器崩溃: {e}")
    finally:
        cleanup()
        logger.info("✅ 服务器已关闭")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()