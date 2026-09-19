"""Runtime configuration loaded from environment variables.

Do not put credentials in source code. Copy ``.env.example`` to the
deployment's protected environment/configuration system instead.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _path(name: str, default: Path) -> str:
    value = os.getenv(name)
    return str(Path(value).expanduser()) if value else str(default)


SERVER_IP = os.getenv("FACE_SERVER_HOST", "")
WEBSOCKET_PORT = _int("FACE_MONITOR_WS_PORT", 0)
TRAIN_WS_PORT = _int("FACE_TRAIN_WS_PORT", 0)
TRAIN_TOKEN = os.getenv("FACE_TRAIN_TOKEN", "")
MONITOR_TOKEN = os.getenv("FACE_MONITOR_TOKEN", "")
CAMERA_DEVICES = [int(value) for value in os.getenv("FACE_CAMERA_DEVICES", "0,1,2,3").split(",") if value.strip()]
CAMERA_WIDTH = _int("FACE_CAMERA_WIDTH", 1080)
CAMERA_HEIGHT = _int("FACE_CAMERA_HEIGHT", 720)
CAMERA_FPS = _int("FACE_CAMERA_FPS", 30)
CAMERA_FOURCC = os.getenv("FACE_CAMERA_FOURCC", "MJPG")
ENABLE_LOCAL_PREVIEW = _bool("FACE_ENABLE_LOCAL_PREVIEW", False)
TRAIN_IMG_DIR = _path("FACE_TRAIN_IMG_DIR", ROOT_DIR / "data" / "faces")
FEATURE_DB_PATH = _path("FACE_FEATURE_DB_PATH", ROOT_DIR / "data" / "features.pkl")
Path(TRAIN_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(FEATURE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
SCRFD_MODEL = _path("FACE_SCRFD_MODEL", ROOT_DIR / "models" / "scrfd_2.5g_mpolaris.tflite")
Model_MODEL = _path("FACE_RECOGNIZER_MODEL", ROOT_DIR / "models" / "adaface_ir18_webface4m_fix_fp32.tflite")
DETECT_EVERY_N = _int("FACE_DETECT_EVERY_N", 3)
RECOGNIZE_COOLDOWN = _int("FACE_RECOGNIZE_COOLDOWN", 3)
RECOGNIZE_MIN_STABLE = _int("FACE_RECOGNIZE_MIN_STABLE", 1)
RECOGNIZE_MIN_FACE = _int("FACE_RECOGNIZE_MIN_FACE", 24)
RECOGNIZE_GROWTH_RATIO = _float("FACE_RECOGNIZE_GROWTH_RATIO", 1.03)
RECOGNIZE_SHARP_DELTA = _float("FACE_RECOGNIZE_SHARP_DELTA", 1.5)
FUSION_MAX_FEATURES = _int("FACE_FUSION_MAX_FEATURES", 6)
MAX_REPORT_QUEUE = _int("FACE_MAX_REPORT_QUEUE", 60)
MIN_DET_SCORE_FOR_SHOW = _float("FACE_MIN_DET_SCORE_FOR_SHOW", 0.18)
MIN_DET_SCORE_FOR_RECOG = _float("FACE_MIN_DET_SCORE_FOR_RECOG", 0.18)
MAX_CONNECTION_ATTEMPTS = _int("FACE_MAX_CONNECTION_ATTEMPTS", 5)
REPORT_COOLDOWN_SEC = _float("FACE_REPORT_COOLDOWN_SEC", 0.1)
LOW_CONF_REPORT_COOLDOWN_SEC = _float("FACE_LOW_CONF_REPORT_COOLDOWN_SEC", 3.0)
ENABLE_UNKNOWN_ALERT = _bool("FACE_ENABLE_UNKNOWN_ALERT", True)
UNKNOWN_ALERT_FRAMES = _int("FACE_UNKNOWN_ALERT_FRAMES", 1)
UNKNOWN_TRACK_TTL = _int("FACE_UNKNOWN_TRACK_TTL", 60)
ENABLE_SERVO = _bool("FACE_ENABLE_SERVO", True)
ENABLE_FFMPEG_STREAM = _bool("FACE_ENABLE_FFMPEG_STREAM", True)
FFMPEG_BIN = os.getenv("FACE_FFMPEG_BIN", "ffmpeg")
STREAM_URL = os.getenv("FACE_STREAM_URL", "")
STREAM_USE_ANNOTATED = _bool("FACE_STREAM_USE_ANNOTATED", False)
STREAM_WIDTH = _int("FACE_STREAM_WIDTH", 640)
STREAM_HEIGHT = _int("FACE_STREAM_HEIGHT", 320)
STREAM_FPS = _int("FACE_STREAM_FPS", 10)
STREAM_PRESET = os.getenv("FACE_STREAM_PRESET", "ultrafast")
STREAM_GOP = _int("FACE_STREAM_GOP", 12)
STREAM_BITRATE = os.getenv("FACE_STREAM_BITRATE", "1000k")
STREAM_MAXRATE = os.getenv("FACE_STREAM_MAXRATE", "0k")
STREAM_BUFSIZE = os.getenv("FACE_STREAM_BUFSIZE", "0")
ENABLE_INFER_RESIZE = _bool("FACE_ENABLE_INFER_RESIZE", True)
INFER_LONG_SIDE = _int("FACE_INFER_LONG_SIDE", 1280)
ENABLE_SMALL_FACE_X2 = _bool("FACE_ENABLE_SMALL_FACE_X2", True)
SMALL_FACE_MIN_SIZE = _int("FACE_SMALL_FACE_MIN_SIZE", 20)
SMALL_FACE_MAX_SIZE = _int("FACE_SMALL_FACE_MAX_SIZE", 50)
SMALL_FACE_MIN_SHARPNESS = _float("FACE_SMALL_FACE_MIN_SHARPNESS", 20.0)
SMALL_FACE_SCALE = _float("FACE_SMALL_FACE_SCALE", 2.0)
