"""Process entrypoint and lifecycle orchestration."""
import logging
import multiprocessing as mp
import signal
import threading

import cv2


def run() -> None:
    import _legacy_engine as engine

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("drone_face_engine.log"), logging.StreamHandler()],
        force=True,
    )
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    cv2.setLogLevel(0)
    signal.signal(signal.SIGINT, engine.signal_handler)
    engine.logger.info("启动人脸识别引擎")
    threading.Thread(target=engine.start_monitor_client, daemon=True).start()
    threading.Thread(target=engine.start_train_client, daemon=True).start()
    engine.main_recognition_loop()
