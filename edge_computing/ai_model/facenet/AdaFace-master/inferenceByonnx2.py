import os
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime
from PIL import Image

# ===================== 核心配置 =====================
# 1) 人脸识别 ONNX 模型路径（保持不变）
ONNX_MODEL_PATH = "./adaface_ir18_webface4m_fixshape.onnx"

# 2) 人脸检测：OpenCV HaarCascade（不需要 YuNet 模型）
# OpenCV 通常自带这个 xml；若你的环境路径不完整，可手动指定绝对路径
HAAR_XML_PATH = os.environ.get(
    "HAAR_XML_PATH",
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# 3) 阈值
SIMILARITY_THRESHOLD = 0.6

# 4) AdaFace 标准输入尺寸（112x112）
INPUT_SIZE: Tuple[int, int] = (112, 112)

# 5) 裁剪扩展比例（给框周围留点边）
BBOX_EXPAND = float(os.environ.get("BBOX_EXPAND", "1.25"))  # 1.1~1.4 常用


# ===================== 工具函数 =====================
def l2_norm(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    denom = np.linalg.norm(x) + 1e-12
    return x / denom


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _square_bbox(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """把 bbox 调成正方形（以中心为准）"""
    cx = x + w // 2
    cy = y + h // 2
    side = max(w, h)
    nx = cx - side // 2
    ny = cy - side // 2
    return nx, ny, side, side


def _expand_bbox(x: int, y: int, w: int, h: int, expand: float) -> Tuple[int, int, int, int]:
    """按比例放大 bbox（以中心为准）"""
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * expand
    nh = h * expand
    nx = int(round(cx - nw / 2.0))
    ny = int(round(cy - nh / 2.0))
    return nx, ny, int(round(nw)), int(round(nh))


def _clip_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """裁剪到图像范围内"""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    nw = max(1, x2 - x1)
    nh = max(1, y2 - y1)
    return x1, y1, nw, nh


class HaarAligner:
    """
    用 OpenCV HaarCascade 做人脸检测，然后 bbox 裁剪 + resize 到 112x112
    """
    def __init__(self, haar_xml_path: str = HAAR_XML_PATH, crop_size: Tuple[int, int] = (112, 112)):
        if not os.path.exists(haar_xml_path):
            raise FileNotFoundError(
                f"HaarCascade xml 不存在: {haar_xml_path}\n"
                f"请设置环境变量 HAAR_XML_PATH 指向 haarcascade_frontalface_default.xml"
            )
        self.crop_size = crop_size
        self.detector = cv2.CascadeClassifier(haar_xml_path)
        if self.detector.empty():
            raise RuntimeError(f"加载 HaarCascade 失败: {haar_xml_path}")

    def detect_best(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 参数可按场景调整：minSize 太大可能漏检，太小容易误检
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(60, 60),
        )
        if faces is None or len(faces) == 0:
            return None

        # 选面积最大的脸
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(w), int(h)

    def get_aligned_face(self, image_path: str, rgb_pil_image: Optional[Image.Image] = None) -> Optional[Image.Image]:
        img = rgb_pil_image if rgb_pil_image is not None else Image.open(image_path).convert("RGB")
        bgr = _pil_to_bgr(img)
        H, W = bgr.shape[:2]

        det = self.detect_best(bgr)
        if det is None:
            return None

        x, y, w, h = det

        # 扩大 + 变正方形，提升鲁棒性
        x, y, w, h = _expand_bbox(x, y, w, h, BBOX_EXPAND)
        x, y, w, h = _square_bbox(x, y, w, h)
        x, y, w, h = _clip_bbox(x, y, w, h, W, H)

        face_bgr = bgr[y:y + h, x:x + w]
        face_bgr = cv2.resize(face_bgr, self.crop_size, interpolation=cv2.INTER_LINEAR)

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)


# ===================== 预处理 + ONNX 推理 =====================
_aligner: Optional[HaarAligner] = None


def preprocess_face(img_path: str) -> Optional[np.ndarray]:
    global _aligner
    if _aligner is None:
        _aligner = HaarAligner(HAAR_XML_PATH, crop_size=INPUT_SIZE)

    try:
        aligned_rgb_img = _aligner.get_aligned_face(img_path)
        if aligned_rgb_img is None:
            print(f"❌ 图片 {img_path} 未检测到人脸")
            return None
    except Exception as e:
        print(f"❌ 人脸裁剪失败 {img_path}：{str(e)}")
        return None

    aligned_rgb_img = np.array(aligned_rgb_img)  # (112,112,3) uint8

    # 保持与你原来一致的归一化
    img = aligned_rgb_img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)         # CHW
    img = np.expand_dims(img, axis=0)    # 1x3x112x112
    return img


def load_onnx_model(onnx_path: str) -> onnxruntime.InferenceSession:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件不存在：{onnx_path}")

    providers = ["CPUExecutionProvider"]
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_path, providers=providers, sess_options=sess_options)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ ONNX模型加载成功：")
    print(f"   输入节点名：{input_name} | 输入形状：{session.get_inputs()[0].shape}")
    print(f"   输出节点名：{output_name} | 输出形状：{session.get_outputs()[0].shape}")
    return session


def extract_onnx_feature(session: onnxruntime.InferenceSession, img_input: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    feature = session.run([output_name], {input_name: img_input})[0]
    return l2_norm(feature)


def calculate_cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
    return float(np.dot(feature1, feature2).item())


def face_recognition_onnx(img_path1: str, img_path2: str):
    if not hasattr(face_recognition_onnx, "session"):
        face_recognition_onnx.session = load_onnx_model(ONNX_MODEL_PATH)
    session = face_recognition_onnx.session

    print(f"\n📸 处理图片 {img_path1}...")
    img1_input = preprocess_face(img_path1)
    print(f"📸 处理图片 {img_path2}...")
    img2_input = preprocess_face(img_path2)

    if img1_input is None or img2_input is None:
        return False, 0.0

    feature1 = extract_onnx_feature(session, img1_input)
    feature2 = extract_onnx_feature(session, img2_input)

    similarity = calculate_cosine_similarity(feature1, feature2)
    is_match = similarity >= SIMILARITY_THRESHOLD

    print("\n✅ 人脸识别结果：")
    print(f"   相似度：{similarity:.4f}")
    print(f"   阈值：{SIMILARITY_THRESHOLD}")
    print(f"   判定：{'同一个人' if is_match else '不同的人'}")

    return is_match, similarity


if __name__ == "__main__":
    TEST_IMG1 = "face_alignment/test_images/person_081.bmp"
    TEST_IMG2 = "face_alignment/test_images/person_082.bmp"
    TEST_IMG3 = "face_alignment/test_images/test1.jpg"

    print("===== 测试1：同一个人比对 =====")
    face_recognition_onnx(TEST_IMG1, TEST_IMG2)

    print("\n===== 测试2：不同人比对 =====")
    face_recognition_onnx(TEST_IMG1, TEST_IMG3)