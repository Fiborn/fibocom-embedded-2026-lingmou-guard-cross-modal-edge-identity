import os
import cv2
import numpy as np
import onnxruntime
from face_alignment import align  # AdaFace人脸对齐模块
from PIL import Image  # 新增：导入PIL处理

# ===================== 核心配置 =====================
# ONNX模型路径（替换为你转换后的ONNX文件路径）
ONNX_MODEL_PATH = "./adaface_ir18_webface4m.onnx"
# 人脸识别阈值（范围0~1，越大越严格）
SIMILARITY_THRESHOLD = 0.6
# AdaFace标准输入尺寸
INPUT_SIZE = (112, 112)


# ===================== 工具函数 =====================
def l2_norm(x):
    """L2归一化（和模型训练/转换时一致）"""
    x = x.flatten()  # 确保一维
    return x / np.linalg.norm(x)


def preprocess_face(img_path):
    """
    人脸对齐+预处理（适配ONNX模型输入）
    :param img_path: 图片路径
    :return: 预处理后的张量（np.array, 1x3x112x112）/ None
    """
    # 1. 人脸对齐（AdaFace官方接口，输出PIL Image格式的RGB图片）
    try:
        aligned_rgb_img = align.get_aligned_face(img_path)
        if aligned_rgb_img is None:
            print(f"❌ 图片 {img_path} 未检测到人脸")
            return None
    except Exception as e:
        print(f"❌ 人脸对齐失败 {img_path}：{str(e)}")
        return None

    # 2. 修复核心：PIL Image → numpy 数组
    aligned_rgb_img = np.array(aligned_rgb_img)  # (112,112,3) uint8

    # 3. 预处理（和ONNX转换时一致）
    # - 归一化：(像素值/255 - 0.5) / 0.5 → 范围[-1, 1]
    # - 转置：HWC(112,112,3) → CHW(3,112,112)
    # - 添加batch维度：1x3x112x112
    img = aligned_rgb_img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)  # CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img


def load_onnx_model(onnx_path):
    """加载ONNX模型，返回推理session"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件不存在：{onnx_path}")

    # 构建ONNX推理session（启用优化）
    providers = ['CPUExecutionProvider']  # CPU推理；有GPU可加 'CUDAExecutionProvider'
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=providers,
        sess_options=onnxruntime.SessionOptions()
    )
    # 打印输入输出信息（调试用）
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ ONNX模型加载成功：")
    print(f"   输入节点名：{input_name} | 输入形状：{session.get_inputs()[0].shape}")
    print(f"   输出节点名：{output_name} | 输出形状：{session.get_outputs()[0].shape}")
    return session


def extract_onnx_feature(session, img_input):
    """
    用ONNX模型提取人脸特征
    :param session: ONNX推理session
    :param img_input: 预处理后的输入（1x3x112x112）
    :return: 归一化后的特征向量（np.array, 512维）
    """
    # 获取输入输出节点名
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ONNX推理
    feature = session.run([output_name], {input_name: img_input})[0]
    # L2归一化（提升相似度计算准确性）
    feature_norm = l2_norm(feature)
    return feature_norm


def calculate_cosine_similarity(feature1, feature2):
    """计算两个特征向量的余弦相似度（标量）"""
    similarity = np.dot(feature1, feature2)
    return similarity.item()  # 转为浮点数


# ===================== 人脸识别主函数 =====================
def face_recognition_onnx(img_path1, img_path2):
    """
    基于ONNX模型的人脸识别
    :param img_path1/img_path2: 两张人脸图片路径
    :return: (是否匹配, 相似度值)
    """
    # 1. 加载ONNX模型（仅首次调用时加载）
    if not hasattr(face_recognition_onnx, "session"):
        face_recognition_onnx.session = load_onnx_model(ONNX_MODEL_PATH)
    session = face_recognition_onnx.session

    # 2. 预处理两张图片
    print(f"\n📸 处理图片 {img_path1}...")
    img1_input = preprocess_face(img_path1)
    print(f"📸 处理图片 {img_path2}...")
    img2_input = preprocess_face(img_path2)

    # 3. 预处理失败则返回
    if img1_input is None or img2_input is None:
        return False, 0.0

    # 4. 提取特征
    feature1 = extract_onnx_feature(session, img1_input)
    feature2 = extract_onnx_feature(session, img2_input)

    # 5. 计算相似度并判断
    similarity = calculate_cosine_similarity(feature1, feature2)
    is_match = similarity >= SIMILARITY_THRESHOLD

    # 6. 输出结果
    print(f"\n✅ 人脸识别结果：")
    print(f"   相似度：{similarity:.4f}")
    print(f"   阈值：{SIMILARITY_THRESHOLD}")
    print(f"   判定：{'同一个人' if is_match else '不同的人'}")

    return is_match, similarity


# ===================== 测试示例 =====================
if __name__ == "__main__":
    # 可选：消除numpy的rcond警告（非必需，仅美化输出）
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    # 替换为你的测试图片路径
    TEST_IMG1 = "face_alignment/test_images/test1.jpg"  # 同一个人的图片1
    TEST_IMG2 = "face_alignment/test_images/test2.jpg"  # 同一个人的图片2
    TEST_IMG3 = "face_alignment/test_images/1.bmp"  # 其他人的图片

    # 测试1：同一个人
    print("===== 测试1：同一个人比对 =====")
    face_recognition_onnx(TEST_IMG1, TEST_IMG2)

    # 测试2：不同的人
    print("\n===== 测试2：不同人比对 =====")
    face_recognition_onnx(TEST_IMG1, TEST_IMG3)