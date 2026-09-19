import net
import torch
import os
from face_alignment import align
import numpy as np

# ===================== 保留官方原生配置 =====================
adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
    'ir_18': "adaface_ir18_webface4m.ckpt",
}


def load_pretrained_model(architecture='ir_18'):
    # 官方原生模型加载逻辑，完全保留
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    # 官方原生输入转换逻辑，完全保留（修复张量创建警告）
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    # 优化：先转numpy数组再创建张量，消除PyTorch效率警告
    img_array = np.array([brg_img.transpose(2, 0, 1)])
    tensor = torch.from_numpy(img_array).float()
    return tensor


# ===================== 新增人脸识别核心函数 =====================
def extract_face_feature(model, img_path):
    """
    从单张图片提取人脸特征（基于官方逻辑封装）
    参数：
        model: 加载好的AdaFace模型
        img_path: 图片路径
    返回：
        feature: 人脸特征向量（torch张量，1×512）；None（图片/人脸异常时）
    """
    # 1. 人脸对齐（官方接口）
    try:
        aligned_rgb_img = align.get_aligned_face(img_path)
        if aligned_rgb_img is None:
            print(f"错误：{img_path} 未检测到人脸")
            return None
    except Exception as e:
        print(f"人脸对齐失败 {img_path}：{e}")
        return None

    # 2. 转换为模型输入格式
    bgr_tensor_input = to_input(aligned_rgb_img)

    # 3. 提取特征（禁用梯度，提升速度+消除grad_fn）
    with torch.no_grad():
        feature, _ = model(bgr_tensor_input)

    return feature


def calculate_cosine_similarity(feature1, feature2):
    """
    计算两个特征向量的余弦相似度（和官方矩阵乘法逻辑一致）
    参数：
        feature1/feature2: 人脸特征向量（torch张量，1×512）
    返回：
        similarity: 余弦相似度值（float，范围[-1,1]）
    """
    # 官方原生相似度计算逻辑（矩阵乘法）
    similarity = (feature1 @ feature2.T).item()
    return similarity


def face_recognition(model, img_path1, img_path2, threshold=0.6):
    """
    人脸识别主函数：对比两张图片是否为同一个人
    参数：
        model: AdaFace模型
        img_path1/img_path2: 两张人脸图片路径
        threshold: 相似度阈值（默认0.6，越大越严格）
    返回：
        is_match: 是否为同一个人（bool）
        similarity: 相似度值（float）
    """
    # 1. 提取两张人脸的特征
    feature1 = extract_face_feature(model, img_path1)
    feature2 = extract_face_feature(model, img_path2)

    # 2. 特征提取失败时返回默认值
    if feature1 is None or feature2 is None:
        return False, 0.0

    # 3. 计算相似度
    similarity = calculate_cosine_similarity(feature1, feature2)

    # 4. 判断是否匹配
    is_match = similarity >= threshold

    return is_match, similarity


# ===================== 测试示例 =====================
if __name__ == '__main__':
    # 1. 加载官方模型（可选ir_18/ir_50，需确保权重文件路径正确）
    print("加载AdaFace模型...")
    model = load_pretrained_model('ir_18')  # 或 'ir_50'

    # 2. 替换为你的测试图片路径
    img1 = "face_alignment/test_images/test1.jpg"  # 同一个人的图片1
    img2 = "face_alignment/test_images/test2.jpg"  # 同一个人的图片2
    img3 = "test_face_other.jpg"  # 其他人的图片

    # 3. 对比同一张人的两张图片
    print(f"\n对比 {img1} 和 {img2}：")
    is_match, sim = face_recognition(model, img1, img2)
    print(f"相似度：{sim:.4f} | 是否为同一个人：{'是' if is_match else '否'}")

    # 4. 对比不同人的图片
    print(f"\n对比 {img1} 和 {img3}：")
    is_match, sim = face_recognition(model, img1, img3)
    print(f"相似度：{sim:.4f} | 是否为同一个人：{'是' if is_match else '否'}")

    # （可选）保留官方原有的批量相似度矩阵输出逻辑
    # test_image_path = 'face_alignment/test_images'
    # features = []
    # for fname in sorted(os.listdir(test_image_path)):
    #     path = os.path.join(test_image_path, fname)
    #     feature = extract_face_feature(model, path)
    #     if feature is not None:
    #         features.append(feature)
    # if features:
    #     similarity_matrix = torch.cat(features) @ torch.cat(features).T
    #     print("\n批量人脸相似度矩阵：")
    #     print(similarity_matrix)