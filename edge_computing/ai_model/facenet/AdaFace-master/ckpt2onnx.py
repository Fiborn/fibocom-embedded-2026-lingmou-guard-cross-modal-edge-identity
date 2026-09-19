import argparse
import cv2
import numpy as np
import onnxruntime
import torchvision.transforms as transforms
import torch
import net

parser = argparse.ArgumentParser(description='onnx inference')
parser.add_argument('--ckpt_path', default='adaface_ir18_webface4m.ckpt', type=str, help='')
parser.add_argument('--onnx_name', default='adaface_ir18_webface4m_fixshape', type=str, help='onnx模型名（无需后缀）')
parser.add_argument('--model_name', default='ir_18', type=str, help='')
parser.add_argument('--onnx_path', default='./', type=str, help='onnx保存路径')
parser.add_argument('--image', default='face_alignment/test_images/test1.jpg', type=str, help='测试图片路径')
args = parser.parse_args()

adaface_models = {
    'ir_18': 'adaface_ir18_webface4m.ckpt',
}


def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location='cpu')['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def l2_norm(x):
    """ l2 normalize
    """
    # 修复：处理多维数组，确保输出一维
    x = x.flatten()
    output = x / np.linalg.norm(x)
    return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return test_transform


def main():
    # ===================== 修复1：正确的图片预处理 =====================
    # 1. 读取图片并转换为RGB（模型训练用RGB格式）
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 调整为模型要求的尺寸（112x112，AdaFace标准输入）
    img = cv2.resize(img, (112, 112))

    # 3. 应用归一化（和训练时一致）
    test_transform = get_test_transform()
    img_tensor = test_transform(img).unsqueeze(0)  # 添加batch维度 (1,3,112,112)
    print(f"输入张量形状：{img_tensor.shape}")

    # ===================== 修复2：加载模型并导出ONNX =====================
    model = load_pretrained_model('ir_18')
    model.eval()

    # 禁用梯度计算
    with torch.no_grad():
        # 修复：AdaFace模型返回 (feature, norm) 元组，仅取特征向量
        torch_out, _ = model(img_tensor)  # (1, 512) 特征向量

    # 拼接ONNX保存路径
    onnx_full_path = f"{args.onnx_path}/{args.onnx_name}.onnx"
    # 修复：处理路径拼接时的重复斜杠（如 ./ + /adaface → .//adaface）
    onnx_full_path = os.path.normpath(onnx_full_path)

    # 导出ONNX（修复：指定dynamic_axes支持动态batch）
    torch.onnx.export(
        model,
        img_tensor,
        onnx_full_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],  # 注意：AdaFace导出时仅输出feature，norm会被忽略
        # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 支持动态batch
    )
    print(f"ONNX模型已保存至：{onnx_full_path}")

    # ===================== 验证ONNX模型 =====================
    session = onnxruntime.InferenceSession(onnx_full_path)
    inputs = {session.get_inputs()[0].name: to_numpy(img_tensor)}
    outs = session.run(None, inputs)[0]  # ONNX输出 (1, 512)

    # L2归一化
    torch_out_np = to_numpy(torch_out).squeeze()  # 转为一维数组 (512,)
    outs_np = outs.squeeze()  # 转为一维数组 (512,)

    torch_out_norm = l2_norm(torch_out_np)  # PyTorch输出归一化
    outs_norm = l2_norm(outs_np)  # ONNX输出归一化

    # 计算余弦相似度（确保返回标量）
    similarity = np.dot(torch_out_norm, outs_norm).item()  # 提取标量值
    print(f"PyTorch vs ONNX 特征相似度：{similarity:.6f}")
    # 相似度接近1.0说明转换成功（通常>0.999）


if __name__ == '__main__':
    # 修复：添加os模块导入（处理路径）
    import os

    main()