# coding: utf-8
import os
import time
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quant_pre_process,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

def log(msg: str) -> None:
    print(msg, flush=True)


def find_image_files(image_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, filenames in os.walk(image_dir):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def get_model_input_info(model_path: str) -> Tuple[str, List]:
    sess = create_session(model_path)
    inputs = sess.get_inputs()
    if not inputs:
        raise RuntimeError("模型没有输入节点")

    inp = inputs[0]
    return inp.name, inp.shape


def get_model_opset_version(model_path: str) -> int:
    model = onnx.load(model_path)
    if not model.opset_import:
        raise RuntimeError("模型没有 opset 信息")
    return model.opset_import[0].version


def preprocess_scrfd_image(
    image_path: str,
    input_size: Tuple[int, int],
    mean: float = 127.5,
    std: float = 128.0,
) -> np.ndarray:
    """
    SCRFD 常见预处理：
    1. BGR 读取
    2. resize 到固定尺寸
    3. 转 float32
    4. (img - 127.5) / 128.0
    5. HWC -> CHW
    6. 增加 batch 维
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    width, height = input_size
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> NCHW
    return img.astype(np.float32)


class SCRFDCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_dir: str,
        input_name: str,
        input_size: Tuple[int, int],
        max_samples: int = 100,
    ):
        self.image_dir = image_dir
        self.input_name = input_name
        self.input_size = input_size

        image_files = find_image_files(image_dir)
        if not image_files:
            raise RuntimeError(f"校准目录没有可用图片: {image_dir}")

        self.image_files = image_files[:max_samples]
        self.index = 0

        log(f"[INFO] 校准图片数量: {len(self.image_files)}")
        log(f"[INFO] 模型输入名: {self.input_name}")
        log(f"[INFO] 输入尺寸: {self.input_size[0]}x{self.input_size[1]}")

    def get_next(self):
        while self.index < len(self.image_files):
            image_path = self.image_files[self.index]
            self.index += 1
            try:
                tensor = preprocess_scrfd_image(image_path, self.input_size)
                return {self.input_name: tensor}
            except Exception as e:
                log(f"[WARN] 跳过图片 {image_path}: {e}")
                continue
        return None
def upgrade_opset(model_in: str, model_out: str, target_opset: int = 13):
    import onnx
    from onnx import version_converter

    log(f"[INFO] 升级 ONNX opset -> {target_opset}")
    model = onnx.load(model_in)
    model = version_converter.convert_version(model, target_opset)
    onnx.save(model, model_out)
    log(f"[OK] opset 升级完成: {model_out}")

def quantize_scrfd_static(
    model_fp32: str,
    model_int8: str,
    calib_dir: str,
    max_samples: int,
    force_per_channel: bool = True,
):
    model_prep = model_fp32.replace(".onnx", "_prep.onnx")
    # 0️⃣ 先升级opset
    model_op = model_fp32.replace(".onnx", "_op13.onnx")
    upgrade_opset(model_fp32, model_op, 13)

    # 1️⃣ 再预处理
    model_prep = model_op.replace(".onnx", "_prep.onnx")
    log("[INFO] 开始模型预处理（graph优化）...")
    quant_pre_process(
        input_model=model_op,
        output_model_path=model_prep,   # 这里改对
        skip_symbolic_shape=True,       # 对检测模型通常更稳，避免某些 shape infer 问题
    )
    log(f"[OK] 预处理完成: {model_prep}")

    input_name, input_shape = get_model_input_info(model_prep)
    opset = get_model_opset_version(model_prep)

    log(f"[INFO] ONNX opset version: {opset}")

    if len(input_shape) != 4:
        raise RuntimeError(f"不支持的输入形状: {input_shape}")

    h = input_shape[2]
    w = input_shape[3]

    if not isinstance(h, int) or not isinstance(w, int):
        raise RuntimeError(
            f"模型输入尺寸不是静态值，当前为: {input_shape}。"
            f"请先导出固定输入尺寸的 ONNX。"
        )

    reader = SCRFDCalibrationDataReader(
        image_dir=calib_dir,
        input_name=input_name,
        input_size=(w, h),
        max_samples=max_samples,
    )

    use_per_channel = force_per_channel and opset >= 13
    if force_per_channel and opset < 13:
        log("[WARN] 当前模型 opset < 13，QDQ 格式下不支持 per_channel，已自动关闭")

    log(f"[INFO] per_channel = {use_per_channel}")
    log("[INFO] 开始静态量化（基于预处理模型）...")

    quantize_static(
        model_input=model_prep,
        model_output=model_int8,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        per_channel=use_per_channel,
    )

    log(f"[OK] 静态量化完成: {model_int8}")


def get_model_file_size_mb(model_path: str) -> float:
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024.0 * 1024.0)


def print_model_size_compare(model_fp32: str, model_int8: str):
    fp32_mb = get_model_file_size_mb(model_fp32)
    int8_mb = get_model_file_size_mb(model_int8)
    ratio = int8_mb / fp32_mb if fp32_mb > 1e-9 else 0.0
    reduce_pct = (1.0 - ratio) * 100.0 if fp32_mb > 1e-9 else 0.0

    log("[INFO] ===== 模型大小对比 =====")
    log(f"[INFO] FP32 大小: {fp32_mb:.2f} MB")
    log(f"[INFO] INT8 大小: {int8_mb:.2f} MB")
    log(f"[INFO] INT8 / FP32: {ratio:.3f}")
    log(f"[INFO] 大小减少: {reduce_pct:.2f}%")


def inspect_qdq_nodes(model_path: str):
    model = onnx.load(model_path)

    q_count = 0
    dq_count = 0
    qlinear_conv_count = 0
    qlinear_matmul_count = 0

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            q_count += 1
        elif node.op_type == "DequantizeLinear":
            dq_count += 1
        elif node.op_type == "QLinearConv":
            qlinear_conv_count += 1
        elif node.op_type == "QLinearMatMul":
            qlinear_matmul_count += 1

    log("[INFO] ===== 量化节点统计 =====")
    log(f"[INFO] QuantizeLinear 数量: {q_count}")
    log(f"[INFO] DequantizeLinear 数量: {dq_count}")
    log(f"[INFO] QLinearConv 数量: {qlinear_conv_count}")
    log(f"[INFO] QLinearMatMul 数量: {qlinear_matmul_count}")


def create_session(model_path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    return sess
def run_model_outputs(model_path: str, image_path: str):
    input_name, input_shape = get_model_input_info(model_path)

    if len(input_shape) != 4:
        raise RuntimeError(f"不支持的输入形状: {input_shape}")

    h = input_shape[2]
    w = input_shape[3]
    if not isinstance(h, int) or not isinstance(w, int):
        raise RuntimeError(f"模型输入尺寸异常: {input_shape}")

    sess = create_session(model_path)

    x = preprocess_scrfd_image(image_path, (w, h))
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, {input_name: x})

    return output_names, outputs
def compare_model_outputs(model_fp32: str, model_int8: str, test_image: str):
    names_fp32, outputs_fp32 = run_model_outputs(model_fp32, test_image)
    names_int8, outputs_int8 = run_model_outputs(model_int8, test_image)

    if names_fp32 != names_int8:
        raise RuntimeError(
            f"FP32 和 INT8 输出名不一致:\n"
            f"FP32: {names_fp32}\n"
            f"INT8: {names_int8}"
        )

    log("[INFO] ===== 输出误差对比 =====")

    global_abs_max = 0.0
    global_mean_abs = 0.0
    valid_count = 0

    for name, a, b in zip(names_fp32, outputs_fp32, outputs_int8):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        diff = np.abs(a - b)
        abs_max = float(np.max(diff))
        abs_mean = float(np.mean(diff))

        denom = np.maximum(np.abs(a), 1e-6)
        rel_diff = diff / denom
        rel_max = float(np.max(rel_diff))
        rel_mean = float(np.mean(rel_diff))

        log(
            f"[INFO] {name:10s} | "
            f"shape={list(a.shape)} | "
            f"abs_max={abs_max:.6f} | "
            f"abs_mean={abs_mean:.6f} | "
            f"rel_max={rel_max:.6f} | "
            f"rel_mean={rel_mean:.6f}"
        )

        global_abs_max = max(global_abs_max, abs_max)
        global_mean_abs += abs_mean
        valid_count += 1

    if valid_count > 0:
        global_mean_abs /= valid_count

    log("[INFO] ===== 误差汇总 =====")
    log(f"[INFO] 所有输出头最大 abs_max: {global_abs_max:.6f}")
    log(f"[INFO] 所有输出头平均 abs_mean: {global_mean_abs:.6f}")


def benchmark_model(model_path: str, test_image: str, warmup: int = 10, runs: int = 50):
    input_name, input_shape = get_model_input_info(model_path)

    h = input_shape[2]
    w = input_shape[3]
    if not isinstance(h, int) or not isinstance(w, int):
        raise RuntimeError(f"模型输入尺寸异常: {input_shape}")

    sess = create_session(model_path)

    x = preprocess_scrfd_image(test_image, (w, h))
    outputs = [o.name for o in sess.get_outputs()]

    for _ in range(warmup):
        _ = sess.run(outputs, {input_name: x})

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(outputs, {input_name: x})
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / avg_ms if avg_ms > 1e-6 else 0.0

    log(f"[INFO] 模型: {model_path}")
    log(f"[INFO] 平均推理耗时: {avg_ms:.2f} ms")
    log(f"[INFO] 估算FPS: {fps:.2f}")

    return avg_ms, fps

def validate_quantized_model(
    model_fp32: str,
    model_int8: str,
    test_image: str,
    warmup: int = 10,
    runs: int = 50,
):
    if not os.path.isfile(model_fp32):
        raise FileNotFoundError(f"找不到 FP32 模型: {model_fp32}")
    if not os.path.isfile(model_int8):
        raise FileNotFoundError(f"找不到 INT8 模型: {model_int8}")
    if not os.path.isfile(test_image):
        raise FileNotFoundError(f"找不到测试图片: {test_image}")

    print_model_size_compare(model_fp32, model_int8)
    inspect_qdq_nodes(model_int8)
    compare_model_outputs(model_fp32, model_int8, test_image)

    log("[INFO] ===== Benchmark 对比 =====")
    fp32_ms, fp32_fps = benchmark_model(model_fp32, test_image, warmup=warmup, runs=runs)
    int8_ms, int8_fps = benchmark_model(model_int8, test_image, warmup=warmup, runs=runs)

    speedup = fp32_ms / int8_ms if int8_ms > 1e-9 else 0.0
    fps_gain = int8_fps / fp32_fps if fp32_fps > 1e-9 else 0.0

    log("[INFO] ===== 性能汇总 =====")
    log(f"[INFO] FP32 平均耗时: {fp32_ms:.2f} ms, FPS: {fp32_fps:.2f}")
    log(f"[INFO] INT8 平均耗时: {int8_ms:.2f} ms, FPS: {int8_fps:.2f}")
    log(f"[INFO] 耗时加速比 FP32/INT8: {speedup:.3f}x")
    log(f"[INFO] FPS 提升比 INT8/FP32: {fps_gain:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="SCRFD ONNX Runtime 静态 INT8 量化工具")
    parser.add_argument(
        "--model_fp32",
        type=str,
        default="./models/scrfd_500m_bnkps_shape640x640.onnx",
        help="原始 FP32 ONNX 模型路径",
    )
    parser.add_argument(
        "--model_int8",
        type=str,
        default="./models/scrfd_500m_bnkps_shape640x640_int8.onnx",
        help="输出 INT8 ONNX 模型路径",
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=r"Y:\survey_demo\datasets\VSDN\VSDN\NIRX128\X8",
        help="校准图片目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=60000,
        help="最多使用多少张校准图，默认 100",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="量化后顺便做模型大小/QDQ节点/误差/测速验证",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default="imagesX8test/person_27/14.bmp",
        help="测速和误差对比用的单张测试图片路径",
    )
    parser.add_argument(
        "--force_per_channel",
        action="store_true",
        help="尝试启用 per_channel；若 opset < 13 会自动关闭",
    )
    parser.add_argument(
        "--skip_quant",
        action="store_true",
        help="跳过量化，只对已有 INT8 模型做验证",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_fp32):
        raise FileNotFoundError(f"找不到模型: {args.model_fp32}")

    log(f"[INFO] ONNX Runtime version: {ort.__version__}")
    log(f"[INFO] 开始处理模型: {args.model_fp32}")

    # 先检查 ONNX 是否可读
    _ = onnx.load(args.model_fp32)

    if not args.skip_quant:
        if not os.path.isdir(args.calib_dir):
            raise NotADirectoryError(f"找不到校准目录: {args.calib_dir}")

        quantize_scrfd_static(
            model_fp32=args.model_fp32,
            model_int8=args.model_int8,
            calib_dir=args.calib_dir,
            max_samples=args.max_samples,
            force_per_channel=args.force_per_channel,
        )
    else:
        log("[INFO] 已跳过量化阶段，直接验证已有 INT8 模型")

    if args.benchmark:
        if not args.test_image:
            raise ValueError("开启 --benchmark 时必须提供 --test_image，且必须是单张图片路径")
        if not os.path.isfile(args.test_image):
            raise FileNotFoundError(f"找不到测试图片: {args.test_image}")

        validate_quantized_model(
            model_fp32=args.model_fp32,
            model_int8=args.model_int8,
            test_image=args.test_image,
            warmup=10,
            runs=50,
        )


if __name__ == "__main__":
    main()