import numpy as np
import cv2
from typing import Optional, List, Tuple


def l2_norm(x: np.ndarray) -> np.ndarray:
    """
    对向量做L2归一化（单位向量）。
    目的：
    - 余弦相似度 = dot(a, b)（当a、b都L2归一化时）
    - 提升不同样本embedding尺度一致性
    """
    x = x.reshape(-1).astype(np.float32)
    n = np.linalg.norm(x) + 1e-12  # 防止除0
    return (x / n).astype(np.float32)


# 112x112 的 5点关键点模板（ArcFace常用）
# 顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
TEMPLATE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AdaFaceONNX:
    """
    AdaFace embedding extractor（ONNXRuntime推理版）

    输入：
    - face_bgr: 人脸ROI（OpenCV读到的BGR图像，uint8）
    - kps5（可选）: 5点关键点(5,2)
        * 必须与 face_bgr 的坐标系一致（ROI内坐标）
        * 如果你拿到的是整帧关键点，需要先减去ROI左上角做坐标转换

    输出：
    - embedding: L2归一化后的特征向量（float32）
    """

    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (112, 112),
        use_cuda: bool = False,
        num_threads: Optional[int] = None,
    ):
        # 模型路径与输入尺寸
        self.onnx_path = onnx_path
        self.input_size = tuple(map(int, input_size))

        # 延迟导入，避免无ORT环境时报错影响其他模块
        import onnxruntime as ort
        self._ort = ort

        # 默认用CPU执行
        self.providers = ["CPUExecutionProvider"]

        # 如果用户希望用CUDA且本机ORT支持CUDA provider，则启用
        if use_cuda:
            if "CUDAExecutionProvider" in self._ort.get_available_providers():
                # CUDA优先，CPU兜底
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Session配置：图优化 + 线程数
        sess_opts = self._ort.SessionOptions()
        sess_opts.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None:
            # intra_op_num_threads：单算子内部并行线程数（CPU优化常用）
            sess_opts.intra_op_num_threads = int(num_threads)

        # 创建推理Session
        self.sess = self._ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=self.providers)

        # 一般AdaFace模型只有一个输入（1,3,112,112），这里取第一个输入名
        self.input_name = self.sess.get_inputs()[0].name

    def _align(self, face_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
        """
        使用5点关键点对齐人脸（仿射变换）到标准模板坐标（112x112）。
        好处：
        - 统一眼睛/嘴巴位置，降低姿态/裁剪差异带来的embedding漂移
        - 通常比“直接resize”更稳定

        参数：
        - kps5: (5,2)，必须在 face_bgr 的坐标系下（ROI坐标）
        """
        dst = TEMPLATE_5PTS_112.copy()         # 目标模板点
        src = kps5.astype(np.float32)          # 源点（来自检测器）

        # 估计“部分仿射”(不含透视)，LMEDS对离群点更鲁棒一些
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            # 如果关键点异常/退化导致仿射估计失败，退回到简单resize
            return cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)

        # warpAffine：输出大小为 input_size（112x112），边界填0（黑）
        aligned = cv2.warpAffine(face_bgr, M, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0)
        return aligned

    @staticmethod
    def _preprocess_bgr(face_bgr: np.ndarray) -> np.ndarray:
        """
        模型输入预处理：
        - BGR(uint8) -> RGB(float32)
        - /255 到 [0,1]
        - (x-0.5)/0.5 到 [-1,1]
        - HWC -> CHW

        返回：
        - img: (3,H,W) float32
        """
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return img

    def extract(self, face_bgr: np.ndarray, kps5: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        单张人脸提取embedding。

        流程：
        1) 无效输入检查
        2) 有关键点则对齐，否则直接resize
        3) 预处理到 (1,3,112,112)
        4) ONNXRuntime推理
        5) L2归一化输出
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        if kps5 is not None:
            # 若kps来自整帧，调用者必须先转为ROI坐标后再传入
            face_in = self._align(face_bgr, kps5)
        else:
            # 无关键点：只做resize（效果通常不如对齐，但更稳不会失败）
            face_in = cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)

        # (1,3,112,112)
        inp = self._preprocess_bgr(face_in)[None, ...].astype(np.float32)

        # sess.run(None, {...}) 返回所有输出；[0]取第一个输出张量
        out = self.sess.run(None, {self.input_name: inp})[0]

        # 输出L2归一化，便于余弦相似度匹配
        return l2_norm(out)

    def extract_batch(
        self,
        faces_bgr: List[np.ndarray],
        kps5_list: Optional[List[Optional[np.ndarray]]] = None
    ) -> List[Optional[np.ndarray]]:
        """
        批量提取embedding（比逐个sess.run更高效，减少ORT调度开销）。

        设计点：
        - 允许输入里有无效face（None/size=0），这些位置返回None
        - 仅把有效样本stack成一个batch送入模型
        - 输出再按原索引回填，保证与输入顺序一致
        """
        if len(faces_bgr) == 0:
            return []

        # 若未提供关键点列表，则默认全部None（不对齐只resize）
        if kps5_list is None:
            kps5_list = [None] * len(faces_bgr)

        batch = []       # 存放每个样本对齐/resize后的112x112人脸（其中无效样本用None占位）
        valid_idx = []   # 记录有效样本索引，用于后续stack与回填

        # 逐个样本做对齐/resize（对齐会产生不同仿射矩阵，无法在模型里一次性做）
        for i, (fb, kp) in enumerate(zip(faces_bgr, kps5_list)):
            if fb is None or fb.size == 0:
                batch.append(None)
                continue

            if kp is not None:
                face_in = self._align(fb, kp)
            else:
                face_in = cv2.resize(fb, self.input_size, interpolation=cv2.INTER_LINEAR)

            batch.append(face_in)
            valid_idx.append(i)

        # 全部无效则直接返回全None
        if len(valid_idx) == 0:
            return [None] * len(faces_bgr)

        # 把有效样本预处理后stack成 (N,3,112,112)
        inp = np.stack([self._preprocess_bgr(batch[i]) for i in valid_idx], axis=0).astype(np.float32)

        # 推理得到 (N, D)
        outs = self.sess.run(None, {self.input_name: inp})[0]
        outs = outs.astype(np.float32)

        # 回填结果：保持与输入faces_bgr相同的列表长度与索引对应
        res = [None] * len(faces_bgr)
        for k, i in enumerate(valid_idx):
            res[i] = l2_norm(outs[k])

        return res