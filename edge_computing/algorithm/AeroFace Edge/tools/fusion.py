import numpy as np
import cv2
from typing import Deque, Tuple, Optional, Iterable, List


def l2_norm(x: np.ndarray) -> np.ndarray:
    """
    对向量做 L2 归一化。
    用途：
    - 使 embedding 成为单位向量
    - 便于用点积直接计算余弦相似度
    """
    x = x.reshape(-1).astype(np.float32)
    n = np.linalg.norm(x) + 1e-12  # 防止除0
    return (x / n).astype(np.float32)


def calc_sharpness_laplacian(face_bgr: np.ndarray) -> float:
    """
    使用拉普拉斯方差作为清晰度指标（经典简单方法）。
    原理：
    - 图像越清晰，高频信息越多
    - 拉普拉斯算子对边缘敏感
    - 方差越大，通常表示越清晰

    返回：
    - 一个非负实数（未归一化）
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0

    # 转灰度（清晰度通常不需要彩色信息）
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # 拉普拉斯响应的方差
    v = cv2.Laplacian(gray, cv2.CV_64F).var()

    return float(v)


def calc_quality(face_bgr: np.ndarray, det_score: float = 0.0) -> float:
    """
    计算人脸质量分数（大致在 [0,1] 范围内）。

    综合三项因素：
      1) 人脸大小（面积 proxy）
      2) 清晰度（拉普拉斯方差）
      3) 检测置信度（detector score）

    返回：
      q ∈ [0,1]
    """

    if face_bgr is None or face_bgr.size == 0:
        return 0.0

    h, w = face_bgr.shape[:2]
    area = float(h * w)

    # -------------------------
    # 1) 尺寸项（越大越好）
    # 在 120x120 左右开始饱和（再大提升有限）
    # -------------------------
    size_term = min(1.0, area / (120.0 * 120.0))

    # -------------------------
    # 2) 清晰度项（越清晰越好）
    # 在拉普拉斯方差 ~200 左右开始饱和
    # -------------------------
    sharp = calc_sharpness_laplacian(face_bgr)
    sharp_term = min(1.0, sharp / 200.0)

    # -------------------------
    # 3) 检测置信项
    # 限制在 [0,1]
    # -------------------------
    conf_term = float(det_score)
    if conf_term > 1.0:
        conf_term = 1.0
    if conf_term < 0.0:
        conf_term = 0.0

    # -------------------------
    # 加权融合（经验权重）
    # size 权重大（面积直接影响可识别性）
    # sharp 次之
    # det_score 作为辅助
    # -------------------------
    q = (0.45 * size_term) + (0.35 * sharp_term) + (0.20 * conf_term)

    # 最终裁剪到 [0,1]
    return float(max(0.0, min(1.0, q)))


def fuse_embeddings(hist: Iterable[Tuple[np.ndarray, float, int]]) -> Optional[np.ndarray]:
    """
    多帧 embedding 融合函数。

    输入：
      hist: 可迭代对象，每个元素为
            (embedding, quality_score, frame_id)

    融合策略：
      - 根据 quality_score 做加权平均
      - 再做 L2 归一化
      - 提升识别稳定性，降低单帧噪声影响

    输出：
      融合后的 L2-normalized embedding
      若无有效embedding则返回 None
    """

    embs = []
    qs = []

    # 收集有效embedding与质量分
    for emb, q, _ in hist:
        if emb is None:
            continue
        embs.append(emb.reshape(-1).astype(np.float32))
        qs.append(float(q))

    if len(embs) == 0:
        return None

    # 转为numpy数组
    qs = np.asarray(qs, dtype=np.float32)

    # 避免某些权重为0导致完全被忽略
    qs = np.clip(qs, 1e-3, None)

    # 归一化权重（加权平均）
    w = qs / (qs.sum() + 1e-12)

    # 加权求和
    fused = np.sum(np.stack(embs, axis=0) * w[:, None], axis=0)

    # 最后再做一次 L2 归一化
    return l2_norm(fused)