import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class SCRFDDetector:
    """
    基于 OpenCV DNN (ONNX) 的 SCRFD 人脸检测器。

    输出：
      - dets: (N,5) float32  [x1,y1,x2,y2,score]（原图坐标）
      - kpss: (N,5,2) float32 关键点（原图坐标），若模型支持

    注意：
    - 输入图像会先 resize + pad 到固定尺寸（如640x640）
    - 检测结果最终会映射回原始图像坐标
    """

    def __init__(
        self,
        onnx_path: str,
        conf: float = 0.4,             # 置信度阈值
        nms: float = 0.4,              # NMS阈值
        input_size: int = 640,         # 模型输入尺寸
        num_anchors: int = 2,          # 每个位置的anchor数量（视模型而定）
        feat_strides: Tuple[int, ...] = (8, 16, 32),  # FPN特征层步长
        topk: int = 200,               # NMS前保留最高topK
    ):
        self.inpWidth = int(input_size)
        self.inpHeight = int(input_size)
        self.confThreshold = float(conf)
        self.nmsThreshold = float(nms)
        self._feat_stride_fpn = list(feat_strides)
        self._num_anchors = int(num_anchors)
        self.fmc = len(self._feat_stride_fpn)  # 特征层数量
        self.topk = int(topk)

        # 加载ONNX模型
        self.net = cv2.dnn.readNet(onnx_path)

        # 预生成 anchor 中心点（固定输入尺寸下可缓存）
        self._anchor_cache: Dict[int, np.ndarray] = {}
        for s in self._feat_stride_fpn:
            self._anchor_cache[s] = self._make_anchors(
                self.inpHeight, self.inpWidth, s, self._num_anchors
            )

    @staticmethod
    def _make_anchors(inp_h: int, inp_w: int, stride: int, num_anchors: int) -> np.ndarray:
        """
        生成某一FPN层的anchor中心点。

        每个stride对应一个特征图：
            H = inp_h / stride
            W = inp_w / stride

        返回：
            (H*W*num_anchors, 2) 中心坐标
        """
        height = inp_h // stride
        width = inp_w // stride

        # 网格生成
        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1
        ).astype(np.float32)

        # 转换到原图尺度
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))

        # 多anchor复制
        if num_anchors > 1:
            anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

        return anchor_centers  # (M,2)

    def resize_image(self, srcimg: np.ndarray):
        """
        等比例缩放 + 居中padding到 (inpHeight, inpWidth)

        返回：
            padded_img
            newh,neww   缩放后尺寸
            padh,padw   padding偏移
            scale       缩放比例
        """
        h, w = srcimg.shape[:2]
        scale = min(self.inpWidth / w, self.inpHeight / h)

        neww, newh = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)

        padw = (self.inpWidth - neww) // 2
        padh = (self.inpHeight - newh) // 2

        img_padded = np.zeros((self.inpHeight, self.inpWidth, 3), dtype=np.uint8)
        img_padded[padh:padh + newh, padw:padw + neww] = img

        return img_padded, newh, neww, padh, padw, scale

    @staticmethod
    def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape=None):
        """
        将网络预测的距离 (l,t,r,b) 解码为 bbox (x1,y1,x2,y2)。

        原理：
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b
        """
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)
        if len(distance.shape) == 1:
            distance = distance.reshape(-1, 4)

        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])

        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def distance2kps(points, distance, max_shape=None):
        """
        解码关键点预测（5点 = 10维）。

        每两个数表示一个点的偏移：
            px = cx + dx
            py = cy + dy
        """
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)

        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]

            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])

            preds.append(px)
            preds.append(py)

        return np.stack(preds, axis=-1)

    def detect(self, srcimg: np.ndarray):
        img, newh, neww, padh, padw, scale = self.resize_image(srcimg)

        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 128,
            (self.inpWidth, self.inpHeight),
            (127.5, 127.5, 127.5),
            swapRB=True
        )

        self.net.setInput(blob)
        out_names = self.net.getUnconnectedOutLayersNames()
        outs = self.net.forward(out_names)

        # 按名字建立映射，不能按 outs 下标硬取
        out_map = {}
        for name, out in zip(out_names, outs):
            out_map[name] = out

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for stride in self._feat_stride_fpn:
            score_name = f"score_{stride}"
            bbox_name = f"bbox_{stride}"
            kps_name = f"kps_{stride}"

            if score_name not in out_map or bbox_name not in out_map:
                continue

            scores = out_map[score_name].reshape(-1, 1)
            bbox_preds = out_map[bbox_name].reshape(-1, 4) * stride

            anchor_centers = self._anchor_cache[stride]

            m = min(anchor_centers.shape[0], scores.shape[0], bbox_preds.shape[0])
            if m <= 0:
                continue

            anchor_centers_use = anchor_centers[:m]
            scores = scores[:m]
            bbox_preds = bbox_preds[:m]

            pos_inds = np.where(scores.ravel() >= self.confThreshold)[0]
            if pos_inds.size == 0:
                continue

            bboxes = self.distance2bbox(anchor_centers_use, bbox_preds)

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if kps_name in out_map:
                kps_preds = out_map[kps_name].reshape(-1, 10) * stride
                kps_preds = kps_preds[:m]
                kpss = self.distance2kps(anchor_centers_use, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos_inds])

        if len(scores_list) == 0:
            return np.zeros((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list).ravel().astype(np.float32)
        bboxes = np.vstack(bboxes_list).astype(np.float32)

        kpss = None
        if len(kpss_list) > 0:
            kpss = np.vstack(kpss_list).astype(np.float32)

        # 映射回原图坐标
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - padw) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - padh) / scale

        if kpss is not None:
            kpss[:, :, 0] = (kpss[:, :, 0] - padw) / scale
            kpss[:, :, 1] = (kpss[:, :, 1] - padh) / scale

        h0, w0 = srcimg.shape[:2]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w0 - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w0 - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h0 - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h0 - 1)

        if scores.size > self.topk:
            order = np.argsort(-scores)[:self.topk]
            scores = scores[order]
            bboxes = bboxes[order]
            if kpss is not None:
                kpss = kpss[order]

        bboxes_xywh = bboxes.copy()
        bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
        bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]

        idxs = cv2.dnn.NMSBoxes(
            bboxes_xywh[:, :4].tolist(),
            scores.tolist(),
            self.confThreshold,
            self.nmsThreshold
        )

        if len(idxs) == 0:
            return np.zeros((0, 5), dtype=np.float32), None

        keep = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]

        dets = np.concatenate(
            [bboxes[keep], scores[keep, None]],
            axis=1
        ).astype(np.float32)

        if kpss is not None:
            kpss = kpss[keep]

        return dets, kpss


    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个xyxy框的IoU。"""
        x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter + 1e-6
        return inter / union
    def associate_dets_to_tracks(self, dets, kpss, tracks, iou_thr=0.5):
        """
        将检测信息（score / kps）通过 IoU 最大匹配关联到 track。

        对每个track：
            找 IoU 最大的 detection
            若 IoU >= 阈值，则绑定

        返回：
            {track_id: {"score":..., "kps":...}}
        """
        out = {}
        if dets is None or len(dets) == 0 or tracks is None or len(tracks) == 0:
            return out

        det_boxes = dets[:, :4]

        for trk in tracks:
            tb = np.array(trk.tlbr, dtype=np.float32)

            best_iou = 0.0
            best_j = -1

            for j in range(det_boxes.shape[0]):
                iou = self._iou_xyxy(tb, det_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= iou_thr:
                info = {"score": float(dets[best_j, 4])}
                info["kps"] = kpss[best_j] if kpss is not None else None
                out[int(trk.track_id)] = info

        return out