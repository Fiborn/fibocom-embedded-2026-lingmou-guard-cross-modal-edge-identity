import numpy as np
from typing import List, Optional


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个检测框（xyxy格式）的 IoU（Intersection over Union）。
    输入：
      a, b: [x1,y1,x2,y2]
    输出：
      IoU ∈ [0,1]

    说明：
    - xyxy 坐标：左上(x1,y1)，右下(x2,y2)
    - union 加 1e-6 防止除0
    """
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    area_b = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    union = area_a + area_b - inter + 1e-6
    return inter / union


class STrack:
    """
    一个极简轨迹对象（单目标跟踪）。
    仅保存：
    - tlbr: 当前框（xyxy）
    - score: 最近一次匹配到的检测置信
    - track_id: 轨迹唯一ID（自增）
    - time_since_update: 距离上次被匹配更新过去了多少帧（用于lost/清理）
    - hits: 被成功匹配更新的次数（可用于稳定性判断）
    """
    _count = 0  # 全局自增ID计数器

    def __init__(self, tlbr, score: float, cls_id: int = 0):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)
        self.cls_id = int(cls_id)

        # 分配唯一轨迹ID
        self.track_id = int(STrack._count)
        STrack._count += 1

        # 轨迹状态统计
        self.frame_id = 0
        self.time_since_update = 0  # 每帧+1，匹配到检测则归0
        self.hits = 1               # 创建即算一次命中

    @property
    def tlwh(self):
        """
        将 xyxy 转为 tlwh 格式：
          [x1, y1, w, h]
        有些下游算法/显示会用到。
        """
        x1, y1, x2, y2 = self.tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def update(self, tlbr, score: float, frame_id: int):
        """
        当该轨迹匹配到某个检测框时调用：
        - 更新位置（tlbr）
        - 更新置信（score）
        - 刷新frame_id
        - time_since_update 归0
        - hits + 1
        """
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)
        self.frame_id = int(frame_id)
        self.time_since_update = 0
        self.hits += 1

    def mark_missed(self):
        """
        每过一帧，如果没有匹配到检测，就调用一次：
        - time_since_update += 1
        用于判断轨迹是否“丢失”以及是否需要清理。
        """
        self.time_since_update += 1


class BYTETracker:
    """
    CPU-friendly 的简化版 ByteTrack 风格跟踪器（IoU + 贪心匹配）。

    输入 dets 格式：
      (N,6) float32 [x1,y1,x2,y2,score,cls_id]

    核心流程（每帧 update）：
      1) 所有轨迹先“变老”（time_since_update += 1）
      2) 过滤检测（低置信/小框剔除）
      3) tracked_stracks 与 dets 做 IoU 贪心匹配（优先保持稳定ID）
      4) 未匹配的 tracked -> 放入 lost
      5) lost_stracks 再尝试与剩余 dets 匹配（重激活）
      6) 剩余未匹配 dets -> 新建轨迹
      7) lost 太久的轨迹清理
    """

    def __init__(
        self,
        track_thresh: float = 0.3,   # 检测置信阈值：低于此分数不参与跟踪
        track_buffer: int = 120,     # 轨迹保留时长（单位：秒/或相对值，结合frame_rate使用）
        match_thresh: float = 0.6,   # IoU匹配阈值：低于此IoU不匹配
        min_box_area: float = 400.0, # 小框过滤：面积小于此值丢弃
        frame_rate: int = 30,
    ):
        # tracked：当前活跃轨迹（本帧可用于输出/显示）
        self.tracked_stracks: List[STrack] = []
        # lost：暂时丢失但还保留的轨迹（允许在buffer内被重新匹配回来）
        self.lost_stracks: List[STrack] = []

        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.min_box_area = float(min_box_area)

        # max_time_lost：允许lost轨迹最多“失联”多少帧
        # 注意：这里使用 frame_rate * track_buffer
        # 如果 track_buffer 传入的是“秒”，那就是保留 track_buffer 秒；
        # 如果 track_buffer 只是“帧数量”，那这里会被放大（需与你的主程序保持一致）。
        self.max_time_lost = int(frame_rate * track_buffer)

        self.frame_id = 0

    @staticmethod
    def _area_xyxy(b):
        """计算 xyxy 框面积（做小框过滤用）。"""
        return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

    def _greedy_match(self, tracks: List[STrack], dets_xyxy: np.ndarray) -> List[tuple]:
        """
        IoU 贪心匹配：
        - 计算每个 track 与每个 det 的 IoU 矩阵
        - 将所有(track,det)配对按IoU从大到小排序
        - 依次取未使用过的track与det配对，直到IoU < match_thresh

        返回：
          matches: [(ti, dj), ...]
          ti：tracks里的索引
          dj：dets里的索引
        """
        if len(tracks) == 0 or dets_xyxy.shape[0] == 0:
            return []

        # IoU矩阵：行=tracks，列=dets
        iou_mat = np.zeros((len(tracks), dets_xyxy.shape[0]), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j in range(dets_xyxy.shape[0]):
                iou_mat[i, j] = iou_xyxy(t.tlbr, dets_xyxy[j])

        matches = []
        used_t = set()
        used_d = set()

        # 枚举所有配对并按 IoU 降序
        pairs = [(i, j, float(iou_mat[i, j])) for i in range(iou_mat.shape[0]) for j in range(iou_mat.shape[1])]
        pairs.sort(key=lambda x: x[2], reverse=True)

        # 贪心选取
        for i, j, v in pairs:
            # 因为已经按降序排序，一旦 v < match_thresh，后面只会更小，可直接break
            if v < self.match_thresh:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            matches.append((i, j))
        return matches

    def update(self, dets: Optional[np.ndarray]) -> List[STrack]:
        """
        每帧调用一次：
        输入 dets：当前帧的检测结果（可能为空/None）
        输出：当前活跃tracked_stracks（用于绘制/下游处理）
        """
        self.frame_id += 1

        # 1) 所有轨迹先“老化”：默认假设本帧没匹配到检测，time_since_update+1
        for t in self.tracked_stracks:
            t.mark_missed()
        for t in self.lost_stracks:
            t.mark_missed()

        # 2) 准备检测：None/空 -> 统一成 shape=(0,6)
        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)

            # 过滤检测：低分数 & 小框直接丢弃（减少误检引发的ID抖动）
            keep = []
            for i in range(dets.shape[0]):
                x1, y1, x2, y2, s, _ = dets[i]
                if s < self.track_thresh:
                    continue
                if self._area_xyxy(dets[i, :4]) < self.min_box_area:
                    continue
                keep.append(i)
            dets = dets[keep] if len(keep) > 0 else np.zeros((0, 6), dtype=np.float32)

        dets_xyxy = dets[:, :4] if dets.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32)

        # 3) 优先将 dets 与 tracked_stracks 匹配（保持稳定ID）
        matches = self._greedy_match(self.tracked_stracks, dets_xyxy)

        matched_t = set(i for i, _ in matches)  # matched track indices
        matched_d = set(j for _, j in matches)  # matched det indices

        # 将匹配到的轨迹更新到本帧检测框
        for ti, dj in matches:
            t = self.tracked_stracks[ti]
            bb = dets[dj, :4]
            sc = float(dets[dj, 4])
            t.update(bb, sc, self.frame_id)

        # 4) 未匹配到检测的 tracked 轨迹 -> 转入 lost（短期保留，等待重激活）
        new_tracked = []
        for i, t in enumerate(self.tracked_stracks):
            if i in matched_t:
                new_tracked.append(t)
            else:
                self.lost_stracks.append(t)
        self.tracked_stracks = new_tracked

        # 5) lost 轨迹尝试重激活：再与 dets 做一次匹配
        #    目的：遮挡/漏检后恢复时尽量沿用原track_id
        if len(self.lost_stracks) > 0 and dets_xyxy.shape[0] > 0:
            matches2 = self._greedy_match(self.lost_stracks, dets_xyxy)
            matched_l = set(i for i, _ in matches2)  # matched lost indices

            for li, dj in matches2:
                t = self.lost_stracks[li]
                bb = dets[dj, :4]
                sc = float(dets[dj, 4])
                t.update(bb, sc, self.frame_id)
                self.tracked_stracks.append(t)  # 重回tracked
                matched_d.add(dj)               # 标记该det已被使用，避免后续新建轨迹

            # 未匹配到的lost继续保留（直到超时清理）
            self.lost_stracks = [t for i, t in enumerate(self.lost_stracks) if i not in matched_l]

        # 6) 剩余未匹配 dets -> 新建轨迹
        for j in range(dets.shape[0]):
            if j in matched_d:
                continue
            bb = dets[j, :4]
            sc = float(dets[j, 4])
            cls_id = int(dets[j, 5]) if dets.shape[1] >= 6 else 0
            self.tracked_stracks.append(STrack(bb, sc, cls_id))

        # 7) 清理：lost 轨迹超过 max_time_lost 帧仍未更新则删除（释放内存/避免ID爆炸）
        self.lost_stracks = [t for t in self.lost_stracks if t.time_since_update <= self.max_time_lost]

        # 返回当前活跃轨迹（用于绘制/识别等）
        return list(self.tracked_stracks)