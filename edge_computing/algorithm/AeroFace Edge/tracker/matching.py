import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import time


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1, dtype=np.int32)
    m2 = np.asarray(m2, dtype=np.int32)

    if len(m1) == 0 or len(m2) == 0:
        unmatched_O = tuple(range(O))
        unmatched_Q = tuple(range(Q))
        return [], unmatched_O, unmatched_Q

    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P)
    )
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q)
    )

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    if len(indices) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )

    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )

    matches = []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    matches = np.asarray(matches, dtype=np.int32)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute IoU matrix
    atlbrs: list[np.ndarray] or np.ndarray, shape [N,4]
    btlbrs: list[np.ndarray] or np.ndarray, shape [M,4]
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    atlbrs = np.asarray(atlbrs, dtype=np.float64).reshape(-1, 4)
    btlbrs = np.asarray(btlbrs, dtype=np.float64).reshape(-1, 4)

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)

    ious = bbox_ious(atlbrs, btlbrs)
    return np.asarray(ious, dtype=np.float64)


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    atracks: list[STrack] or np.ndarray
    btracks: list[STrack] or np.ndarray
    return: cost_matrix np.ndarray
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray):
        atlbrs = np.asarray(atracks, dtype=np.float64).reshape(-1, 4)
    else:
        atlbrs = np.asarray([track.tlbr for track in atracks], dtype=np.float64).reshape(-1, 4) \
            if len(atracks) > 0 else np.zeros((0, 4), dtype=np.float64)

    if len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        btlbrs = np.asarray(btracks, dtype=np.float64).reshape(-1, 4)
    else:
        btlbrs = np.asarray([track.tlbr for track in btracks], dtype=np.float64).reshape(-1, 4) \
            if len(btracks) > 0 else np.zeros((0, 4), dtype=np.float64)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1.0 - _ious
    return np.asarray(cost_matrix, dtype=np.float64)


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU using pred_bbox
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray):
        atlbrs = np.asarray(atracks, dtype=np.float64).reshape(-1, 4)
    else:
        atlbrs = np.asarray(
            [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks],
            dtype=np.float64
        ).reshape(-1, 4) if len(atracks) > 0 else np.zeros((0, 4), dtype=np.float64)

    if len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        btlbrs = np.asarray(btracks, dtype=np.float64).reshape(-1, 4)
    else:
        btlbrs = np.asarray(
            [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks],
            dtype=np.float64
        ).reshape(-1, 4) if len(btracks) > 0 else np.zeros((0, 4), dtype=np.float64)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1.0 - _ious
    return np.asarray(cost_matrix, dtype=np.float64)


def embedding_distance(tracks, detections, metric='cosine'):
    """
    tracks: list[STrack]
    detections: list[BaseTrack]
    return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)

    cost_matrix = np.maximum(
        0.0,
        cdist(track_features, det_features, metric).astype(np.float64)
    )
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections], dtype=np.float64)

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections], dtype=np.float64)

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric='maha'
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1.0 - lambda_) * gating_distance

    return cost_matrix.astype(np.float64)


def fuse_iou(cost_matrix, tracks, detections):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    if cost_matrix.size == 0:
        return cost_matrix

    reid_sim = 1.0 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1.0 - iou_dist
    fuse_sim = reid_sim * (1.0 + iou_sim) / 2.0

    det_scores = np.array([det.score for det in detections], dtype=np.float64)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    fuse_cost = 1.0 - fuse_sim
    return fuse_cost.astype(np.float64)


def fuse_score(cost_matrix, detections):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections], dtype=np.float64)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    fuse_sim = iou_sim * det_scores
    fuse_cost = 1.0 - fuse_sim
    return fuse_cost.astype(np.float64)