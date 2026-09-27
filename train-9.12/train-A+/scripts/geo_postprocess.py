from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# 几何先验的可调权重；可在实验中改变它们并比较效果。
@dataclass
class GeoConfig:
    alpha: float = 0.85
    beta: float = 0.35
    gamma: float = 0.55
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.12


def sigmoid(x: float) -> float:
    """将任意实数压到 0~1；截断输入以避免 exp 溢出。"""
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def logit(p: float) -> float:
    """sigmoid 的反函数，将 0~1 的置信度转为可加减的分数空间。"""
    p = max(1e-5, min(1.0 - 1e-5, float(p)))
    return math.log(p / (1.0 - p))


def estimate_axis(gray: np.ndarray):
    """由边缘点的主方向估计绝缘子纹理轴线，供后续一维统计使用。"""
    edges = cv2.Canny(gray, 50, 150)
    pts = np.column_stack(np.where(edges > 0))
    if len(pts) < 20:
        return np.array([1.0, 0.0]), edges
    pts = pts.astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    # SVD 的第一主方向就是这些边缘点延展最明显的方向。
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis_yx = vt[0]
    axis_xy = np.array([axis_yx[1], axis_yx[0]], dtype=np.float32)
    norm = np.linalg.norm(axis_xy) + 1e-6
    return axis_xy / norm, edges


def periodic_anomaly_map(image: np.ndarray) -> np.ndarray:
    """生成纹理异常热图：绝缘子规律结构中突出的边缘变化可能是缺陷。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.equalizeHist(gray)
    axis, edges = estimate_axis(gray)
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx - w / 2.0, yy - h / 2.0], axis=-1)
    # 将二维像素投影到主轴，把沿绝缘子方向的纹理变成一维曲线。
    projection = coords[..., 0] * axis[0] + coords[..., 1] * axis[1]
    bins = np.linspace(projection.min(), projection.max(), 96)
    profile = np.zeros(len(bins) - 1, dtype=np.float32)
    for i in range(len(bins) - 1):
        mask = (projection >= bins[i]) & (projection < bins[i + 1])
        if mask.any():
            profile[i] = edges[mask].mean()
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, 9), 0).ravel()
    residual = np.abs(profile - smooth)
    z = (residual - residual.mean()) / (residual.std() + 1e-6)
    z = np.clip(z, 0, 4) / 4.0
    anomaly = np.zeros((h, w), dtype=np.float32)
    idx = np.clip(np.digitize(projection, bins) - 1, 0, len(z) - 1)
    anomaly[:] = z[idx]
    return cv2.GaussianBlur(anomaly, (19, 19), 0)


def score_box_with_geo(image: np.ndarray, box_xyxy, yolo_score: float, cfg: Optional[GeoConfig] = None) -> float:
    """融合 YOLO 置信度、纹理异常和候选框面积合理性，得到最终分数。"""
    cfg = cfg or GeoConfig()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
    area_penalty = 0.0
    if area_ratio < cfg.min_area_ratio:
        area_penalty = (cfg.min_area_ratio - area_ratio) / cfg.min_area_ratio
    elif area_ratio > cfg.max_area_ratio:
        area_penalty = (area_ratio - cfg.max_area_ratio) / cfg.max_area_ratio
    anomaly = periodic_anomaly_map(image)
    geo = float(anomaly[y1:y2, x1:x2].mean()) if (x2 > x1 and y2 > y1) else 0.0
    inside = 1.0 if 0 <= (x1 + x2) / 2.0 <= w and 0 <= (y1 + y2) / 2.0 <= h else 0.0
    # 在 logit 空间相加：异常越明显越加分，面积明显异常则扣分。
    fused = logit(yolo_score) + cfg.alpha * geo + cfg.beta * inside - cfg.gamma * area_penalty
    return sigmoid(fused)


def fuse_predictions(image: np.ndarray, predictions, cfg: Optional[GeoConfig] = None, min_score: float = 0.15):
    """对所有阶段二预测重新打分并滤除低分框。"""
    fused = []
    for pred in predictions:
        x1, y1, x2, y2, score, cls = pred
        final_score = score_box_with_geo(image, (x1, y1, x2, y2), score, cfg)
        if final_score >= min_score:
            fused.append((x1, y1, x2, y2, final_score, cls))
    return fused
