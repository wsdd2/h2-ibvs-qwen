from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ibvs_pad_touch.types import Intrinsics


def sample_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 2,
    percentile: float = 50.0,
) -> Optional[float]:
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None
    patch = depth_m[max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 2.5)]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, max(0.0, min(100.0, float(percentile)))))


def point_from_uvz(u: float, v: float, z_m: float, K: Intrinsics, source: str = "") -> tuple[float, float, float]:
    return K.backproject(u, v, z_m)


def exposed_panel_depth(
    depth_m: np.ndarray | None,
    corners: Sequence[Sequence[float]] | None,
    exclude_polygons: Sequence[Sequence[Sequence[float]]] = (),
) -> Optional[float]:
    """Median depth of the panel face that is not covered by the gripper or the screen."""
    if depth_m is None or not corners:
        return None
    height, width = depth_m.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    quad = np.round(np.asarray(corners, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    if quad.shape[0] < 3:
        return None
    cv2.fillConvexPoly(mask, quad, 255)
    for poly in exclude_polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 0)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    step = max(1, int(xs.size // 4000))
    vals = depth_m[ys[::step], xs[::step]]
    valid = vals[np.isfinite(vals) & (vals > 0.05) & (vals < 2.5)]
    if valid.size < 20:
        return None
    return float(np.median(valid))
