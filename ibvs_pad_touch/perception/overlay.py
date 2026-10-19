from __future__ import annotations

import cv2
import numpy as np

from ibvs_pad_touch.panel.render import draw_rect_on_image
from ibvs_pad_touch.types import Localization

COLORS = {
    "button_1": (255, 180, 0),
    "button_2": (0, 220, 255),
    "button_3": (0, 90, 255),
}
LABELS = {"button_1": "B1 <", "button_2": "B2 >", "button_3": "B3 OK"}
def draw_seg_masks(bgr: np.ndarray, masks) -> np.ndarray:
    """Contours only. Model confidences live in the reserved HUD panel."""
    if not masks:
        return bgr
    vis = bgr.copy()
    colors = {"panel": (40, 200, 70), "gripper": (255, 200, 0), "screen": (0, 180, 255)}
    for mask in masks:
        poly = np.asarray(mask.polygon, dtype=np.int32).reshape(-1, 1, 2)
        if poly.shape[0] < 3:
            continue
        color = colors.get(mask.name, (200, 200, 200))
        cv2.polylines(vis, [poly], True, color, 2, cv2.LINE_AA)
    return vis


def draw_pose_buttons(bgr: np.ndarray, points, skip=None, tol_px: float = 8.0, conf_min: float = 0.0) -> np.ndarray:
    """Mark pose keypoints that were detected, including ones not used for motion."""
    if not points:
        return bgr
    kept = skip or {}
    vis = None
    height, width = bgr.shape[:2]
    for name in ("button_1", "button_2", "button_3"):
        point = points.get(name)
        if point is None or float(getattr(point, "conf", 1.0)) < float(conf_min):
            continue
        u, v = float(point.u), float(point.v)
        same = kept.get(name)
        if same is not None and abs(float(same[0]) - u) <= tol_px and abs(float(same[1]) - v) <= tol_px:
            continue
        px, py = int(round(u)), int(round(v))
        if not (0 <= px < width and 0 <= py < height):
            continue
        if vis is None:
            vis = bgr.copy()
        color = COLORS.get(name, (0, 255, 0))
        marker = cv2.MARKER_TILTED_CROSS if name == "button_3" else cv2.MARKER_CROSS
        cv2.drawMarker(vis, (px, py), color, marker, 16, 2)
        cv2.circle(vis, (px, py), 8, color, 1, cv2.LINE_AA)
        label = {"button_1": "B1", "button_2": "B2", "button_3": "B3"}.get(name, name)
        cv2.putText(
            vis,
            label,
            (min(width - 28, px + 10), max(16, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return bgr if vis is None else vis


def draw_localization(bgr: np.ndarray, loc: Localization) -> np.ndarray:
    vis = bgr.copy()
    if loc.corners_px:
        pts = np.asarray(loc.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (60, 180, 80))
        vis = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)
        draw_rect_on_image(vis, loc.corners_px, (40, 200, 70), "")
    for name, point in loc.buttons.items():
        color = COLORS.get(name, (0, 255, 0))
        u, v = int(round(point.u)), int(round(point.v))
        marker = cv2.MARKER_TILTED_CROSS if name == "button_3" else cv2.MARKER_CROSS
        cv2.drawMarker(vis, (u, v), color, marker, 16, 2)
        cv2.circle(vis, (u, v), 8, color, 1, cv2.LINE_AA)
    _draw_button_column(vis, loc)
    return vis


def _draw_button_column(vis: np.ndarray, loc: Localization) -> None:
    """Names sit outside the panel so they do not cover the screen or each other."""
    names = [name for name in ("button_1", "button_2", "button_3") if name in loc.buttons]
    if not names:
        return
    height, width = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    rows = []
    max_w = 0
    line_h = 0
    for name in names:
        tag = {
            "pose": " pose",
            "scale": " scale",
            "scale+yellow": " scale",
            "scale+pnp": " pnp",
            "scale+yellow+pnp": " pnp",
        }.get(loc.buttons[name].source, "")
        text = f"{LABELS.get(name, name)}{tag}"
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        rows.append((name, text, th))
        max_w = max(max_w, tw)
        line_h = max(line_h, th)
    if loc.corners_px:
        xs = [float(p[0]) for p in loc.corners_px]
        ys = [float(p[1]) for p in loc.corners_px]
        left, right, top = min(xs), max(xs), min(ys)
    else:
        left = min(loc.buttons[name].u for name in names)
        right = max(loc.buttons[name].u for name in names)
        top = min(loc.buttons[name].v for name in names)
    gap = 14
    if right + gap + max_w + 6 <= width:
        x = int(right + gap)
    else:
        x = int(max(8, left - gap - max_w))
    y = int(max(line_h + 10, top + line_h))
    if x < 260:
        y = max(y, 84)
    step = line_h + 14
    for name, text, _th in rows:
        if y >= height - 4:
            break
        color = COLORS.get(name, (0, 255, 0))
        cv2.putText(vis, text, (x, y), font, scale, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
        y += step
