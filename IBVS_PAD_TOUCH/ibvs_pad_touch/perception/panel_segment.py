from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from ibvs_pad_touch.perception.localizer import detect_lcd_corners, order_corners
from ibvs_pad_touch.perception.yellow_buttons import assign_poke_buttons


def _border_stats(hsv: np.ndarray, margin: int = 16) -> tuple[float, float]:
    top = hsv[:margin]
    bot = hsv[-margin:]
    left = hsv[:, :margin]
    right = hsv[:, -margin:]
    sample = np.concatenate(
        [top.reshape(-1, 3), bot.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    return float(np.median(sample[:, 1])), float(np.median(sample[:, 2]))


def _landmark_points(bgr: np.ndarray) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    lcd = detect_lcd_corners(bgr)
    if lcd is not None:
        points.extend((float(x), float(y)) for x, y in lcd)
        points.append((float(lcd[:, 0].mean()), float(lcd[:, 1].mean())))
    yellow = assign_poke_buttons(bgr)
    if yellow:
        points.extend(yellow.values())
    return points


def _wall_foam_mask(hsv: np.ndarray) -> np.ndarray:
    """White wall + styrofoam: high value, very low saturation."""
    sat, val = _border_stats(hsv)
    bright = hsv[:, :, 2] >= max(165.0, val - 12.0)
    pale = hsv[:, :, 1] <= max(18.0, sat + 12.0)
    return bright & pale


def _seed_mask(bgr: np.ndarray, points: Sequence[tuple[float, float]]) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(10, int(0.02 * min(h, w)))
    mask[:border] = cv2.GC_BGD
    mask[-border:] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    mask[_wall_foam_mask(hsv)] = cv2.GC_BGD

    if len(points) >= 3:
        pts = np.asarray(points, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang = rect
        rw = max(rw, 1.0) * 2.4
        rh = max(rh, 1.0) * 2.6
        probable = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
        probable_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(probable_mask, probable, 1)
        grow = mask == cv2.GC_PR_BGD
        mask[(probable_mask == 1) & grow] = cv2.GC_PR_FGD

    lcd = detect_lcd_corners(bgr)
    if lcd is not None:
        cv2.fillConvexPoly(mask, np.round(lcd).astype(np.int32), int(cv2.GC_FGD))
    for u, v in points:
        cv2.circle(mask, (int(round(u)), int(round(v))), 10, int(cv2.GC_FGD), -1)
    return mask


def _largest_quad(fg: np.ndarray, points: Sequence[tuple[float, float]]) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.03 * fg.shape[0] * fg.shape[1]:
        return None
    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(hull))
    ordered = order_corners(corners)
    if points and not _contains_points(ordered, points):
        return None
    return ordered


def _contains_points(corners: np.ndarray, points: Sequence[tuple[float, float]], pad: float = 8.0) -> bool:
    poly = corners.reshape(-1, 1, 2).astype(np.float32)
    for u, v in points:
        if cv2.pointPolygonTest(poly, (float(u), float(v)), False) < 0:
            # allow a few pixels outside the rounded corners
            if cv2.pointPolygonTest(poly, (float(u), float(v)), True) < -pad:
                return False
    return True


def detect_panel_body(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Segment the beige enclosure from the white wall and styrofoam backer."""
    points = _landmark_points(bgr)
    if len(points) < 3:
        return None
    work = bgr
    scale = 1.0
    if max(bgr.shape[:2]) > 900:
        scale = 800.0 / max(bgr.shape[:2])
        work = cv2.resize(bgr, (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)))
        points = [(u * scale, v * scale) for u, v in points]
    mask = _seed_mask(work, points)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(work, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    fg[_wall_foam_mask(hsv)] = 0
    corners = _largest_quad(fg, points)
    if corners is None:
        return None
    if scale != 1.0:
        corners = corners / scale
    return order_corners(corners)


def panel_corners_from_landmarks(
    bgr: np.ndarray,
    extra_points: Iterable[tuple[float, float]] | None = None,
) -> Optional[np.ndarray]:
    del extra_points
    return detect_panel_body(bgr)
