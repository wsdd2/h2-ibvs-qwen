# -*- coding: utf-8 -*-
"""Left-arm side camera: gap between the right jaw and the panel face.

The pixel gap from the jaw to the panel face is the visual error for chest Z.
This view does not use hand-eye and does not servo chest X/Y.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ibvs_pad_touch.panel.model import PanelModel
from ibvs_pad_touch.perception.localizer import detect_panel_corners


@dataclass(frozen=True)
class SideContact:
    measured: bool
    contact: bool
    gap_px: float | None
    tip_uv: tuple[float, float] | None
    message: str
    blocked: bool = False
    target_uv: tuple[float, float] | None = None


def estimate_side_gap(
    bgr: np.ndarray,
    *,
    model: PanelModel | None = None,
    contact_px: float = 14.0,
    panel_corners: np.ndarray | None = None,
    tip_uv: tuple[float, float] | None = None,
) -> SideContact:
    given = None if panel_corners is None else np.asarray(panel_corners, dtype=np.float32).reshape(-1, 2)
    tip_near_given = (
        given is not None
        and tip_uv is not None
        and _outside_distance(np.asarray(tip_uv, dtype=np.float64), given) <= _near_panel_limit(given)
    )
    if given is not None and (_side_panel_plausible(given, bgr.shape) or tip_near_given):
        corners = given
    else:
        corners = detect_panel_corners(bgr, model)
    if corners is None or not (_side_panel_plausible(corners, bgr.shape) or tip_near_given):
        if tip_uv is not None:
            return SideContact(
                False,
                False,
                None,
                (float(tip_uv[0]), float(tip_uv[1])),
                "侧视未看到面板，先退开",
                blocked=True,
            )
        return SideContact(False, False, None, None, "侧视未看到面板")
    tip = None if tip_uv is None else np.asarray(tip_uv, dtype=np.float64)
    if tip is None:
        tip = _gripper_point_toward_panel(bgr, corners)
    if tip is None:
        return SideContact(False, False, None, None, "侧视未看到右夹爪")
    gap = _outside_distance(tip, corners)
    height = bgr.shape[0]
    if tip_uv is None and (gap > _near_panel_limit(corners) or float(tip[1]) < 0.30 * height):
        return SideContact(False, False, None, None, "侧视未看到右夹爪")
    contact = gap <= float(contact_px)
    message = "侧视接触" if contact else f"侧视伺服 间隙 {gap:.0f}px"
    nearest = _closest_boundary_point(tip, corners)
    return SideContact(
        True,
        contact,
        float(gap),
        (float(tip[0]), float(tip[1])),
        message,
        target_uv=(float(nearest[0]), float(nearest[1])),
    )


def _side_panel_plausible(corners: np.ndarray, shape: tuple[int, ...]) -> bool:
    """A ceiling vent is a dark rectangle. The faceplate sits with the arm, not overhead."""
    height, width = int(shape[0]), int(shape[1])
    quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if float(quad[:, 1].mean()) < 0.32 * height:
        return False
    if float(cv2.contourArea(quad)) < 0.06 * width * height:
        return False
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    short, long = sorted(edges)[0], sorted(edges)[-1]
    if short < 1.0:
        return False
    aspect = long / short
    return 1.6 <= aspect <= 3.6


def _near_panel_limit(corners: np.ndarray) -> float:
    """The closed jaw sits against the faceplate. A dark speck across the room is not it."""
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    span = max(
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[3] - quad[0])),
    )
    return max(48.0, 0.35 * span)


def _closest_boundary_point(point: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Point on the panel outline closest to the jaw. That is the side-view target."""
    quad = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    tip = np.asarray(point, dtype=np.float64).reshape(2)
    best = quad[0]
    best_d = float("inf")
    for index in range(len(quad)):
        start = quad[index]
        end = quad[(index + 1) % len(quad)]
        edge = end - start
        length2 = float(np.dot(edge, edge))
        scale = 0.0 if length2 < 1e-9 else float(np.clip(np.dot(tip - start, edge) / length2, 0.0, 1.0))
        projected = start + scale * edge
        dist = float(np.dot(projected - tip, projected - tip))
        if dist < best_d:
            best_d = dist
            best = projected
    return best


def _outside_distance(point: np.ndarray, corners: np.ndarray) -> float:
    polygon = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    signed = float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True))
    return max(0.0, -signed)


def _gripper_point_toward_panel(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 75)
    panel = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(panel, np.round(corners).astype(np.int32), 255)
    mask[panel > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    best_point = None
    best_gap = None
    min_area = 0.005 * width * height
    max_area = 0.45 * width * height
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        _center, rect_size, _angle = cv2.minAreaRect(contour)
        short, long = sorted((float(rect_size[0]), float(rect_size[1])))
        if short < 8.0 or long / max(short, 1.0) > 3.5:
            continue
        pts = contour.reshape(-1, 2).astype(np.float64)
        gaps = [_outside_distance(pt, corners) for pt in pts[:: max(1, len(pts) // 40)]]
        if not gaps:
            continue
        index = int(np.argmin(gaps))
        gap = gaps[index]
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_point = pts[:: max(1, len(pts) // 40)][index]
    return best_point
