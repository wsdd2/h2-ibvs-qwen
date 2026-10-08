from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ibvs_pad_touch.types import CameraPoint


def _centroid(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        return contour.reshape(-1, 2).mean(axis=0)
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([0.0, -1.0], dtype=np.float64)
    return vec / norm


def _elongation(contour: np.ndarray) -> float:
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    return float(max(rw, rh) / max(min(rw, rh), 1.0))


def _touches_bottom(contour: np.ndarray, height: int, margin: float) -> bool:
    return bool(np.any(contour.reshape(-1, 2)[:, 1] >= height - margin))


def _touches_right(contour: np.ndarray, width: int, margin: float) -> bool:
    return bool(np.any(contour.reshape(-1, 2)[:, 0] >= width - margin))


def _touches_top(contour: np.ndarray, height: int, margin: float) -> bool:
    return bool(np.any(contour.reshape(-1, 2)[:, 1] <= margin))


def point_inside_panel(
    point: Sequence[float] | np.ndarray,
    panel_corners: Sequence[Sequence[float]] | None,
) -> bool:
    if panel_corners is None or len(panel_corners) == 0:
        return False
    polygon = np.asarray(panel_corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        True,
    ) >= -4.0


def _inside_panel(
    point: np.ndarray,
    panel_corners: Sequence[Sequence[float]] | None,
) -> bool:
    return point_inside_panel(point, panel_corners)
    if panel_corners is None or len(panel_corners) == 0:
        return False
    polygon = np.asarray(panel_corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        True,
    ) >= -4.0


def _border_entry_tip(
    contour: np.ndarray,
    height: int,
    width: int,
    dark_bin: np.ndarray,
) -> Optional[np.ndarray]:
    """Prefer the closed-jaw sharp corner; a transverse bracket may extend farther."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    margin = max(6.0, 0.018 * max(height, width))
    contact = pts[
        (pts[:, 0] >= width - margin)
        | (pts[:, 1] >= height - margin)
        | (pts[:, 1] <= margin)
    ]
    if len(contact) < 2:
        return None
    origin = np.mean(contact, axis=0)
    sharp = _v_tip(contour, origin, dark_bin)
    if sharp is not None:
        return sharp
    # The visible wrist can enter through a broad bottom edge; its contact
    # centroid may lie left of the jaw. Use the camera's bottom-right side as
    # the body reference when choosing the distal end along the PCA axis.
    body_reference = np.array([width - 1.0, height - 1.0], dtype=np.float64)
    return _principal_axis_tip(contour, body_reference)


def _principal_axis_tip(contour: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Ignore transverse pins: select the distal end along the body PCA axis."""
    x, y, width, height = cv2.boundingRect(contour)
    roi = np.zeros((height, width), dtype=np.uint8)
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(roi, [shifted], -1, 255, -1)
    rows, cols = np.where(roi > 0)
    if len(rows) < 8:
        pts = contour.reshape(-1, 2).astype(np.float64)
        return pts[int(np.argmax(np.linalg.norm(pts - origin, axis=1)))]
    pixels = np.column_stack([cols + x, rows + y]).astype(np.float64)
    center = np.mean(pixels, axis=0)
    values, vectors = np.linalg.eigh(np.cov(pixels.T))
    axis = vectors[:, int(np.argmax(values))]
    toward_body = center - origin
    if float(np.dot(axis, toward_body)) < 0.0:
        axis = -axis
    pts = contour.reshape(-1, 2).astype(np.float64)
    projection = (pts - origin) @ axis
    return pts[int(np.argmax(projection))]


def dark_mask(bgr: np.ndarray) -> np.ndarray:
    """Closed Dex1 plastic is dark. No close(): that welds jaws to speckled floor."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray < 88, 255, 0).astype(np.uint8)


def silver_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pale = cv2.inRange(hsv, (0, 0, 125), (180, 50, 230))
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return pale


def _forearm_contour(silver_bin: np.ndarray) -> Optional[np.ndarray]:
    h, w = silver_bin.shape[:2]
    cnts, _ = cv2.findContours(silver_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [c for c in cnts if 0.02 * h * w <= cv2.contourArea(c) <= 0.26 * h * w]
    if not usable:
        return None
    return max(usable, key=cv2.contourArea)


def _outward_extreme(contour: np.ndarray, origin: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    dist = np.linalg.norm(pts - origin, axis=1)
    return pts[int(np.argmax(dist))]


def _resample_contour(contour: np.ndarray, step: float = 3.0) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return pts
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < step * 4:
        return pts
    samples = np.arange(0.0, total, step)
    out = []
    j = 0
    for s in samples:
        while j + 1 < len(arc) and arc[j + 1] < s:
            j += 1
        t = 0.0 if arc[j + 1] == arc[j] else (s - arc[j]) / (arc[j + 1] - arc[j])
        out.append(closed[j] * (1.0 - t) + closed[j + 1] * t)
    return np.asarray(out, dtype=np.float64)


def _v_tip(contour: np.ndarray, origin: np.ndarray, dark_bin: np.ndarray) -> Optional[np.ndarray]:
    """Closed-jaw V: sharp local corner, medium thickness just inward of the point."""
    pts = _resample_contour(contour, 3.0)
    if len(pts) < 12:
        return None
    dt = cv2.distanceTransform(dark_bin, cv2.DIST_L2, 5)
    win = 8
    best_pt = None
    best = -1.0
    n = len(pts)
    for i, point in enumerate(pts):
        prev = pts[(i - win) % n] - point
        nxt = pts[(i + win) % n] - point
        n1 = float(np.linalg.norm(prev))
        n2 = float(np.linalg.norm(nxt))
        if n1 < 10 or n2 < 10:
            continue
        cos = float(np.clip(np.dot(prev, nxt) / (n1 * n2), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos)))
        if angle > 62.0:
            continue
        inward = _unit(origin - point)
        probe = point + 20.0 * inward
        pr, pc = int(round(probe[1])), int(round(probe[0]))
        if not (0 <= pr < dt.shape[0] and 0 <= pc < dt.shape[1]):
            continue
        thick = float(dt[pr, pc])
        if thick < 5.0 or thick > 13.5:
            continue
        score = (62.0 - angle) * float(np.linalg.norm(point - origin))
        if score > best:
            best = score
            best_pt = point
    return best_pt


def detect_dex1_tip(
    bgr: np.ndarray,
    panel_corners: Sequence[Sequence[float]] | None = None,
) -> Optional[CameraPoint]:
    """Poke point = distal tip of the closed black Dex1 jaws.

    Ignore the side steel pin, the white L-bracket, the forearm pulley, and clothing.

    On the chest camera (light wall / beige panel) the closed Dex1 is a single
    dark silhouette; the tip is the pointed extreme away from the silver wrist.
    Handheld black-floor photos are a harder fallback: find the wrist barrel
    or the hanging V on the large wrist/pulley blob.
    """
    h, w = bgr.shape[:2]
    dark_bin = dark_mask(bgr)
    silver = _forearm_contour(silver_mask(bgr))
    silver_c = _centroid(silver) if silver is not None else None
    dark_cnts, _ = cv2.findContours(dark_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, np.ndarray]] = []
    for contour in dark_cnts:
        area = float(cv2.contourArea(contour))
        frac = area / float(h * w)
        if frac < 0.0038 or frac > 0.18:
            continue
        if float(np.mean(contour.reshape(-1, 2)[:, 1] <= 3)) > 0.10 and frac > 0.02:
            continue
        cen = _centroid(contour)
        if _inside_panel(cen, panel_corners):
            continue
        touches_entry = (
            _touches_bottom(contour, h, 0.07 * h)
            or _touches_right(contour, w, 0.07 * w)
            or _touches_top(contour, h, 0.07 * h)
        )
        if (
            touches_entry
            and panel_corners is not None
            and len(panel_corners) > 0
            and 0.006 <= frac <= 0.08
        ):
            tip_pt = _border_entry_tip(contour, h, w, dark_bin)
            if tip_pt is not None and not _inside_panel(tip_pt, panel_corners):
                scored.append(
                    (
                        180.0 + float(np.linalg.norm(tip_pt - cen)),
                        tip_pt,
                    )
                )
            continue
        if _touches_bottom(contour, h, 0.07 * h) and frac > 0.012:
            continue
        if silver is not None:
            dist = cv2.pointPolygonTest(silver, (float(cen[0]), float(cen[1])), True)
            if dist < -0.09 * max(h, w) and frac < 0.08:
                continue
        origin = silver_c if silver_c is not None else cen
        elong = _elongation(contour)

        # Full Dex1 silhouette (typical chest-cam case): pointed extreme away from wrist.
        if 0.006 <= frac <= 0.07 and elong >= 1.35:
            tip_pt = _outward_extreme(contour, origin)
            if silver is None or cv2.pointPolygonTest(silver, (float(tip_pt[0]), float(tip_pt[1])), False) < 0:
                scored.append((80.0 + 400.0 * frac + 15.0 * elong, tip_pt))

        # Side view: pulley+housing+jaws merge. Take the hanging V, not the pin.
        if frac >= 0.08:
            tip_pt = _v_tip(contour, cen, dark_bin)
            if tip_pt is not None:
                scored.append((120.0 + float(np.linalg.norm(tip_pt - cen)), tip_pt))

    if not scored:
        return None
    tip_pt = max(scored, key=lambda item: item[0])[1]
    if not (3.0 <= tip_pt[0] < w - 3.0 and 3.0 <= tip_pt[1] < h - 3.0):
        return None
    return CameraPoint(u=float(tip_pt[0]), v=float(tip_pt[1]), source="dex1_tip")
