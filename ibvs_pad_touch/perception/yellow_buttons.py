from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class YellowBar:
    cx: float
    cy: float
    width: float
    height: float
    contour: np.ndarray

    @property
    def left(self) -> float:
        return self.cx - 0.5 * self.width

    @property
    def right(self) -> float:
        return self.cx + 0.5 * self.width


@dataclass(frozen=True)
class OrientedYellowBar:
    cx: float
    cy: float
    major: float
    minor: float
    axis_x: float
    axis_y: float


def detect_red_bar(bgr: np.ndarray) -> bool:
    """The long red key at the bottom of the panel. A small red LED does not count."""
    if bgr is None or bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 70), (12, 255, 255)),
        cv2.inRange(hsv, (168, 90, 70), (180, 255, 255)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80.0:
            continue
        _center, (side_a, side_b), _angle = cv2.minAreaRect(contour)
        major = max(float(side_a), float(side_b))
        minor = max(1.0, min(float(side_a), float(side_b)))
        if major >= 28.0 and major / minor >= 2.2:
            return True
    return False


def yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 90, 90), (40, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


# A clicked key can be partly in frame. These gates are looser than the long-bar layout.
YELLOW_MIN_AREA = 80.0
YELLOW_MAX_AREA_FRACTION = 0.35
YELLOW_PICK_PX = 28.0
YELLOW_TRACK_PX = 72.0
YELLOW_TRACK_MISS_FRAMES = 8


@dataclass(frozen=True)
class YellowBlob:
    cx: float
    cy: float
    area: float
    contour: np.ndarray


def yellow_blobs(bgr: np.ndarray) -> list[YellowBlob]:
    """Connected yellow regions. A close or cropped key does not have to be a long bar."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return []
    mask = yellow_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    max_area = YELLOW_MAX_AREA_FRACTION * float(width * height)
    blobs: list[YellowBlob] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < YELLOW_MIN_AREA or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        blobs.append(
            YellowBlob(
                cx=float(moments["m10"] / moments["m00"]),
                cy=float(moments["m01"] / moments["m00"]),
                area=area,
                contour=contour,
            )
        )
    return blobs


def _blob_contains(blob: YellowBlob, u: float, v: float) -> bool:
    return cv2.pointPolygonTest(blob.contour, (float(u), float(v)), False) >= 0


def _nearest_blob(blobs: list[YellowBlob], u: float, v: float) -> tuple[YellowBlob, float] | None:
    if not blobs:
        return None
    ranked = min(blobs, key=lambda blob: math.hypot(blob.cx - float(u), blob.cy - float(v)))
    return ranked, math.hypot(ranked.cx - float(u), ranked.cy - float(v))


def yellow_blob_at(bgr: np.ndarray, u: float, v: float, pick_px: float = YELLOW_PICK_PX) -> YellowBlob | None:
    """The yellow region under the click, or the nearest one within a few pixels."""
    blobs = yellow_blobs(bgr)
    inside = [blob for blob in blobs if _blob_contains(blob, u, v)]
    if inside:
        return min(inside, key=lambda blob: math.hypot(blob.cx - float(u), blob.cy - float(v)))
    nearest = _nearest_blob(blobs, u, v)
    if nearest is None or nearest[1] > float(pick_px):
        return None
    return nearest[0]


def match_yellow_blob(
    blobs: list[YellowBlob],
    u: float,
    v: float,
    track_px: float = YELLOW_TRACK_PX,
) -> YellowBlob | None:
    """Follow the same region through a small image shift. Do not hop to a distant key."""
    inside = [blob for blob in blobs if _blob_contains(blob, u, v)]
    if inside:
        return min(inside, key=lambda blob: math.hypot(blob.cx - float(u), blob.cy - float(v)))
    nearest = _nearest_blob(blobs, u, v)
    if nearest is None or nearest[1] > float(track_px):
        return None
    return nearest[0]


def detect_yellow_bars(bgr: np.ndarray) -> list[YellowBar]:
    mask = yellow_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    bars: list[YellowBar] = []
    # Wrist view can be close enough for one key to occupy several percent.
    max_area = 0.04 * width * height
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 40 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h < 3 or w < 10 or (w / max(h, 1)) < 2.2:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        bars.append(
            YellowBar(
                cx=moments["m10"] / moments["m00"],
                cy=moments["m01"] / moments["m00"],
                width=float(w),
                height=float(h),
                contour=contour,
            )
        )
    bars.sort(key=lambda item: (item.cy, item.cx))
    return bars


def detect_oriented_yellow_bars(bgr: np.ndarray) -> list[OrientedYellowBar]:
    """Detect long yellow controls independent of camera roll."""
    mask = yellow_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    bars: list[OrientedYellowBar] = []
    # Wrist view can be close enough for one key to occupy several percent.
    max_area = 0.04 * width * height
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 40 or area > max_area:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
        lengths = [float(np.linalg.norm(edge)) for edge in edges]
        major_index = int(np.argmax(lengths))
        major = lengths[major_index]
        minor = min(lengths)
        if minor < 3.0 or major < 10.0 or major / minor < 2.2:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        axis = edges[major_index] / max(major, 1e-6)
        bars.append(
            OrientedYellowBar(
                cx=float(moments["m10"] / moments["m00"]),
                cy=float(moments["m01"] / moments["m00"]),
                major=major,
                minor=minor,
                axis_x=float(axis[0]),
                axis_y=float(axis[1]),
            )
        )
    return bars


def _distance(a: OrientedYellowBar, b: OrientedYellowBar) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _parallel(a: OrientedYellowBar, b: OrientedYellowBar) -> float:
    return abs(a.axis_x * b.axis_x + a.axis_y * b.axis_y)


def _rotated_partial_layout(
    bgr: np.ndarray,
) -> dict[str, tuple[float, float]] | None:
    """Rocker + nearby OK + farther RESET; panel border may be outside frame."""
    bars = sorted(
        detect_oriented_yellow_bars(bgr),
        key=lambda item: item.major * item.minor,
        reverse=True,
    )[:7]
    if len(bars) < 3:
        return None
    best: tuple[float, OrientedYellowBar, OrientedYellowBar] | None = None
    for triple in itertools.combinations(bars, 3):
        for near_a, near_b, reset in (
            (triple[0], triple[1], triple[2]),
            (triple[0], triple[2], triple[1]),
            (triple[1], triple[2], triple[0]),
        ):
            pair_gap = _distance(near_a, near_b)
            scale = 0.5 * (near_a.major + near_b.major)
            if not (0.20 * scale <= pair_gap <= 1.40 * scale):
                continue
            if _parallel(near_a, near_b) < 0.75:
                continue
            rocker, ok = (
                (near_a, near_b)
                if _distance(near_a, reset) > _distance(near_b, reset)
                else (near_b, near_a)
            )
            reset_gap = _distance(rocker, reset)
            ratio = reset_gap / max(pair_gap, 1e-6)
            if not (1.45 <= ratio <= 4.5):
                continue
            if min(_parallel(rocker, reset), _parallel(ok, reset)) < 0.60:
                continue
            rx, ry = reset.cx - rocker.cx, reset.cy - rocker.cy
            ox, oy = ok.cx - rocker.cx, ok.cy - rocker.cy
            off_line = abs(rx * oy - ry * ox) / max(reset_gap, 1e-6)
            if off_line > 0.75 * pair_gap:
                continue
            score = ratio + 0.5 * (_parallel(rocker, ok) + _parallel(ok, reset))
            if best is None or score > best[0]:
                best = (score, rocker, ok)
    if best is None:
        return None
    _, rocker, ok = best
    axis_x, axis_y = rocker.axis_x, rocker.axis_y
    down_x, down_y = ok.cx - rocker.cx, ok.cy - rocker.cy
    if axis_x * down_y - axis_y * down_x < 0.0:
        axis_x, axis_y = -axis_x, -axis_y
    half = 0.24 * rocker.major
    b1 = (rocker.cx - half * axis_x, rocker.cy - half * axis_y)
    b2 = (rocker.cx + half * axis_x, rocker.cy + half * axis_y)
    return {
        "button_1": b1,
        "button_2": b2,
        "button_3": (ok.cx, ok.cy),
    }


def _split_pair(bar: YellowBar, mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x0 = max(0, int(round(bar.left)))
    x1 = min(mask.shape[1], int(round(bar.right)))
    y0 = max(0, int(round(bar.cy - bar.height)))
    y1 = min(mask.shape[0], int(round(bar.cy + bar.height)))
    if x1 - x0 < 8 or y1 - y0 < 2:
        return (bar.cx - 0.25 * bar.width, bar.cy), (bar.cx + 0.25 * bar.width, bar.cy)
    profile = mask[y0:y1, x0:x1].sum(axis=0).astype(np.float32)
    mid0 = int(0.35 * profile.size)
    mid1 = int(0.65 * profile.size)
    if mid1 <= mid0:
        split = 0.5 * bar.width
    else:
        split = float(np.argmin(profile[mid0:mid1]) + mid0)
    left = mask[y0:y1, x0 : x0 + max(int(split), 1)]
    right = mask[y0:y1, x0 + max(int(split), 1) : x1]
    return _mask_centroid(left, x0, y0, (bar.cx - 0.25 * bar.width, bar.cy)), _mask_centroid(
        right, x0 + max(int(split), 1), y0, (bar.cx + 0.25 * bar.width, bar.cy)
    )


def _mask_centroid(roi: np.ndarray, x0: int, y0: int, fallback: tuple[float, float]) -> tuple[float, float]:
    moments = cv2.moments(roi)
    if moments["m00"] <= 1e-6:
        return fallback
    return x0 + moments["m10"] / moments["m00"], y0 + moments["m01"] / moments["m00"]


def _cluster_rows(bars: list[YellowBar]) -> list[list[YellowBar]]:
    if not bars:
        return []
    heights = [item.height for item in bars]
    gap = max(8.0, float(np.median(heights)) * 1.8)
    rows: list[list[YellowBar]] = [[bars[0]]]
    for bar in bars[1:]:
        if abs(bar.cy - rows[-1][-1].cy) <= gap:
            rows[-1].append(bar)
        else:
            rows.append([bar])
    for row in rows:
        row.sort(key=lambda item: item.cx)
    return rows


def assign_poke_buttons(bgr: np.ndarray) -> dict[str, tuple[float, float]] | None:
    """Layout: top yellow row is Button 1|2, the next row is Button 3 (OK)."""
    rotated = _rotated_partial_layout(bgr)
    if rotated is not None:
        return rotated
    bars = detect_yellow_bars(bgr)
    rows = _cluster_rows(bars)
    if len(rows) < 2:
        return _rotated_partial_layout(bgr)
    # Right-hand column: keep rows whose x is near the rightmost cluster.
    xs = [bar.cx for row in rows for bar in row]
    if not xs:
        return None
    right = float(np.median(sorted(xs)[-min(3, len(xs)) :]))
    column = [row for row in rows if abs(np.mean([bar.cx for bar in row]) - right) < 0.18 * bgr.shape[1]]
    if len(column) < 2:
        column = rows
    top, second = column[0], column[1]
    mask = yellow_mask(bgr)
    if len(top) >= 2:
        b1 = (top[0].cx, top[0].cy)
        b2 = (top[1].cx, top[1].cy)
    else:
        b1, b2 = _split_pair(top[0], mask)
    ok = max(second, key=lambda item: item.width)
    b3 = (ok.cx, ok.cy)
    if not (b1[0] < b2[0] and b3[1] > b1[1]):
        return _rotated_partial_layout(bgr)
    return {"button_1": b1, "button_2": b2, "button_3": b3}
