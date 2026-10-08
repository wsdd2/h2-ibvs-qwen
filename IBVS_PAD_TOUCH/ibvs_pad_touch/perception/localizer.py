from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ibvs_pad_touch.panel.model import PanelModel

CANONICAL_NORM = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)


def order_corners(corners: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    summed = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[int(np.argmin(summed))]
    ordered[2] = pts[int(np.argmax(summed))]
    ordered[1] = pts[int(np.argmin(diff))]
    ordered[3] = pts[int(np.argmax(diff))]
    return ordered


def homography_from_corners(corners_px: Sequence[Sequence[float]]) -> np.ndarray:
    return cv2.getPerspectiveTransform(CANONICAL_NORM, order_corners(corners_px))


def homography_from_meter_corners(model: PanelModel, corners_px: Sequence[Sequence[float]]) -> np.ndarray:
    src = np.array(model.canonical_corners_m(), dtype=np.float32)
    return cv2.getPerspectiveTransform(src, order_corners(corners_px))


def project_xy(xy_m: tuple[float, float], H_meter: np.ndarray) -> tuple[float, float]:
    src = np.array([[[float(xy_m[0]), float(xy_m[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_meter)[0, 0]
    return float(dst[0]), float(dst[1])


def project_norm(uv: tuple[float, float], H_norm: np.ndarray) -> tuple[float, float]:
    src = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_norm)[0, 0]
    return float(dst[0]), float(dst[1])


class PanelLocalizer:
    """Fixed-panel homography. Corners may be supplied or detected as a beige quad."""

    def __init__(self, model: PanelModel):
        self.model = model
        self._corners: Optional[np.ndarray] = None

    def from_corners(self, corners_px: Sequence[Sequence[float]]) -> dict:
        corners = order_corners(corners_px)
        self._corners = corners
        return {
            "corners_px": [(float(x), float(y)) for x, y in corners],
            "H_norm": homography_from_corners(corners),
            "H_meter": homography_from_meter_corners(self.model, corners),
            "visible": True,
        }

    def last_corners(self) -> Optional[np.ndarray]:
        return None if self._corners is None else self._corners.copy()

    def detect(self, bgr: np.ndarray) -> Optional[dict]:
        corners = detect_panel_corners(bgr, self.model)
        if corners is None:
            return None
        return self.from_corners(corners)

    def button_uv(self, pose: dict, key: int | str) -> tuple[float, float]:
        button = self.model.button(key)
        return project_xy((button.rect.cx, button.rect.cy), pose["H_meter"])


def _border_hits(corners: np.ndarray, width: int, height: int, margin_frac: float = 0.04) -> int:
    mx = margin_frac * width
    my = margin_frac * height
    xs, ys = corners[:, 0], corners[:, 1]
    hits = 0
    if float(xs.min()) <= mx:
        hits += 1
    if float(xs.max()) >= width - mx:
        hits += 1
    if float(ys.min()) <= my:
        hits += 1
    if float(ys.max()) >= height - my:
        hits += 1
    return hits


def panel_quad_acceptable(corners: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Reject tiny blobs and full-frame / full-page quads."""
    height, width = shape[:2]
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return False
    xs, ys = pts[:, 0], pts[:, 1]
    box_w = float(xs.max() - xs.min())
    box_h = float(ys.max() - ys.min())
    if box_w < 40 or box_h < 20:
        return False
    if box_w > 0.88 * width or box_h > 0.88 * height:
        return False
    area = box_w * box_h
    if not (0.04 * width * height <= area <= 0.70 * width * height):
        return False
    if _border_hits(pts, width, height) >= 3:
        return False
    return True


def has_panel_landmarks(bgr: np.ndarray) -> bool:
    """Real panel has a dark LCD and/or the yellow poke keys. A blank wall does not."""
    if detect_lcd_corners(bgr) is not None:
        return True
    from ibvs_pad_touch.perception.yellow_buttons import assign_poke_buttons

    return assign_poke_buttons(bgr) is not None


def _panel_corners_sane(corners: np.ndarray, shape: tuple[int, ...]) -> bool:
    return panel_quad_acceptable(corners, shape)


def _lcd_from_mask(mask: np.ndarray, gray: np.ndarray | None = None) -> Optional[np.ndarray]:
    """Prefer a dark window near the measured LCD aspect, not the largest dark blob."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    target_aspect = 0.175 / 0.072
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.004 * width * height or area > 0.35 * width * height:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        short, long = sorted((float(rw), float(rh)))
        if short < 1 or long / short < 1.6 or long / short > 3.4:
            continue
        corners = order_corners(cv2.boxPoints(rect))
        if _border_hits(corners, width, height, 0.02) >= 2:
            continue
        aspect_err = abs((long / short) - target_aspect)
        contrast = 30.0 if gray is None else _lcd_contrast_score(gray, corners)
        if contrast < 18.0:
            continue
        score = contrast * area / (1.0 + 14.0 * aspect_err)
        if score > best_score:
            best = corners
            best_score = score
    return best


def _lcd_contrast_score(gray: np.ndarray, corners: np.ndarray) -> float:
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    ring = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
    inside = gray[inner > 0]
    outside = gray[ring > 0]
    if inside.size < 20 or outside.size < 20:
        return 0.0
    return float(np.median(outside)) - float(np.median(inside))


def detect_lcd_adaptive(bgr: np.ndarray) -> Optional[np.ndarray]:
    """LCD is a darker window on a brighter housing; global thresholds merge it with the arm."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    best = None
    best_score = 0.0
    for block, bias in ((31, 10), (51, 15), (71, 18)):
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            bias,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 0.004 * width * height or area > 0.12 * width * height:
                continue
            rect = cv2.minAreaRect(contour)
            short, long = sorted((float(rect[1][0]), float(rect[1][1])))
            if short < 8 or long / short < 1.5 or long / short > 3.4:
                continue
            cy = float(rect[0][1])
            if cy < 0.18 * height:
                continue
            corners = order_corners(cv2.boxPoints(rect))
            contrast = _lcd_contrast_score(gray, corners)
            if contrast < 25.0:
                continue
            aspect_err = abs((long / short) - (0.175 / 0.072))
            score = contrast * area / (1.0 + 8.0 * aspect_err)
            if score > best_score:
                best = corners
                best_score = score
    return best


def detect_lcd_corners(bgr: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Wrap and indoor light lift the LCD above the old V=75 cutoff.
    # Keep every threshold and pick the aspect match. The first dark blob is
    # often the shadow under the housing, not the screen.
    best = None
    best_score = 0.0
    target_aspect = 0.175 / 0.072
    masks = [
        cv2.inRange(hsv, (0, 0, 0), (180, 90, 75)),
        cv2.inRange(hsv, (0, 0, 0), (180, 80, 140)),
        cv2.inRange(hsv, (0, 0, 0), (180, 100, 170)),
    ]
    _, dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    masks.append(dark)
    for mask in masks:
        found = _lcd_from_mask(mask, gray)
        if found is None:
            continue
        short, long = _quad_short_long(found)
        aspect_err = abs((long / max(short, 1.0)) - target_aspect)
        score = _lcd_contrast_score(gray, found) / (1.0 + 14.0 * aspect_err)
        if score > best_score:
            best = found
            best_score = score
    adaptive = detect_lcd_adaptive(bgr)
    if adaptive is not None:
        short, long = _quad_short_long(adaptive)
        aspect_err = abs((long / max(short, 1.0)) - target_aspect)
        score = _lcd_contrast_score(gray, adaptive) / (1.0 + 14.0 * aspect_err)
        if score > best_score:
            best = adaptive
    return best


def _quad_short_long(corners: np.ndarray) -> tuple[float, float]:
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    ordered = sorted(edges)
    return ordered[0], ordered[-1]


def lcd_rect_m(model: PanelModel) -> Optional[np.ndarray]:
    display = model.display
    if display is None:
        return None
    return np.array(
        [
            [display.left_m, display.top_m],
            [display.right_m, display.top_m],
            [display.right_m, display.bottom_m],
            [display.left_m, display.bottom_m],
        ],
        dtype=np.float32,
    )


def panel_corners_from_lcd(lcd_px: Sequence[Sequence[float]], model: PanelModel) -> Optional[np.ndarray]:
    """Expand a seen LCD quad to the full faceplate using the tape-measure model."""
    lcd_m = lcd_rect_m(model)
    if lcd_m is None:
        return None
    lcd_img = order_corners(lcd_px)
    homography = cv2.getPerspectiveTransform(lcd_m, lcd_img)
    panel_m = np.asarray(model.canonical_corners_m(), dtype=np.float32).reshape(1, 4, 2)
    panel_px = cv2.perspectiveTransform(panel_m, homography)[0]
    return order_corners(panel_px)


def _point_in_quad(point: Sequence[float], corners: np.ndarray) -> bool:
    poly = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def _has_dark_interior(gray: np.ndarray, corners: np.ndarray) -> bool:
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    inner = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=2)
    if int(cv2.countNonZero(inner)) < 80:
        inner = mask
    ring = cv2.bitwise_and(mask, cv2.bitwise_not(inner))
    inside = gray[inner > 0]
    border = gray[ring > 0]
    if inside.size < 20 or border.size < 20:
        return False
    return float(np.median(inside)) + 25.0 < float(np.median(border))


def _screen_sits_above(bgr: np.ndarray, corners: np.ndarray) -> bool:
    """True when this quad starts under the dark screen, so it is only the lower housing."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = order_corners(corners)
    height, width = gray.shape[:2]
    top = float(np.mean(quad[:2, 1]))
    x0 = int(np.clip(float(quad[:, 0].min()) + 8, 0, width - 2))
    x1 = int(np.clip(float(quad[:, 0].max()) - 8, x0 + 1, width))
    above_y1 = int(max(0, top - 6))
    above_y0 = int(max(0, top - 36))
    in_y0 = int(np.clip(top + 6, 0, height - 2))
    in_y1 = int(np.clip(top + 32, in_y0 + 1, height))
    if above_y1 - above_y0 < 10 or in_y1 - in_y0 < 8:
        return False
    above = gray[above_y0:above_y1, x0:x1]
    inside = gray[in_y0:in_y1, x0:x1]
    if above.size < 40 or inside.size < 40:
        return False
    return float(np.median(above)) + 18.0 < float(np.median(inside))


def _accept_panel_quad(bgr: np.ndarray, corners: np.ndarray | None) -> Optional[np.ndarray]:
    if corners is None or not panel_quad_acceptable(corners, bgr.shape):
        return None
    if _screen_sits_above(bgr, corners):
        return None
    return corners


def detect_panel_by_outline(bgr: np.ndarray) -> Optional[np.ndarray]:
    """White housing on a white wall: use the rectangular outline, not yellow keys."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lcd = detect_lcd_corners(bgr)
    lcd_c = None if lcd is None else (float(lcd[:, 0].mean()), float(lcd[:, 1].mean()))
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.03 * bgr.shape[0] * bgr.shape[1]:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        short, long = sorted((float(rw), float(rh)))
        if short < 1 or long / short < 1.6 or long / short > 3.6:
            continue
        corners = order_corners(cv2.boxPoints(rect))
        if not panel_quad_acceptable(corners, bgr.shape):
            continue
        if lcd_c is not None and not _point_in_quad(lcd_c, corners):
            continue
        if lcd_c is None and not _has_dark_interior(gray, corners):
            continue
        score = area + (4000.0 if lcd_c is not None else 0.0)
        if score > best_score:
            best = corners
            best_score = score
    return best


def detect_panel_corners(bgr: np.ndarray, model: PanelModel | None = None) -> Optional[np.ndarray]:
    """Find the faceplate even when the yellow keys are hidden by the gripper."""
    from ibvs_pad_touch.perception.panel_segment import detect_panel_body

    lcd = detect_lcd_corners(bgr)
    if lcd is not None and model is not None and model.display is not None:
        expanded = panel_corners_from_lcd(lcd, model)
        if expanded is not None:
            lcd_c = (float(lcd[:, 0].mean()), float(lcd[:, 1].mean()))
            if _point_in_quad(lcd_c, expanded):
                accepted = _accept_panel_quad(bgr, expanded)
                if accepted is not None:
                    return accepted

    outline = _accept_panel_quad(bgr, detect_panel_by_outline(bgr))
    if outline is not None:
        return outline

    body = detect_panel_body(bgr)
    body = _accept_panel_quad(bgr, None if body is None else order_corners(body))
    if body is not None:
        return body

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (40, 70, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = bgr.shape[:2]
    min_area = 0.08 * width * height
    best = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            approx = cv2.boxPoints(rect)
        else:
            approx = approx.reshape(4, 2)
        if area > best_area:
            best = approx
            best_area = area
    if best is None:
        return None
    corners = order_corners(best)
    if not has_panel_landmarks(bgr):
        return None
    return _accept_panel_quad(bgr, corners)


def refine_yellow_centroid(bgr: np.ndarray, center_uv: tuple[float, float], half_wh: tuple[int, int]) -> Optional[tuple[float, float]]:
    h, w = bgr.shape[:2]
    cx, cy = int(round(center_uv[0])), int(round(center_uv[1]))
    hx, hy = half_wh
    x0, x1 = max(0, cx - hx), min(w, cx + hx)
    y0, y1 = max(0, cy - hy), min(h, cy + hy)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 90, 90), (40, 255, 255))
    if int(cv2.countNonZero(mask)) < 8:
        return None
    moments = cv2.moments(mask)
    if moments["m00"] <= 1e-6:
        return None
    return x0 + moments["m10"] / moments["m00"], y0 + moments["m01"] / moments["m00"]
