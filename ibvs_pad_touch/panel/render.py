from __future__ import annotations

import cv2
import numpy as np

from ibvs_pad_touch.panel.model import PanelModel, Rect, iter_named_rects

BEIGE = (214, 220, 226)
LCD = (28, 28, 28)
YELLOW = (0, 210, 235)
RED = (40, 40, 200)
GREEN = (70, 200, 70)
INK = (40, 40, 40)
DIM = (120, 120, 120)
ACCENT = (40, 90, 220)


def _to_px(x_m: float, y_m: float, scale: float, origin: tuple[int, int]) -> tuple[int, int]:
    return int(round(origin[0] + x_m * scale)), int(round(origin[1] + y_m * scale))


def _color(name: str) -> tuple[int, int, int]:
    key = str(name).lower()
    if key in {"yellow"}:
        return YELLOW
    if key in {"red", "alarm"}:
        return RED
    if key in {"green", "power_led"}:
        return GREEN
    if key in {"lcd"}:
        return LCD
    return DIM


def render_panel(
    model: PanelModel,
    *,
    scale_px_per_m: float = 4000.0,
    margin_px: int = 70,
    highlight: int | str | None = None,
) -> np.ndarray:
    """Orthographic scale drawing. Default 4 px/mm."""
    width = int(round(model.width_m * scale_px_per_m)) + 2 * margin_px
    height = int(round(model.height_m * scale_px_per_m)) + 2 * margin_px + 48
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    origin = (margin_px, margin_px + 36)
    x0, y0 = origin
    x1 = x0 + int(round(model.width_m * scale_px_per_m))
    y1 = y0 + int(round(model.height_m * scale_px_per_m))
    cv2.rectangle(image, (x0, y0), (x1, y1), BEIGE, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (170, 170, 170), 2)

    target = None
    if highlight is not None:
        target = model.button(highlight).name

    for name, rect, meta in iter_named_rects(model):
        color = _color(meta.get("color", "dim"))
        p0 = _to_px(rect.left_m, rect.top_m, scale_px_per_m, origin)
        p1 = _to_px(rect.right_m, rect.bottom_m, scale_px_per_m, origin)
        thickness = -1
        if name == "display":
            cv2.rectangle(image, p0, p1, LCD, -1)
            continue
        if meta.get("kind") == "led" or name.startswith("led_"):
            center = _to_px(rect.cx, rect.cy, scale_px_per_m, origin)
            radius = max(3, int(round(0.5 * min(rect.width_m, rect.height_m) * scale_px_per_m)))
            cv2.circle(image, center, radius, color, -1)
            continue
        if meta.get("poke"):
            fill = color if name != target else (0, 160, 255)
            cv2.rectangle(image, p0, p1, fill, -1)
            cv2.rectangle(image, p0, p1, ACCENT if name == target else INK, 2)
        else:
            cv2.rectangle(image, p0, p1, color, -1)
        label = str(meta.get("label", name))
        cv2.putText(image, label, (p0[0], p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)

    _draw_dimensions(image, model, scale_px_per_m, origin)
    cv2.putText(
        image,
        "IBVS_PAD_TOUCH  300 x 130 mm   Button1=<  Button2=>  Button3=OK",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        INK,
        1,
        cv2.LINE_AA,
    )
    return image


def _draw_dimensions(
    image: np.ndarray,
    model: PanelModel,
    scale: float,
    origin: tuple[int, int],
) -> None:
    b1 = model.button(1).rect
    b2 = model.button(2).rect
    b3 = model.button(3).rect
    x_right = _to_px(model.width_m, 0.0, scale, origin)[0]
    y_top = origin[1]
    # right-edge distances
    _h_dim(image, _to_px(b2.right_m, b2.cy, scale, origin), (x_right, _to_px(0, b2.cy, scale, origin)[1]), "35mm")
    _h_dim(image, _to_px(b1.left_m, b1.top_m - 0.008, scale, origin), (x_right, _to_px(0, b1.top_m - 0.008, scale, origin)[1]), "70mm")
    _v_dim(image, (int(0.5 * (_to_px(b1.cx, 0, scale, origin)[0] + _to_px(b2.cx, 0, scale, origin)[0])), y_top), _to_px(b1.cx, b1.cy, scale, origin), "30mm")
    _v_dim(image, _to_px(b2.right_m + 0.012, b1.cy, scale, origin), _to_px(b3.right_m + 0.012, b3.cy, scale, origin), "20mm")


def _h_dim(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], text: str) -> None:
    cv2.line(image, p0, p1, ACCENT, 1, cv2.LINE_AA)
    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2 - 4)
    cv2.putText(image, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT, 1, cv2.LINE_AA)


def _v_dim(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], text: str) -> None:
    cv2.line(image, p0, p1, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(image, text, (p1[0] + 4, (p0[1] + p1[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT, 1, cv2.LINE_AA)


def draw_rect_on_image(image: np.ndarray, corners_px: list[tuple[float, float]], color: tuple[int, int, int], label: str = "") -> None:
    pts = np.asarray(corners_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(image, label, (int(corners_px[0][0]), int(corners_px[0][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def project_rect(rect: Rect, homography: np.ndarray) -> list[tuple[float, float]]:
    src = np.array([[[x, y]] for x, y in rect.corners_m()], dtype=np.float32)
    # corners_m are in meters; homography is built from meter-canonical or norm-canonical.
    dst = cv2.perspectiveTransform(src, homography)
    return [(float(p[0]), float(p[1])) for p in dst.reshape(-1, 2)]
