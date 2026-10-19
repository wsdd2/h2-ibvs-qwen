# -*- coding: utf-8 -*-
"""Chest detection: YOLO12-seg panel, YOLO26s-pose buttons and gripper tip.

A regular panel rectangle keeps the tape-measure buttons. Pose points are used
only when that rectangle is missing, and only if the point is confident and inside.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ibvs_pad_touch.perception.features import sample_depth
from ibvs_pad_touch.perception.gripper_tip import point_inside_panel
from ibvs_pad_touch.perception.localizer import order_corners
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization

ROOT = Path(__file__).resolve().parents[2]
SEG_WEIGHTS = ROOT / "outputs" / "runs" / "yolo12_seg_20261014_1045" / "weights" / "best.pt"
POSE_WEIGHTS = ROOT / "outputs" / "runs" / "yolo26s_pose_20261014_1045" / "weights" / "best.pt"
KPT_NAMES = ("gripper_tip", "button_1", "button_2", "button_3")
BUTTON_NAMES = ("button_1", "button_2", "button_3")


@dataclass
class YoloPoint:
    name: str
    u: float
    v: float
    conf: float


@dataclass
class YoloMask:
    name: str
    conf: float
    polygon: list[tuple[float, float]]


@dataclass
class YoloTouch:
    panel_corners: list[tuple[float, float]] | None = None
    panel_conf: float = 0.0
    points: dict[str, YoloPoint] = field(default_factory=dict)
    masks: list[YoloMask] = field(default_factory=list)

    def point(self, name: str) -> YoloPoint | None:
        return self.points.get(name)


class YoloTouchDetector:
    """Lazy ultralytics models. Missing weights or import leaves detection classical."""

    def __init__(
        self,
        seg_weights: str | Path = SEG_WEIGHTS,
        pose_weights: str | Path = POSE_WEIGHTS,
        imgsz: int = 640,
    ) -> None:
        self.seg_weights = Path(seg_weights)
        self.pose_weights = Path(pose_weights)
        self.imgsz = int(imgsz)
        self._seg = None
        self._pose = None
        self._ready = False
        self._failed = ""

    def status(self) -> str:
        if self._failed:
            return self._failed
        if self._ready:
            return "yolo"
        if not self.seg_weights.is_file() or not self.pose_weights.is_file():
            return "weights missing"
        return "idle"

    def predict(self, bgr: np.ndarray) -> YoloTouch | None:
        if not self._ensure():
            return None
        seg = self._seg.predict(bgr, imgsz=self.imgsz, conf=0.15, verbose=False)[0]
        pose = self._pose.predict(bgr, imgsz=self.imgsz, conf=0.15, verbose=False)[0]
        return YoloTouch(
            panel_corners=_panel_quad(seg),
            panel_conf=_panel_conf(seg),
            points=_pose_points(pose),
            masks=_seg_masks(seg),
        )


    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        if not self.seg_weights.is_file() or not self.pose_weights.is_file():
            self._failed = "weights missing"
            print(
                f"[ibvs-yolo] weights missing seg={self.seg_weights} pose={self.pose_weights}",
                flush=True,
            )
            return False
        try:
            from ultralytics import YOLO

            self._seg = YOLO(str(self.seg_weights))
            self._pose = YOLO(str(self.pose_weights))
            self._ready = True
            print(f"[ibvs-yolo] loaded {self.seg_weights} + {self.pose_weights}", flush=True)
            return True
        except Exception as exc:
            self._failed = f"yolo unavailable: {exc}"
            print(f"[ibvs-yolo] {self._failed}", flush=True)
            return False


def panel_is_regular_rectangle(corners, aspect_min: float = 1.55, aspect_max: float = 8.0) -> bool:
    """Image quad of the faceplate: near right angles, matched opposite sides, not a square."""
    if corners is None:
        return False
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (4, 2):
        return False
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    lengths = np.array([float(np.linalg.norm(edge)) for edge in edges], dtype=np.float64)
    if float(lengths.min()) < 30.0:
        return False
    for a, b in ((0, 2), (1, 3)):
        longer = max(float(lengths[a]), float(lengths[b]))
        if abs(float(lengths[a]) - float(lengths[b])) > 0.4 * longer:
            return False
    for i in range(4):
        prev = edges[i - 1]
        edge = edges[i]
        denom = float(np.linalg.norm(prev) * np.linalg.norm(edge))
        if denom < 1e-6:
            return False
        cosine = float(np.dot(prev, edge) / denom)
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if abs(angle - 90.0) > 28.0:
            return False
    pair_a = 0.5 * (float(lengths[0]) + float(lengths[2]))
    pair_b = 0.5 * (float(lengths[1]) + float(lengths[3]))
    aspect = max(pair_a, pair_b) / max(min(pair_a, pair_b), 1e-6)
    return float(aspect_min) <= aspect <= float(aspect_max)


def pose_button_ok(point: YoloPoint | None, corners, conf_min: float) -> bool:
    if point is None or float(point.conf) < float(conf_min):
        return False
    if panel_is_regular_rectangle(corners):
        return False
    return point_inside_panel((point.u, point.v), corners)


def apply_pose_buttons(
    loc: Localization,
    touch: YoloTouch | None,
    *,
    conf_min: float = 0.3,
    depth_m: np.ndarray | None = None,
    K: Intrinsics | None = None,
) -> Localization:
    """Use a complete high-confidence pose triplet; otherwise keep the scale prior."""
    if touch is None:
        return loc
    regular = panel_is_regular_rectangle(loc.corners_px)
    if regular and not _regular_pose_triplet_ok(
        touch,
        loc,
        max(0.65, float(conf_min)),
    ):
        return loc
    buttons = dict(loc.buttons)
    changed = False
    for name in BUTTON_NAMES:
        pt = touch.point(name)
        valid = (
            pt is not None
            and float(pt.conf) >= (max(0.65, float(conf_min)) if regular else float(conf_min))
            and point_inside_panel((pt.u, pt.v), loc.corners_px)
        )
        if not valid:
            continue
        z_m = sample_depth(depth_m, pt.u, pt.v) if depth_m is not None else None
        old = buttons.get(name)
        if z_m is None and old is not None:
            z_m = old.z_m
        point = CameraPoint(u=pt.u, v=pt.v, source="pose", confidence=float(pt.conf))
        if z_m is not None and K is not None:
            point = point.with_depth(K, z_m)
        buttons[name] = point
        changed = True
    if not changed:
        return loc
    return Localization(
        visible=loc.visible,
        corners_px=loc.corners_px,
        buttons=buttons,
        message=loc.message,
    )


def _regular_pose_triplet_ok(
    touch: YoloTouch,
    loc: Localization,
    conf_min: float,
) -> bool:
    points = [touch.point(name) for name in BUTTON_NAMES]
    if any(point is None or float(point.conf) < float(conf_min) for point in points):
        return False
    if not loc.corners_px:
        return False
    if any(not point_inside_panel((point.u, point.v), loc.corners_px) for point in points):
        return False
    quad = np.asarray(loc.corners_px, dtype=np.float64).reshape(4, 2)
    panel_w = max(1.0, float(np.ptp(quad[:, 0])))
    panel_h = max(1.0, float(np.ptp(quad[:, 1])))
    panel_diag = float(np.hypot(panel_w, panel_h))
    array = np.asarray([[point.u, point.v] for point in points], dtype=np.float64)
    pair_distances = [
        float(np.linalg.norm(array[i] - array[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    if min(pair_distances) < 3.0 or max(pair_distances) > 0.35 * panel_diag:
        return False
    # STULZ keys occupy the right portion of the faceplate.
    if float(np.mean(array[:, 0])) < float(quad[:, 0].min()) + 0.55 * panel_w:
        return False
    for name, point in zip(BUTTON_NAMES, points):
        prior = loc.button(name)
        if prior is None:
            continue
        if float(np.hypot(point.u - prior.u, point.v - prior.v)) > 0.30 * panel_diag:
            return False
    return True


def pose_tip(touch: YoloTouch | None, conf_min: float) -> CameraPoint | None:
    if touch is None:
        return None
    pt = touch.point("gripper_tip")
    if pt is None or float(pt.conf) < float(conf_min):
        return None
    return CameraPoint(u=pt.u, v=pt.v, source="pose", confidence=float(pt.conf))


def _class_name(result, index: int) -> str:
    names = getattr(result, "names", None) or {}
    if isinstance(names, dict):
        return str(names.get(int(index), index))
    if isinstance(names, (list, tuple)) and int(index) < len(names):
        return str(names[int(index)])
    return str(index)


def _panel_conf(result) -> float:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return 0.0
    best = 0.0
    for idx, cls_id in enumerate(boxes.cls.tolist()):
        if _class_name(result, int(cls_id)) != "panel":
            continue
        conf = float(boxes.conf[idx])
        if conf > best:
            best = conf
    return best


def _panel_quad(result) -> list[tuple[float, float]] | None:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.cls is None or len(boxes) == 0:
        return None
    polygons = masks.xy
    best_i = None
    best_conf = -1.0
    for idx, cls_id in enumerate(boxes.cls.tolist()):
        if _class_name(result, int(cls_id)) != "panel":
            continue
        conf = float(boxes.conf[idx])
        if conf > best_conf:
            best_conf = conf
            best_i = idx
    if best_i is None or best_i >= len(polygons):
        return None
    poly = np.asarray(polygons[best_i], dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None
    quad = order_corners(cv2.boxPoints(cv2.minAreaRect(poly)))
    return [(float(x), float(y)) for x, y in quad]


def keep_side_masks(masks, gripper_min: float = 0.5):
    """One mask per class. A low-confidence gripper is usually a person, not the jaw."""
    best = {}
    for mask in masks or []:
        if mask.name == "gripper" and float(mask.conf) < float(gripper_min):
            continue
        current = best.get(mask.name)
        if current is None or float(mask.conf) > float(current.conf):
            best[mask.name] = mask
    return list(best.values())


def tip_on_mask(point, mask, margin_px: float = 40.0):
    if point is None or mask is None:
        return False
    poly = np.asarray(mask.polygon, dtype=np.float32).reshape(-1, 1, 2)
    if poly.shape[0] < 3:
        return False
    return float(cv2.pointPolygonTest(poly, (float(point.u), float(point.v)), True)) >= -float(margin_px)


def gripper_toward_panel_tip(mask, panel_corners) -> CameraPoint | None:
    """Chest fallback: the jaw point nearest the panel, not an end of the long axis.

    On this view the gripper stands upright beside the faceplate. The long axis
    runs from the top of the jaw to the wrist pin. The poke face is the side
    toward the panel.
    """
    if mask is None or not panel_corners:
        return None
    pts = np.asarray(mask.polygon, dtype=np.float32).reshape(-1, 2)
    panel = np.asarray(panel_corners, dtype=np.float32).reshape(-1, 1, 2)
    if pts.shape[0] < 3 or panel.shape[0] < 3:
        return None
    step = max(1, int(pts.shape[0] // 80))
    best = None
    best_d = None
    for u, v in pts[::step]:
        dist = float(cv2.pointPolygonTest(panel, (float(u), float(v)), True))
        if best_d is None or dist > best_d:
            best_d = dist
            best = (float(u), float(v))
    if best is None:
        return None
    return CameraPoint(u=best[0], v=best[1], source="face", confidence=float(mask.conf))


def gripper_axis_tip(mask) -> CameraPoint | None:
    """Distal end of the gripper mask when pose has no tip.

    The long axis has two ends. The jaw is the narrower one; the wrist end is wider.
    """
    if mask is None:
        return None
    poly = np.asarray(mask.polygon, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None
    contour = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    x, y, width, height = cv2.boundingRect(contour)
    if width < 2 or height < 2:
        return None
    roi = np.zeros((height, width), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.fillPoly(roi, [shifted], 255)
    rows, cols = np.where(roi > 0)
    if len(rows) < 8:
        return None
    pixels = np.column_stack([cols + x, rows + y]).astype(np.float64)
    center = np.mean(pixels, axis=0)
    values, vectors = np.linalg.eigh(np.cov(pixels.T))
    axis = vectors[:, int(np.argmax(values))]
    proj = (pixels - center) @ axis
    lo = float(proj.min())
    hi = float(proj.max())
    span = hi - lo
    if span < 8.0:
        return None
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    band = max(4.0, 0.15 * span)

    def end(high: bool) -> tuple[np.ndarray, float]:
        chosen = proj >= hi - band if high else proj <= lo + band
        chunk = pixels[chosen]
        cross = (chunk - center) @ normal
        extreme = int(np.argmax(proj[chosen]) if high else np.argmin(proj[chosen]))
        return chunk[extreme], float(cross.max() - cross.min())

    high_pt, high_width = end(True)
    low_pt, low_width = end(False)
    tip = low_pt if low_width <= high_width else high_pt
    return CameraPoint(
        u=float(tip[0]),
        v=float(tip[1]),
        source="axis",
        confidence=float(mask.conf),
    )


def _seg_masks(result) -> list[YoloMask]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.cls is None or len(boxes) == 0:
        return []
    found: list[YoloMask] = []
    for idx, cls_id in enumerate(boxes.cls.tolist()):
        if idx >= len(masks.xy):
            break
        poly = np.asarray(masks.xy[idx], dtype=np.float32).reshape(-1, 2)
        if poly.shape[0] < 3:
            continue
        found.append(
            YoloMask(
                name=_class_name(result, int(cls_id)),
                conf=float(boxes.conf[idx]),
                polygon=[(float(x), float(y)) for x, y in poly],
            )
        )
    return found


def _pose_points(result) -> dict[str, YoloPoint]:
    kpts = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kpts is None or kpts.data is None or len(kpts.data) == 0:
        return {}
    data = kpts.data
    pick = 0
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(data):
        pick = int(np.argmax(boxes.conf.detach().cpu().numpy()))
    row = data[pick].detach().cpu().numpy()
    found: dict[str, YoloPoint] = {}
    for i, name in enumerate(KPT_NAMES):
        if i >= row.shape[0]:
            break
        u, v, conf = (float(row[i, 0]), float(row[i, 1]), float(row[i, 2]))
        if conf <= 0.0:
            continue
        found[name] = YoloPoint(name=name, u=u, v=v, conf=conf)
    return found
