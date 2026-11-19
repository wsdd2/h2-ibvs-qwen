# -*- coding: utf-8 -*-
"""Chest detection: YOLO12-seg panel, YOLO26s-pose buttons and gripper tip.

A regular panel rectangle keeps the tape-measure buttons. Pose points are used
only when that rectangle is missing, and only if the point is confident and inside.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ibvs_pad_touch.perception.features import sample_depth
from ibvs_pad_touch.perception.gripper_tip import point_inside_panel
from ibvs_pad_touch.perception.localizer import order_corners
from ibvs_pad_touch.platform import gpu_name, prepare_torch_gpu
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization

ROOT = Path(__file__).resolve().parents[2]
SEG_WEIGHTS = ROOT / "outputs" / "runs" / "yolo12_seg_20261014_1045" / "weights" / "best.pt"
POSE_WEIGHTS = ROOT / "outputs" / "runs" / "yolo26s_pose_20261103_1130" / "weights" / "best.pt"


def onnx_weights(path: str | Path) -> Path:
    """Sibling ONNX next to a .pt checkpoint. best.pt -> best.onnx."""
    return Path(path).with_suffix(".onnx")


def rknn_weights(path: str | Path) -> Path:
    """Sibling RKNN next to a .pt checkpoint. best.pt -> best.rknn."""
    return Path(path).with_suffix(".rknn")


def _model_imgsz(model, weights: Path, fallback: int) -> int:
    """ONNX graphs are fixed-size. PyTorch checkpoints keep the caller's imgsz."""
    if weights.suffix.lower() != ".onnx":
        return int(fallback)
    size = (getattr(model, "overrides", None) or {}).get("imgsz")
    if isinstance(size, (list, tuple)):
        size = size[0] if size else None
    if isinstance(size, (int, float)) and int(size) > 0:
        return int(size)
    return int(fallback)
KPT_NAMES = ("gripper_tip", "button_1", "button_2", "button_3", "FingerTip")
BUTTON_NAMES = ("button_1", "button_2", "button_3")
# Chest and wrist each run seg + pose. Two forwards at a time keep the Orin GPU busy
# without stacking four networks on one frame.
_GPU_SLOTS = threading.BoundedSemaphore(2)
_USE_QUANTIZE: bool | None = None


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


def normalize_button_layout(touch: YoloTouch | None) -> YoloTouch | None:
    """Button 3 is the OK key below Button 1. Swap a pose pair that violates that."""
    if touch is None:
        return None
    upper = touch.point("button_1")
    lower = touch.point("button_3")
    if upper is None or lower is None:
        return touch
    if float(upper.v) <= float(lower.v) + 8.0:
        return touch
    points = dict(touch.points)
    points["button_1"] = YoloPoint("button_1", lower.u, lower.v, lower.conf)
    points["button_3"] = YoloPoint("button_3", upper.u, upper.v, upper.conf)
    return YoloTouch(
        panel_corners=touch.panel_corners,
        panel_conf=touch.panel_conf,
        points=points,
        masks=list(touch.masks),
    )


def without_gripper(touch: YoloTouch | None) -> YoloTouch | None:
    """Drop the Dex1 jaw and gripper_tip. FingerTip, panel, screen, and buttons stay."""
    if touch is None:
        return None
    return YoloTouch(
        panel_corners=touch.panel_corners,
        panel_conf=touch.panel_conf,
        points={name: point for name, point in touch.points.items() if name != "gripper_tip"},
        masks=[mask for mask in touch.masks if mask.name != "gripper"],
    )


class YoloTouchDetector:
    """Lazy ultralytics models. Missing weights or import leaves detection classical."""

    def __init__(
        self,
        seg_weights: str | Path = SEG_WEIGHTS,
        pose_weights: str | Path = POSE_WEIGHTS,
        imgsz: int = 640,
        name: str = "",
        cpu_pose: bool = False,
        use_onnx: bool = False,
        use_rknn: bool = False,
    ) -> None:
        self.name = str(name or "")
        self.use_onnx = bool(use_onnx) and not bool(use_rknn)
        self.use_rknn = bool(use_rknn)
        if self.use_rknn:
            self.seg_weights = rknn_weights(seg_weights)
            self.pose_weights = rknn_weights(pose_weights)
        elif self.use_onnx:
            self.seg_weights = onnx_weights(seg_weights)
            self.pose_weights = onnx_weights(pose_weights)
        else:
            self.seg_weights = Path(seg_weights)
            self.pose_weights = Path(pose_weights)
        self.imgsz = int(imgsz)
        self._seg_imgsz = self.imgsz
        self._pose_imgsz = self.imgsz
        self._cpu_pose = bool(cpu_pose)
        self._infer_lock = threading.Lock()
        self._cpu_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._wrist_waiting = False
        self._seg = None
        self._pose = None
        self._rknn_seg = None
        self._rknn_pose = None
        self._seg_disabled = False
        self._device = "cpu"
        self._half = False
        self._ready = False
        self._failed = ""
        self._precision_key: str | None = None

    def status(self) -> str:
        if self._failed:
            return self._failed
        if self._ready:
            if self.use_rknn:
                return "yolo rknn"
            precision = "fp16" if self._half else "fp32"
            return f"yolo {self._device} {precision}"
        pose_only = self._cpu_pose and self.pose_weights.is_file()
        if self._cpu_pose and not self.pose_weights.is_file():
            return "weights missing"
        if not pose_only and (not self.seg_weights.is_file() or not self.pose_weights.is_file()):
            return "weights missing"
        return "idle"

    def predict(self, bgr: np.ndarray, *, pose_only: bool = False) -> YoloTouch | None:
        if self.use_rknn:
            return self._predict_rknn(bgr, pose_only=pose_only or self._cpu_pose)
        if self._cpu_pose:
            return self._predict_cpu_pose(bgr)
        with self._load_lock:
            if not self._ensure():
                return None
        image = np.ascontiguousarray(bgr)
        if pose_only:
            # Chest seg+pose used to hold the GPU for the whole pair, so the
            # wrist fingertip only updated after both of those finished.
            self._wrist_waiting = True
            try:
                pose = self._forward(self._pose, image, polite=False, imgsz=self._pose_imgsz)
            finally:
                self._wrist_waiting = False
            return YoloTouch(
                panel_corners=None,
                panel_conf=0.0,
                points=_pose_points(pose),
                masks=[],
            )
        seg = self._forward(self._seg, image, polite=True, imgsz=self._seg_imgsz)
        pose = self._forward(self._pose, image, polite=True, imgsz=self._pose_imgsz)
        return YoloTouch(
            panel_corners=_panel_quad(seg),
            panel_conf=_panel_conf(seg),
            points=_pose_points(pose),
            masks=_seg_masks(seg),
        )

    def _predict_rknn(self, bgr: np.ndarray, *, pose_only: bool = False) -> YoloTouch | None:
        from ibvs_pad_touch.perception.rknn_yolo import pose_points, seg_instances

        with self._load_lock:
            if not self._ensure_rknn():
                return None
        image = np.ascontiguousarray(bgr)
        height, width = image.shape[:2]
        outputs, ratio, pad_x, pad_y = self._rknn_pose.infer(image)
        pose_only_touch = YoloTouch(
            panel_corners=None,
            panel_conf=0.0,
            points=_rknn_pose_points(pose_points(outputs[0], ratio, pad_x, pad_y, self._rknn_pose.imgsz)),
            masks=[],
        )
        if pose_only or self._rknn_seg is None or self._seg_disabled:
            return pose_only_touch
        seg_box: dict = {}

        def _run_seg() -> None:
            try:
                seg_box["value"] = self._rknn_seg.infer(image)
            except Exception as exc:
                seg_box["error"] = exc

        seg_thread = threading.Thread(target=_run_seg, name=f"rknn-seg-{self.name}", daemon=True)
        seg_thread.start()
        seg_thread.join(1.5)
        if seg_thread.is_alive() or "error" in seg_box or "value" not in seg_box:
            self._seg_disabled = True
            detail = seg_box.get("error", "timeout 1.5s")
            print(f"[ibvs-yolo] {self.name} seg skipped, pose still drawn: {detail}", flush=True)
            return pose_only_touch
        seg_out, seg_ratio, seg_px, seg_py = seg_box["value"]
        pose_out, pose_ratio, pose_px, pose_py = outputs, ratio, pad_x, pad_y
        det, proto = _seg_outputs(seg_out)
        instances = seg_instances(det, proto, seg_px, seg_py, (height, width), self._rknn_seg.imgsz)
        masks = [
            YoloMask(name=name, conf=conf, polygon=polygon)
            for name, conf, polygon in instances
        ]
        panel = next((item for item in instances if item[0] == "panel"), None)
        corners = None
        panel_conf = 0.0
        if panel is not None and len(panel[2]) >= 4:
            panel_conf = float(panel[1])
            quad = order_corners(cv2.boxPoints(cv2.minAreaRect(np.asarray(panel[2], dtype=np.float32))))
            corners = [(float(x), float(y)) for x, y in quad]
        points = pose_points(pose_out[0], pose_ratio, pose_px, pose_py, self._rknn_pose.imgsz)
        return YoloTouch(
            panel_corners=corners,
            panel_conf=panel_conf,
            points=_rknn_pose_points(points),
            masks=masks,
        )

    def _ensure_rknn(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        needed = [self.pose_weights] if self._cpu_pose else [self.seg_weights, self.pose_weights]
        missing = [path for path in needed if not path.is_file()]
        if missing:
            self._failed = "weights missing"
            print(f"[ibvs-yolo] weights missing {missing}", flush=True)
            return False
        try:
            from ibvs_pad_touch.perception.rknn_yolo import RknnSession

            pose_core = 2 if self._cpu_pose else 1
            self._rknn_pose = RknnSession(self.pose_weights, imgsz=640, core=pose_core)
            self._pose_imgsz = 640
            label = f"{self.name} " if self.name else ""
            print(f"[ibvs-yolo] {label}pose loaded npu core {pose_core} {self.pose_weights.name}", flush=True)
            if not self._cpu_pose:
                try:
                    self._rknn_seg = RknnSession(self.seg_weights, imgsz=1280, core=0)
                    self._seg_imgsz = 1280
                    print(f"[ibvs-yolo] {label}seg loaded npu core 0 {self.seg_weights.name}", flush=True)
                except Exception as exc:
                    self._rknn_seg = None
                    print(f"[ibvs-yolo] {label}seg failed, pose still runs: {exc}", flush=True)
            self._ready = True
            return True
        except Exception as exc:
            self._failed = f"rknn unavailable: {exc}"
            print(f"[ibvs-yolo] {self._failed}", flush=True)
            return False

    def _predict_cpu_pose(self, bgr: np.ndarray) -> YoloTouch | None:
        """Right-wrist pose. A second model on CPU, so the chest GPU run is not moved."""
        with self._load_lock:
            if not self._ensure():
                return None
        image = np.ascontiguousarray(bgr)
        with self._cpu_lock:
            pose = self._predict(self._pose, image, device="cpu", half=False, imgsz=self._pose_imgsz)
        return YoloTouch(
            panel_corners=None,
            panel_conf=0.0,
            points=_pose_points(pose),
            masks=[],
        )

    def _forward_pair(self, image: np.ndarray):
        """One GPU. Two Ultralytics predicts at once queue behind each other and run slower than one after the other."""
        seg = self._forward(self._seg, image, polite=False, imgsz=self._seg_imgsz)
        pose = self._forward(self._pose, image, polite=False, imgsz=self._pose_imgsz)
        return seg, pose

    def _forward(self, model, image: np.ndarray, *, polite: bool = False, imgsz: int | None = None):
        deadline = time.monotonic() + 0.4
        while True:
            if polite and self._wrist_waiting and time.monotonic() < deadline:
                time.sleep(0.005)
                continue
            with self._infer_lock:
                if polite and self._wrist_waiting and time.monotonic() < deadline:
                    continue
                with _GPU_SLOTS:
                    return self._predict(model, image, imgsz=imgsz)

    def _predict(
        self,
        model,
        image: np.ndarray,
        *,
        device: str | None = None,
        half: bool | None = None,
        imgsz: int | None = None,
    ):
        use_device = self._device if device is None else device
        use_half = self._half if half is None else half
        kwargs = {
            "imgsz": self.imgsz if imgsz is None else int(imgsz),
            "conf": 0.15,
            "device": use_device,
            "verbose": False,
        }
        if use_half:
            kwargs.update(self._precision_kwargs())
        try:
            return model.predict(image, **kwargs)[0]
        except TypeError as exc:
            if "quantize" not in kwargs or "quantize" not in str(exc):
                raise
            self._precision_key = "half"
            kwargs.pop("quantize", None)
            kwargs["half"] = True
            return model.predict(image, **kwargs)[0]
        except RuntimeError as exc:
            text = str(exc).lower()
            if not self._half or not any(token in text for token in ("half", "float16", "fp16", "quantize")):
                raise
            self._half = False
            self._precision_key = None
            kwargs.pop("half", None)
            kwargs.pop("quantize", None)
            print(f"[ibvs-yolo] FP16 不可用，改用 FP32: {exc}", flush=True)
            return model.predict(image, **kwargs)[0]

    def _precision_kwargs(self) -> dict:
        """FP16 on CUDA. New Ultralytics calls this quantize=16; half= warns every frame."""
        if not self._half:
            return {}
        if self._precision_key is None:
            self._precision_key = "quantize" if _ultralytics_uses_quantize() else "half"
        if self._precision_key == "quantize":
            return {"quantize": 16}
        return {"half": True}

    def _ensure_cpu_pose(self) -> bool:
        if not self.pose_weights.is_file():
            self._failed = "weights missing"
            print(f"[ibvs-yolo] weights missing pose={self.pose_weights}", flush=True)
            return False
        try:
            from ultralytics import YOLO

            _silence_half_deprecation()
            self._device = "cpu"
            self._half = False
            self._pose = YOLO(str(self.pose_weights))
            self._pose_imgsz = _model_imgsz(self._pose, self.pose_weights, self.imgsz)
            blank = np.zeros((self._pose_imgsz, self._pose_imgsz, 3), dtype=np.uint8)
            self._predict(self._pose, blank, device="cpu", half=False, imgsz=self._pose_imgsz)
            self._ready = True
            label = f"{self.name} " if self.name else ""
            kind = "onnx " if self.use_onnx else ""
            print(
                f"[ibvs-yolo] {label}loaded cpu fp32 {kind}{self.pose_weights.name}",
                flush=True,
            )
            return True
        except Exception as exc:
            self._failed = f"yolo unavailable: {exc}"
            print(f"[ibvs-yolo] {self._failed}", flush=True)
            return False

    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        if self.use_rknn:
            return self._ensure_rknn()
        if self._cpu_pose:
            return self._ensure_cpu_pose()
        if not self.seg_weights.is_file() or not self.pose_weights.is_file():
            self._failed = "weights missing"
            print(
                f"[ibvs-yolo] weights missing seg={self.seg_weights} pose={self.pose_weights}",
                flush=True,
            )
            return False
        try:
            from ultralytics import YOLO

            _silence_half_deprecation()
            self._device = prepare_torch_gpu()
            self._half = self._device.startswith("cuda") and not self.use_onnx
            self._seg = YOLO(str(self.seg_weights))
            self._pose = YOLO(str(self.pose_weights))
            self._seg_imgsz = _model_imgsz(self._seg, self.seg_weights, self.imgsz)
            self._pose_imgsz = _model_imgsz(self._pose, self.pose_weights, self.imgsz)
            blank_side = max(self._seg_imgsz, self._pose_imgsz)
            blank = np.zeros((blank_side, blank_side, 3), dtype=np.uint8)
            try:
                self._forward_pair(blank)
            except Exception as exc:
                if not self._device.startswith("cuda"):
                    raise
                print(f"[ibvs-yolo] GPU 预热失败，改用 CPU: {exc}", flush=True)
                self._device = "cpu"
                self._half = False
                self._forward_pair(blank)
            self._ready = True
            label = f"{self.name} " if self.name else ""
            where = self._device if self._device == "cpu" else f"{self._device} {gpu_name()}"
            precision = "fp16" if self._half else "fp32"
            kind = "onnx " if self.use_onnx else ""
            print(
                f"[ibvs-yolo] {label}loaded {where} {precision} {kind}"
                f"{self.seg_weights.name} + {self.pose_weights.name}",
                flush=True,
            )
            return True
        except Exception as exc:
            self._failed = f"yolo unavailable: {exc}"
            print(f"[ibvs-yolo] {self._failed}", flush=True)
            return False


def _silence_half_deprecation() -> None:
    """Ultralytics prints this on every predict(). FP16 itself stays enabled."""
    import logging

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "'half' is deprecated" not in record.getMessage()

    logger = logging.getLogger("ultralytics")
    if not any(isinstance(item, _Filter) for item in logger.filters):
        logger.addFilter(_Filter())


def _ultralytics_uses_quantize() -> bool:
    """predict(half=True) warns on Ultralytics 8.4.90+. quantize=16 is the same FP16."""
    global _USE_QUANTIZE
    if _USE_QUANTIZE is not None:
        return _USE_QUANTIZE
    try:
        import ultralytics

        parts: list[int] = []
        for piece in str(ultralytics.__version__).split("."):
            digits = "".join(ch for ch in piece if ch.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        while len(parts) < 3:
            parts.append(0)
        _USE_QUANTIZE = tuple(parts[:3]) >= (8, 4, 90)
    except Exception:
        _USE_QUANTIZE = False
    return _USE_QUANTIZE


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
    return _pose_keypoint(touch, "gripper_tip", conf_min, "pose")


def pose_finger_tip(touch: YoloTouch | None, conf_min: float) -> CameraPoint | None:
    """Revo2 index fingertip. This replaces gripper_tip in the wrist approach."""
    return _pose_keypoint(touch, "FingerTip", conf_min, "finger")


def _pose_keypoint(
    touch: YoloTouch | None,
    name: str,
    conf_min: float,
    source: str,
) -> CameraPoint | None:
    if touch is None:
        return None
    pt = touch.point(name)
    if pt is None or float(pt.conf) < float(conf_min):
        return None
    return CameraPoint(u=pt.u, v=pt.v, source=source, confidence=float(pt.conf))


def _rknn_pose_points(points: dict[str, tuple[float, float, float]]) -> dict[str, YoloPoint]:
    return {
        name: YoloPoint(name=name, u=u, v=v, conf=conf)
        for name, (u, v, conf) in points.items()
    }


def _seg_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    det = next(item for item in outputs if item.shape[-1] == 38)
    proto = next(item for item in outputs if item.shape[-1] != 38)
    return det, proto


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
    """Keep the best keypoint of each name across every detected person.

    The wrist view often contains the feet as well as the hand. Keeping only
    the highest box drops a lower-scoring FingerTip.
    """
    kpts = getattr(result, "keypoints", None)
    if kpts is None or kpts.data is None or len(kpts.data) == 0:
        return {}
    data = kpts.data.detach().cpu().numpy()
    found: dict[str, YoloPoint] = {}
    for row in data:
        for i, name in enumerate(KPT_NAMES):
            if i >= row.shape[0]:
                break
            u, v, conf = (float(row[i, 0]), float(row[i, 1]), float(row[i, 2]))
            if conf <= 0.0:
                continue
            current = found.get(name)
            if current is None or conf > current.conf:
                found[name] = YoloPoint(name=name, u=u, v=v, conf=conf)
    return found
