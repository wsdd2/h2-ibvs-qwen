"""Background ViT-VS comparison. The result is for display and is never a command."""

from __future__ import annotations

import threading
from typing import Callable

import cv2
import numpy as np

from ibvs_pad_touch.servo.vit_shadow import (
    ShadowView,
    VitShadowObserver,
    comparison_payload,
)

INPUT_SIZE = 224
PATCH_STRIDE = 14


class Dinov2PatchExtractor:
    """Lazy DINOv2 ViT-S/14. The model loads on the first describe call."""

    name = "dinov2-vits14"

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"

    def describe(self, bgr: np.ndarray) -> np.ndarray:
        model, device = self._load()
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        batch = (tensor.to(device) - mean) / std
        with torch.no_grad():
            layers = model.get_intermediate_layers(batch, n=1, reshape=True)
        grid = layers[0][0].detach().float().cpu().permute(1, 2, 0).numpy()
        return np.ascontiguousarray(grid, dtype=np.float64)

    def _load(self):
        if self._model is not None:
            return self._model, self._device
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            model.eval()
            model.to(device)
        except Exception:
            if device == "cpu":
                raise
            device = "cpu"
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            model.eval()
            model.to(device)
        self._model = model
        self._device = device
        return model, device


class ShadowMonitor:
    """Keep a reference wrist image and publish a comparison beside the live step."""

    def __init__(
        self,
        frame_fn: Callable[[], np.ndarray | None],
        depth_fn: Callable[[], np.ndarray | None],
        actual_fn: Callable[[], tuple[float, float, float] | list[float] | None],
        intrinsics_fn: Callable[[], dict[str, float]],
        extractor=None,
    ) -> None:
        self._frame_fn = frame_fn
        self._depth_fn = depth_fn
        self._actual_fn = actual_fn
        self._intrinsics_fn = intrinsics_fn
        self._extractor = extractor
        self._observer = VitShadowObserver()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._once = False
        self._reference: np.ndarray | None = None
        self._desired: np.ndarray | None = None
        self._generation = 0
        self._payload = comparison_payload(
            None,
            None,
            enabled=False,
            reference=False,
            extractor=self.extractor_name,
            message="还没记下参考腕图",
        )

    @property
    def extractor_name(self) -> str:
        extractor = self._extractor
        if extractor is None:
            return Dinov2PatchExtractor.name
        return str(getattr(extractor, "name", "custom"))

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._payload)

    def capture_reference(self, bgr: np.ndarray) -> dict[str, object]:
        frame = np.ascontiguousarray(bgr)
        if frame.ndim != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
            raise ValueError("参考腕图太小")
        with self._lock:
            self._reference = frame.copy()
            self._desired = None
            self._generation += 1
            self._observer.reset()
            self._payload = comparison_payload(
                None,
                None,
                enabled=self._running,
                reference=True,
                extractor=self.extractor_name,
                message="已记下参考腕图。点「算一拍」或「开始对照」后才计算，不控制手臂",
            )
            return dict(self._payload)

    def clear_reference(self) -> dict[str, object]:
        with self._lock:
            self._reference = None
            self._desired = None
            self._generation += 1
            self._observer.reset()
            self._payload = comparison_payload(
                None,
                None,
                enabled=self._running,
                reference=False,
                extractor=self.extractor_name,
                message="已清除参考腕图",
            )
            return dict(self._payload)

    def set_running(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            self._running = bool(enabled)
            self._payload["enabled"] = self._running
            if self._running:
                self._payload["message"] = "对照进行中，结果只显示，不发给手臂"
            else:
                self._payload["message"] = "对照已暂停，手臂仍走当前方案"
            payload = dict(self._payload)
        if self._running:
            self._ensure_thread()
            self._wake.set()
        return payload

    def request_once(self) -> dict[str, object]:
        with self._lock:
            self._once = True
            self._payload["message"] = "正在计算对照，手臂不会使用这个结果"
            payload = dict(self._payload)
        self._ensure_thread()
        self._wake.set()
        return payload

    def refresh(self) -> dict[str, object]:
        """Compute one comparison on the caller thread. Tests use this directly."""
        with self._lock:
            reference = None if self._reference is None else self._reference
            desired = self._desired
            generation = self._generation
            running = self._running
        if reference is None:
            return self._store(
                comparison_payload(
                    None,
                    None,
                    enabled=running,
                    reference=False,
                    extractor=self.extractor_name,
                    message="还没记下参考腕图",
                )
            )
        current = self._frame_fn()
        if current is None:
            return self._store(
                comparison_payload(
                    None,
                    None,
                    enabled=running,
                    reference=True,
                    extractor=self.extractor_name,
                    message="还没有右腕画面",
                )
            )
        try:
            extractor = self._extractor_instance()
            if desired is None:
                desired = extractor.describe(reference)
            current_desc = extractor.describe(current)
            with self._lock:
                if generation != self._generation:
                    return dict(self._payload)
                self._desired = desired
            info = self._intrinsics_fn() or {}
            depth = self._depth_fn()
            view = _view_for(current.shape, info)
            depths = patch_depths(
                depth,
                current_desc.shape[0],
                current_desc.shape[1],
                info.get("plane_z_m"),
            )
            report = self._observer.observe(desired, current_desc, depths, view)
            actual = self._actual_fn()
            depth_note = depth is None and bool(info.get("depth_assumed"))
        except Exception as exc:
            return self._store(
                comparison_payload(
                    None,
                    None,
                    enabled=running,
                    reference=True,
                    extractor=self.extractor_name,
                    message=f"对照失败：{exc}",
                )
            )
        with self._lock:
            if generation != self._generation:
                return dict(self._payload)
        payload = comparison_payload(
            actual,
            report,
            enabled=running,
            reference=True,
            extractor=self.extractor_name,
        )
        if depth_note and payload.get("shadow_step_mm") is not None:
            payload["message"] = str(payload["message"]) + "；没有腕部深度，shadow 用 0.20 m 标称深度"
        return self._store(payload)

    def close(self) -> None:
        self._running = False
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.2)

    def _ensure_thread(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vit-shadow", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(1.2)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                should = self._running or self._once
                self._once = False
            if should:
                self.refresh()

    def _extractor_instance(self):
        with self._lock:
            if self._extractor is None:
                self._extractor = Dinov2PatchExtractor()
            return self._extractor

    def _store(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._payload = payload
            return dict(payload)


def patch_depths(
    depth_m: np.ndarray | None,
    rows: int,
    cols: int,
    plane_z_m: float | None,
) -> np.ndarray:
    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    depth = None if depth_m is None else np.asarray(depth_m)
    if depth is not None and depth.ndim == 2 and depth.size:
        height, width = depth.shape[:2]
        for row in range(rows):
            for col in range(cols):
                v = min(height - 1, int((row + 0.5) * height / rows))
                u = min(width - 1, int((col + 0.5) * width / cols))
                grid[row, col] = float(depth[v, u])
    if plane_z_m is not None and np.isfinite(float(plane_z_m)):
        missing = ~np.isfinite(grid) | (grid <= 0.02) | (grid >= 2.5)
        grid[missing] = float(plane_z_m)
    return grid


def _view_for(shape: tuple[int, ...], intrinsics: dict[str, float]) -> ShadowView:
    height = max(1, int(shape[0]))
    width = max(1, int(shape[1]))
    scale_x = INPUT_SIZE / float(width)
    scale_y = INPUT_SIZE / float(height)
    fx = float(intrinsics.get("fx") or 615.0) * scale_x
    fy = float(intrinsics.get("fy") or 615.0) * scale_y
    cx = float(intrinsics.get("cx") or (width * 0.5)) * scale_x
    cy = float(intrinsics.get("cy") or (height * 0.5)) * scale_y
    return ShadowView(fx=fx, fy=fy, cx=cx, cy=cy, stride=PATCH_STRIDE)
