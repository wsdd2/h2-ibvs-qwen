# -*- coding: utf-8 -*-
"""RK3588 NPU inference for the exported YOLO26 end-to-end graphs.

The ONNX heads are already decoded to 300 candidates:
  pose (1, 300, 21) = xyxy, conf, cls, 5 keypoints x (x, y, conf)
  seg  (1, 300, 38) = xyxy, conf, cls, 32 mask coefficients
       plus proto (1, 32, 320, 320)
Coordinates are in the letterboxed network input. The RKNN file applies
mean 0 and std 255, so the runtime input is RGB uint8 in NHWC.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SEG_NAMES = ("panel", "gripper", "screen")
POSE_NAMES = ("gripper_tip", "button_1", "button_2", "button_3", "FingerTip")


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Match the Ultralytics center pad used when the ONNX files were checked."""
    height, width = bgr.shape[:2]
    ratio = min(float(size) / float(height), float(size) / float(width))
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    resized = cv2.resize(bgr, (resized_w, resized_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, ratio, pad_x, pad_y


def unletter(x: float, y: float, ratio: float, pad_x: int, pad_y: int) -> tuple[float, float]:
    return (float(x) - float(pad_x)) / ratio, (float(y) - float(pad_y)) / ratio


class RknnSession:
    """One RKNN model bound to a single NPU core."""

    def __init__(self, path: str | Path, imgsz: int, core: int) -> None:
        from rknnlite.api import RKNNLite

        self.path = Path(path)
        self.imgsz = int(imgsz)
        self.core = int(core)
        masks = {
            0: RKNNLite.NPU_CORE_0,
            1: RKNNLite.NPU_CORE_1,
            2: RKNNLite.NPU_CORE_2,
        }
        if self.core not in masks:
            raise ValueError(f"NPU core must be 0, 1, or 2, got {self.core}")
        session = RKNNLite(verbose=False)
        if session.load_rknn(str(self.path)) != 0:
            raise RuntimeError(f"RKNN load failed: {self.path}")
        if session.init_runtime(core_mask=masks[self.core]) != 0:
            raise RuntimeError(f"RKNN runtime init failed on core {self.core}: {self.path}")
        self.session = session

    def infer(self, bgr: np.ndarray) -> tuple[list[np.ndarray], float, int, int]:
        canvas, ratio, pad_x, pad_y = letterbox(bgr, self.imgsz)
        rgb = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        # Runtime 2.5 accepts a 4-D NHWC uint8 buffer and applies /255 itself.
        batch = np.ascontiguousarray(rgb[None])
        outputs = self.session.inference(inputs=[batch], data_format="nhwc")
        if not outputs:
            raise RuntimeError(f"RKNN inference returned no outputs: {self.path}")
        return [np.asarray(item) for item in outputs], ratio, pad_x, pad_y

    def release(self) -> None:
        self.session.release()


def _candidate_rows(output: np.ndarray, width: int, imgsz: int) -> np.ndarray:
    """Return (N, width) detections. RKNN may emit either (1, 300, C) or (1, C, 300)."""
    array = np.squeeze(np.asarray(output, dtype=np.float32))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[-1] != width and array.shape[0] == width:
        array = np.transpose(array, (1, 0))
    rows = array.reshape(-1, width).copy()
    if rows.shape[0] and float(np.max(np.abs(rows[:, :4]))) <= 2.0:
        rows[:, 0:4] *= float(imgsz)
        for index in range(5):
            rows[:, 6 + index * 3] *= float(imgsz)
            rows[:, 7 + index * 3] *= float(imgsz)
    return rows


def pose_points(
    output: np.ndarray,
    ratio: float,
    pad_x: int,
    pad_y: int,
    imgsz: int,
    conf_min: float = 0.15,
) -> dict[str, tuple[float, float, float]]:
    """Best keypoint of each name. Coordinates are returned in the camera frame."""
    rows = _candidate_rows(output, 21, imgsz)
    found: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        if float(row[4]) < float(conf_min):
            continue
        for index, name in enumerate(POSE_NAMES):
            base = 6 + index * 3
            key_u, key_v, key_conf = (float(row[base]), float(row[base + 1]), float(row[base + 2]))
            if key_conf <= 0.0:
                continue
            u, v = unletter(key_u, key_v, ratio, pad_x, pad_y)
            current = found.get(name)
            if current is None or key_conf > current[2]:
                found[name] = (u, v, key_conf)
    return found


def seg_instances(
    det: np.ndarray,
    proto: np.ndarray,
    pad_x: int,
    pad_y: int,
    orig_shape: tuple[int, int],
    imgsz: int,
    conf_min: float = 0.15,
) -> list[tuple[str, float, list[tuple[float, float]]]]:
    """Mask polygons in the original camera frame."""
    rows = _candidate_rows(det, 38, imgsz)
    proto_map = np.asarray(proto, dtype=np.float32)
    proto_map = np.squeeze(proto_map)
    if proto_map.ndim == 3 and proto_map.shape[-1] == 32:
        proto_map = np.transpose(proto_map, (2, 0, 1))
    proto_map = proto_map.reshape(32, -1)
    side = int(round(proto_map.shape[1] ** 0.5))
    proto_map = proto_map.reshape(32, side, side)
    height, width = orig_shape
    found: list[tuple[str, float, list[tuple[float, float]]]] = []
    for row in rows:
        conf = float(row[4])
        if conf < float(conf_min):
            continue
        class_id = int(round(float(row[5])))
        if class_id < 0 or class_id >= len(SEG_NAMES):
            continue
        polygon = _mask_polygon(row[6:38], proto_map, row[:4], pad_x, pad_y, (height, width), imgsz)
        if polygon is None:
            continue
        found.append((SEG_NAMES[class_id], conf, polygon))
    return found


def _mask_polygon(
    coeff: np.ndarray,
    proto: np.ndarray,
    box: np.ndarray,
    pad_x: int,
    pad_y: int,
    orig_shape: tuple[int, int],
    imgsz: int,
) -> list[tuple[float, float]] | None:
    mask = coeff.astype(np.float32) @ proto.reshape(32, -1)
    mask = 1.0 / (1.0 + np.exp(-mask))
    mask = mask.reshape(proto.shape[-2], proto.shape[-1])
    mask = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = int(np.clip(min(x1, x2), 0, imgsz - 1))
    x2 = int(np.clip(max(x1 + 1, max(x1, int(round(float(box[2]))))), 0, imgsz))
    y1 = int(np.clip(min(y1, y2), 0, imgsz - 1))
    y2 = int(np.clip(max(y1 + 1, max(y1, int(round(float(box[3]))))), 0, imgsz))
    cropped = np.zeros((imgsz, imgsz), dtype=np.float32)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    bottom = imgsz - pad_y
    right = imgsz - pad_x
    if bottom <= pad_y or right <= pad_x:
        return None
    cropped = cropped[pad_y:bottom, pad_x:right]
    height, width = orig_shape
    cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (cropped > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 16.0:
        return None
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return None
    return [(float(x), float(y)) for x, y in approx]
