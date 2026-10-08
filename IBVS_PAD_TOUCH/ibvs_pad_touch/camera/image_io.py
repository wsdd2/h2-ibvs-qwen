"""Sidecar frames: raw BGR (no cv2), PPM fallback, JPEG when cv2 exists."""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

BGR_MAGIC = b"IBVSBGR1"
DEPTH_MAGIC = b"IBVSDP01"


def write_ppm(path: Path, bgr: np.ndarray) -> Path:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    tmp = path.with_suffix(".tmp.ppm")
    out = path.with_suffix(".ppm")
    tmp.write_bytes(f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes())
    tmp.replace(out)
    return out


def read_ppm(data: bytes) -> Optional[np.ndarray]:
    if not data.startswith(b"P6"):
        return None
    try:
        header, raw = data.split(b"\n255\n", 1)
    except ValueError:
        return None
    parts = header.decode("ascii", errors="ignore").split()
    if len(parts) < 3:
        return None
    try:
        width, height = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    need = width * height * 3
    if len(raw) < need or width <= 0 or height <= 0:
        return None
    rgb = np.frombuffer(raw[:need], dtype=np.uint8).reshape(height, width, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def write_bgr(path: Path, bgr: np.ndarray) -> Path:
    arr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = arr.shape[:2]
    tmp = path.with_suffix(".tmp.bgr")
    out = path.with_suffix(".bgr")
    tmp.write_bytes(BGR_MAGIC + struct.pack("<II", w, h) + arr.tobytes())
    tmp.replace(out)
    return out


def read_bgr(data: bytes) -> Optional[np.ndarray]:
    if not data.startswith(BGR_MAGIC) or len(data) < 16:
        return None
    width, height = struct.unpack("<II", data[8:16])
    need = width * height * 3
    raw = data[16:]
    if width <= 0 or height <= 0 or len(raw) < need:
        return None
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape(height, width, 3).copy()


def write_depth(path: Path, depth_raw: np.ndarray, scale_m: float) -> Path:
    arr = np.ascontiguousarray(depth_raw, dtype=np.uint16)
    h, w = arr.shape[:2]
    tmp = path.with_suffix(".tmp.depth")
    tmp.write_bytes(
        DEPTH_MAGIC
        + struct.pack("<IIf", w, h, float(scale_m))
        + arr.tobytes()
    )
    tmp.replace(path)
    return path


def read_depth(path: Path, max_age_s: float = 2.0) -> Optional[np.ndarray]:
    try:
        stat = path.stat()
        if max_age_s > 0 and time.time() - stat.st_mtime > max_age_s:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(DEPTH_MAGIC) or len(data) < 20:
        return None
    width, height, scale_m = struct.unpack("<IIf", data[8:20])
    need = int(width) * int(height) * 2
    raw = data[20:]
    if width <= 0 or height <= 0 or len(raw) < need or scale_m <= 0:
        return None
    depth = np.frombuffer(raw[:need], dtype=np.uint16).reshape(height, width)
    return depth.astype(np.float32) * float(scale_m)


def read_image(path: Path) -> Optional[np.ndarray]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 16:
        return None
    if data.startswith(BGR_MAGIC):
        return read_bgr(data)
    if data.startswith(b"P6"):
        return read_ppm(data)
    try:
        import cv2

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def pick_sidecar(paths: list[Path], max_age_s: float = 4.0) -> Optional[Path]:
    newest: Optional[Path] = None
    newest_mtime = 0.0
    now = time.time()
    for cand in paths:
        try:
            stat = cand.stat()
        except OSError:
            continue
        if stat.st_size < 200:
            continue
        if max_age_s > 0 and now - stat.st_mtime > max_age_s:
            continue
        if stat.st_mtime >= newest_mtime:
            newest = cand
            newest_mtime = stat.st_mtime
    return newest
