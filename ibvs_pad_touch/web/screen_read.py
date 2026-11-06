# -*- coding: utf-8 -*-
"""Read the solid black block on a panel crop and compare two crops.

The selection is the filled dark bar with light glyphs inside it. An outline
and a dithered hatch stay out of this result. Text uses RapidOCR when that
package is installed; the position and the image comparison do not need it.
"""
from __future__ import annotations

import re
import threading
from typing import Any

import cv2
import numpy as np

from ibvs_pad_touch.platform import gpu_name, onnx_cuda

LEXICON = (
    "返回",
    "信息",
    "操作",
    "配置",
    "统计",
    "数值",
    "部件",
    "系统",
    "数据记录",
    "事件记录",
    "运行时间",
    "室内温度",
    "设备启动",
    "设备起动",
    "非本地停机",
    "开机",
    "关机",
    "设备停止",
    "ON",
)
# Blurred upper-panel difference. The footer clock is left out, so a ticking
# clock alone does not count as a new screen.
SCREEN_DIFF_THRESHOLD = 8.0
_ENGINE: Any = None
_ENGINE_FAILED = False
_ENGINE_LOCK = threading.RLock()


def _engine() -> Any:
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE_FAILED:
        return None
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE_FAILED:
            return None
        if _ENGINE is not None:
            return _ENGINE
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            _ENGINE_FAILED = True
            return None
        if onnx_cuda():
            try:
                _ENGINE = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                print(f"[ibvs-ocr] RapidOCR CUDA {gpu_name()}", flush=True)
                return _ENGINE
            except Exception as exc:
                print(f"[ibvs-ocr] CUDA 失败，改用 CPU: {exc}", flush=True)
        try:
            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE_FAILED = True
            return None
        print("[ibvs-ocr] RapidOCR CPU", flush=True)
        return _ENGINE


def run_ocr(image: np.ndarray) -> Any:
    """One RapidOCR call. None means the package is not available."""
    engine = _engine()
    if engine is None:
        return None
    with _ENGINE_LOCK:
        return engine(image)


def luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask.tolist() + [0]):
        if value and start is None:
            start = index
        if not value and start is not None:
            found.append((start, index))
            start = None
    return found


def edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 2:
        return 99
    prev = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(
                min(
                    prev[other_index] + 1,
                    current[other_index - 1] + 1,
                    prev[other_index - 1] + (char != other),
                )
            )
        prev = current
    return prev[-1]


def snap_word(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%.:/°℃+\-]", "", cleaned)
    if cleaned in LEXICON:
        return cleaned
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,6}", cleaned):
        return cleaned
    ranked = sorted((edit_distance(cleaned, word), word) for word in LEXICON if word != "ON")
    distance, word = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 99
    if distance <= 1 and distance < second and len(word) == len(cleaned):
        return word
    return cleaned


def _scaled_level(image: np.ndarray) -> tuple[np.ndarray, float]:
    level = luminance(image)
    height, width = level.shape[:2]
    if width < 2 or height < 2:
        return level, 1.0
    scale = 960.0 / float(width)
    if abs(scale - 1.0) < 0.04:
        return level, 1.0
    resized = cv2.resize(
        level,
        (960, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale


def _unscale(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    if scale == 1.0:
        return tuple(int(value) for value in box)
    x, y, w, h = box
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        max(1, int(round(w / scale))),
        max(1, int(round(h / scale))),
    )


def idle_buttons(level: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Home screen: ON is the filled key, the temperature cell is the dither beside it."""
    height, _width = level.shape
    if height < 160:
        return None
    background = float(np.percentile(level[: int(height * 0.82)], 75))
    if float((level[180:420] < background - 32).mean()) > 0.012:
        return None
    column = np.median(level[40:125], axis=0)
    if column.size < 360:
        return None
    split = 180 + int(np.argmax(column[210:360]))
    if not 260 < split < 340:
        return None

    def tighten(x0: int, x1: int) -> tuple[int, int, int, int] | None:
        x0 = max(0, x0)
        x1 = min(level.shape[1], x1)
        if x1 - x0 < 20:
            return None
        fraction = (level[30:140, x0:x1] < background - 30).mean(axis=0)
        cols = np.where(fraction > 0.18)[0]
        if cols.size < 15:
            return None
        left, right = x0 + int(cols[0]), x0 + int(cols[-1]) + 1
        rows = np.where((level[:, left:right] < background - 28).mean(axis=1) > 0.18)[0]
        rows = rows[(rows >= 20) & (rows <= 150)]
        if rows.size < 12:
            return None
        return left, int(rows.min()), right - left, int(rows.max() - rows.min() + 1)

    temperature = tighten(split - 130, split - 4)
    power = tighten(split + 4, split + 90)
    if temperature is None or power is None:
        return None
    return power, temperature


def find_selection(level: np.ndarray) -> tuple[int, int, int, int] | None:
    idle = idle_buttons(level)
    if idle is not None:
        return idle[0]
    height, _width = level.shape
    if height < 48:
        return None
    background = float(np.percentile(level[: int(height * 0.82)], 75))
    best: tuple[float, int, int, int] | None = None
    window = 32
    limit = int(height * 0.80) - window
    if limit <= 0:
        return None
    for top in range(0, limit, 2):
        column = np.median(level[top : top + window], axis=0)
        closed = cv2.morphologyEx(
            (column < background - 16).astype(np.uint8).reshape(1, -1),
            cv2.MORPH_CLOSE,
            np.ones((1, 28), np.uint8),
        ).reshape(-1)
        for left, right in _runs(closed):
            span = right - left
            if span < 48 or span > 360:
                continue
            median = float(np.median(column[left:right]))
            if median > background - 24:
                continue
            score = (background - median) * min(span, 240)
            if best is None or score > best[0]:
                best = (score, left, right, top)
    if best is None:
        return None
    _, left, right, top = best
    row_median = np.median(level[:, left:right], axis=1)
    rows = np.where(row_median < background - 16)[0]
    rows = rows[(rows >= top - 12) & (rows <= top + window + 14)]
    if rows.size < 8:
        return left, top, right - left, window
    box = (int(left), int(rows.min()), int(right - left), int(rows.max() - rows.min() + 1))
    box = _trim_selection(level, box, background)
    return _expand_selection(level, box, background)


def _trim_selection(
    level: np.ndarray, box: tuple[int, int, int, int], background: float
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    fraction = (level[:, x : x + w] < background - 30).mean(axis=1)
    rows = np.where(fraction > 0.32)[0]
    rows = rows[(rows >= y - 4) & (rows <= y + h + 4)]
    if rows.size < 8:
        return box
    breaks = np.where(np.diff(rows) > 6)[0]
    groups = np.split(rows, breaks + 1)
    center = y + h / 2
    group = min(groups, key=lambda item: abs(float(item.mean()) - center))
    if group.size < 8:
        return box
    return x, int(group.min()), w, int(group.max() - group.min() + 1)


def _expand_selection(
    level: np.ndarray, box: tuple[int, int, int, int], background: float
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    if h < 8:
        return box
    fraction = (level[y : y + h] < background - 26).mean(axis=0)
    closed = cv2.morphologyEx(
        (fraction > 0.16).astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 24), np.uint8),
    ).reshape(-1)
    center = x + w / 2
    for left, right in _runs(closed):
        if left - 8 <= center <= right + 8 and 48 <= right - left <= 360:
            return left, y, right - left, h
    return box


def _recognize_crop(gray: np.ndarray) -> tuple[str, float]:
    engine = _engine()
    if engine is None or gray.size == 0:
        return "", 0.0
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(canvas, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    with _ENGINE_LOCK:
        recognized = engine.text_recognizer([canvas])[0]
    if not recognized:
        return "", 0.0
    text, score = recognized[0]
    return snap_word(str(text)), float(score)


def selection_text(level: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, float]:
    x, y, w, h = box
    crop = level[y : y + h, x : x + w]
    if crop.size == 0:
        return "", 0.0
    votes: dict[str, float] = {}
    fallback = ("", 0.0)
    for percent in (42, 50, 58, 66):
        glyphs = (crop > np.percentile(crop, percent)).astype(np.uint8) * 255
        glyphs = cv2.morphologyEx(glyphs, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        glyphs = cv2.resize(255 - glyphs, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        text, score = _recognize_crop(glyphs)
        if text in LEXICON and score >= votes.get(text, 0.0):
            votes[text] = score
        elif score > fallback[1]:
            fallback = (text, score)
    if votes:
        word = max(votes, key=votes.get)
        return word, votes[word]
    return fallback


def read_panel(image: np.ndarray | None) -> dict[str, Any]:
    """Solid block on this crop. xywh is in the crop's own pixels."""
    empty = {"text": "", "score": 0.0, "xywh": None}
    if image is None or getattr(image, "size", 0) == 0:
        return empty
    level, scale = _scaled_level(image)
    idle = idle_buttons(level)
    if idle is not None:
        box = _unscale(idle[0], scale)
        return {"text": "ON", "score": 1.0, "xywh": list(box)}
    found = find_selection(level)
    if found is None:
        return empty
    text, score = selection_text(level, found)
    box = _unscale(found, scale)
    return {"text": text, "score": float(score), "xywh": list(box)}


def format_block(info: dict[str, Any] | None) -> str:
    box = None if not info else info.get("xywh")
    if not box or len(box) != 4:
        return "没有看到实心黑块"
    text = str(info.get("text") or "").strip() or "文字未识别"
    x, y, w, h = (int(value) for value in box)
    return f"黑块 {text}  x={x} y={y} 宽={w} 高={h}"


def _gray_small(image: np.ndarray) -> np.ndarray:
    gray = luminance(image)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 1.2)


def upper_diff(before: np.ndarray, after: np.ndarray) -> float:
    left = _gray_small(before)
    right = _gray_small(after)
    rows = int(left.shape[0] * 0.82)
    delta = cv2.absdiff(left[:rows], right[:rows])
    return float(delta.mean())


def _center(info: dict[str, Any], shape: tuple[int, ...]) -> tuple[float, float] | None:
    box = info.get("xywh")
    if not box or len(box) != 4 or shape[0] < 1 or shape[1] < 1:
        return None
    x, y, w, h = (float(value) for value in box)
    return ((x + w / 2.0) / float(shape[1]), (y + h / 2.0) / float(shape[0]))


def blocks_differ(
    before: np.ndarray,
    after: np.ndarray,
    before_info: dict[str, Any],
    after_info: dict[str, Any],
) -> bool:
    """True when the panel content changed since the previous crop."""
    prev_text = str(before_info.get("text") or "")
    next_text = str(after_info.get("text") or "")
    if prev_text and next_text and prev_text != next_text:
        return True
    prev_center = _center(before_info, before.shape)
    next_center = _center(after_info, after.shape)
    moved = False
    if prev_center is not None and next_center is not None:
        moved = abs(prev_center[0] - next_center[0]) > 0.06 or abs(prev_center[1] - next_center[1]) > 0.06
    if moved:
        return True
    diff = upper_diff(before, after)
    if prev_text and next_text and prev_text == next_text and diff < SCREEN_DIFF_THRESHOLD:
        return False
    return diff >= SCREEN_DIFF_THRESHOLD
