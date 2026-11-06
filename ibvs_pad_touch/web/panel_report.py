# -*- coding: utf-8 -*-
"""Turn a panel photo into a climate or event report.

RapidOCR supplies the lines. The page kind comes from the words on the
screen: 室内 plus a temperature or humidity is the status page, and rows such
as 开机 / 关机 are the event log. The home screen also shows a temperature,
but it has no 室内 label, so it stays an ordinary screen shot.
"""
from __future__ import annotations

import re
from typing import Any

EVENT_NAMES = (
    "本地停机",
    "非本地停机",
    "设备起动",
    "设备启动",
    "设备停止",
    "开机",
    "关机",
    "停机",
)
MENU_MARKS = {"数据记录", "运行时间", "统计", "数值", "部件"}
SKIP_NAMES = MENU_MARKS | {
    "返回",
    "信息",
    "操作",
    "配置",
    "系统",
    "事件记录",
    "室内",
    "室内温度",
    "室内温",
}
PAGE_NAMES = {"climate": "温湿度现状", "events": "事件记录", "screen": "屏幕"}


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


def clean_text(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("℃", "C").replace("°C", "C").replace("°", "")
    value = value.replace("*", "").replace("＊", "")
    return re.sub(r"\s+", "", value)


def hanzi_name(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", str(text or ""))


def _clock(hour: int, minute: int) -> str | None:
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def _date(month: int, day: int) -> str | None:
    if 1 <= month <= 12 and 1 <= day <= 31:
        return f"{month:02d}/{day:02d}"
    return None


def parse_date(text: str) -> str | None:
    source = re.sub(r"[^0-9/]", "", str(text or ""))
    if not source:
        return None
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})", source)
    if match:
        return _date(int(match.group(1)), int(match.group(2)))
    digits = re.sub(r"\D", "", source)
    if len(digits) == 4:
        return _date(int(digits[:2]), int(digits[2:]))
    if len(digits) == 3:
        return _date(int(digits[0]), int(digits[1:]))
    return None


def parse_clock(text: str) -> str | None:
    source = re.sub(r"[^0-9:.]", "", str(text or ""))
    if not source or "/" in source:
        return None
    match = re.fullmatch(r"(\d{1,2})[:.](\d{2})", source)
    if match:
        return _clock(int(match.group(1)), int(match.group(2)))
    digits = re.sub(r"\D", "", source)
    if len(digits) == 3:
        return _clock(int(digits[0]), int(digits[1:]))
    if len(digits) == 4:
        return _clock(int(digits[:2]), int(digits[2:]))
    if len(digits) == 5:
        first = _clock(int(digits[:2]), int(digits[2:4]))
        if first:
            return first
        return _clock(int(digits[1:3]), int(digits[3:]))
    return None


def match_event(text: str) -> str:
    name = hanzi_name(text)
    if not name or name in SKIP_NAMES:
        return ""
    if name in EVENT_NAMES:
        return name
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,6}", name):
        return ""
    distance = min(edit_distance(name, word) for word in EVENT_NAMES)
    if distance > 1:
        return ""
    candidates = [
        word for word in EVENT_NAMES if len(word) == len(name) and edit_distance(name, word) == distance
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(name) >= 4 and candidates:
        return candidates[0]
    return ""


def _box(item: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    raw = item.get("xywh")
    if not raw or len(raw) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return x, y, width, height, x + width / 2.0, y + height / 2.0


def _footer(item: dict[str, Any], image_height: float | None) -> bool:
    if "unit" in str(item.get("text") or "").lower():
        return True
    box = _box(item)
    if image_height and box is not None and box[5] > float(image_height) * 0.82:
        return True
    return False


def _measure(text: str, kind: str) -> float | None:
    cleaned = clean_text(text)
    if kind == "temp":
        match = re.search(r"(-?\d+(?:\.\d+)?)C\b", cleaned) or re.search(r"(-?\d+(?:\.\d+)?)C", cleaned)
    else:
        match = re.search(r"(\d+(?:\.\d+)?)%", cleaned)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_measure(value: float, raw: str) -> str:
    cleaned = clean_text(raw)
    match = re.search(r"\d+\.(\d+)", cleaned)
    if match:
        decimals = min(len(match.group(1)), 2)
        return f"{value:.{decimals}f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _items(items: list[dict[str, Any]], image_shape: tuple[int, ...] | None) -> list[dict[str, Any]]:
    height = None if not image_shape else float(image_shape[0])
    usable = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if _footer(item, height):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        usable.append(item)
    return usable


def _pick(candidates: list[dict[str, Any]], indoor_cy: float | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if indoor_cy is None:
        return max(candidates, key=lambda item: float((_box(item) or (0, 0, 0, 0, 0, 0))[3]))
    below = []
    for item in candidates:
        box = _box(item)
        cy = 0.0 if box is None else box[5]
        if cy >= indoor_cy - 8:
            below.append(item)
    pool = below or candidates

    def distance(item: dict[str, Any]) -> float:
        box = _box(item)
        cy = 0.0 if box is None else box[5]
        return abs(cy - indoor_cy)

    return min(pool, key=distance)


def _climate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    texts = {hanzi_name(item.get("text") or "") for item in items}
    if texts & MENU_MARKS:
        return None
    indoor = [item for item in items if hanzi_name(item.get("text") or "") == "室内"]
    if not indoor:
        return None
    indoor_box = _box(indoor[0])
    indoor_cy = None if indoor_box is None else indoor_box[5]
    temps = [item for item in items if _measure(str(item.get("text") or ""), "temp") is not None]
    hums = [item for item in items if _measure(str(item.get("text") or ""), "hum") is not None]
    if not temps and not hums:
        return None
    temp_item = _pick(temps, indoor_cy)
    hum_item = _pick(hums, indoor_cy)
    temp_value = None if temp_item is None else _measure(str(temp_item.get("text") or ""), "temp")
    hum_value = None if hum_item is None else _measure(str(hum_item.get("text") or ""), "hum")
    temp_text = "" if temp_value is None or temp_item is None else _format_measure(temp_value, str(temp_item.get("text") or ""))
    hum_text = "" if hum_value is None or hum_item is None else _format_measure(hum_value, str(hum_item.get("text") or ""))
    lines = [
        "温湿度现状",
        f"室内温度 {temp_text} ℃" if temp_text else "室内温度 没有读出",
        f"室内湿度 {hum_text} %" if hum_text else "室内湿度 没有读出",
    ]
    return {
        "page_kind": "climate",
        "report_text": "\n".join(lines),
        "report": {
            "temperature_c": temp_value,
            "humidity_pct": hum_value,
            "temperature_text": temp_text,
            "humidity_text": hum_text,
        },
    }


def _time_token(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "")
    if "C" in clean_text(text) or "%" in text:
        return False
    source = re.sub(r"[^0-9]", "", text)
    return len(source) >= 3


def _event_rows(items: list[dict[str, Any]], image_shape: tuple[int, ...] | None) -> list[dict[str, str]]:
    width = None if not image_shape or len(image_shape) < 2 else float(image_shape[1])
    names: list[dict[str, Any]] = []
    times: list[dict[str, Any]] = []
    for item in items:
        box = _box(item)
        cx = 0.0 if box is None else box[4]
        cy = 0.0 if box is None else box[5]
        name = match_event(str(item.get("text") or ""))
        if name and (width is None or box is None or cx < width * 0.58):
            height = 40.0 if box is None else max(8.0, box[3])
            names.append({"name": name, "cx": cx, "cy": cy, "h": height, "item": item})
            continue
        if _time_token(item):
            times.append({"text": str(item.get("text") or ""), "cx": cx, "cy": cy})
    rows: list[dict[str, str]] = []
    used: set[int] = set()
    for name in sorted(names, key=lambda item: item["cy"]):
        tol = max(24.0, min(float(name["h"]), 80.0) * 0.55)
        near = []
        for index, token in enumerate(times):
            if index in used:
                continue
            if token["cx"] + 8 < float(name["cx"]):
                continue
            if abs(float(token["cy"]) - float(name["cy"])) <= tol:
                near.append((index, token))
        if not near:
            rows.append({"name": str(name["name"]), "when": "时间未识别", "date": "", "time": ""})
            continue
        ordered = sorted(near, key=lambda pair: float(pair[1]["cx"]))
        for index, _token in ordered:
            used.add(index)
        left = ordered[0][1]
        right = ordered[-1][1] if len(ordered) > 1 else None
        left_kind, left_val = _left_value(str(left["text"]))
        right_kind, right_val = ("", "") if right is None else _right_value(str(right["text"]))
        date = left_val if left_kind == "date" else (right_val if right_kind == "date" else "")
        clock = right_val if right_kind == "clock" else (left_val if left_kind == "clock" else "")
        raws = []
        if left_kind == "raw":
            raws.append(left_val)
        if right_kind == "raw" and right_val not in raws:
            raws.append(right_val)
        if date and clock:
            when = f"{date} {clock}"
        elif date and raws:
            when = f"{date} {' '.join(raws)}"
        elif clock:
            when = clock
        elif raws:
            when = f"时间未识别（{' '.join(raws)}）"
        else:
            when = "时间未识别"
        rows.append({"name": str(name["name"]), "when": when, "date": date, "time": clock})
    return rows


def _left_value(text: str) -> tuple[str, str]:
    date = parse_date(text)
    if date:
        return "date", date
    clock = parse_clock(text)
    if clock:
        return "clock", clock
    return "raw", str(text or "").strip()


def _right_value(text: str) -> tuple[str, str]:
    clock = parse_clock(text)
    if clock:
        return "clock", clock
    date = parse_date(text)
    if date:
        return "date", date
    return "raw", str(text or "").strip()


def _events(items: list[dict[str, Any]], image_shape: tuple[int, ...] | None) -> dict[str, Any] | None:
    texts = {hanzi_name(item.get("text") or "") for item in items}
    rows = _event_rows(items, image_shape)
    title = "事件记录" in texts
    if not rows and (not title or texts & MENU_MARKS):
        return None
    lines = ["事件记录"]
    if not rows:
        lines.append("已到事件记录页面，本页没有读出事件行。")
    else:
        for index, row in enumerate(rows, 1):
            lines.append(f"{index}. {row['name']}  {row['when']}")
    return {
        "page_kind": "events",
        "report_text": "\n".join(lines),
        "report": {"rows": rows},
    }


def build_report(items: list[dict[str, Any]] | None, image_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    usable = _items(list(items or []), image_shape)
    ocr_text = "\n".join(str(item.get("text") or "").strip() for item in usable if str(item.get("text") or "").strip())
    chosen = _climate(usable) or _events(usable, image_shape)
    if chosen is None:
        chosen = {"page_kind": "screen", "report_text": "", "report": {}}
    kind = str(chosen["page_kind"])
    return {
        "page_kind": kind,
        "page_name": PAGE_NAMES.get(kind, "屏幕"),
        "report_text": str(chosen.get("report_text") or ""),
        "report": dict(chosen.get("report") or {}),
        "ocr_text": ocr_text,
    }


def _xywh(box: Any) -> list[float] | None:
    try:
        points = list(box)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _items_from_outcome(outcome: Any) -> list[dict[str, Any]]:
    if outcome is None:
        return []
    result = outcome[0] if isinstance(outcome, tuple) else outcome
    if result is None:
        return []
    if hasattr(result, "txts"):
        boxes = list(getattr(result, "boxes", None) or [])
        texts = list(result.txts or [])
        scores = list(getattr(result, "scores", None) or [])
        found = []
        for index, text in enumerate(texts):
            box = boxes[index] if index < len(boxes) else None
            score = float(scores[index]) if index < len(scores) else 0.0
            found.append({"text": str(text), "score": score, "xywh": None if box is None else _xywh(box)})
        return found
    if not isinstance(result, (list, tuple)):
        return []
    found = []
    for item in result:
        if not item or len(item) < 2:
            continue
        score = 0.0
        if len(item) > 2:
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                score = 0.0
        found.append({"text": str(item[1]), "score": score, "xywh": _xywh(item[0])})
    return found


def ocr_items(image: Any) -> list[dict[str, Any]] | None:
    """Full-page RapidOCR. None means the package is not available."""
    if image is None or getattr(image, "size", 0) == 0:
        return []
    from ibvs_pad_touch.web.screen_read import run_ocr

    try:
        outcome = run_ocr(image)
    except Exception:
        return []
    if outcome is None:
        return None
    return _items_from_outcome(outcome)
