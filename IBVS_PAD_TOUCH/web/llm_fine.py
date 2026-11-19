"""Remote LLM for Hybrid wrist fine-tuning after chest PBVS."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Sequence

DEFAULT_URL = os.environ.get("H2_LLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("H2_LLM_MODEL", "Qwen3.8-27B-FP8")
# Connection probe only. Model inference uses a separate, longer timeout.
CONNECT_TIMEOUT_S = 60.0
INFER_TIMEOUT_S = 120.0
MAX_STEP_M = 0.015
# Fingertip is still short of the panel. Do not accept a press inside this gap.
X_PRESS_GAP_M = 0.008
# Inside this Y/Z band the fingertip is lined up. Further lateral steps only orbit.
YZ_FOCUS_M = 0.012
# Kinematic X can already read "arrived" while the real tip is still short.
X_COMMIT_M = 0.015
X_CREEP_FLOOR_M = -0.004
# Wrist or chest pixels still show the fingertip off the button.
PIXEL_MISS_PX = 40.0
# Finger and the target button count as aligned inside this pixel radius.
FINGER_ON_BUTTON_PX = 25.0
PANEL_RULES = (
    "只找三个按键。对齐的唯一标准是视觉检测里的 Finger 和目标 Button 重合。\n"
    "检测不到 Button 时，视野里最多两条黄条："
    "上面黄条左半边是 Button1，右半边是 Button2，下面那条黄条是 Button3。\n"
    "如果是一条黄条加一条红条，手指偏低，已经到了下面的区域，要向上，不要往前戳。\n"
)


class LlmUnavailable(RuntimeError):
    pass


def resolve_url(url: str | None = None) -> str:
    return str(url or DEFAULT_URL).rstrip("/")


def check_llm_connection(
    url: str | None = None,
    *,
    timeout_s: float = CONNECT_TIMEOUT_S,
) -> tuple[bool, str]:
    """Probe the remote OpenAI-compatible endpoint.

    ``timeout_s`` bounds only the TCP/HTTP connection attempt. It does not
    cover later tip-delta inference time.
    """
    base = resolve_url(url)
    deadline = max(0.5, float(timeout_s))
    request = urllib.request.Request(base + "/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=deadline) as response:
            body = response.read(65536)
            if int(response.status) >= 400:
                return False, f"远程 LLM 拒绝连接：HTTP {response.status}"
    except urllib.error.HTTPError as error:
        # Some gateways answer /models with 401/404 while still reachable.
        if int(error.code) in {401, 403, 404}:
            return True, f"远程 LLM 已连通（{base}，HTTP {error.code}）"
        return False, f"远程 LLM 拒绝连接：HTTP {error.code}"
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        return False, f"远程 LLM 连接失败（{deadline:.0f}s 超时内）：{reason}"
    except TimeoutError:
        return False, f"远程 LLM 连接超时（{deadline:.0f}s），未计入推理时间"
    except OSError as error:
        return False, f"远程 LLM 连接失败：{error}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return True, f"远程 LLM 已连通（{base}）"
    models = payload.get("data") if isinstance(payload, dict) else None
    count = len(models) if isinstance(models, list) else 0
    if count:
        return True, f"远程 LLM 已连通（{base}，{count} 个模型）"
    return True, f"远程 LLM 已连通（{base}）"


def clip_delta(delta: Sequence[float], limit_m: float = MAX_STEP_M) -> list[float]:
    if len(delta) != 3:
        raise ValueError("delta_xyz 需要 3 个数")
    out: list[float] = []
    for value in delta:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("delta_xyz 必须是有限数值")
        out.append(max(-limit_m, min(limit_m, number)))
    return out


def _flatten_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in value
        )
    return str(value)


def message_text(message: dict[str, Any]) -> str:
    """Visible answer plus the thinking trace. Qwen often leaves content empty."""
    parts = [
        _flatten_content(message.get("content")),
        _flatten_content(message.get("reasoning_content") or message.get("reasoning")),
    ]
    text = "\n".join(part for part in parts if part).strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        raise ValueError("LLM 没有返回内容")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if not match:
        preview = raw.replace("\n", " ")[:80]
        raise ValueError(f"LLM 返回里没有 JSON 对象：{preview}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON 不是对象")
    return payload


def align_delta(
    delta: Sequence[float],
    error_m: Sequence[float] | None,
    *,
    limit_m: float = MAX_STEP_M,
    z_hold_m: float = 0.004,
) -> list[float]:
    """Keep a step from increasing the measured button-minus-tip error.

    Positive error means the fingertip has not reached the button on that axis.
    A negative Z error means the fingertip is already above the button, so an
    upward command is replaced by a step back down.
    """
    clipped = clip_delta(delta, limit_m=limit_m)
    if error_m is None or len(error_m) != 3:
        return clipped
    aligned: list[float] = []
    for index, (step, gap) in enumerate(zip(clipped, error_m)):
        gap_m = float(gap)
        if not math.isfinite(gap_m):
            aligned.append(step)
            continue
        if index == 2 and abs(gap_m) <= z_hold_m:
            aligned.append(0.0)
            continue
        if abs(gap_m) <= 1e-4:
            aligned.append(0.0)
            continue
        if step * gap_m < 0.0:
            step = math.copysign(min(limit_m, abs(gap_m)), gap_m)
        else:
            step = math.copysign(min(abs(step), limit_m, abs(gap_m)), gap_m)
        aligned.append(step)
    return aligned


def keep_pressing(
    delta: Sequence[float],
    error_m: Sequence[float] | None,
    done: bool,
    *,
    limit_m: float = MAX_STEP_M,
    x_gap_m: float = X_PRESS_GAP_M,
) -> tuple[list[float], bool]:
    """Refuse a finished press while the fingertip is still short in X."""
    step = align_delta(delta, error_m, limit_m=limit_m)
    if error_m is None or len(error_m) != 3:
        return step, bool(done)
    try:
        gap_x = float(error_m[0])
    except (TypeError, ValueError):
        return step, bool(done)
    if not math.isfinite(gap_x) or gap_x <= x_gap_m:
        return step, bool(done)
    step[0] = min(limit_m, gap_x)
    return step, False


def pixel_gap_px(button_uv: Sequence[float] | None, tip_uv: Sequence[float] | None) -> float | None:
    if button_uv is None or tip_uv is None or len(button_uv) < 2 or len(tip_uv) < 2:
        return None
    try:
        du = float(button_uv[0]) - float(tip_uv[0])
        dv = float(button_uv[1]) - float(tip_uv[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(du) or not math.isfinite(dv):
        return None
    return math.hypot(du, dv)


def panel_scene(
    *,
    target: str,
    finger_uv: Sequence[float] | None,
    button_uv: Sequence[float] | None,
    yellow_bars: Sequence[dict[str, float]] | None = None,
    red_bar: bool = False,
) -> dict[str, Any]:
    """Map the wrist view onto Button1/2/3. Image v grows downward."""
    bars = sorted(
        [bar for bar in (yellow_bars or []) if "cy" in bar and "left" in bar and "right" in bar],
        key=lambda bar: float(bar["cy"]),
    )
    finger_low = bool(red_bar) and len(bars) >= 1
    aim = _uv_pair(button_uv)
    layout = ""
    if aim is None and bars and not finger_low:
        upper = bars[0]
        lower = bars[-1] if len(bars) >= 2 else None
        width = float(upper["right"]) - float(upper["left"])
        if target == "button_3" and lower is not None:
            aim = (float(lower.get("cx", (float(lower["left"]) + float(lower["right"])) / 2.0)), float(lower["cy"]))
            layout = "没有 Button 点，用下面黄条作为 Button3"
        elif target == "button_2":
            aim = (float(upper["right"]) - 0.25 * width, float(upper["cy"]))
            layout = "没有 Button 点，用上面黄条右半边作为 Button2"
        else:
            aim = (float(upper["left"]) + 0.25 * width, float(upper["cy"]))
            layout = "没有 Button 点，用上面黄条左半边作为 Button1"
    gap = pixel_gap_px(aim, finger_uv)
    on_button = gap is not None and gap <= FINGER_ON_BUTTON_PX and not finger_low
    if finger_low:
        text = "一条黄条加一条红条：手指偏低，先向上离开下面的区域。"
    elif on_button:
        text = f"Finger 与 {target} 已重合，像素差 {gap:.0f}。可以沿 X 向前 15 mm。"
    elif gap is not None:
        text = f"Finger 还没对上 {target}，像素差 {gap:.0f}。先让两者重合，不要结束。"
    else:
        text = "还没同时看到 Finger 和目标位置。"
    if layout:
        text = layout + "。" + text
    return {
        "finger_on_button": on_button,
        "finger_low": finger_low,
        "aim_uv": None if aim is None else [float(aim[0]), float(aim[1])],
        "pixel_gap_px": gap,
        "scene": text,
    }


def image_still_off(vision: dict[str, Any] | None, *, limit_px: float = PIXEL_MISS_PX) -> bool:
    if not vision:
        return False
    gap = vision.get("pixel_gap_px")
    try:
        return gap is not None and float(gap) > limit_px
    except (TypeError, ValueError):
        return False


def _uv_pair(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None or len(value) < 2:
        return None
    try:
        u, v = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return u, v


def focus_press(
    delta: Sequence[float],
    error_m: Sequence[float] | None,
    done: bool,
    *,
    allow_creep: bool = False,
    image_open: bool = False,
    finger_on_button: bool = False,
    finger_low: bool = False,
    limit_m: float = MAX_STEP_M,
) -> tuple[list[float], bool]:
    """Press +X only after the finger visually covers the target button.

    A red bar under the yellow keys means the finger has dropped too low.
    """
    if finger_low:
        return [0.0, 0.0, min(limit_m, 0.012)], False
    if finger_on_button:
        return [min(limit_m, X_COMMIT_M), 0.0, 0.0], False
    _ = (allow_creep, error_m, done, image_open)
    return clip_delta(delta, limit_m=limit_m), False


def _fmt_xyz(value: Sequence[float] | None, scale: float = 1000.0) -> str:
    if value is None or len(value) != 3:
        return "无"
    return " ".join(f"{axis} {float(value[index]) * scale:+.1f}" for index, axis in enumerate("XYZ"))


def _fmt_uv(value: Sequence[float] | None) -> str:
    if value is None or len(value) < 2:
        return "无"
    return f"u={float(value[0]):.0f} v={float(value[1]):.0f}"


def vision_prompt(vision: dict[str, Any] | None) -> str:
    if not vision:
        return ""
    error = vision.get("error_m")
    height = "无"
    if isinstance(error, (list, tuple)) and len(error) == 3:
        try:
            gap = float(error[2]) * 1000.0
        except (TypeError, ValueError):
            gap = float("nan")
        if math.isfinite(gap) and gap > 4.0:
            height = f"指尖低于按键 {gap:.1f} mm，只允许向上，且不超过这个差值"
        elif math.isfinite(gap) and gap < -4.0:
            height = f"指尖已经高于按键 {abs(gap):.1f} mm，禁止向上，必须向下靠近"
        elif math.isfinite(gap):
            height = "高度已在 4 mm 内，Z 必须为 0"
    forward = ""
    if isinstance(error, (list, tuple)) and len(error) == 3:
        try:
            gap_x = float(error[0]) * 1000.0
        except (TypeError, ValueError):
            gap_x = float("nan")
        if math.isfinite(gap_x) and gap_x > X_PRESS_GAP_M * 1000.0:
            forward = (
                f"前方还差 {gap_x:.1f} mm，指尖还没碰到面板。"
                "done 必须是 false，X 必须为正。"
            )
    lateral = ""
    if isinstance(error, (list, tuple)) and len(error) == 3:
        try:
            gap_y = abs(float(error[1]))
            gap_z = abs(float(error[2]))
        except (TypeError, ValueError):
            gap_y = gap_z = float("nan")
        if image_still_off(vision):
            lateral = (
                f"画面里指尖离按键还有 {float(vision.get('pixel_gap_px')):.0f} 像素。"
                "关节坐标这时候不可信。按画面把指尖移到按键上，done 必须是 false。"
            )
        elif math.isfinite(gap_y) and math.isfinite(gap_z) and gap_y <= YZ_FOCUS_M and gap_z <= YZ_FOCUS_M:
            lateral = "左右和高度已经对齐。这一步 X 固定向前 15 mm，Y 和 Z 为 0。"
    scene = str(vision.get("scene") or "")
    return (
        PANEL_RULES
        + (scene + "\n" if scene else "")
        + "下面是测量值。Finger 没对上 Button 时，以画面为准。\n"
        f"PBVS 落点 mm：{_fmt_xyz(vision.get('pbvs_xyz_m'))}\n"
        f"胸口当前按键 mm：{_fmt_xyz(vision.get('chest_xyz_m'))}\n"
        f"当前指尖 mm：{_fmt_xyz(vision.get('tip_xyz_m'))}\n"
        f"误差（按键减指尖）mm：{_fmt_xyz(vision.get('error_m'))}\n"
        f"高度：{height}\n"
        f"前方：{forward or 'X 已进入 8 mm'}\n"
        f"{lateral}\n"
        f"胸口按键像素：{_fmt_uv(vision.get('chest_button_uv'))}，胸口指尖像素：{_fmt_uv(vision.get('chest_tip_uv'))}\n"
        f"右腕按键像素：{_fmt_uv(vision.get('wrist_button_uv'))}，右腕指尖像素：{_fmt_uv(vision.get('wrist_tip_uv'))}\n"
        f"右腕识别：{vision.get('wrist_message') or '无'}\n"
        "图像 v 越小越靠上。误差 Z 为负时指尖已经偏高。不要把偏高判断成偏低。\n"
    )


def parse_tip_command(payload: dict[str, Any], *, limit_m: float = MAX_STEP_M) -> dict[str, Any]:
    done = bool(payload.get("done"))
    delta_raw = payload.get("delta_xyz", [0.0, 0.0, 0.0])
    if not isinstance(delta_raw, (list, tuple)) or len(delta_raw) != 3:
        raise ValueError("delta_xyz 需要 3 个数")
    delta = clip_delta(delta_raw, limit_m=limit_m)
    return {"delta_xyz": delta, "done": done, "reason": str(payload.get("reason") or "")}


def request_tip_delta(
    jpeg: bytes,
    *,
    tip_xyz: Sequence[float] | None,
    button: str,
    depth_ok: bool,
    vision: dict[str, Any] | None = None,
    url: str | None = None,
    model: str | None = None,
    timeout_s: float = INFER_TIMEOUT_S,
) -> dict[str, Any]:
    """Ask the remote model for one torso-frame fingertip nudge.

    Inference timeout is independent from :func:`check_llm_connection`.
    """
    if not jpeg:
        raise LlmUnavailable("没有右腕 JPEG，不能请求 LLM 微调")
    base = resolve_url(url)
    chosen = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    tip = [float(value) for value in (tip_xyz or (0.0, 0.0, 0.0))]
    depth_note = "有效" if depth_ok else "已丢失（近距）"
    prompt = (
        "你在控制宇树 H2 右食指按空调面板。胸口 PBVS 粗定位已完成，现在只做末端微调。\n"
        f"目标按键：{button}\n"
        f"当前指尖躯干坐标米：{tip}\n"
        f"右腕深度：{depth_note}\n"
        "坐标系：躯干系 +X 向前朝面板，+Y 向左，+Z 向上。\n"
        + vision_prompt(vision)
        + "只返回一个 JSON 对象，不要其它文字："
        '{"delta_xyz":[dx,dy,dz],"done":false,"reason":"..."}\n'
        f"delta_xyz 单位米，每轴绝对值不超过 {MAX_STEP_M}。"
        "未对准时 done=false；认为已经贴上按键时 done=true 且 delta 接近 0。"
    )
    encoded = base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": chosen,
        "temperature": 0.1,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LlmUnavailable(f"LLM 推理失败：HTTP {error.code} {detail}") from error
    except urllib.error.URLError as error:
        raise LlmUnavailable(f"LLM 推理连接失败：{getattr(error, 'reason', error)}") from error
    except TimeoutError as error:
        raise LlmUnavailable("LLM 推理超时") from error
    try:
        reply = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LlmUnavailable("LLM 返回不是 JSON") from error
    try:
        message = reply["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LlmUnavailable("LLM 返回缺少 choices[].message") from error
    if not isinstance(message, dict):
        raise LlmUnavailable("LLM 返回的 message 不是对象")
    command = parse_tip_command(_extract_json(message_text(message)))
    error = None if not vision else vision.get("error_m")
    delta, done = keep_pressing(command["delta_xyz"], error, command["done"])
    command["delta_xyz"] = delta
    command["done"] = done
    return command
