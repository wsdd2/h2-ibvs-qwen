# -*- coding: utf-8 -*-
"""Chest image IBVS plus a left-arm side view for contact.

The chest camera is fixed (eye-to-hand). It servos the real fingertip onto the
button in the image. That corrects left/right and up/down without a Cartesian
target, so bad joint readings cannot shift the goal.

The left-arm camera stays put and looks across at the right hand. Its pixel
gap is the visual error for the axis the chest image cannot see: forward and
back along the chest optical axis. Chest pixels still servo left/right and
up/down. Contact stops the arm. The old right-wrist hand-eye solution is not used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ibvs_pad_touch.servo.delta import clip_delta, image_xy_delta
from ibvs_pad_touch.types import CameraPoint, Intrinsics


@dataclass(frozen=True)
class DualViewCommand:
    dx: float
    dy: float
    dz: float
    phase: str
    aligned: bool
    contact: bool
    message: str


def servo_depth_m(
    measured: float | None,
    fallback: float,
    low: float = 0.08,
    high: float = 1.20,
) -> float:
    """Use a measured range when it is finite. Otherwise keep the nominal depth."""
    if measured is None:
        return float(fallback)
    try:
        value = float(measured)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(value):
        return float(fallback)
    return float(min(float(high), max(float(low), value)))


def side_pixels_to_z(
    gap_px: float,
    K: Intrinsics,
    *,
    setpoint_px: float,
    nominal_depth_m: float,
    gain: float,
    max_step_m: float,
) -> float:
    """Turn a side-view pixel gap into a chest-camera Z step.

    Positive means the jaw is still outside the setpoint, so the tip moves
    forward along the chest optical axis. The side camera is not mapped into
    chest X/Y; those axes belong to the chest image.
    """
    err_px = float(gap_px) - float(setpoint_px)
    depth = max(0.05, float(nominal_depth_m))
    raw = err_px * depth / max(float(K.fx), 1.0)
    return float(np.clip(raw * float(gain), -abs(float(max_step_m)), abs(float(max_step_m))))


def compose_dual_view_ibvs(
    button_uv: tuple[float, float] | None,
    tip_uv: tuple[float, float] | None,
    K: Intrinsics,
    *,
    side_measured: bool = False,
    side_contact: bool = False,
    side_gap_px: float | None = None,
    pixel_tol: float = 8.0,
    nominal_depth_m: float = 0.60,
    side_depth_m: float | None = None,
    gain: float = 0.35,
    max_step_m: float = 0.006,
    approach_step_m: float = 0.004,
    standoff_px: float = 36.0,
    contact_px: float = 14.0,
    side_blocked: bool = False,
) -> DualViewCommand:
    """Chest pixels drive camera X/Y. Side pixels drive camera Z.

    ``nominal_depth_m`` scales the chest image. ``side_depth_m`` is the left
    camera's range to the jaw and scales only the forward step.
    """
    side_range = float(nominal_depth_m if side_depth_m is None else side_depth_m)
    if tip_uv is None:
        return DualViewCommand(
            0.0,
            0.0,
            0.0,
            "SEEK_GRIPPER",
            False,
            False,
            "看不到夹爪，右臂保持，左臂找角度",
        )
    if side_blocked:
        dx, dy = 0.0, 0.0
        if button_uv is not None and tip_uv is not None:
            button = CameraPoint(u=float(button_uv[0]), v=float(button_uv[1]))
            tip = CameraPoint(u=float(tip_uv[0]), v=float(tip_uv[1]))
            raw = image_xy_delta(button, tip, K, nominal_depth_m)
            step = clip_delta(raw, max_step_m, gain=gain)
            dx, dy = float(step[0]), float(step[1])
        return DualViewCommand(
            dx,
            dy,
            -float(approach_step_m),
            "BACKOFF",
            False,
            False,
            "侧视未看到面板，先退开",
        )
    if button_uv is None:
        return DualViewCommand(0.0, 0.0, 0.0, "SEEK_PANEL", False, False, "等待胸视看到面板")
    button = CameraPoint(u=float(button_uv[0]), v=float(button_uv[1]))
    tip = CameraPoint(u=float(tip_uv[0]), v=float(tip_uv[1]))
    raw = image_xy_delta(button, tip, K, nominal_depth_m)
    step = clip_delta(raw, max_step_m, gain=gain)
    px_err = float(np.hypot(button.u - tip.u, button.v - tip.v))
    aligned = px_err <= float(pixel_tol)
    if not aligned:
        too_close = bool(side_contact) or (
            side_measured and side_gap_px is not None and float(side_gap_px) < float(standoff_px)
        )
        if too_close and side_gap_px is not None:
            dz = side_pixels_to_z(
                float(side_gap_px),
                K,
                setpoint_px=float(standoff_px),
                nominal_depth_m=side_range,
                gain=gain,
                max_step_m=max_step_m,
            )
            message = "侧视过近，先退开一小段"
        elif too_close:
            dz = -float(approach_step_m)
            message = "侧视过近，先退开一小段"
        elif side_measured and side_gap_px is not None:
            dz = side_pixels_to_z(
                float(side_gap_px),
                K,
                setpoint_px=float(contact_px),
                nominal_depth_m=side_range,
                gain=gain,
                max_step_m=max_step_m,
            )
            message = "胸视对准左右/上下，侧视间隙同时前进"
        else:
            dz = 0.0
            message = "胸视 IBVS：对准左右/上下"
        return DualViewCommand(
            float(step[0]),
            float(step[1]),
            dz,
            "ALIGN_XY",
            False,
            False,
            message,
        )
    if side_contact:
        return DualViewCommand(0.0, 0.0, 0.0, "CONTACT", True, True, "左手侧视：已接触按键")
    if not side_measured or side_gap_px is None:
        return DualViewCommand(
            0.0,
            0.0,
            0.0,
            "NEED_SIDE",
            True,
            False,
            "胸视已对准，等待左手侧视间隙",
        )
    dz = side_pixels_to_z(
        float(side_gap_px),
        K,
        setpoint_px=float(contact_px),
        nominal_depth_m=side_range,
        gain=gain,
        max_step_m=max_step_m,
    )
    return DualViewCommand(
        0.0,
        0.0,
        dz,
        "APPROACH",
        True,
        False,
        f"侧视 IBVS 间隙 {float(side_gap_px):.0f}px，沿胸视光轴",
    )
