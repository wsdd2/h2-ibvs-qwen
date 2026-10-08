from __future__ import annotations

import numpy as np

from ibvs_pad_touch.types import CameraPoint, Intrinsics


def tip_to_button_delta(
    button: CameraPoint,
    tip: CameraPoint,
    K: Intrinsics,
) -> tuple[np.ndarray, str]:
    """Camera-frame vector that moves the closed fingertip onto the button.

    Chest camera is fixed, so the panel/button barely moves. Servo the tip.
    No hand-eye: this vector is what motion control should apply as Cartesian xyz.
    """
    if button.xyz_m is None:
        if button.z_m is None:
            return np.zeros(3), "button_depth_missing"
        button = button.with_depth(K, button.z_m)
    if tip.xyz_m is None:
        if tip.z_m is None:
            return np.zeros(3), "tip_depth_missing"
        tip = tip.with_depth(K, tip.z_m)
    delta = np.asarray(button.xyz_m, dtype=np.float64) - np.asarray(tip.xyz_m, dtype=np.float64)
    return delta, "ok"


def image_xy_delta(
    button: CameraPoint,
    tip: CameraPoint,
    K: Intrinsics,
    nominal_depth_m: float,
) -> np.ndarray:
    """Depth-free 2D IBVS approximation; forward motion is always zero."""
    depth = max(0.05, float(nominal_depth_m))
    return np.asarray(
        [
            (float(button.u) - float(tip.u)) * depth / float(K.fx),
            (float(button.v) - float(tip.v)) * depth / float(K.fy),
            0.0,
        ],
        dtype=np.float64,
    )


def gate_xy_then_depth(
    dx: float,
    dy: float,
    dz: float,
    xy_err: float,
    xy_tol_m: float,
) -> tuple[float, float, float, str]:
    """Align the switch in-plane first, then go in to the panel — not the wall.

    Camera frame: X right, Y down, Z forward (depth). Do not mix axes.
    """
    if float(xy_err) > float(xy_tol_m):
        return float(dx), float(dy), 0.0, "ALIGN_XY"
    return 0.0, 0.0, float(dz), "APPROACH_Z"


def clip_delta(delta: np.ndarray, max_step: float, gain: float = 1.0) -> np.ndarray:
    step = np.asarray(delta, dtype=np.float64) * float(gain)
    norm = float(np.linalg.norm(step))
    if norm > max_step > 0:
        step *= max_step / norm
    return step
