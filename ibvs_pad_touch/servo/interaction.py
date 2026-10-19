from __future__ import annotations

import numpy as np


def point_interaction_matrix(x: float, y: float, z_m: float) -> np.ndarray:
    """3x3 L mapping camera twist [vx, vy, vz] to [x_dot, y_dot, Z_dot]."""
    z = float(z_m)
    return np.array(
        [
            [-1.0 / z, 0.0, x / z],
            [0.0, -1.0 / z, y / z],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def camera_velocity(
    x: float,
    y: float,
    z_m: float,
    x_star: float,
    y_star: float,
    z_star: float,
    *,
    lambda_xy: float,
    lambda_z: float,
    damping: float = 1e-6,
) -> np.ndarray:
    error = np.array([x - x_star, y - y_star, z_m - z_star], dtype=np.float64)
    gain = np.diag([lambda_xy, lambda_xy, lambda_z])
    L = point_interaction_matrix(x, y, z_m)
    ltl = L.T @ L + float(damping) * np.eye(3)
    velocity = -np.linalg.solve(ltl, L.T @ (gain @ error))
    return velocity


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > float(limit) > 0.0:
        return vector * (float(limit) / norm)
    return vector


def eye_in_hand_ibvs_step(
    button_uv: tuple[float, float],
    tip_uv: tuple[float, float],
    button_z_m: float,
    tip_z_m: float | None,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    lambda_xy: float = 0.8,
    lambda_z: float = 0.6,
    dt_s: float = 0.25,
    max_step_m: float = 0.004,
    pixel_tol: float = 8.0,
    depth_tol_m: float = 0.004,
    aligned_approach_m: float = 0.003,
) -> dict[str, object]:
    """Drive the wrist-camera button feature onto the gripper tip.

    s is the button's normalized image point plus its depth. s* is the tip
    pixel and, when the depth image sees the jaw, the tip depth. The law is
    v = -λ L⁺ (s - s*) for camera translation. A missing tip depth still
    creeps forward once the pixels already coincide.
    """
    z = float(button_z_m)
    if not np.isfinite(z) or z < 0.05 or z > 0.80:
        raise ValueError(f"button depth {z} is outside the IBVS range")
    x = (float(button_uv[0]) - float(cx)) / float(fx)
    y = (float(button_uv[1]) - float(cy)) / float(fy)
    x_star = (float(tip_uv[0]) - float(cx)) / float(fx)
    y_star = (float(tip_uv[1]) - float(cy)) / float(fy)
    has_tip_z = tip_z_m is not None and np.isfinite(float(tip_z_m)) and float(tip_z_m) > 0.02
    z_star = float(tip_z_m) if has_tip_z else z
    velocity = camera_velocity(
        x,
        y,
        z,
        x_star,
        y_star,
        z_star,
        lambda_xy=float(lambda_xy),
        lambda_z=float(lambda_z),
    )
    step = _clip_norm(velocity * float(dt_s), float(max_step_m))
    pixel_err = float(np.hypot(float(button_uv[0]) - float(tip_uv[0]), float(button_uv[1]) - float(tip_uv[1])))
    depth_err = float(z - float(tip_z_m)) if has_tip_z else None
    if not has_tip_z and pixel_err <= float(pixel_tol):
        step = _clip_norm(step + np.array([0.0, 0.0, float(aligned_approach_m)]), float(max_step_m))
    arrived = bool(
        pixel_err <= float(pixel_tol)
        and has_tip_z
        and depth_err is not None
        and abs(depth_err) <= float(depth_tol_m)
    )
    return {
        "camera_step_m": (float(step[0]), float(step[1]), float(step[2])),
        "pixel_err": pixel_err,
        "depth_err_m": depth_err,
        "arrived": arrived,
    }


def eye_in_hand_close_align_step(
    button_uv: tuple[float, float],
    tip_uv: tuple[float, float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    nominal_z_m: float = 0.04,
    lambda_xy: float = 0.8,
    dt_s: float = 0.25,
    max_step_m: float = 0.004,
) -> dict[str, object]:
    """Slide the camera in X/Y until the button pixel meets the gripper tip.

    The wrist depth is empty inside about 5 cm, so this step must not move
    along the optical axis. ``nominal_z_m`` only scales the lateral gain.
    """
    z = float(nominal_z_m)
    if not np.isfinite(z) or z <= 0.005 or z > 0.05:
        raise ValueError(f"close-range nominal depth {z} is outside 5 cm")
    x = (float(button_uv[0]) - float(cx)) / float(fx)
    y = (float(button_uv[1]) - float(cy)) / float(fy)
    x_star = (float(tip_uv[0]) - float(cx)) / float(fx)
    y_star = (float(tip_uv[1]) - float(cy)) / float(fy)
    velocity = np.array(
        [
            float(lambda_xy) * z * (x - x_star),
            float(lambda_xy) * z * (y - y_star),
            0.0,
        ],
        dtype=np.float64,
    )
    step = _clip_norm(velocity * float(dt_s), float(max_step_m))
    step[2] = 0.0
    pixel_err = float(
        np.hypot(float(button_uv[0]) - float(tip_uv[0]), float(button_uv[1]) - float(tip_uv[1]))
    )
    return {
        "camera_step_m": (float(step[0]), float(step[1]), float(step[2])),
        "pixel_err": pixel_err,
    }
