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
