# -*- coding: utf-8 -*-
"""Right index fingertip from h2_with_revo2_fixed.urdf.

The chain is fixed in that file, with the index already straight. The poke
point is right_index_tip_link, expressed in right_wrist_yaw_link. The pad
frame remains available for checks. The wrist orientation is the reachable
press pose, which aims this fingertip toward the panel.
"""

from __future__ import annotations

import math

import numpy as np

# right_wrist_yaw_link -> right_index_pad_ee_link, all fixed joints.
_CHAIN_XYZ_RPY = (
    ((0.05824125, 0.0, -0.00017221), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.023, 0.0, 0.0), (0.0, 1.5707963267948966, 0.0)),
    ((-0.0021181, 0.029568, 0.080876), (-0.11819, 0.14743, -0.17422)),
    ((0.0, 0.0, 0.032), (0.0, 0.0, 0.0)),
    ((0.014497, 0.0, 0.023303), (1.7145, 0.4756, 0.0)),
    ((-0.0004169476, 0.0030809231, -0.0004458390), (1.5769375186, 0.1314468814, 1.8511640697)),
)
# right_index_tip_joint, used only to keep the pad behind the tip.
_TIP_XYZ_RPY = ((0.011097, 0.0, 0.035739), (0.0, 0.4756, 0.0))


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _frame(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy(*rpy)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def _pad_frame_in_wrist() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for xyz, rpy in _CHAIN_XYZ_RPY:
        transform = transform @ _frame(xyz, rpy)
    return transform


def _tip_in_wrist() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for xyz, rpy in _CHAIN_XYZ_RPY[:5]:
        transform = transform @ _frame(xyz, rpy)
    return (transform @ _frame(*_TIP_XYZ_RPY))[:3, 3]


def index_pad_in_wrist() -> tuple[float, float, float]:
    """Pad origin in right_wrist_yaw_link."""
    point = _pad_frame_in_wrist()[:3, 3]
    return (float(point[0]), float(point[1]), float(point[2]))


def index_tip_in_wrist() -> tuple[float, float, float]:
    """Fingertip origin in right_wrist_yaw_link. This is the Revo2 poke point."""
    point = _tip_in_wrist()
    return (float(point[0]), float(point[1]), float(point[2]))


def index_tip_tool_rpy_deg() -> tuple[float, float, float]:
    """Wrist pose that aims the fingertip toward the panel."""
    return REACHABLE_PRESS_RPY_DEG


def index_pad_axes_in_wrist() -> tuple[np.ndarray, np.ndarray]:
    """Outward pad normal and finger tangent, both in right_wrist_yaw_link."""
    rotation = _pad_frame_in_wrist()[:3, :3]
    palm = rotation[:, 2].copy()
    finger = rotation[:, 0].copy()
    return palm, finger


# The Dex1 press orientation. It reaches this panel; the fully parallel pad pose does not.
REACHABLE_PRESS_RPY_DEG = (-88.8133, -3.98969, 3.20691)
# Turn from that pose toward the pad, without requiring the palm to lie flat on the panel.
PAD_TILT_FROM_REACHABLE_DEG = 45.0


def _rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(rotation[2, 0]) < 0.999999:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _parallel_pad_rotation() -> np.ndarray:
    """Wrist orientation whose pad normal is torso +X and whose finger tangent is torso +Z."""
    palm, finger = index_pad_axes_in_wrist()
    side = np.cross(palm, finger)
    wrist_axes = np.column_stack([palm, finger, side])
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    upward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    torso_axes = np.column_stack([forward, upward, np.cross(forward, upward)])
    return torso_axes @ wrist_axes.T


def _rotate_toward(start: np.ndarray, target: np.ndarray, angle_rad: float) -> np.ndarray:
    relative = start.T @ target
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    full = math.acos(cosine)
    if full < 1e-6 or angle_rad <= 0.0:
        return start
    step = min(float(angle_rad), full)
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / np.linalg.norm(axis)
    x, y, z = (float(value) for value in axis)
    c, s = math.cos(step), math.sin(step)
    one = 1.0 - c
    local = np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )
    return start @ local


def index_pad_tool_rpy_deg() -> tuple[float, float, float]:
    """Reachable wrist pose. The pad meets the panel at an angle, not flat."""
    reachable = _rpy(*(math.radians(value) for value in REACHABLE_PRESS_RPY_DEG))
    tilted = _rotate_toward(
        reachable,
        _parallel_pad_rotation(),
        math.radians(PAD_TILT_FROM_REACHABLE_DEG),
    )
    return _rpy_from_matrix(tilted)
