"""Hold the three waist joints at one captured pose.

The capture happens once. Later measurements can stop an arm move, and they
can shift the arm command so the wrist stays at the pose planned on the
captured waist. They do not replace the captured waist target.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ibvs_pad_touch.servo.upper_body_model import (
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
    UpperBodyModel,
)

WAIST_MOTORS = (12, 13, 14)
WAIST_MOTOR_NAMES = dict(zip(WAIST_MOTORS, WAIST_JOINTS))
ARM_MOTORS = (22, 23, 24, 25, 26, 27, 28)
WAIST_KP = 300.0
WAIST_KD = 3.0
WAIST_DRIFT_TOLERANCE_DEG = (1.0, 1.0, 1.0)
ARM_CORRECTION_LIMIT_RAD = math.radians(2.0)
_DAMPING = 0.005
_TASK_SCALE = np.diag((1.0, 1.0, 1.0, 0.3, 0.3, 0.3))


def _finite_rad(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("waist angle must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("waist angle must be a finite number")
    return number


def waist_lock_from_q(q_by_motor: Mapping[int, float]) -> dict[int, float]:
    """Copy motors 12, 13, 14. Missing or non-finite values are refused."""
    locked: dict[int, float] = {}
    for motor in WAIST_MOTORS:
        if motor not in q_by_motor:
            raise ValueError(f"waist motor {motor} is missing")
        locked[motor] = _finite_rad(q_by_motor[motor])
    return locked


def stamp_waist_command(
    command: Mapping[int, float],
    dq: Mapping[int, float] | None,
    lock: Mapping[int, float] | None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Force the captured waist position and zero waist velocity."""
    stamped = {int(motor): float(value) for motor, value in command.items()}
    velocity = {} if dq is None else {int(motor): float(value) for motor, value in dq.items()}
    if not lock:
        return stamped, velocity
    for motor, angle in waist_lock_from_q(lock).items():
        stamped[motor] = angle
        velocity[motor] = 0.0
    return stamped, velocity


def waist_drift(
    lock: Mapping[int, float],
    measured: Mapping[int, float] | None,
    tolerance_deg: Sequence[float] = WAIST_DRIFT_TOLERANCE_DEG,
) -> dict[str, object]:
    """Compare a later reading with the capture. The capture is not updated."""
    tolerances = [float(value) for value in tolerance_deg]
    if len(tolerances) != 3 or any(value <= 0 or not math.isfinite(value) for value in tolerances):
        raise ValueError("three positive finite waist tolerances are required")
    reference = waist_lock_from_q(lock)
    if not measured:
        return {"ok": False, "reason": "missing waist state", "delta_deg": None}
    try:
        delta = [
            math.degrees(_finite_rad(measured[motor]) - reference[motor])
            for motor in WAIST_MOTORS
        ]
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "missing waist state", "delta_deg": None}
    good = all(abs(error) <= limit for error, limit in zip(delta, tolerances))
    return {
        "ok": good,
        "reason": "within tolerance" if good else "waist moved; hold approach and relocalize",
        "delta_deg": dict(zip(WAIST_JOINTS, delta)),
    }


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return np.zeros(3)
    skew = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=float,
    )
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        axis = np.sqrt(np.maximum(0.0, np.diag(rotation)))
        return axis * angle
    return skew * (angle / (2.0 * sine))


def _tool_transform(offset_m: Sequence[float]) -> np.ndarray:
    offset = np.asarray(offset_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("tool offset must be a finite 3-vector")
    transform = np.eye(4)
    transform[:3, 3] = offset
    return transform


def _pose(lock: Mapping[int, float], arm_rad: Sequence[float]) -> dict[str, float]:
    arm = [float(value) for value in arm_rad]
    if len(arm) != 7 or not all(math.isfinite(value) for value in arm):
        raise ValueError("right arm command must be seven finite angles")
    pose = {WAIST_MOTOR_NAMES[motor]: angle for motor, angle in waist_lock_from_q(lock).items()}
    pose.update(zip(RIGHT_ARM_JOINTS, arm))
    return pose


def arm_delta_for_measured_waist(
    model: UpperBodyModel,
    lock: Mapping[int, float],
    arm_rad: Sequence[float],
    measured: Mapping[int, float],
    tool_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
    tip_link: str = "right_wrist_yaw_link",
) -> np.ndarray:
    """Arm increment that keeps the wrist at the pose planned on the captured waist.

    The returned seven values are added to the right arm. Each is limited to
    ±2 degrees. The waist target itself stays at ``lock``.
    """
    nominal = _pose(lock, arm_rad)
    adjusted = dict(nominal)
    adjusted.update(
        {WAIST_MOTOR_NAMES[motor]: _finite_rad(measured[motor]) for motor in WAIST_MOTORS}
    )
    tool = _tool_transform(tool_offset_m)
    ee = model.relative("torso_link", tip_link, adjusted)
    tip = ee @ tool
    jacobian = model.jacobian("torso_link", tip_link, adjusted, RIGHT_ARM_JOINTS).copy()
    jacobian[:3] += np.cross(jacobian[3:].T, tip[:3, 3] - ee[:3, 3]).T
    torso_R_pelvis = model.relative("pelvis", "torso_link", adjusted)[:3, :3]
    actual = model.relative("pelvis", tip_link, adjusted) @ tool
    desired = model.relative("pelvis", tip_link, nominal) @ tool
    correction = np.r_[
        torso_R_pelvis.T @ (desired[:3, 3] - actual[:3, 3]),
        torso_R_pelvis.T @ _rotation_vector(desired[:3, :3] @ actual[:3, :3].T),
    ]
    scaled = _TASK_SCALE @ jacobian
    step = scaled.T @ np.linalg.solve(
        scaled @ scaled.T + _DAMPING**2 * np.eye(6),
        _TASK_SCALE @ correction,
    )
    return np.clip(step, -ARM_CORRECTION_LIMIT_RAD, ARM_CORRECTION_LIMIT_RAD)


def resolve_h2_urdf() -> Path | None:
    here = Path(__file__).resolve()
    candidates = (
        Path("/home/unitree/MscapeTech/urdf/H2.urdf"),
        here.parents[3] / "unitree_ros" / "robots" / "h2_description" / "H2.urdf",
        here.parents[3] / "urdf" / "H2.urdf",
        here.parents[3] / "Hand_Eye_Calib" / "urdf" / "H2.urdf",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def tip_link_name(model: UpperBodyModel) -> str:
    if "right_ee_link" in model.links:
        return "right_ee_link"
    if "right_wrist_yaw_link" in model.links:
        return "right_wrist_yaw_link"
    raise ValueError("URDF has no right wrist link")
