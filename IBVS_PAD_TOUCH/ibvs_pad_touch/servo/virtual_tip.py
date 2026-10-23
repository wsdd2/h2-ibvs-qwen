# -*- coding: utf-8 -*-
"""FK virtual tip for chest-camera PBVS/IBVS when the real Dex1 tip is occluded.

Chest D435 is fixed on torso. The closed fingertip is often outside that
image, so control uses a model tip: wrist_yaw FK + Dex1 offset + optional
one-shot visual bias. This is virtual visual servoing, not eye-to-hand
feature tracking of a real pixel.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ibvs_pad_touch.types import CameraPoint, Intrinsics


# Motion-control URDF: right_wrist_yaw_link -> right_ee_link is +0.205 m
# along wrist X with identity rotation. Vision must not add a second offset.
DEFAULT_TIP_OFFSET_EE_M = (0.205, 0.0, 0.0)
CHEST_CAMERA_HEIGHT_M = 0.40
CHEST_CAMERA_FORWARD_M = 0.025


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX, matching H2CameraIK._rpy_from_matrix."""
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


def apply_ee_offset(
    ee_xyz_m: Sequence[float],
    ee_rpy_rad: Sequence[float],
    offset_ee_m: Sequence[float] = DEFAULT_TIP_OFFSET_EE_M,
) -> tuple[float, float, float]:
    rotation = rpy_to_matrix(float(ee_rpy_rad[0]), float(ee_rpy_rad[1]), float(ee_rpy_rad[2]))
    offset = rotation @ np.asarray(offset_ee_m, dtype=np.float64)
    return (
        float(ee_xyz_m[0]) + float(offset[0]),
        float(ee_xyz_m[1]) + float(offset[1]),
        float(ee_xyz_m[2]) + float(offset[2]),
    )


def torso_to_chest_camera(
    xyz_torso_m: Sequence[float],
    camera_height_m: float = CHEST_CAMERA_HEIGHT_M,
    camera_forward_m: float = CHEST_CAMERA_FORWARD_M,
) -> tuple[float, float, float]:
    """Torso X forward/Y left/Z up -> chest camera X right/Y down/Z forward.

    The camera sits above torso_link and a little forward. Its optical axis is
    horizontal, parallel to the ground.
    """
    x, y, z = (float(value) for value in xyz_torso_m)
    return (-y, float(camera_height_m) - z, x - float(camera_forward_m))


def chest_camera_to_torso(
    xyz_cam_m: Sequence[float],
    camera_height_m: float = CHEST_CAMERA_HEIGHT_M,
    camera_forward_m: float = CHEST_CAMERA_FORWARD_M,
) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in xyz_cam_m)
    return (z + float(camera_forward_m), -x, float(camera_height_m) - y)


def project_camera_xyz(
    xyz_cam_m: Sequence[float],
    K: Intrinsics,
) -> Optional[tuple[float, float]]:
    x, y, z = (float(value) for value in xyz_cam_m)
    if z <= 1e-4:
        return None
    return K.fx * x / z + K.cx, K.fy * y / z + K.cy


def virtual_tip_from_ee(
    ee_xyz_m: Sequence[float],
    ee_rpy_rad: Sequence[float],
    K: Intrinsics,
    *,
    offset_ee_m: Sequence[float] = DEFAULT_TIP_OFFSET_EE_M,
    bias_cam_m: Sequence[float] | None = None,
    camera_height_m: float = CHEST_CAMERA_HEIGHT_M,
    camera_forward_m: float = CHEST_CAMERA_FORWARD_M,
    torso_T_camera: np.ndarray | None = None,
) -> CameraPoint:
    tip_torso = apply_ee_offset(ee_xyz_m, ee_rpy_rad, offset_ee_m)
    if torso_T_camera is None:
        tip_cam = np.asarray(
            torso_to_chest_camera(tip_torso, camera_height_m, camera_forward_m),
            dtype=np.float64,
        )
    else:
        point = np.asarray([tip_torso[0], tip_torso[1], tip_torso[2], 1.0], dtype=np.float64)
        tip_cam = (np.linalg.inv(np.asarray(torso_T_camera, dtype=np.float64).reshape(4, 4)) @ point)[:3]
    if bias_cam_m is not None:
        tip_cam = tip_cam + np.asarray(bias_cam_m, dtype=np.float64)
    xyz = (float(tip_cam[0]), float(tip_cam[1]), float(tip_cam[2]))
    uv = project_camera_xyz(xyz, K)
    if uv is None:
        return CameraPoint(u=K.cx, v=K.cy, z_m=xyz[2], xyz_m=xyz, source="virtual_fk")
    return CameraPoint(u=float(uv[0]), v=float(uv[1]), z_m=xyz[2], xyz_m=xyz, source="virtual_fk")


def visual_bias_cam_m(
    visual: CameraPoint,
    virtual: CameraPoint,
) -> Optional[tuple[float, float, float]]:
    if visual.xyz_m is None or virtual.xyz_m is None:
        return None
    return (
        float(visual.xyz_m[0] - virtual.xyz_m[0]),
        float(visual.xyz_m[1] - virtual.xyz_m[1]),
        float(visual.xyz_m[2] - virtual.xyz_m[2]),
    )


def snap_visual_to_virtual_depth(
    visual: CameraPoint,
    virtual: CameraPoint,
    K: Intrinsics,
) -> CameraPoint:
    """A click gives the true image ray; keep FK/virtual depth if the click has none."""
    z = visual.z_m if visual.z_m is not None else virtual.z_m
    if z is None:
        return visual
    return CameraPoint(
        u=visual.u,
        v=visual.v,
        z_m=float(z),
        xyz_m=K.backproject(visual.u, visual.v, float(z)),
        source=visual.source or "manual_snap",
        confidence=visual.confidence,
    )


def choose_control_tip(
    *,
    virtual: CameraPoint | None,
    manual: CameraPoint | None,
    K: Intrinsics,
    tracker_manual: bool = False,
    panel_corners: Sequence[Sequence[float]] | None = None,
) -> CameraPoint | None:
    """Manual click wins unless it landed on the panel. Auto detections never drive."""
    from ibvs_pad_touch.perception.gripper_tip import point_inside_panel

    if tracker_manual and manual is not None:
        if point_inside_panel((manual.u, manual.v), panel_corners):
            return virtual
        if manual.xyz_m is None and virtual is not None:
            return snap_visual_to_virtual_depth(manual, virtual, K)
        return manual
    return virtual


def wrist_cam_delta_to_torso(
    delta_cam_m: Sequence[float],
    ee_rpy_rad: Sequence[float],
    hand_R_cam: np.ndarray,
) -> tuple[float, float, float]:
    """Map a wrist-camera translational IBVS step into torso XYZ."""
    r_torso_hand = rpy_to_matrix(float(ee_rpy_rad[0]), float(ee_rpy_rad[1]), float(ee_rpy_rad[2]))
    v_hand = np.asarray(hand_R_cam, dtype=np.float64) @ np.asarray(delta_cam_m, dtype=np.float64)
    v_torso = r_torso_hand @ v_hand
    return float(v_torso[0]), float(v_torso[1]), float(v_torso[2])
