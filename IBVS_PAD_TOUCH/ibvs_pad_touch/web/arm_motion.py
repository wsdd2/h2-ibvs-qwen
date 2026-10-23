# -*- coding: utf-8 -*-
"""Chest IBVS + right-arm poke. Left arm holds a side camera to watch contact."""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


HANDEYE_ROOTS = (
    Path("/home/unitree/MscapeTech/Hand_Eye_Calib"),
    Path(__file__).resolve().parents[3] / "Hand_Eye_Calib",
)
# Between the too-low -60° start and the -96° start that put the jaw above the buttons.
# Image feedback then raises or lowers until the real tip meets the button row.
# 130° folded the forearms down in front of the body. 46° was nearly straight
# and reached the wall. 65° keeps a horizontal raise with the hand short of the wall.
READY_DEG = (-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0)
# Keep the forearm in front of the chest camera. Shoulder -90° with elbow 85°
# folded the arm back into that bracket and broke it.
# pitch, roll, yaw, elbow, wrist roll, wrist pitch, wrist yaw.
# Roll more negative swings the right arm out, away from the chest.
FOREARM_LEVEL_DEG = (-62.0, -48.0, 0.0, 48.0, 0.0, 0.0, 0.0)
WRIST_VIEW_DEG = READY_DEG
REST_DEG = (10.0, -8.0, 0.0, 15.0, 0.0, 0.0, 0.0)
# Joint zero points the forearms straight forward. Elbow +90° folds them down
# beside the thigh; a small outward roll keeps the hands off the hips.
# pitch, roll, yaw, elbow, wrist roll, wrist pitch, wrist yaw.
LEFT_HANG_DEG = (0.0, 10.0, 0.0, 90.0, 0.0, 0.0, 0.0)
RIGHT_HANG_DEG = (0.0, -10.0, 0.0, 90.0, 0.0, 0.0, 0.0)
# Left arm horizontal, camera looks across to the right hand / panel.
# pitch/roll/yaw/elbow/w_roll/w_pitch/w_yaw.
# Wrist pitch more positive tilts this camera down. +10 still framed the ceiling.
LEFT_SIDE_VIEW_DEG = (-65.0, 55.0, 35.0, 65.0, 0.0, 22.0, -45.0)
# When the gripper leaves the image, hold the right arm and sweep these left
# joints around the pose that lost the view. Wrist pitch/yaw and shoulder yaw
# change the side-camera look angle; wrist roll rolls that view.
LEFT_SEEK_JOINTS = (5, 6, 2, 4)
LEFT_SEEK_LABELS = {5: "腕 pitch", 6: "腕 yaw", 2: "肩 yaw", 4: "腕 roll"}
LEFT_SEEK_STEP_DEG = 4.0
LEFT_SEEK_SPAN_DEG = 16.0
LEFT_SEEK_RAMP_S = 0.45
LEFT_REST_DEG = (10.0, 8.0, 0.0, 15.0, 0.0, 0.0, 0.0)
LEFT_CAMERA_PAYLOAD_KG = 0.25
# Loaded arm tracks roughly 25-28° behind arm_sdk command on this H2.
# Command farther, but reverse from measured feedback before the physical arm
# exceeds the safe high pose.
SWEEP_HIGH_SHOULDER_DEG = -112.0
MEASURED_HIGH_SHOULDER_DEG = -88.0
SWEEP_LOW_SHOULDER_DEG = -58.0
SWEEP_HIGH_ELBOW_DEG = 61.0
SWEEP_LOW_ELBOW_DEG = 67.0
SWEEP_PITCH_STEP_DEG = 2.0
# Chest Y is down. Stop the raise when the real tip is this close to the button row.
SCAN_TIP_HEIGHT_PX = 90.0
SWEEP_MAX_STEPS = 40
SEEK_DWELL_S = 0.50
SEEK_RAMP_S = 1.6
JOG_STEP_M = 0.006
FEATURE_READY_FRAMES = 3
# Photos: the hold pose is already a few cm short of the yellow buttons.
# Do not keep a 8-20cm air gap; finish by poking from the current pose.
PBVS_STANDOFF_M = 0.03
PBVS_XY_TOL_M = 0.01
PBVS_Z_TOL_M = 0.01
PBVS_MAX_STEP_M = 0.020
PBVS_STEP_SECONDS = 0.45
# Coarse PBVS is one 6-DoF move to the 3 cm standoff, same pose path as
# h2_arm_service singlearmcartesian: right_ee XYZ plus DIRECT_EE_RPY_DEG.
PBVS_DIRECT_SPEED_M_S = 0.10
PBVS_DIRECT_MIN_SECONDS = 1.2
PBVS_DIRECT_MAX_SECONDS = 3.5
PBVS_DIRECT_MAX_JOINT_DEG = 120.0
POKE_EXTRA_M = 0.10
POKE_STEP_M = 0.006
POKE_UP_STEP_M = 0.002
POKE_RIGHT_STEP_M = 0.0015
RIGHT_EE_OFFSET_WRIST_M = (0.205, 0.0, 0.0)
# URDF right_elbow_joint. Positive bend brings the poke toward horizontal.
RIGHT_ELBOW_INDEX = 3
RIGHT_ELBOW_LIMIT_RAD = (-0.986, 3.071)
DIRECT_EE_RPY_DEG = (-88.8133, -3.98969, 3.20691)
# 20 mm treated a 2 cm Y miss as arrived, so the next shot was never sent.
DIRECT_PBVS_ARRIVE_TOL_M = 0.005
# First gap in front of the panel. It is not a fixed stop: if Y/Z stops
# getting closer, the gap shrinks by one step per shot.
DIRECT_PBVS_X_CLEAR_M = 0.08
DIRECT_PBVS_X_LEAD_M = 0.05
DIRECT_PBVS_X_RELAX_M = 0.02
DIRECT_PBVS_YZ_STALL_M = 0.002
# The service plans one open-loop Cartesian shot. From a distant pose its IK
# may stop inside the 25 mm gate, and the arm then holds that miss. Later
# shots start from the joints actually reached. A 2 cm Y miss is not done.
DIRECT_PBVS_MAX_SHOTS = 3
# A hang-to-level lift is about 130° on one joint. A low arm reaching a
# chest-height button is about 150°. The wrist often stops ~20° short of
# the exact motion-control roll.
# 15 mm rejected a pitch-8 solution at 16.0 mm / 9° / 99°.
# 25 mm / 150° then rejected the same kind of lift at 38.9 mm / 24.6° / 151°.
# 80 mm is still a bad solve and must not be sent. Arrival stays at 5 mm.
DIRECT_PBVS_POSE_POS_TOL_M = 0.045
DIRECT_PBVS_POSE_ROT_TOL_DEG = 25.0
DIRECT_PBVS_MIN_JOINT_DEG = 165.0
# Wrist and chest button positions closer than this are averaged. Farther ones stay chest-only.
EYE_IN_HAND_AGREE_M = 0.02
EYE_IN_HAND_DWELL_S = 1.0
# After a press, back away from the panel and raise the jaw before the next button.
SEQUENCE_RETRACT_X_M = 0.10
SEQUENCE_LIFT_Z_M = 0.08
BUTTON_NAMES = ("button_1", "button_2", "button_3")
DIRECT_EE_PREALIGN_STANDOFF_M = 0.15
# Seeds and small RPY changes for one exact goal, or this search budget runs out.
PBVS_PATH_TIMEOUT_S = 8.0
PBVS_STAGE_TIMEOUT_S = 24.0
# /h2_arm/singlearmcartesian calls Plan(), not PlanJointSpace(). The hand
# follows one Cartesian quintic. feedrate 0.3 applies only to the unused
# Ruckig joint-space branch. Cartesian time uses the raw 80/140 deg/s limits
# and is at least the yaml move_duration of 2 s.
MOTION_FEEDRATE = 0.3
MOTION_DT_S = 0.02
MOTION_MIN_SECONDS = 2.0
# IK knots along the Cartesian curve. Playback still runs at 50 Hz.
MOTION_KNOT_S = 0.20
MOTION_ARM_KP = 140.0
MOTION_ARM_KD = 3.0
MOTION_TRACK_GAIN = 0.18
MOTION_TRACK_FILTER = 0.12
MOTION_TRACK_MAX_RAD = 0.012
MOTION_TRACK_MAX_STEP_RAD = 0.00018
MOTION_TRACK_RAMP = 0.20
MOTION_TRACK_FADE = 0.20
RIGHT_ARM_JOINTS = (22, 23, 24, 25, 26, 27, 28)
MOTION_TRACK_TOL_DEG = 8.0
MOTION_TRACK_WAIT_S = 1.0
MOTION_MAX_VEL_DEG_S = (80.0, 80.0, 80.0, 80.0, 140.0, 140.0, 140.0)
MOTION_MAX_ACC_DEG_S2 = (240.0, 240.0, 240.0, 240.0, 450.0, 450.0, 450.0)
TRAJECTORY_JOINT_INDEX = {
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}


def target_features_ready(telem: dict[str, Any]) -> bool:
    """Chest panel + chosen button, or the last locked target after a close-up loss."""
    return bool(
        telem.get("target_uv")
        and (telem.get("panel_visible") or telem.get("using_last_target"))
    )


_VISUAL_TIP_SOURCES = {"chest_visual", "chest_click"}


def _has_visual_tip(telem: dict[str, Any]) -> bool:
    if str(telem.get("tip_source") or "") in _VISUAL_TIP_SOURCES:
        return True
    return bool(
        telem.get("tip_visible")
        and (telem.get("tip_tracker") or {}).get("mode") == "manual_flow"
    )


def tip_height_error_px(telem: dict[str, Any]) -> float | None:
    """Chest image v: positive means the real tip is below the button row."""
    if not _has_visual_tip(telem):
        return None
    target = telem.get("target_uv")
    tip = telem.get("tip_uv")
    if not target or not tip:
        return None
    try:
        return float(tip[1]) - float(target[1])
    except (TypeError, ValueError, IndexError):
        return None


def raise_direction(telem: dict[str, Any]) -> int:
    """-1 raises the shoulder. +1 lowers it when the jaw is already above the buttons."""
    error = tip_height_error_px(telem)
    if error is not None and error < -SCAN_TIP_HEIGHT_PX:
        return 1
    return -1


def scan_lock_ready(telem: dict[str, Any]) -> bool:
    """Stop moving when the real chest tip reaches the button row.

    The chest camera sees the panel before the arm has moved, so a visible
    panel must not end the sweep. FK does not count.
    """
    error = tip_height_error_px(telem)
    return error is not None and abs(error) <= SCAN_TIP_HEIGHT_PX


def ibvs_features_ready(telem: dict[str, Any]) -> bool:
    """Chest button plus a real image tip, or a measured left-side gap."""
    return bool(
        target_features_ready(telem)
        and (_has_visual_tip(telem) or telem.get("side_measured"))
    )


def pbvs_features_ready(telem: dict[str, Any]) -> bool:
    return bool(
        target_features_ready(telem)
        and telem.get("target_torso_m")
        and telem.get("eye_to_hand")
    )


def pbvs_coarse_delta(
    error_cam_m: list[float] | tuple[float, float, float],
) -> tuple[tuple[float, float, float], bool]:
    """Move near the button but preserve a camera-Z standoff."""
    ex, ey, ez = (float(value) for value in error_cam_m)
    residual_z = ez - PBVS_STANDOFF_M
    done = (
        abs(ex) <= PBVS_XY_TOL_M
        and abs(ey) <= PBVS_XY_TOL_M
        and abs(residual_z) <= PBVS_Z_TOL_M
    )
    return (ex, ey, residual_z), done


def should_jog_ibvs(telem: dict[str, Any], height_locked: bool) -> bool:
    """Jog from chest image error or a measured side-view approach. FK does not count."""
    if not height_locked:
        return False
    if telem.get("side_contact") or telem.get("phase") == "CONTACT":
        return False
    return ibvs_features_ready(telem)


def direct_pbvs_yz_error(current: Sequence[float], button: Sequence[float]) -> float:
    return math.hypot(float(button[1]) - float(current[1]), float(button[2]) - float(current[2]))


def relax_x_clearance(
    x_clear_m: float,
    yz_before_m: float | None,
    yz_after_m: float,
    relax_m: float = DIRECT_PBVS_X_RELAX_M,
    stall_m: float = DIRECT_PBVS_YZ_STALL_M,
) -> float:
    """Shrink the X gap by one step when Y/Z did not get closer."""
    if yz_before_m is None or float(yz_after_m) <= DIRECT_PBVS_ARRIVE_TOL_M:
        return float(x_clear_m)
    if float(yz_after_m) <= float(yz_before_m) - float(stall_m):
        return float(x_clear_m)
    return max(0.0, float(x_clear_m) - float(relax_m))


def direct_pbvs_motion_goal(
    current: Sequence[float],
    button: Sequence[float],
    yz_tol_m: float = DIRECT_PBVS_ARRIVE_TOL_M,
    x_clear_m: float = DIRECT_PBVS_X_CLEAR_M,
    x_lead_m: float = DIRECT_PBVS_X_LEAD_M,
) -> tuple[float, float, float]:
    """Align Y and Z first. X may come forward a few centimetres, not the whole way.

    ``x_clear_m`` is the gap still kept in front of the button. It starts at
    8 cm and is reduced one step at a time when Y/Z stops improving. A hand
    already inside the current gap backs up to that plane.
    """
    if direct_pbvs_yz_error(current, button) <= float(yz_tol_m):
        return (float(button[0]), float(button[1]), float(button[2]))
    clear_x = float(button[0]) - max(0.0, float(x_clear_m))
    current_x = float(current[0])
    if current_x > clear_x:
        held_x = clear_x
    else:
        held_x = min(clear_x, current_x + float(x_lead_m))
    return (held_x, float(button[1]), float(button[2]))


def level_right_waypoints(current: Sequence[float] | None) -> list[tuple[float, float, float, float, float, float, float]]:
    """Clear the chest camera before raising. Do not pitch or fold further back on the way."""
    goal = tuple(float(value) for value in FOREARM_LEVEL_DEG)
    if current is None or len(current) != 7:
        return [goal]
    clear = (
        max(float(current[0]), goal[0]),
        min(float(current[1]), goal[1]),
        float(current[2]),
        min(float(current[3]), goal[3]),
        float(current[4]),
        float(current[5]),
        float(current[6]),
    )
    if all(abs(clear[index] - float(current[index])) < 1.0 for index in range(7)):
        return [goal]
    if all(abs(clear[index] - goal[index]) < 1.0 for index in range(7)):
        return [goal]
    return [clear, goal]


def sequence_lift_goal(
    button: Sequence[float],
    retract_m: float = SEQUENCE_RETRACT_X_M,
    lift_m: float = SEQUENCE_LIFT_Z_M,
) -> tuple[float, float, float]:
    """Pull the jaw off the panel, then up, so the next press does not drag."""
    return (
        float(button[0]) - float(retract_m),
        float(button[1]),
        float(button[2]) + float(lift_m),
    )


def _xyz_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def fuse_pbvs_buttons(
    chest: dict[str, Sequence[float]],
    wrist: dict[str, Sequence[float]],
    tol_m: float = EYE_IN_HAND_AGREE_M,
) -> tuple[dict[str, tuple[float, float, float]], str]:
    """Average buttons both cameras agree on. Any larger gap keeps the chest map."""
    chest_out = {
        name: (float(value[0]), float(value[1]), float(value[2]))
        for name, value in chest.items()
        if value is not None and len(value) == 3
    }
    paired: list[tuple[str, float]] = []
    for name, chest_xyz in chest_out.items():
        wrist_xyz = wrist.get(name)
        if wrist_xyz is None or len(wrist_xyz) != 3:
            continue
        paired.append((name, _xyz_distance(chest_xyz, wrist_xyz)))
    if not paired or max(distance for _name, distance in paired) >= float(tol_m):
        return chest_out, "chest"
    fused = dict(chest_out)
    for name, _distance in paired:
        wrist_xyz = wrist[name]
        chest_xyz = chest_out[name]
        fused[name] = tuple(
            (float(chest_xyz[index]) + float(wrist_xyz[index])) / 2.0 for index in range(3)
        )
    return fused, "average"


def clip_torso_delta(torso: tuple[float, float, float], limit_m: float = JOG_STEP_M) -> tuple[float, float, float]:
    lim = abs(float(limit_m))
    return (
        max(-lim, min(lim, float(torso[0]))),
        max(-lim, min(lim, float(torso[1]))),
        max(-lim, min(lim, float(torso[2]))),
    )


def chest_panel_seen(telem: dict[str, Any]) -> bool:
    return bool(telem.get("panel_visible") or telem.get("panel_partial") or telem.get("using_last_target"))


def wrist_panel_seen(telem: dict[str, Any]) -> bool:
    """Left camera no longer locks height; kept so old telemetry keys stay harmless."""
    _ = telem
    return False


def panel_seen(telem: dict[str, Any]) -> bool:
    return chest_panel_seen(telem)


def chest_gripper_missing(telem: dict[str, Any]) -> bool:
    """True only when this frame explicitly has no chest gripper.

    Callers that omit ``tip_uv`` are left alone so older jogs still run.
    """
    if str(telem.get("phase") or "") == "SEEK_GRIPPER":
        return True
    if "tip_uv" not in telem:
        return False
    tip = telem.get("tip_uv")
    return tip is None or tip == ""


def side_gripper_missing(telem: dict[str, Any]) -> bool:
    return "未看到右夹爪" in str(telem.get("side_message") or "")


def next_left_gripper_seek(
    origin_deg: Sequence[float],
    joint_i: int,
    direction: int,
    offset_deg: float,
) -> tuple[list[float], int, int, float]:
    """One look-around step. One left joint moves, then the next, within ±span."""
    origin = [float(value) for value in origin_deg]
    if len(origin) != 7:
        raise ValueError("left seek origin must have 7 joints")
    count = len(LEFT_SEEK_JOINTS)
    index = int(joint_i) % count
    sign = -1 if int(direction) < 0 else 1
    step = LEFT_SEEK_STEP_DEG * sign
    new_offset = float(offset_deg) + step
    if abs(new_offset) > LEFT_SEEK_SPAN_DEG + 1e-6:
        sign = -sign
        new_offset = float(offset_deg) + LEFT_SEEK_STEP_DEG * sign
        if sign > 0:
            index = (index + 1) % count
            new_offset = 0.0
    pose = list(origin)
    pose[LEFT_SEEK_JOINTS[index]] = origin[LEFT_SEEK_JOINTS[index]] + new_offset
    return pose, index, sign, new_offset


def next_sweep_joints_deg(
    current_deg: list[float],
    direction: int,
) -> tuple[list[float], int, bool] | None:
    """One bounded scan step. direction -1 raises, +1 lowers."""
    if not current_deg:
        return None
    out = [float(v) for v in current_deg]
    if len(out) < 4:
        return None
    scan_direction = -1 if direction < 0 else 1
    turned = False
    if scan_direction < 0:
        out[0] = max(
            SWEEP_HIGH_SHOULDER_DEG,
            out[0] - SWEEP_PITCH_STEP_DEG,
        )
        if out[0] <= SWEEP_HIGH_SHOULDER_DEG + 1e-6:
            scan_direction = 1
            turned = True
    else:
        out[0] = min(
            SWEEP_LOW_SHOULDER_DEG,
            out[0] + SWEEP_PITCH_STEP_DEG,
        )
        if out[0] >= SWEEP_LOW_SHOULDER_DEG - 1e-6:
            scan_direction = -1
            turned = True
    ratio = (SWEEP_LOW_SHOULDER_DEG - out[0]) / (
        SWEEP_LOW_SHOULDER_DEG - SWEEP_HIGH_SHOULDER_DEG
    )
    ratio = max(0.0, min(1.0, ratio))
    out[3] = SWEEP_LOW_ELBOW_DEG + ratio * (
        SWEEP_HIGH_ELBOW_DEG - SWEEP_LOW_ELBOW_DEG
    )
    return out, scan_direction, turned


def ensure_handeye_calib() -> Path:
    for root in HANDEYE_ROOTS:
        if (root / "handeye_calib" / "h2_arm_sdk.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise RuntimeError(
        "找不到 Hand_Eye_Calib。应在 /home/unitree/MscapeTech/Hand_Eye_Calib"
    )


def quintic_joint_seconds(
    start_rad: Sequence[float],
    target_rad: Sequence[float],
    minimum_s: float = MOTION_MIN_SECONDS,
    feedrate: float = MOTION_FEEDRATE,
) -> float:
    """Duration of one synchronized quintic, using the motion-control vel/acc bounds."""
    duration = float(minimum_s)
    rate = float(feedrate)
    for index, (start, target) in enumerate(zip(start_rad, target_rad)):
        diff = abs(float(target) - float(start))
        if diff <= 1e-5:
            continue
        vmax = math.radians(MOTION_MAX_VEL_DEG_S[index]) * rate
        amax = math.radians(MOTION_MAX_ACC_DEG_S2[index]) * rate
        duration = max(duration, 1.875 * diff / vmax, math.sqrt(5.7735 * diff / amax))
    return duration


def quintic_blend(phase: float) -> float:
    phase = min(1.0, max(0.0, float(phase)))
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def expand_quintic_knots(
    knots: Sequence[Sequence[float]],
    total_s: float,
    dt: float,
) -> list[list[float]]:
    """50 Hz joint samples through Cartesian IK knots. One quintic per segment."""
    if not knots:
        return []
    if len(knots) == 1:
        return [[float(value) for value in knots[0]]]
    intervals = len(knots) - 1
    steps = max(1, int(round(max(float(total_s), dt) / intervals / dt)))
    samples: list[list[float]] = []
    for start, end in zip(knots, knots[1:]):
        for step in range(steps):
            blend = quintic_blend(step / steps)
            samples.append(
                [
                    float(start[index]) * (1.0 - blend) + float(end[index]) * blend
                    for index in range(len(start))
                ]
            )
    samples.append([float(value) for value in knots[-1]])
    return samples


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    """URDF fixed-axis R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_to_rpy(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    sy = math.hypot(float(matrix[0][0]), float(matrix[1][0]))
    if sy > 1e-8:
        roll = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
        pitch = math.atan2(-float(matrix[2][0]), sy)
        yaw = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
    else:
        roll = math.atan2(-float(matrix[1][2]), float(matrix[1][1]))
        pitch = math.atan2(-float(matrix[2][0]), sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def _quat_normalize(quat: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quat))
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in quat)


def _quat_from_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    trace = float(matrix[0][0]) + float(matrix[1][1]) + float(matrix[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (float(matrix[2][1]) - float(matrix[1][2])) / scale,
            (float(matrix[0][2]) - float(matrix[2][0])) / scale,
            (float(matrix[1][0]) - float(matrix[0][1])) / scale,
        )
    elif float(matrix[0][0]) > float(matrix[1][1]) and float(matrix[0][0]) > float(matrix[2][2]):
        scale = math.sqrt(1.0 + float(matrix[0][0]) - float(matrix[1][1]) - float(matrix[2][2])) * 2.0
        quat = (
            (float(matrix[2][1]) - float(matrix[1][2])) / scale,
            0.25 * scale,
            (float(matrix[0][1]) + float(matrix[1][0])) / scale,
            (float(matrix[0][2]) + float(matrix[2][0])) / scale,
        )
    elif float(matrix[1][1]) > float(matrix[2][2]):
        scale = math.sqrt(1.0 + float(matrix[1][1]) - float(matrix[0][0]) - float(matrix[2][2])) * 2.0
        quat = (
            (float(matrix[0][2]) - float(matrix[2][0])) / scale,
            (float(matrix[0][1]) + float(matrix[1][0])) / scale,
            0.25 * scale,
            (float(matrix[1][2]) + float(matrix[2][1])) / scale,
        )
    else:
        scale = math.sqrt(1.0 + float(matrix[2][2]) - float(matrix[0][0]) - float(matrix[1][1])) * 2.0
        quat = (
            (float(matrix[1][0]) - float(matrix[0][1])) / scale,
            (float(matrix[0][2]) + float(matrix[2][0])) / scale,
            (float(matrix[1][2]) + float(matrix[2][1])) / scale,
            0.25 * scale,
        )
    return _quat_normalize(quat)


def _matrix_from_quat(quat: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = _quat_normalize(quat)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def slerp_rpy(
    start: Sequence[float],
    goal: Sequence[float],
    blend: float,
) -> tuple[float, float, float]:
    """Orientation blend used by H2ArmControl::Plan()."""
    if blend <= 0.0:
        return tuple(float(value) for value in start)
    if blend >= 1.0:
        return tuple(float(value) for value in goal)
    start_q = _quat_from_matrix(_rpy_matrix(float(start[0]), float(start[1]), float(start[2])))
    goal_q = _quat_from_matrix(_rpy_matrix(float(goal[0]), float(goal[1]), float(goal[2])))
    dot = sum(left * right for left, right in zip(start_q, goal_q))
    if dot < 0.0:
        goal_q = tuple(-value for value in goal_q)
        dot = -dot
    if dot > 0.9995:
        mixed = tuple(start_q[index] + blend * (goal_q[index] - start_q[index]) for index in range(4))
    else:
        theta = math.acos(min(1.0, dot))
        sine = math.sin(theta)
        start_weight = math.sin((1.0 - blend) * theta) / sine
        goal_weight = math.sin(blend * theta) / sine
        mixed = tuple(start_weight * start_q[index] + goal_weight * goal_q[index] for index in range(4))
    return _matrix_to_rpy(_matrix_from_quat(mixed))


def camera_delta_to_torso(
    dx: float,
    dy: float,
    dz: float,
    camera_R_torso=None,
) -> tuple[float, float, float]:
    """Chest D435: X right, Y down, Z forward. Torso: X forward, Y left, Z up."""
    if camera_R_torso is not None:
        vector = (float(dx), float(dy), float(dz))
        return tuple(
            sum(float(camera_R_torso[row][col]) * vector[col] for col in range(3))
            for row in range(3)
        )
    return (float(dz), -float(dx), -float(dy))


class ArmMover:
    def __init__(
        self,
        *,
        iface: str = "eth0",
        telemetry: Callable[[], dict[str, Any]],
        confirm: bool = False,
        log: Callable[[str], None] = print,
        on_ee_pose: Callable[[list[float], list[float]], None] | None = None,
        direct_pbvs_poke: bool = False,
        direct_pbvs_max_m: float = 1.20,
        direct_pbvs_max_joint_deg: float = 150.0,
        pbvs_first: bool = False,
        camera_R_torso=None,
        direct_ee_poke: bool = False,
        direct_ee_poke_depth_m: float = 0.005,
        direct_ee_max_m: float = 0.0,
        direct_ee_max_joint_deg: float = 130.0,
        pbvs_step_m: float = PBVS_MAX_STEP_M,
        pbvs_step_seconds: float = PBVS_STEP_SECONDS,
        pbvs_max_joint_deg: float = 20.0,
    ) -> None:
        self.iface = iface
        self.telemetry = telemetry
        self.confirm = bool(confirm)
        self.log = log
        self.on_ee_pose = on_ee_pose
        self.wrist_watch = None
        self.wrist_hold = None
        self.wrist_view = None
        self.cam2hand_matrix = None
        self.on_screen_capture = None
        self.wrist_dwell_s = EYE_IN_HAND_DWELL_S
        self.direct_pbvs_poke = bool(direct_pbvs_poke)
        self.direct_pbvs_max_m = max(0.02, float(direct_pbvs_max_m))
        self.direct_pbvs_max_joint_deg = max(5.0, float(direct_pbvs_max_joint_deg))
        self.pbvs_first = bool(pbvs_first)
        self.camera_R_torso = camera_R_torso
        self.direct_ee_poke = bool(direct_ee_poke)
        self.direct_ee_poke_depth_m = max(0.0, min(0.02, float(direct_ee_poke_depth_m)))
        self.direct_ee_max_m = max(0.0, float(direct_ee_max_m))
        self.direct_ee_max_joint_deg = max(10.0, float(direct_ee_max_joint_deg))
        self.pbvs_step_m = max(0.005, min(0.04, float(pbvs_step_m)))
        self.pbvs_step_seconds = max(0.20, min(1.0, float(pbvs_step_seconds)))
        self.pbvs_max_joint_deg = max(5.0, min(30.0, float(pbvs_max_joint_deg)))
        self.controller: Any = None
        self.ik: Any = None
        self._stop = threading.Event()
        self._running = threading.Event()
        self._scanning = threading.Event()
        self._return_ready = threading.Event()
        self._hang_down = threading.Event()
        self._level_right = threading.Event()
        self._service_node_probe = None
        self._stop_service_node = None
        self._start_service_node = None
        self._service_node_cached: tuple[float, bool] | None = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self.phase = "idle"
        self.message = "arm idle"
        self._height_locked = False
        self._seek_done = False
        self._seek_steps = 0
        self._sweep_direction = -1
        self._sweep_turns = 0
        self._ready_streak = 0
        self._ibvs_enabled = False
        self._scan_needs_prepare = False
        self._left_side_ready = False
        self._last_visible_q: list[float] | None = None
        self._last_camera_delta: tuple[float, float, float] | None = None
        self._poke_remaining_m = 0.0
        self._control_stage = "IDLE"
        self._last_tip_v: float | None = None
        self._awaiting_raise_v: float | None = None
        self._up_stuck = 0
        self._ee_xyz_m: list[float] | None = None
        self._ee_rpy_deg: list[float] | None = None
        self._right_ee_xyz_m: list[float] | None = None
        self._right_ee_rpy_deg: list[float] | None = None
        self._direct_ee_align_steps = 0
        self._direct_ee_orient_steps = 0
        self._direct_ee_approach_steps = 0
        self._playback_stop = threading.Event()
        self._playback_thread: threading.Thread | None = None
        self._playback_id = ""
        self._playback_speed = 1.0
        self._pbvs_deadline: float | None = None
        self._service_pbvs_sent = False
        self._pbvs_release = True
        self._pbvs_wrist_checked = False
        self._pbvs_source = ""
        self._goal_name = ""
        self._locked_buttons: dict[str, tuple[float, float, float]] = {}
        self._button_sequence: list[str] = []
        self._sequence_run = False
        self._motion_backend = "arm_sdk"
        self._left_seek_origin: list[float] | None = None
        self._left_seek_joint = 0
        self._left_seek_dir = 1
        self._left_seek_offset = 0.0

    def status(self) -> dict[str, Any]:
        with self._lock:
            ee_xyz_m = None if self._ee_xyz_m is None else list(self._ee_xyz_m)
            ee_rpy_deg = None if self._ee_rpy_deg is None else list(self._ee_rpy_deg)
            right_ee_xyz_m = (
                None if self._right_ee_xyz_m is None else list(self._right_ee_xyz_m)
            )
            right_ee_rpy_deg = (
                None if self._right_ee_rpy_deg is None else list(self._right_ee_rpy_deg)
            )
        return {
            "enabled": self.confirm,
            "phase": self.phase,
            "message": self.message,
            "running": self._running.is_set(),
            "scanning": self._scanning.is_set(),
            "leveling_right": self._level_right.is_set(),
            "scan_needs_prepare": self._scan_needs_prepare,
            "left_side_ready": self._left_side_ready,
            "ibvs_enabled": self._ibvs_enabled,
            "control_stage": self._control_stage,
            "height_locked": self._height_locked,
            "seek_done": self._seek_done,
            "seek_steps": self._seek_steps,
            "sweep_direction": "up" if self._sweep_direction < 0 else "down",
            "sweep_turns": self._sweep_turns,
            "features_ready": ibvs_features_ready(self.telemetry() or {}),
            "target_ready": target_features_ready(self.telemetry() or {}),
            "has_recovery_pose": self._last_visible_q is not None,
            "gravity": bool(self.controller is not None and getattr(self.controller, "_gravity", None)),
            "sdk_active": self.controller is not None,
            "manual_enabled": self._control_stage == "MANUAL",
            "direct_pbvs_poke": self.direct_pbvs_poke,
            "pbvs_first": self.pbvs_first,
            "direct_ee_poke": self.direct_ee_poke,
            "servo_mode": self.servo_mode(),
            "motion_backend": self._motion_backend,
            "pbvs_step_m": self.pbvs_step_m,
            "pbvs_step_seconds": self.pbvs_step_seconds,
            "pbvs_max_joint_deg": self.pbvs_max_joint_deg,
            "ee_xyz_m": ee_xyz_m,
            "ee_rpy_deg": ee_rpy_deg,
            "right_ee_xyz_m": right_ee_xyz_m,
            "right_ee_rpy_deg": right_ee_rpy_deg,
            "playback": bool(self._playback_thread is not None and self._playback_thread.is_alive()),
            "playback_id": self._playback_id,
            "playback_speed": self._playback_speed,
            "button_sequence": list(self._button_sequence),
            "sequence_run": self._sequence_run,
            "locked_buttons": {name: list(value) for name, value in self._locked_buttons.items()},
            "pbvs_source": self._pbvs_source,
        }

    def servo_mode(self) -> str:
        if self.direct_pbvs_poke:
            return "direct_pbvs"
        if self.pbvs_first:
            return "hybrid"
        if self.direct_ee_poke:
            return "direct_ee"
        return "dual"

    def set_servo_mode(self, mode: str) -> str:
        """Choose the next Start path. Does not move the arm."""
        if self._ibvs_enabled or self._running.is_set() or self._return_ready.is_set() or self._hang_down.is_set() or self._level_right.is_set():
            self.message = "先 Pause 或等 Reset 完成，再切换模式"
            self.log(f"[ARM] {self.message}")
            return self.message
        chosen = str(mode or "").strip().lower()
        if chosen == "direct_pbvs":
            self.direct_pbvs_poke = True
            self.pbvs_first = False
            self.direct_ee_poke = False
            self.message = "模式 Direct PBVS：选按钮后 Start，一次到位后保持"
        elif chosen == "hybrid":
            self.direct_pbvs_poke = False
            self.pbvs_first = True
            self.direct_ee_poke = False
            self.message = "模式 Hybrid VS：选按钮后 Start，PBVS 到位后抬左臂并切双视 IBVS"
        else:
            self.message = "未知伺服模式"
        self.log(f"[ARM] {self.message}")
        return self.message

    def motion_released(self) -> bool:
        """True only after 释放双臂 has dropped the controller and the motion thread."""
        thread_alive = self._thread is not None and self._thread.is_alive()
        playback_alive = self._playback_thread is not None and self._playback_thread.is_alive()
        return (
            self.phase == "released"
            and self._control_stage == "RELEASED"
            and self.controller is None
            and not thread_alive
            and not playback_alive
            and not self._running.is_set()
            and not self._ibvs_enabled
            and not self._return_ready.is_set()
            and not self._hang_down.is_set()
            and not self._level_right.is_set()
        )

    def _service_node_up(self) -> bool:
        now = time.monotonic()
        cached = self._service_node_cached
        if cached is not None and now - cached[0] < 0.4:
            return cached[1]
        probe = self._service_node_probe
        if probe is None:
            from ibvs_pad_touch.web.arm_service import service_node_running

            probe = service_node_running
        try:
            up = bool(probe())
        except Exception:
            up = False
        self._service_node_cached = (now, up)
        return up

    def _sdk_holding(self) -> bool:
        """This page publishes rt/arm_sdk only through the SDK controller."""
        controller = self.controller
        if controller is None:
            return False
        return not callable(getattr(controller, "move_cartesian", None))

    def _motion_busy(self) -> bool:
        """True while a move this page started is still running."""
        if self._ibvs_enabled or self._running.is_set() or self._return_ready.is_set() or self._hang_down.is_set() or self._level_right.is_set():
            return True
        if self.phase in {"hanging", "returning_ready", "raising", "leveling_right", "playback"}:
            return True
        playback = self._playback_thread
        return playback is not None and playback.is_alive()

    def _can_switch_backend(self) -> bool:
        """Idle, released, or a finished hold. Not while a move is in progress."""
        return not self._motion_busy()

    def _drop_page_holder(self, *, quiet: bool = False) -> None:
        """Drop this page's controller without publishing a new arm_sdk command."""
        self._stop_playback()
        self._ibvs_enabled = False
        self._running.clear()
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._hang_down.clear()
        self._level_right.clear()
        self._return_ready.clear()
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and callable(getattr(thread, "join", None)):
            thread.join(timeout=2.0)
        controller = self.controller
        self.controller = None
        if controller is not None:
            if quiet:
                # The service node is already publishing. A weight ramp would fight it.
                stop_servo = getattr(controller, "stop_servo", None)
                if callable(stop_servo):
                    try:
                        stop_servo()
                    except Exception as exc:
                        self.log(f"[ARM] 切换前停伺服失败: {exc}")
            else:
                try:
                    controller.release(seconds=0.4)
                except Exception as exc:
                    self.log(f"[ARM] 切换前松权失败: {exc}")
        self.phase = "released"
        self._control_stage = "RELEASED"

    def _call_stop_service_node(self) -> bool:
        fn = self._stop_service_node
        if fn is None:
            from ibvs_pad_touch.web.arm_service import stop_service_node

            fn = stop_service_node
        try:
            return bool(fn())
        except Exception as exc:
            self.log(f"[ARM] 停止运控服务失败: {exc}")
            return False

    def _call_start_service_node(self) -> bool:
        fn = self._start_service_node
        if fn is None:
            from ibvs_pad_touch.web.arm_service import start_service_node

            fn = start_service_node
        try:
            return bool(fn())
        except Exception as exc:
            self.log(f"[ARM] 启动运控服务失败: {exc}")
            return False

    def publisher_conflict(self, publisher: str) -> str:
        """Refuse when this page and h2_arm_service_node would both publish rt/arm_sdk."""
        which = "service" if str(publisher) == "service" else "arm_sdk"
        if which == "arm_sdk" and self._service_node_up():
            return (
                "冲突：h2_arm_service_node 正在占用 rt/arm_sdk，本页不能再发 arm_sdk。"
                "已拒绝。点 Service 改走运控；或再点一次 arm_sdk，本页会先停掉该节点。"
            )
        if which == "service" and self._sdk_holding():
            return (
                "冲突：本页正在发布 arm_sdk，不能再调用运控服务。"
                "已拒绝。请先点释放双臂。"
            )
        return ""

    def _refuse_conflict(self, publisher: str | None = None) -> str:
        which = publisher or ("service" if self._motion_backend == "service" else "arm_sdk")
        message = self.publisher_conflict(which)
        if message:
            self.message = message
            self.log(f"[ARM][WARN] {message}")
        return message

    def set_motion_backend(self, backend: str) -> str:
        """Pick arm_sdk or the arm service. Idle pages can switch; a running move cannot."""
        chosen = str(backend or "").strip().lower().replace("-", "_")
        if chosen in {"sdk", "arm"}:
            chosen = "arm_sdk"
        if chosen not in {"arm_sdk", "service"}:
            self.message = "未知控制链路。可选 arm_sdk 或运控服务。"
            self.log(f"[ARM] {self.message}")
            return self.message
        if self._motion_busy():
            if chosen == self._motion_backend:
                self.message = (
                    "控制链路已是 arm_sdk"
                    if chosen == "arm_sdk"
                    else "控制链路已是运控服务"
                )
                return self.message
            self.message = "请先点释放双臂，再切换 arm_sdk / 运控服务。当前仍占用控制，已拒绝。"
            self.log(f"[ARM][WARN] {self.message}")
            return self.message
        node_up = self._service_node_up()
        if (
            chosen == self._motion_backend == "arm_sdk"
            and not node_up
            and not self._sdk_holding()
        ):
            self.message = "控制链路已是 arm_sdk"
            return self.message
        if (
            chosen == self._motion_backend == "service"
            and node_up
            and not self._sdk_holding()
        ):
            self.message = "控制链路已是运控服务"
            return self.message
        self._drop_page_holder(quiet=node_up and self._sdk_holding())
        self._service_node_cached = None
        if chosen == "arm_sdk":
            if self._service_node_up():
                stopped = self._call_stop_service_node()
                self._service_node_cached = None
                if not stopped or self._service_node_up():
                    self._motion_backend = "service"
                    self.message = (
                        "冲突：停不掉 h2_arm_service_node，它仍占用 rt/arm_sdk。"
                        "已拒绝，仍是运控服务。"
                    )
                    self.log(f"[ARM][WARN] {self.message}")
                    return self.message
            self._motion_backend = "arm_sdk"
            self.message = "已停止运控服务。控制链路：arm_sdk。下次 Raise/Start 由本页发布 arm_sdk。"
        else:
            if not self._service_node_up():
                self._call_start_service_node()
                self._service_node_cached = None
            self._motion_backend = "service"
            if self._service_node_up():
                self.message = "控制链路：运控服务。下次 Raise/Start 调用 h2_arm_service，本页不发 arm_sdk。"
            else:
                self.message = "控制链路：运控服务。h2_arm_service_node 没有起来，Start 前请确认节点。"
        self.log(f"[ARM] {self.message}")
        return self.message

    def return_to_ready(self) -> str:
        """Ramp both arms to the Ctrl+C lowered pose on the active link."""
        self._stop_playback()
        conflict = self._refuse_conflict()
        if conflict:
            return conflict
        if not self.confirm:
            self.message = "启动网页时要加 --confirm-arm-motion 才会回放下位"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._ibvs_enabled = False
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._control_stage = "RESET"
        self._pbvs_deadline = None
        self._level_right.clear()
        self._return_ready.set()
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="ibvs-arm", daemon=True)
            self._thread.start()
        self.phase = "returning_ready"
        if self._motion_backend == "service":
            self.message = "Reset：运控服务正在回到放下位"
        else:
            self.message = "Reset：正在回到放下位，arm_sdk 保持"
        self.log(f"[ARM] {self.message}")
        return self.message

    def hang_arms(self) -> str:
        """Hang both arms beside the body through the service. This page does not publish arm_sdk."""
        self._stop_playback()
        if self._motion_backend != "service":
            self.message = "自然下垂只走运控服务。当前是 arm_sdk，已拒绝。"
            self.log(f"[ARM][WARN] {self.message}")
            return self.message
        conflict = self._refuse_conflict("service")
        if conflict:
            return conflict
        if not self.confirm:
            self.message = "启动网页时要加 --confirm-arm-motion 才会下垂"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._ibvs_enabled = False
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._control_stage = "HANG"
        self._pbvs_deadline = None
        self._return_ready.clear()
        self._level_right.clear()
        self._hang_down.set()
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="ibvs-arm", daemon=True)
            self._thread.start()
        self.phase = "hanging"
        self.message = "下垂：运控服务将双臂收到身体两侧，本页不发 arm_sdk"
        self.log(f"[ARM] {self.message}")
        return self.message

    def scan(self) -> str:
        self._stop_playback()
        conflict = self._refuse_conflict()
        if conflict:
            return conflict
        if not self.confirm:
            self.message = "启动网页时要加 --confirm-arm-motion 才会抬臂"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._ibvs_enabled = False
        self._control_stage = "SCAN"
        self._level_right.clear()
        self._scanning.set()
        self._scan_needs_prepare = True
        self._height_locked = False
        self._seek_done = False
        self._seek_steps = 0
        self._sweep_direction = -1
        self._sweep_turns = 0
        self._ready_streak = 0
        self._running.set()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="ibvs-arm", daemon=True)
            self._thread.start()
        self.message = "Raise：双臂只到固定预备位，不再自动向上扫描"
        return self.message

    def level_right_arm(self) -> str:
        """Raise only the right forearm to horizontal, so the wrist camera can be checked."""
        self._stop_playback()
        conflict = self._refuse_conflict()
        if conflict:
            return conflict
        if not self.confirm:
            self.message = "启动网页时要加 --confirm-arm-motion 才会平举右臂"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._ibvs_enabled = False
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._return_ready.clear()
        self._hang_down.clear()
        self._control_stage = "LEVEL_RIGHT"
        self._pbvs_deadline = None
        self._level_right.set()
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="ibvs-arm", daemon=True)
            self._thread.start()
        self.phase = "leveling_right"
        self.message = "右臂平举：只抬右臂，前臂水平朝前，左臂不动"
        self.log(f"[ARM] {self.message}")
        return self.message

    def set_button_sequence(self, steps: Sequence[str]) -> list[str]:
        """Ordered presses. The same button may appear more than once."""
        cleaned: list[str] = []
        for step in steps or []:
            name = str(step or "").strip()
            if name in BUTTON_NAMES:
                cleaned.append(name)
        self._button_sequence = cleaned
        return list(cleaned)

    def start_sequence(self) -> str:
        """PBVS the queued buttons. An empty queue is refused before any motion."""
        return self.start(sequence=True)

    def start(self, *, sequence: bool = False) -> str:
        self._sequence_run = False
        if sequence and not self._button_sequence:
            self.message = "动作组合为空，请先添加 Button 再点组合启动"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._stop_playback()
        conflict = self._refuse_conflict()
        if conflict:
            return conflict
        if not self.confirm:
            self.message = "启动网页时要加 --confirm-arm-motion"
            return self.message
        if (self.direct_ee_poke or self.direct_pbvs_poke or self.pbvs_first) and (
            self.controller is None or self._thread is None or not self._thread.is_alive()
        ):
            self._stop.clear()
            self._scan_needs_prepare = False
            self._scanning.clear()
            self._thread = threading.Thread(target=self._loop, name="direct-ee-arm", daemon=True)
            self._thread.start()
            deadline = time.monotonic() + 8.0
            while self.controller is None and self._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
        if self.controller is None or self._thread is None or not self._thread.is_alive():
            self.message = (
                "Direct EE 接管失败"
                if self.direct_ee_poke
                else (
                    "PBVS 接管失败"
                    if self.pbvs_first or self.direct_pbvs_poke
                    else "请先点 Raise/Scan 举臂"
                )
            )
            return self.message
        # Start always freezes the current scan pose first. It must never leave
        # the arm oscillating merely because one visual condition is missing.
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._height_locked = True
        self._seek_done = True
        self.phase = "hold_for_start"
        telem = self.telemetry() or {}
        self._publish_ee()
        telem = self.telemetry() or telem
        # Near the panel the chest/wrist views are already washed out. Start
        # must still reach from this pose; waiting for features leaves the
        # same short hold as the last two runs.
        if self.direct_ee_poke or self.direct_pbvs_poke or self.pbvs_first:
            target_torso = telem.get("target_torso_m")
            if not target_torso or len(target_torso) != 3:
                self._ibvs_enabled = False
                self._control_stage = "HOLD"
                self.message = "PBVS 需要眼在手外标定和胸视 RGB-D 按键三维坐标"
                return self.message
        self._ibvs_enabled = True
        self._last_visible_q = None
        self._poke_remaining_m = 0.0
        self._last_camera_delta = None
        self._last_tip_v = None
        self._awaiting_raise_v = None
        self._up_stuck = 0
        self._direct_ee_align_steps = 0
        self._direct_ee_orient_steps = 0
        self._direct_ee_approach_steps = 0
        self._pbvs_deadline = None
        self._service_pbvs_sent = False
        self._pbvs_release = True
        self._pbvs_wrist_checked = False
        self._pbvs_source = ""
        self._goal_name = ""
        self._locked_buttons = {}
        self._sequence_run = bool(sequence)
        self._reset_left_seek()
        self._running.set()
        if sequence:
            labels = {"button_1": "Button 1", "button_2": "Button 2", "button_3": "Button 3"}
            names = " → ".join(labels.get(name, name) for name in self._button_sequence)
            self._control_stage = "DIRECT_PBVS"
            self.message = f"组合启动：{names}"
            return self.message
        if self.direct_ee_poke:
            self._control_stage = "DIRECT_EE_ALIGN_YZ"
            self.message = "Start：先在面板前 15cm 对齐 Y/Z，再沿 X 点触"
        elif self.direct_pbvs_poke:
            self._control_stage = "DIRECT_PBVS"
            self.message = "Start：Direct PBVS 交给 /h2_arm/singlearmcartesian"
        elif self.pbvs_first:
            self._control_stage = "PBVS_COARSE"
            self.message = "Start：眼在手外 PBVS 一次到位，随后抬左臂并切换双视 IBVS"
        else:
            self._control_stage = "DUAL_IBVS"
            self.message = "Start：胸视 IBVS 对准左右上下，左手侧视决定前进和接触"
        return self.message

    def pause(self) -> str:
        self._stop_playback()
        self._ibvs_enabled = False
        self._control_stage = "MANUAL"
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._pbvs_deadline = None
        self._reset_left_seek()
        self.phase = "manual"
        self.message = "Pause：冻结当前位置，已进入右臂六轴细调"
        self.log(f"[ARM] {self.message}")
        return self.message

    def stop(self) -> str:
        """Compatibility alias. The web Stop/Pause action keeps arm_sdk holding."""
        return self.pause()

    def start_playback(
        self,
        trajectory_id: str,
        samples: list[dict[str, Any]],
        speed: float = 2.0,
    ) -> dict[str, Any]:
        if not self.confirm:
            return {"ok": False, "error": "arm motion is not confirmed"}
        conflict = self._refuse_conflict("arm_sdk")
        if conflict:
            return {"ok": False, "error": conflict}
        if len(samples) < 2:
            return {"ok": False, "error": "trajectory has fewer than two samples"}
        self._stop_playback()
        if self.controller is None or self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._scan_needs_prepare = False
            self._scanning.clear()
            self._thread = threading.Thread(target=self._loop, name="playback-arm", daemon=True)
            self._thread.start()
            deadline = time.monotonic() + 8.0
            while self.controller is None and self._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
        if self.controller is None:
            return {"ok": False, "error": "arm_sdk connection failed"}
        self._ibvs_enabled = False
        self._running.clear()
        self._control_stage = "PLAYBACK"
        self.phase = "playback"
        self._playback_id = str(trajectory_id)
        self._playback_speed = max(1.0, min(4.0, float(speed)))
        self._playback_stop.clear()
        self.message = f"轨迹回放 {self._playback_id} ×{self._playback_speed:.1f}"
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(list(samples),),
            name="motion-trajectory-playback",
            daemon=True,
        )
        self._playback_thread.start()
        return {
            "ok": True,
            "id": self._playback_id,
            "speed": self._playback_speed,
            "samples": len(samples),
        }

    def _stop_playback(self) -> None:
        self._playback_stop.set()
        thread = self._playback_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread is not None and not thread.is_alive():
            self._playback_thread = None

    def _playback_loop(self, samples: list[dict[str, Any]]) -> None:
        conflict = self._refuse_conflict("arm_sdk")
        if conflict:
            self.phase = "playback_error"
            self._control_stage = "HOLD"
            self.message = conflict
            self._playback_stop.set()
            return
        try:
            current = self.controller.hold_positions()
            first = self._trajectory_command(samples[0], current)
            ramp_s = max(0.4, 1.5 / self._playback_speed)
            steps = max(1, int(ramp_s / 0.02))
            for step in range(1, steps + 1):
                if self._playback_stop.is_set():
                    return
                ratio = step / steps
                command = {
                    joint: float(current[joint]) * (1.0 - ratio) + float(first.get(joint, current[joint])) * ratio
                    for joint in current
                }
                self.controller.write(command, weight=1.0)
                time.sleep(0.02)
            first_t = float(samples[0].get("t") or 0.0)
            started = time.monotonic()
            last_command = first
            for sample in samples[1:]:
                if self._playback_stop.is_set():
                    return
                due = max(0.0, float(sample.get("t") or first_t) - first_t) / self._playback_speed
                while not self._playback_stop.is_set() and time.monotonic() - started < due:
                    self._playback_stop.wait(min(0.02, due - (time.monotonic() - started)))
                if self._playback_stop.is_set():
                    return
                last_command = self._trajectory_command(sample, last_command)
                self.controller.write(last_command, weight=1.0)
            self.phase = "playback_complete"
            self._control_stage = "HOLD"
            self.message = f"轨迹回放完成 {self._playback_id} ×{self._playback_speed:.1f}"
        except Exception as exc:
            self.phase = "playback_error"
            self._control_stage = "HOLD"
            self.message = f"轨迹回放失败: {exc}"
            self.log(f"[ARM][WARN] {self.message}")
        finally:
            self._playback_stop.set()

    @staticmethod
    def _trajectory_command(
        sample: dict[str, Any],
        base: dict[int, float],
    ) -> dict[int, float]:
        command = dict(base)
        for joint in sample.get("joints") or []:
            index = TRAJECTORY_JOINT_INDEX.get(str(joint.get("name") or ""))
            if index is None:
                continue
            value = joint.get("q")
            if value is None and joint.get("q_deg") is not None:
                value = math.radians(float(joint["q_deg"]))
            if value is not None:
                command[index] = float(value)
        return command

    def manual_jog(
        self,
        dxyz_m: tuple[float, float, float],
        drpy_deg: tuple[float, float, float],
    ) -> dict[str, Any]:
        if not self.confirm:
            return {"ok": False, "error": "arm motion is not confirmed"}
        conflict = self._refuse_conflict()
        if conflict:
            return {"ok": False, "error": conflict}
        if self.controller is None or self.ik is None:
            return {"ok": False, "error": "请先 Raise"}
        if self._control_stage != "MANUAL":
            return {"ok": False, "error": "请先点 Pause 进入细调"}
        dxyz = tuple(float(value) for value in dxyz_m)
        drpy = tuple(math.radians(float(value)) for value in drpy_deg)
        if max(abs(value) for value in dxyz) > 0.010:
            return {"ok": False, "error": "单次平移不得超过 10mm"}
        if max(abs(value) for value in drpy) > math.radians(5.0):
            return {"ok": False, "error": "单次旋转不得超过 5deg"}
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        q = commanded or measured
        if not q or not measured:
            return {"ok": False, "error": "没有右臂关节状态"}
        apply_loaded = getattr(self.ik, "apply_loaded_delta", None)
        if commanded and callable(apply_loaded):
            q_new, ok, pos_err, ori_err = apply_loaded(
                commanded,
                measured,
                dxyz=dxyz,
                drpy=drpy,
                rotation_frame="base",
            )
        else:
            q_new, ok, pos_err, ori_err = self.ik.apply_delta(
                q,
                dxyz=dxyz,
                drpy=drpy,
                rotation_frame="base",
            )
        jump = max(abs(float(after) - float(before)) for before, after in zip(q, q_new))
        if not ok or jump > math.radians(10.0):
            self.message = f"细调 IK 拒绝 pos={pos_err*1000:.1f}mm rot={math.degrees(ori_err):.1f}deg"
            return {"ok": False, "error": self.message}
        self.controller.apply_arm_rad("right", q_new, seconds=0.35, dt=0.02)
        self.phase = "manual"
        self.message = (
            "细调 "
            f"dxyz_mm=({dxyz[0]*1000:.1f},{dxyz[1]*1000:.1f},{dxyz[2]*1000:.1f}) "
            f"drpy_deg=({drpy_deg[0]:.1f},{drpy_deg[1]:.1f},{drpy_deg[2]:.1f})"
        )
        self._publish_ee()
        return {"ok": True, "message": self.message}

    def manual_elbow(self, delta_deg: float) -> dict[str, Any]:
        """Nudge only the right elbow. Cartesian jog cannot isolate this joint."""
        if not self.confirm:
            return {"ok": False, "error": "arm motion is not confirmed"}
        conflict = self._refuse_conflict()
        if conflict:
            return {"ok": False, "error": conflict}
        if self.controller is None:
            return {"ok": False, "error": "请先 Raise"}
        if self._control_stage != "MANUAL":
            return {"ok": False, "error": "请先点 Pause 进入细调"}
        try:
            delta = float(delta_deg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "肘角度无效"}
        if not math.isfinite(delta) or abs(delta) > 5.0 or abs(delta) < 1e-6:
            return {"ok": False, "error": "单次肘调节不得超过 5deg"}
        commanded = self.controller.commanded_arm_rad("right")
        if not commanded:
            return {"ok": False, "error": "没有右臂关节状态"}
        q_new = [float(value) for value in commanded]
        low, high = RIGHT_ELBOW_LIMIT_RAD
        q_new[RIGHT_ELBOW_INDEX] = min(
            high,
            max(low, q_new[RIGHT_ELBOW_INDEX] + math.radians(delta)),
        )
        if abs(q_new[RIGHT_ELBOW_INDEX] - float(commanded[RIGHT_ELBOW_INDEX])) < 1e-5:
            return {"ok": False, "error": "右肘已到限位"}
        self.controller.apply_arm_rad("right", q_new, seconds=0.35, dt=0.02)
        elbow_deg = math.degrees(q_new[RIGHT_ELBOW_INDEX])
        self.phase = "manual"
        self.message = f"右肘 {elbow_deg:.1f}°"
        self._publish_ee()
        return {"ok": True, "message": self.message, "elbow_deg": round(elbow_deg, 2)}

    def release_for_operator(self) -> str:
        """Drop arm_sdk on both arms. Cameras keep running for the motion engineer."""
        self._stop_playback()
        if self.controller is None and (self._thread is None or not self._thread.is_alive()):
            self.phase = "released"
            self._control_stage = "RELEASED"
            self._left_side_ready = False
            self.message = "双臂已释放。再点 arm_sdk 会先停止运控服务，然后本页接管。"
            return self.message
        self._ibvs_enabled = False
        self._control_stage = "RELEASED"
        self._left_side_ready = False
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._height_locked = False
        self._seek_done = False
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self._thread = None
        controller = self.controller
        self.controller = None
        if controller is not None:
            try:
                self.log("[ARM] 释放：本进程不占用 arm_sdk，运控服务继续持有")
                controller.release(seconds=1.2)
            except Exception as exc:
                self.log(f"[ARM] 松权失败: {exc}")
                try:
                    controller.release(seconds=0.8)
                except Exception as release_exc:
                    self.log(f"[ARM] release: {release_exc}")
        self.phase = "released"
        self.message = "已释放双臂。再点 arm_sdk 会先停止运控服务，然后本页接管。"
        self.log(f"[ARM] {self.message}")
        return self.message

    def shutdown(self) -> None:
        self._stop_playback()
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        self._ibvs_enabled = False
        self._control_stage = "SHUTDOWN"
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
        controller = self.controller
        if controller is None:
            return
        if self._sdk_holding() and self._service_node_up():
            self.log("[ARM][WARN] 退出时运控服务仍占用 rt/arm_sdk，本页先松开，再由服务下垂")
            try:
                controller.release(seconds=0.4)
            except Exception as exc:
                self.log(f"[ARM] 松权失败: {exc}")
            self.controller = None
            try:
                from ibvs_pad_touch.web.arm_service import H2ArmServiceController

                service = H2ArmServiceController()
                service.ramp_arm("left", LEFT_HANG_DEG, seconds=8.0, dt=0.02)
                service.ramp_arm("right", RIGHT_HANG_DEG, seconds=8.0, dt=0.02)
                self.phase = "released"
                self.message = "已松开 arm_sdk，并由运控服务下垂"
                self.log("[ARM] 已由运控服务下垂")
            except Exception as exc:
                self.log(f"[ARM] 服务下垂失败: {exc}")
            return
        try:
            self.log("[ARM] 退出：放下左右臂并交回 arm_sdk")
            controller.ramp_arm("left", LEFT_REST_DEG, seconds=6.0, dt=0.02)
            controller.ramp_arm("right", REST_DEG, seconds=6.0, dt=0.02)
            controller.release(seconds=1.2)
            self.phase = "released"
            self.message = "已放下并交回 arm_sdk"
            self.log("[ARM] 已放下")
        except Exception as exc:
            self.log(f"[ARM] 放下失败，改松权: {exc}")
            try:
                controller.release(seconds=0.8)
            except Exception as release_exc:
                self.log(f"[ARM] release: {release_exc}")
        finally:
            self.controller = None

    def _connect(self) -> None:
        conflict = self._refuse_conflict()
        if conflict:
            raise RuntimeError(conflict)
        if self.controller is not None:
            return
        ensure_handeye_calib()
        from handeye_calib.h2_ee_ik import H2CameraIK

        if self._motion_backend == "service":
            from ibvs_pad_touch.web.arm_service import H2ArmServiceController

            self.controller = H2ArmServiceController()
            self.ik = H2CameraIK(arm="right")
            self.log("[ARM] 运控服务已接，本进程不发 arm_sdk")
            self._publish_ee()
            return
        from ibvs_pad_touch.web.dds_bind import ensure_unitree_sdk2py

        ensure_unitree_sdk2py()
        from handeye_calib.h2_arm_sdk import H2ArmSdkController

        try:
            self.controller = H2ArmSdkController(
                iface=self.iface,
                kp=85.0,
                kd=2.2,
                gravity=True,
                tau_scale=1.0,
                wrist_payload_kg=0.75,
            )
        except TypeError:
            self.controller = H2ArmSdkController(iface=self.iface)
            self.controller.kp = 85.0
            self.controller.kd = 2.2
        self.controller.wait_lowstate()
        self._ensure_gravity()
        self.controller.ensure_ai_mode()
        self.controller.enable_arm_sdk()
        self.ik = H2CameraIK(arm="right")
        self.log("[ARM] arm_sdk 已接，重力补偿开")
        self._publish_ee()

    def _ensure_gravity(self) -> None:
        from handeye_calib.h2_gravity import H2GravityCompensator, format_tau_summary

        if getattr(self.controller, "_gravity", None) is None:
            self.log("[ARM][WARN] 控制器没挂上重力补偿，现场补挂")
            self.controller._gravity = H2GravityCompensator(
                wrist_payload_kg=0.75,
                payload_arm="right",
                tau_scale=1.0,
            )
            self.controller._format_tau = format_tau_summary
        self._add_left_camera_payload()
        q = {22 + i: math.radians(READY_DEG[i]) for i in range(7)}
        q.update({15 + i: math.radians(LEFT_SIDE_VIEW_DEG[i]) for i in range(7)})
        tau = self.controller._gravity.motor_tau(q)
        shoulder = float(tau.get(22, 0.0))
        elbow = float(tau.get(25, 0.0))
        left_shoulder = float(tau.get(16, 0.0))
        self.log(
            f"[ARM] 重力补偿 tau READY  Rshoulder={shoulder:+.2f}Nm Relbow={elbow:+.2f}Nm "
            f"Lroll={left_shoulder:+.2f}Nm"
        )
        if abs(shoulder) < 2.0:
            raise RuntimeError(
                f"重力补偿异常：肩关节 tau={shoulder:.2f} Nm。"
                "请同步 Hand_Eye_Calib 和 /home/unitree/MscapeTech/urdf/H2.urdf"
            )

    def _add_left_camera_payload(self) -> None:
        gravity = getattr(self.controller, "_gravity", None)
        if gravity is None or not hasattr(gravity, "_add_payload"):
            return
        gravity._add_payload(
            "left_wrist_yaw_link",
            LEFT_CAMERA_PAYLOAD_KG,
            (0.05, 0.0, 0.0),
        )

    def _loop(self) -> None:
        try:
            self._connect()
            self._prepare_scan_pose()
            if self._scanning.is_set():
                self._seek_until_panel()
        except Exception as exc:
            self.phase = "error"
            self.message = f"抬臂失败: {exc}"
            self.log(f"[ARM][ERROR] {exc}")
            self._running.clear()
            return
        while not self._stop.is_set():
            if self._level_right.is_set():
                try:
                    self._execute_level_right()
                except Exception as exc:
                    self.phase = "error"
                    self.message = f"右臂平举失败: {exc}"
                    self.log(f"[ARM][ERROR] {exc}")
                    self._level_right.clear()
                continue
            if self._hang_down.is_set():
                try:
                    self._execute_hang()
                except Exception as exc:
                    self.phase = "error"
                    self.message = f"下垂失败: {exc}"
                    self.log(f"[ARM][ERROR] {exc}")
                    self._hang_down.clear()
                continue
            if self._return_ready.is_set():
                try:
                    self._execute_return_ready()
                except Exception as exc:
                    self.phase = "error"
                    self.message = f"回放下位失败: {exc}"
                    self.log(f"[ARM][ERROR] {exc}")
                    self._return_ready.clear()
                continue
            if not self._running.is_set():
                self._publish_ee()
                time.sleep(0.05)
                continue
            try:
                self._publish_ee()
                self._servo_once()
            except Exception as exc:
                self.message = f"伺服: {exc}"
                self.log(f"[ARM][WARN] {exc}")
            time.sleep(0.08)

    def _prepare_scan_pose(self) -> None:
        if not self._scan_needs_prepare:
            return
        if self._refuse_conflict():
            self._scan_needs_prepare = False
            self._scanning.clear()
            self._running.clear()
            return
        self.phase = "raising"
        self.message = "左臂平举侧视，右臂举到点触位"
        self.log("[ARM] 左臂平举、相机朝右看点触；右臂外展举到工作位")
        self.controller.ramp_arm("left", LEFT_SIDE_VIEW_DEG, seconds=8.0, dt=0.02)
        self._left_side_ready = True
        self.controller.ramp_arm("right", READY_DEG, seconds=8.0, dt=0.02)
        self._warn_if_sagging()
        self._scan_needs_prepare = False
        self._scanning.clear()
        self._height_locked = True
        self._seek_done = True
        self.phase = "hold_for_start"
        self.message = "双臂固定预备位；确认胸视夹爪和侧视间隙后再 Start"
        self.log("[ARM] Raise 到固定预备位；禁用自动上下扫描")

    def _execute_return_ready(self) -> None:
        """Same lowered pose as Ctrl+C. The active link stays held."""
        self._return_ready.clear()
        if self._refuse_conflict():
            self._running.clear()
            self._ibvs_enabled = False
            return
        self._ibvs_enabled = False
        self._running.clear()
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._left_side_ready = False
        self._control_stage = "RESET"
        self.phase = "returning_ready"
        via_service = self._motion_backend == "service"
        self.message = "Reset：运控服务回到放下位" if via_service else "Reset：回到放下位，arm_sdk 保持"
        self.log("[ARM] Reset 回 Ctrl+C 放下位，不释放当前链路")
        self.controller.ramp_arm("left", LEFT_REST_DEG, seconds=6.0, dt=0.02)
        self.controller.ramp_arm("right", REST_DEG, seconds=6.0, dt=0.02)
        self._height_locked = False
        self._seek_done = False
        self._control_stage = "HOLD"
        self.phase = "hold_for_start"
        self.message = "已回到放下位，运控服务仍保持" if via_service else "已回到放下位，arm_sdk 仍保持"
        self.log(f"[ARM] {self.message}")

    def _ramp_right_level(self) -> None:
        current = None
        try:
            measured = self.controller.measured_arm_deg("right")
            if measured and len(measured) == 7:
                current = [float(value) for value in measured]
        except Exception:
            current = None
        poses = level_right_waypoints(current)
        seconds = 8.0 if len(poses) == 1 else 4.0
        for pose in poses:
            if self._return_ready.is_set() or self._stop.is_set():
                return
            self.controller.ramp_arm("right", pose, seconds=seconds, dt=0.02)

    def _execute_level_right(self) -> None:
        """Level the right forearm forward. The left arm is not commanded."""
        self._level_right.clear()
        if self._refuse_conflict():
            self._running.clear()
            self._ibvs_enabled = False
            return
        self._ibvs_enabled = False
        self._running.clear()
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._control_stage = "LEVEL_RIGHT"
        self.phase = "leveling_right"
        self.message = "右臂平举：先向外避开胸口相机，再朝前抬"
        self.log("[ARM] 只平举右臂，左臂保持原位，不收到胸口相机后面")
        self._ramp_right_level()
        self._height_locked = False
        self._seek_done = False
        self._control_stage = "HOLD"
        self.phase = "hold_for_start"
        self.message = "右臂已平举，左臂未动。可以看右腕相机视角"
        self.log(f"[ARM] {self.message}")

    def _execute_hang(self) -> None:
        """Hang both forearms beside the thighs. Does not enable arm_sdk here."""
        self._hang_down.clear()
        if self._refuse_conflict("service"):
            self._running.clear()
            self._ibvs_enabled = False
            return
        self._ibvs_enabled = False
        self._running.clear()
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._left_side_ready = False
        self._control_stage = "HANG"
        self.phase = "hanging"
        self.message = "下垂：双臂收到身体两侧"
        self.log("[ARM] 运控服务下垂到身体两侧")
        self.controller.ramp_arm("left", LEFT_HANG_DEG, seconds=8.0, dt=0.02)
        self.controller.ramp_arm("right", RIGHT_HANG_DEG, seconds=8.0, dt=0.02)
        self._height_locked = False
        self._seek_done = False
        self._control_stage = "HOLD"
        self.phase = "hold_for_start"
        self.message = "双臂已自然下垂。再点 arm_sdk 会先停止运控服务，然后本页接管。"
        self.log("[ARM] 下垂完成，本页未发 arm_sdk")

    def _warn_if_sagging(self) -> None:
        try:
            cmd = [math.degrees(v) for v in (self.controller.commanded_arm_rad("right") or [])]
            meas = list(self.controller.measured_arm_deg("right"))
        except Exception:
            return
        if not cmd or not meas:
            return
        sag = meas[0] - cmd[0]
        self.log(f"[ARM] 肩关节 cmd={cmd[0]:+.1f}° meas={meas[0]:+.1f}° sag={sag:+.1f}°")
        if sag > 12.0:
            self.log("[ARM][WARN] 抬不到指令高度，重力补偿可能偏弱")

    def _publish_ee(self) -> None:
        if self.ik is None or self.controller is None:
            return
        try:
            commanded = self.controller.commanded_arm_rad("right")
            measured = self.controller.measured_arm_rad("right")
            display_q = measured or commanded
            if display_q:
                self.update_joint_readback(display_q)
            if self.on_ee_pose is None:
                return
            # Eye-to-hand PBVS must use the physical measured pose. Publishing
            # the arm_sdk target here previously made FK disagree with the jaw.
            q = measured or commanded
            if not q:
                return
            xyz, rpy = self.ik.fk_xyz_rpy(q)
            self.on_ee_pose([float(v) for v in xyz], [float(v) for v in rpy])
        except Exception as exc:
            self.log(f"[ARM][WARN] FK 虚拟尖端失败: {exc}")

    def update_joint_readback(self, q_rad) -> None:
        """Refresh wrist/right_ee pose from read-only DDS joints, even after Release."""
        if self.ik is None or not q_rad or len(q_rad) != 7:
            return
        try:
            xyz, rpy = self.ik.fk_xyz_rpy(q_rad)
            tip_fn = getattr(self.ik, "tip_xyz", None)
            tool_xyz = (
                tip_fn(q_rad, RIGHT_EE_OFFSET_WRIST_M)
                if callable(tip_fn)
                else xyz
            )
        except Exception as exc:
            self.log(f"[ARM][WARN] readback FK 失败: {exc}")
            return
        with self._lock:
            self._ee_xyz_m = [float(value) for value in xyz]
            self._ee_rpy_deg = [math.degrees(float(value)) for value in rpy]
            self._right_ee_xyz_m = [float(value) for value in tool_xyz]
            self._right_ee_rpy_deg = [math.degrees(float(value)) for value in rpy]

    def _current_arm_deg(self) -> list[float]:
        q = self.controller.commanded_arm_rad("right")
        if q:
            return [math.degrees(v) for v in q]
        return list(self.controller.measured_arm_deg("right"))

    def _sweep_a_bit(self) -> bool:
        try:
            measured = self.controller.measured_arm_deg("right")
        except Exception:
            measured = []
        if (
            self._sweep_direction < 0
            and measured
            and float(measured[0]) <= MEASURED_HIGH_SHOULDER_DEG
        ):
            self._sweep_direction = 1
            self._sweep_turns += 1
            self.log(
                "[ARM] 实测肩 pitch 到安全高位 "
                f"{float(measured[0]):.1f}°，转向向下"
            )
        planned = next_sweep_joints_deg(
            self._current_arm_deg(),
            self._sweep_direction,
        )
        if planned is None:
            return False
        nxt, self._sweep_direction, turned = planned
        if turned:
            self._sweep_turns += 1
            self.log(
                "[ARM] 扫描到边界，转向"
                f" {'向上' if self._sweep_direction < 0 else '向下'}"
            )
        self.controller.ramp_arm("right", nxt, seconds=SEEK_RAMP_S, dt=0.02)
        return True

    def _measured_shoulder_at_high(self) -> bool:
        try:
            measured = self.controller.measured_arm_deg("right")
        except Exception:
            return False
        return bool(measured) and float(measured[0]) <= MEASURED_HIGH_SHOULDER_DEG

    def _seek_until_panel(self) -> None:
        """Move the right arm until the chest tip sits on the button row."""
        while self._running.is_set() and self._scanning.is_set() and not self._stop.is_set():
            telem = self.telemetry() or {}
            if scan_lock_ready(telem):
                self._ready_streak += 1
                if self._ready_streak >= FEATURE_READY_FRAMES:
                    self._lock_height("胸视指尖接近按键高度，停止")
                    self.log("[ARM] 胸视指尖接近按键高度，不再改高度")
                    return
                time.sleep(0.08)
                continue
            self._ready_streak = 0
            direction = raise_direction(telem)
            if direction < 0 and self._measured_shoulder_at_high():
                self._lock_height("右臂已到安全高位，可 Start")
                self.log("[ARM] 实测肩 pitch 到安全高位，停住")
                return
            self._sweep_direction = direction
            if self._seek_steps >= SWEEP_MAX_STEPS or not self._sweep_a_bit():
                self._seek_done = True
                self._scanning.clear()
                self.phase = "sweep_complete"
                self.message = "高度调整完成，可 Start；胸视 IBVS，左手侧视接触"
                self.log("[ARM] 高度调整完成，胸视 IBVS Start")
                return
            self._seek_steps += 1
            self.phase = "sweep_height"
            verb = "降低" if direction > 0 else "抬高"
            self.message = f"{verb} {self._seek_steps}/{SWEEP_MAX_STEPS}"
            self.log(f"[ARM] height step={self._seek_steps} {verb}")
        self._seek_done = True
        self._scanning.clear()

    def _lock_height(self, message: str) -> None:
        self._scanning.clear()
        self._height_locked = True
        self._seek_done = True
        self.phase = "panel_seen"
        self.message = message

    def _servo_once(self) -> None:
        if self._refuse_conflict():
            self._ibvs_enabled = False
            self._running.clear()
            self._scanning.clear()
            return
        telem = self.telemetry() or {}
        if self._scanning.is_set() and self._scan_needs_prepare:
            self._prepare_scan_pose()
            if self._scanning.is_set():
                self._seek_until_panel()
            return
        if not self._height_locked and not self._seek_done:
            self._seek_until_panel()
            return
        if not self._ibvs_enabled:
            if self.phase in {
                "direct_pbvs_rejected",
                "direct_pbvs_complete",
                "direct_pbvs_short",
                "direct_ee_rejected",
                "direct_ee_complete",
            }:
                return
            self.phase = "hold_for_start"
            self.message = "双臂就位，可 Start；胸视 IBVS，左手侧视接触"
            return
        if self._control_stage in {
            "DIRECT_EE_ALIGN_YZ",
            "DIRECT_EE_ORIENT_SAFE",
            "DIRECT_EE_APPROACH_X",
        }:
            self._direct_ee_once(telem)
            return
        if self._control_stage == "RAISE_LEFT_FOR_IBVS":
            self._raise_left_for_ibvs()
            return
        if self._control_stage in {"DIRECT_PBVS", "PBVS_COARSE"} and not self._pbvs_wrist_checked:
            self._pbvs_wrist_checked = True
            self._fuse_wrist_pbvs(telem)
            if (
                self._sequence_run
                and self._button_sequence
                and not self._return_ready.is_set()
                and not self._stop.is_set()
            ):
                self._run_button_sequence()
                return
            telem = self.telemetry() or telem
        if self._control_stage == "DIRECT_PBVS":
            self._direct_pbvs_once(telem)
            return
        if self._control_stage == "PBVS_COARSE":
            self._pbvs_once(telem)
            return
        if self._control_stage == "DUAL_IBVS":
            self._dual_once(telem)
            return
        if not should_jog_ibvs(telem, self._height_locked):
            if self._can_poke():
                self._poke_once("vision_gap")
                return
            self.phase = "hold_wait_features"
            self.message = "等待按钮或 FK；近距不再回退"
            return
        phase = str(telem.get("phase") or "")
        if phase in {"SEEK_PANEL", "LOST", ""}:
            if self._can_poke():
                self._poke_once("lost_panel")
                return
            self.phase = "ready_wait_panel"
            self.message = "在等按钮特征；不回退"
            return
        dx = float(telem.get("dx") or 0.0)
        dy = float(telem.get("dy") or 0.0)
        dz = float(telem.get("dz") or 0.0)
        if max(abs(dx), abs(dy), abs(dz)) < 1e-4:
            if self._can_poke():
                self._poke_once("close_gap")
                return
            self.phase = phase
            self.message = "误差很小，点触行程已用完，保持"
            return
        self._apply_camera_delta((dx, dy, dz), seconds=0.45, limit_m=JOG_STEP_M, phase=phase)

    def _reset_left_seek(self) -> None:
        self._left_seek_origin = None
        self._left_seek_joint = 0
        self._left_seek_dir = 1
        self._left_seek_offset = 0.0

    def _step_left_gripper_seek(self) -> str:
        """Move one left joint. The right arm is not commanded here."""
        controller = self.controller
        if controller is None or not hasattr(controller, "ramp_arm"):
            return "看不到夹爪，右臂保持；左臂控制器不可用"
        rad = None
        if hasattr(controller, "commanded_arm_rad"):
            rad = controller.commanded_arm_rad("left")
        if not rad and hasattr(controller, "measured_arm_rad"):
            rad = controller.measured_arm_rad("left")
        if not rad or len(rad) != 7:
            return "看不到夹爪，右臂保持；没有左臂关节"
        if self._left_seek_origin is None:
            self._left_seek_origin = [math.degrees(float(value)) for value in rad]
        pose, joint, direction, offset = next_left_gripper_seek(
            self._left_seek_origin,
            self._left_seek_joint,
            self._left_seek_dir,
            self._left_seek_offset,
        )
        self._left_seek_joint = joint
        self._left_seek_dir = direction
        self._left_seek_offset = offset
        controller.ramp_arm("left", pose, seconds=LEFT_SEEK_RAMP_S, dt=0.02)
        label = LEFT_SEEK_LABELS[LEFT_SEEK_JOINTS[joint]]
        return f"看不到夹爪，右臂保持，左{label} {offset:+.0f}°"

    def _tip_torso_error(self, telem: dict[str, Any]) -> tuple[float, float, float] | None:
        raw = telem.get("pbvs_error_torso_m")
        if not raw or len(raw) != 3:
            return None
        try:
            error = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(error) != 3 or not all(math.isfinite(value) for value in error):
            return None
        return error

    def _close_tip_gap(self, telem: dict[str, Any]) -> bool:
        """Move the right arm by the 3D tip-to-button gap. Image overlap is not arrival."""
        error = self._tip_torso_error(telem)
        if error is None:
            return False
        distance = math.sqrt(sum(value * value for value in error))
        if distance <= DIRECT_PBVS_ARRIVE_TOL_M:
            return False
        self._apply_torso_delta(
            error,
            seconds=max(0.45, min(2.0, distance / 0.15)),
            phase="close_tip",
            limit_m=min(0.04, distance),
        )
        if self.message.startswith("IK 拒绝") or self.message.startswith("IBVS 服务失败"):
            self.message = f"尖端未到按键，剩余 {distance*1000:.0f}mm。{self.message}"
        else:
            self.message = f"尖端未到按键，按三维差靠近，剩余 {distance*1000:.0f}mm"
        self.log(f"[ARM] {self.message}")
        return True

    def _dual_once(self, telem: dict[str, Any]) -> None:
        """Chest image X/Y, then chest +Z only while the left camera still sees a gap."""
        if str(telem.get("phase") or "") == "CONTACT":
            self._reset_left_seek()
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "contact"
            self.message = "左手侧视：已接触按键，停止"
            self.log("[ARM] side contact, stop")
            return
        lost_chest = chest_gripper_missing(telem)
        lost_side = side_gripper_missing(telem)
        if lost_chest or lost_side:
            if lost_chest and self._close_tip_gap(telem):
                return
            self.phase = "seek_left_angle"
            self.message = self._step_left_gripper_seek()
            self.log(f"[ARM] {self.message}")
            if lost_chest:
                return
        else:
            self._reset_left_seek()
        dx = float(telem.get("dx") or 0.0)
        dy = float(telem.get("dy") or 0.0)
        dz = 0.0 if lost_side else float(telem.get("dz") or 0.0)
        self._settle_raise(telem.get("tip_uv"))
        dy, blocked = self._limit_upward(dy)
        if max(abs(dx), abs(dy), abs(dz)) < 1e-4:
            if self._close_tip_gap(telem):
                return
            if not lost_side:
                self.phase = str(telem.get("phase") or "dual_wait")
                self.message = self._up_block_message if blocked else str(telem.get("message") or "胸视对准，或等待左手侧视")
            return
        raised = self._apply_camera_delta(
            (dx, dy, dz),
            seconds=0.45,
            limit_m=JOG_STEP_M,
            phase=str(telem.get("phase") or "DUAL_IBVS"),
        )
        if raised:
            self._awaiting_raise_v = self._tip_row(telem.get("tip_uv"))
        if blocked and not lost_side:
            self.message = self._up_block_message

    @staticmethod
    def _tip_row(tip_uv) -> float | None:
        if not tip_uv or len(tip_uv) < 2:
            return None
        try:
            return float(tip_uv[1])
        except (TypeError, ValueError):
            return None

    def _settle_raise(self, tip_uv) -> None:
        """Judge the previous raise only after the shoulder command actually went up."""
        baseline = self._awaiting_raise_v
        self._awaiting_raise_v = None
        if baseline is None:
            return
        row = self._tip_row(tip_uv)
        if row is None:
            return
        if row >= baseline - 1.0:
            self._up_stuck += 1
        else:
            self._up_stuck = 0

    def _limit_upward(self, dy: float) -> tuple[float, bool]:
        """Camera -Y raises the arm. Stop after real raises fail to lift the tip."""
        self._up_block_message = ""
        if dy >= -1e-5:
            self._up_stuck = 0
            self._awaiting_raise_v = None
            return dy, False
        try:
            measured = self.controller.measured_arm_deg("right") if self.controller is not None else []
        except Exception:
            measured = []
        if measured and float(measured[0]) <= MEASURED_HIGH_SHOULDER_DEG:
            self._up_block_message = "肩已到安全高位，不再上举"
            return 0.0, True
        if self._up_stuck >= 3:
            self._up_block_message = "上举没有让尖端靠近按键，停止上举"
            return 0.0, True
        return dy, False

    def _nominal_tool_rpy(self) -> tuple[float, float, float]:
        return tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)

    def _pbvs_path_candidates(
        self,
        _current_tip: tuple[float, float, float],
        goal: tuple[float, float, float],
        _optical: tuple[float, float, float],
    ):
        """Joint seeds and small RPY changes for this exact goal."""
        nominal = self._nominal_tool_rpy()
        ready = [math.radians(value) for value in READY_DEG]
        level = [math.radians(value) for value in FOREARM_LEVEL_DEG]
        points: list[tuple[str, tuple[float, float, float]]] = [("goal", goal)]
        rpy_variants = (
            (10.0, 0.0, 0.0, "roll+10"),
            (-10.0, 0.0, 0.0, "roll-10"),
            (0.0, 8.0, 0.0, "pitch+8"),
            (0.0, -8.0, 0.0, "pitch-8"),
        )
        seen: set[tuple] = set()
        candidates = []

        def push(label: str, xyz, rpy, seed, seed_tag: str) -> None:
            key = (
                tuple(round(float(value), 3) for value in xyz),
                tuple(round(float(value), 3) for value in rpy),
                seed_tag,
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append((label, tuple(float(value) for value in xyz), tuple(rpy), seed))

        for point_name, xyz in points:
            push(point_name, xyz, nominal, None, "measured")
            push(f"{point_name}/ready", xyz, nominal, ready, "ready")
            push(f"{point_name}/level", xyz, nominal, level, "level")
        for droll, dpitch, dyaw, name in rpy_variants:
            push(
                name,
                goal,
                tuple(nominal[index] + math.radians(delta) for index, delta in enumerate((droll, dpitch, dyaw))),
                None,
                "measured",
            )
        return candidates

    def _call_pose_solver(self, solve_pose, commanded, measured, xyz, rpy, seed):
        kwargs = {
            "target_tool_xyz_m": xyz,
            "target_tool_rpy_rad": rpy,
            "tool_offset_wrist_m": RIGHT_EE_OFFSET_WRIST_M,
        }
        if seed is not None:
            kwargs["seed_q_rad"] = seed
        try:
            return solve_pose(commanded, measured, **kwargs)
        except TypeError:
            if seed is None:
                raise
            kwargs.pop("seed_q_rad", None)
            return solve_pose(commanded, measured, **kwargs)

    def _current_tool_rpy(self, measured) -> tuple[float, float, float]:
        fk = getattr(self.ik, "fk_xyz_rpy", None)
        if callable(fk):
            try:
                _xyz, rpy = fk(measured)
                if len(rpy) == 3 and all(math.isfinite(float(value)) for value in rpy):
                    return tuple(float(value) for value in rpy)
            except Exception:
                pass
        return self._nominal_tool_rpy()

    def _pose_command_ok(self, q_new, baseline, pos_err, ori_err, *, must_move: bool):
        jump = max(abs(float(after) - float(before)) for before, after in zip(baseline, q_new))
        jump_limit = math.radians(
            max(float(self.direct_pbvs_max_joint_deg), DIRECT_PBVS_MIN_JOINT_DEG)
        )
        min_jump = math.radians(0.4) if must_move else 0.0
        ok = (
            float(pos_err) <= DIRECT_PBVS_POSE_POS_TOL_M
            and abs(float(ori_err)) <= math.radians(DIRECT_PBVS_POSE_ROT_TOL_DEG)
            and min_jump <= jump <= jump_limit
        )
        return ok, jump

    def _solve_tool_sample(self, solve_pose, commanded, measured, xyz, rpy, seed):
        detail = getattr(self.ik, "solve_loaded_tool_pose", None)
        if callable(detail):
            kwargs = {
                "target_tool_xyz_m": xyz,
                "target_tool_rpy_rad": rpy,
                "tool_offset_wrist_m": RIGHT_EE_OFFSET_WRIST_M,
            }
            if seed is not None:
                kwargs["seed_q_rad"] = seed
            try:
                result = detail(commanded, measured, **kwargs)
            except TypeError:
                result = None
            if isinstance(result, dict) and "q_command" in result and "q_solved" in result:
                return result
        solved = self._call_pose_solver(solve_pose, commanded, measured, xyz, rpy, seed)
        if solved is None:
            return None
        q_new, _ok, pos_err, ori_err = solved
        q_command = [float(value) for value in q_new]
        return {
            "q_command": q_command,
            "q_solved": list(q_command),
            "pos_err": float(pos_err),
            "ori_err": float(ori_err),
        }

    def _plan_cartesian_tool_move(
        self,
        commanded,
        measured,
        current_tip: tuple[float, float, float],
        goal: tuple[float, float, float],
        solve_pose,
        *,
        must_move: bool,
        stage_limited: bool,
    ):
        """Endpoint IK first, then one Cartesian knot chain. A failed knot still plays."""
        if stage_limited and self._pbvs_deadline is None:
            self._pbvs_deadline = time.monotonic() + PBVS_STAGE_TIMEOUT_S
        deadline = self._pbvs_deadline if stage_limited else None
        start_rpy = self._current_tool_rpy(measured)
        baseline = [float(value) for value in (commanded or measured)]
        optical = (1.0, 0.0, 0.0)
        last = None
        tried = 0
        for label, _xyz, goal_rpy, seed in self._pbvs_path_candidates(current_tip, goal, optical):
            if self._return_ready.is_set():
                return None
            if deadline is not None and time.monotonic() > deadline:
                break
            tried += 1
            self.message = f"PBVS 笛卡尔 {tried} {label}"
            endpoint = self._solve_tool_sample(
                solve_pose, commanded, measured, goal, goal_rpy, seed
            )
            if endpoint is None:
                continue
            pose_ok, jump = self._pose_command_ok(
                endpoint["q_command"], baseline, endpoint["pos_err"], endpoint["ori_err"], must_move=must_move
            )
            last = {
                "q": list(endpoint["q_command"]),
                "samples": [],
                "pos_err": float(endpoint["pos_err"]),
                "ori_err": float(endpoint["ori_err"]),
                "jump": float(jump),
                "label": label,
                "xyz": goal,
                "ok": False,
                "tried": tried,
            }
            if not pose_ok:
                continue
            seconds = quintic_joint_seconds(
                baseline, endpoint["q_command"], feedrate=1.0
            )
            knot_count = max(2, int(math.ceil(seconds / MOTION_KNOT_S)) + 1)
            knots = []
            solved_seed = seed
            for step in range(knot_count - 1):
                if self._return_ready.is_set():
                    return None
                blend = quintic_blend(step / (knot_count - 1))
                xyz = tuple(
                    current_tip[index] * (1.0 - blend) + goal[index] * blend
                    for index in range(3)
                )
                rpy = slerp_rpy(start_rpy, goal_rpy, blend)
                sample_seed = seed if step == 0 else solved_seed
                result = self._solve_tool_sample(
                    solve_pose, commanded, measured, xyz, rpy, sample_seed
                )
                if result is None:
                    break
                sample_ok, _sample_jump = self._pose_command_ok(
                    result["q_command"],
                    baseline,
                    result["pos_err"],
                    result["ori_err"],
                    must_move=False,
                )
                if not sample_ok:
                    break
                knots.append(list(result["q_command"]))
                solved_seed = list(result["q_solved"])
            knots.append(list(endpoint["q_command"]))
            if len(knots) == 1:
                knots.insert(0, list(baseline))
            samples = expand_quintic_knots(knots, seconds, MOTION_DT_S)
            if label != "goal":
                self.log(f"[ARM] PBVS 改走 {label}")
            return {
                "q": list(samples[-1]),
                "samples": samples,
                "pos_err": float(endpoint["pos_err"]),
                "ori_err": float(endpoint["ori_err"]),
                "jump": float(jump),
                "label": label,
                "xyz": goal,
                "ok": True,
                "tried": tried,
                "seconds": seconds,
            }
        if last is not None:
            last["tried"] = tried
        elif tried == 0:
            last = {
                "q": list(baseline),
                "samples": [],
                "pos_err": 0.0,
                "ori_err": 0.0,
                "jump": 0.0,
                "label": "timeout",
                "xyz": goal,
                "ok": False,
                "tried": 0,
            }
        return last

    def _search_pbvs_solution(
        self,
        commanded,
        measured,
        current_tip: tuple[float, float, float],
        goal: tuple[float, float, float],
        optical: tuple[float, float, float],
        solve_pose,
        skip_labels: set[str] | None = None,
        must_move: bool = False,
        stage_limited: bool = True,
    ):
        """Try different 6-DoF paths until one is executable or the budget ends."""
        if stage_limited and self._pbvs_deadline is None:
            self._pbvs_deadline = time.monotonic() + PBVS_STAGE_TIMEOUT_S
        limit = time.monotonic() + PBVS_PATH_TIMEOUT_S
        if stage_limited and self._pbvs_deadline is not None:
            limit = min(limit, self._pbvs_deadline)
        jump_limit = math.radians(
            max(float(self.direct_pbvs_max_joint_deg), DIRECT_PBVS_MIN_JOINT_DEG)
        )
        min_jump = math.radians(0.4) if must_move else 0.0
        skipped = skip_labels or set()
        baseline = commanded or measured
        last = None
        tried = 0
        for label, xyz, rpy, seed in self._pbvs_path_candidates(current_tip, goal, optical):
            if self._return_ready.is_set() or time.monotonic() > limit:
                break
            if label in skipped:
                continue
            tried += 1
            self.message = f"PBVS 试路径 {tried} {label}"
            solved = self._call_pose_solver(solve_pose, commanded, measured, xyz, rpy, seed)
            if solved is None:
                continue
            q_new, _ok, pos_err, ori_err = solved
            jump = max(abs(float(after) - float(before)) for before, after in zip(baseline, q_new))
            pose_ok = (
                float(pos_err) <= DIRECT_PBVS_POSE_POS_TOL_M
                and abs(float(ori_err)) <= math.radians(DIRECT_PBVS_POSE_ROT_TOL_DEG)
                and min_jump <= jump <= jump_limit
            )
            last = {
                "q": [float(value) for value in q_new],
                "pos_err": float(pos_err),
                "ori_err": float(ori_err),
                "jump": float(jump),
                "label": label,
                "xyz": xyz,
                "ok": pose_ok,
                "tried": tried,
            }
            if pose_ok:
                if label != "goal":
                    self.log(f"[ARM] PBVS 改走 {label}")
                return last
        if last is not None:
            last["tried"] = tried
        elif tried == 0:
            last = {
                "q": list(baseline),
                "pos_err": 0.0,
                "ori_err": 0.0,
                "jump": 0.0,
                "label": "timeout",
                "xyz": goal,
                "ok": False,
                "tried": 0,
            }
        return last

    def _pbvs_search_failure(self, prefix: str, found) -> str:
        if found is None or int(found.get("tried") or 0) == 0:
            return f"{prefix} 超时，未再尝试路径"
        return (
            f"{prefix} IK 未通过，试了 {found['tried']} 条路径，"
            f"末次 {found['label']} pos={found['pos_err']*1000:.1f}mm "
            f"rot={math.degrees(found['ori_err']):.1f}deg "
            f"jump={math.degrees(found['jump']):.0f}deg"
        )

    def _pbvs_once(self, telem: dict[str, Any]) -> None:
        target = self._goal_from_telem(telem)
        if target is None:
            self.phase = "pbvs_wait_depth"
            self.message = "PBVS 等待标定后的按键三维坐标"
            return
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        if not measured or not callable(tip_xyz):
            self.phase = "pbvs_wait_fk"
            self.message = "PBVS 等待实测关节 FK 指尖"
            return
        try:
            current_tip = tuple(float(value) for value in tip_xyz(measured))
        except Exception as exc:
            self.phase = "pbvs_wait_fk"
            self.message = f"PBVS FK 失败: {exc}"
            return
        optical = (
            tuple(float(self.camera_R_torso[row][2]) for row in range(3))
            if self.camera_R_torso is not None
            else (1.0, 0.0, 0.0)
        )
        coarse_target = tuple(
            target[index] - PBVS_STANDOFF_M * optical[index]
            for index in range(3)
        )
        error_torso = tuple(
            coarse_target[index] - current_tip[index]
            for index in range(3)
        )
        if self.camera_R_torso is not None:
            error_camera = tuple(
                sum(float(self.camera_R_torso[row][col]) * error_torso[row] for row in range(3))
                for col in range(3)
            )
        else:
            error_camera = (-error_torso[1], -error_torso[2], error_torso[0])
        # Same 6-DoF target as /h2_arm/singlearmcartesian: standoff XYZ plus the
        # right_ee orientation already stored in simple_cartesian.yaml.
        done = all(abs(value) <= PBVS_XY_TOL_M for value in error_torso)
        if done:
            if self._left_side_ready:
                self._control_stage = "DUAL_IBVS"
                self.phase = "pbvs_ready"
                self.message = "运控位姿已到位，切换双视 IBVS 做前后进近"
                self.log("[ARM] PBVS coarse ready -> DUAL_IBVS")
            else:
                self._control_stage = "RAISE_LEFT_FOR_IBVS"
                self.phase = "pbvs_ready_raise_left"
                self.message = "运控位姿已到位，开始抬左臂建立侧视"
                self.log("[ARM] PBVS coarse ready -> raise left side camera")
            return
        distance = math.sqrt(sum(value * value for value in error_torso))
        if callable(getattr(self.controller, "move_cartesian", None)):
            if self._service_pbvs_sent:
                if self._left_side_ready:
                    self._control_stage = "DUAL_IBVS"
                    self.phase = "pbvs_ready"
                    self.message = "运控服务已执行，切换双视 IBVS"
                else:
                    self._control_stage = "RAISE_LEFT_FOR_IBVS"
                    self.phase = "pbvs_ready_raise_left"
                    self.message = "运控服务已执行，开始抬左臂建立侧视"
                return
            seconds = max(2.0, min(6.0, distance / 0.15))
            ok, detail = self.controller.move_cartesian(
                "right",
                coarse_target,
                tuple(float(value) for value in DIRECT_EE_RPY_DEG),
                seconds=seconds,
            )
            if not ok:
                self._ibvs_enabled = False
                self._control_stage = "HOLD"
                self._running.clear()
                self.phase = "pbvs_timeout"
                self.message = f"PBVS 服务失败：{detail}"
                self.log(f"[ARM] {self.message}")
                return
            follow_goal = self._service_coarse_target() or coarse_target
            ok, message = self._service_followup_cartesian(
                follow_goal,
                tuple(float(value) for value in DIRECT_EE_RPY_DEG),
                "PBVS",
            )
            if not ok:
                self._ibvs_enabled = False
                self._control_stage = "HOLD"
                self._running.clear()
                self.phase = "pbvs_timeout"
                self.message = message
                self.log(f"[ARM] {self.message}")
                return
            self._service_pbvs_sent = True
            self.phase = "pbvs_coarse"
            self.message = message
            return
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not commanded or not callable(solve_pose):
            self.phase = "pbvs_ik_rejected"
            self.message = "PBVS 拒绝：运控末端 IK 未更新"
            return
        found = self._plan_cartesian_tool_move(
            commanded,
            measured,
            current_tip,
            coarse_target,
            solve_pose,
            must_move=False,
            stage_limited=True,
        )
        if found is None or not found["ok"]:
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self._running.clear()
            self.phase = "pbvs_timeout"
            self.message = self._pbvs_search_failure("PBVS", found)
            self.log(f"[ARM] {self.message}")
            return
        q_new = found["q"]
        self._last_visible_q = [float(value) for value in (commanded or measured)]
        self._play_cartesian_samples(found["samples"])
        self.phase = "pbvs_coarse"
        via = "" if found["label"] == "goal" else f" 经 {found['label']}"
        self.message = f"运控位姿一次到位 {distance*1000:.0f}mm{via}"

    def _raise_left_for_ibvs(self) -> None:
        """After PBVS, raise only the left camera arm while the right arm holds."""
        self.phase = "raising_left_for_ibvs"
        self.message = "右臂保持 PBVS 粗略点位；左臂抬到侧视位"
        self.controller.ramp_arm("left", LEFT_SIDE_VIEW_DEG, seconds=8.0, dt=0.02)
        self._left_side_ready = True
        time.sleep(0.3)
        self._control_stage = "DUAL_IBVS"
        self.phase = "dual_ibvs_ready"
        self.message = "左臂侧视已就位，双视 IBVS 开始进近"
        self.log("[ARM] left side camera ready -> DUAL_IBVS")

    def _direct_ee_once(self, telem: dict[str, Any]) -> None:
        """Start-without-Raise: solve right_ee_link pose directly to the RGB-D button."""
        target_torso = telem.get("target_torso_m")
        if not target_torso or len(target_torso) != 3:
            self._finish_direct_ee(False, "Direct EE 拒绝：按键三维坐标丢失")
            return
        try:
            target = tuple(float(value) for value in target_torso)
        except (TypeError, ValueError):
            self._finish_direct_ee(False, "Direct EE 拒绝：按键三维坐标无效")
            return
        optical = (
            tuple(float(self.camera_R_torso[row][2]) for row in range(3))
            if self.camera_R_torso is not None
            else (1.0, 0.0, 0.0)
        )
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        if not commanded or not measured:
            self._finish_direct_ee(False, "Direct EE 拒绝：没有右臂关节状态")
            return
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        solve_position = getattr(self.ik, "apply_loaded_tip_target", None)
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not callable(tip_xyz) or not callable(solve_position) or not callable(solve_pose):
            self._finish_direct_ee(False, "Direct EE 拒绝：运控末端 IK 未更新")
            return
        aligning = self._control_stage == "DIRECT_EE_ALIGN_YZ"
        orienting = self._control_stage == "DIRECT_EE_ORIENT_SAFE"
        if aligning or orienting:
            motion_target = tuple(
                target[index] - DIRECT_EE_PREALIGN_STANDOFF_M * optical[index]
                for index in range(3)
            )
        else:
            motion_target = tuple(
                target[index] + self.direct_ee_poke_depth_m * optical[index]
                for index in range(3)
            )
        current = tuple(float(value) for value in tip_xyz(measured, RIGHT_EE_OFFSET_WRIST_M))
        distance = math.sqrt(sum((motion_target[i] - current[i]) ** 2 for i in range(3)))
        if self.direct_ee_max_m > 0.0 and distance > self.direct_ee_max_m:
            self._finish_direct_ee(
                False,
                f"Direct EE 拒绝：距离 {distance*1000:.0f}mm > {self.direct_ee_max_m*1000:.0f}mm",
            )
            return
        if aligning:
            q_new, ok, pos_err, ori_err = solve_position(
                commanded,
                measured,
                target_tip_xyz_m=motion_target,
                offset_m=RIGHT_EE_OFFSET_WRIST_M,
            )
        else:
            rpy = tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
            q_new, ok, pos_err, ori_err = solve_pose(
                commanded,
                measured,
                target_tool_xyz_m=motion_target,
                target_tool_rpy_rad=rpy,
                tool_offset_wrist_m=RIGHT_EE_OFFSET_WRIST_M,
            )
        jump = max(
            abs(float(after) - float(before))
            for before, after in zip(commanded, q_new)
        )
        if not ok or jump > math.radians(self.direct_ee_max_joint_deg):
            self._finish_direct_ee(
                False,
                f"Direct EE IK 拒绝 pos={pos_err*1000:.1f}mm "
                f"rot={math.degrees(ori_err):.1f}deg jump={math.degrees(jump):.0f}deg",
            )
            return
        self._last_visible_q = [float(value) for value in commanded]
        if aligning:
            self._direct_ee_align_steps += 1
        elif orienting:
            self._direct_ee_orient_steps += 1
        else:
            self._direct_ee_approach_steps += 1
        self.controller.apply_arm_rad(
            "right",
            q_new,
            seconds=5.0 if (aligning or orienting) else 3.0,
            dt=0.02,
        )
        measured_after = self.controller.measured_arm_rad("right")
        actual_after = tuple(
            float(value) for value in tip_xyz(measured_after, RIGHT_EE_OFFSET_WRIST_M)
        )
        residual = tuple(motion_target[index] - actual_after[index] for index in range(3))
        residual_norm = math.sqrt(sum(value * value for value in residual))
        if aligning:
            yz_error = math.hypot(residual[1], residual[2])
            if yz_error <= 0.015:
                self._control_stage = "DIRECT_EE_ORIENT_SAFE"
                self.phase = "direct_ee_yz_aligned"
                self.message = (
                    f"Direct EE Y/Z 已对齐 {yz_error*1000:.0f}mm，"
                    "在 15cm 外调整末端姿态"
                )
                return
            if self._direct_ee_align_steps >= 6:
                self._finish_direct_ee(
                    False,
                    f"Direct EE Y/Z 补偿未收敛：{yz_error*1000:.0f}mm",
                )
                return
            self.phase = "direct_ee_aligning_yz"
            self.message = (
                f"Direct EE Y/Z 补偿 {self._direct_ee_align_steps}/6，"
                f"剩余 {yz_error*1000:.0f}mm"
            )
            return
        if orienting:
            _wrist_xyz, actual_rpy = self.ik.fk_xyz_rpy(measured_after)
            desired_rpy = tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
            angle_errors = [
                math.atan2(
                    math.sin(desired_rpy[index] - float(actual_rpy[index])),
                    math.cos(desired_rpy[index] - float(actual_rpy[index])),
                )
                for index in range(3)
            ]
            angle_error_deg = math.degrees(
                math.sqrt(sum(value * value for value in angle_errors))
            )
            if residual_norm <= 0.020 and angle_error_deg <= 12.0:
                self._control_stage = "DIRECT_EE_APPROACH_X"
                self.phase = "direct_ee_oriented"
                self.message = (
                    f"Direct EE 姿态已对齐 {angle_error_deg:.1f}deg，开始沿 X 接近"
                )
                return
            if self._direct_ee_orient_steps >= 6:
                self._finish_direct_ee(
                    False,
                    f"Direct EE 姿态补偿未收敛：位置 {residual_norm*1000:.0f}mm "
                    f"姿态 {angle_error_deg:.1f}deg",
                )
                return
            self.phase = "direct_ee_orienting"
            self.message = (
                f"Direct EE 姿态补偿 {self._direct_ee_orient_steps}/6，"
                f"位置 {residual_norm*1000:.0f}mm 姿态 {angle_error_deg:.1f}deg"
            )
            return
        if residual_norm <= 0.015:
            self._finish_direct_ee(
                True,
                f"Direct EE 已沿 X 到按键，实测残差 {residual_norm*1000:.0f}mm",
            )
            return
        if self._direct_ee_approach_steps >= 6:
            self._finish_direct_ee(
                False,
                f"Direct EE 最终补偿未收敛：{residual_norm*1000:.0f}mm",
            )
            return
        self.phase = "direct_ee_approaching_x"
        self.message = (
            f"Direct EE 最终补偿 {self._direct_ee_approach_steps}/6，"
            f"剩余 {residual_norm*1000:.0f}mm"
        )

    def _finish_direct_ee(self, success: bool, message: str) -> None:
        self._ibvs_enabled = False
        self._control_stage = "HOLD"
        self._running.clear()
        self.phase = "direct_ee_complete" if success else "direct_ee_rejected"
        self.message = message
        self.log(f"[ARM] {message}")

    def _finish_direct_pbvs(self, arrived: bool, message: str) -> None:
        if self._pbvs_release:
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self._running.clear()
        self.phase = "direct_pbvs_complete" if arrived else "direct_pbvs_short"
        self.message = message
        self.log(f"[ARM] {message}")

    def _goal_from_telem(self, telem: dict[str, Any]) -> tuple[float, float, float] | None:
        name = str(telem.get("target") or "")
        locked = self._locked_buttons.get(name)
        if locked is not None and len(locked) == 3:
            try:
                goal = tuple(float(value) for value in locked)
            except (TypeError, ValueError):
                goal = ()
            if len(goal) == 3 and all(math.isfinite(value) for value in goal):
                self._goal_name = name
                return goal
        self._goal_name = ""
        raw = telem.get("target_torso_m")
        if not raw or len(raw) != 3:
            return None
        try:
            goal = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(goal) != 3 or not all(math.isfinite(value) for value in goal):
            return None
        return goal

    def _snapshot_chest_buttons(self, telem: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
        raw = telem.get("buttons_torso_m") or {}
        out: dict[str, tuple[float, float, float]] = {}
        if isinstance(raw, dict):
            for name in BUTTON_NAMES:
                value = raw.get(name)
                if not value or len(value) != 3:
                    continue
                try:
                    goal = tuple(float(item) for item in value)
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(item) for item in goal):
                    out[name] = goal
        if out:
            return out
        goal = self._goal_from_telem(telem)
        name = str(telem.get("target") or "")
        if goal is not None and name in BUTTON_NAMES:
            out[name] = goal
        return out

    def _wrist_buttons_torso(self, buttons_cam: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
        matrix_fn = self.cam2hand_matrix
        cam2hand = matrix_fn() if callable(matrix_fn) else None
        fk_matrix = getattr(self.ik, "fk_matrix", None)
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        if cam2hand is None or not callable(fk_matrix) or not measured:
            return {}
        try:
            torso_T_hand = fk_matrix(measured)
            torso_T_cam = [
                [
                    sum(float(torso_T_hand[row][col]) * float(cam2hand[col][k]) for col in range(4))
                    for k in range(4)
                ]
                for row in range(4)
            ]
        except Exception as exc:
            self.log(f"[ARM][WARN] 眼在手上坐标变换失败: {exc}")
            return {}
        out: dict[str, tuple[float, float, float]] = {}
        for name, xyz in (buttons_cam or {}).items():
            if name not in BUTTON_NAMES or not xyz or len(xyz) != 3:
                continue
            try:
                point = [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0]
                mapped = [
                    sum(float(torso_T_cam[row][col]) * point[col] for col in range(4))
                    for row in range(3)
                ]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in mapped):
                out[str(name)] = (mapped[0], mapped[1], mapped[2])
        return out

    def _fuse_wrist_pbvs(self, telem: dict[str, Any]) -> None:
        """Lock the chest button map, then compare it with one level-pose wrist look."""
        chest = self._snapshot_chest_buttons(telem)
        self._locked_buttons = dict(chest)
        self._pbvs_source = "chest"
        watch = self.wrist_watch
        view = self.wrist_view
        matrix_fn = self.cam2hand_matrix
        ready = callable(watch) and callable(view) and callable(matrix_fn) and matrix_fn() is not None
        if not ready or self.controller is None:
            self.log("[ARM] 眼在手上标定不可用，三个按键只用胸口坐标")
            return
        watch(True)
        try:
            self.phase = "wrist_pbvs"
            self.message = "胸口三个按键已锁定，右臂向外平举核对眼在手上"
            self.log(f"[ARM] {self.message}")
            self._ramp_right_level()
            self._publish_ee()
            if self._return_ready.is_set() or self._stop.is_set():
                return
            seen = bool((view() or {}).get("seen"))
            if not seen:
                self._pbvs_source = "chest"
                self.message = "平举过程中没有看到面板和按键，跳过眼在手上"
                self.log(f"[ARM] {self.message}")
                return
            self.message = "眼在手上看到面板，停留 1 秒"
            self.log(f"[ARM] {self.message}")
            if callable(self.wrist_hold):
                self.wrist_hold()
            time.sleep(max(0.0, float(self.wrist_dwell_s)))
            sample = view() or {}
            wrist = self._wrist_buttons_torso(sample.get("buttons_cam_m") or {})
            if not wrist:
                self._pbvs_source = "chest"
                self.message = "停留时没有算出眼在手上的按键坐标，只用胸口"
                self.log(f"[ARM] {self.message}")
                return
            fused, source = fuse_pbvs_buttons(chest, wrist)
            self._locked_buttons = fused
            self._pbvs_source = source
            if source == "average":
                self.message = "胸口和眼在手上误差小于 2cm，使用两者平均坐标"
            else:
                self.message = "眼在手上和胸口误差不小于 2cm，只用胸口坐标"
            self.log(f"[ARM] {self.message}")
        except Exception as exc:
            self._pbvs_source = "chest"
            self.log(f"[ARM][WARN] 眼在手上核对失败，只用胸口: {exc}")
        finally:
            try:
                watch(False)
            except Exception:
                pass

    def _move_tool_to(self, goal: tuple[float, float, float]) -> bool:
        """One Cartesian move to an absolute torso point, without the X-gap policy."""
        if self.controller is None:
            return False
        rpy = tuple(float(value) for value in DIRECT_EE_RPY_DEG)
        if callable(getattr(self.controller, "move_cartesian", None)):
            ok, detail = self.controller.move_cartesian("right", goal, rpy, seconds=2.0)
            if not ok:
                self.message = f"抬起失败：{detail}"
                return False
            time.sleep(0.35)
            return True
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not commanded or not measured or not callable(tip_xyz) or not callable(solve_pose):
            self.message = "抬起失败：末端 IK 不可用"
            return False
        try:
            current_tip = tuple(float(value) for value in tip_xyz(measured))
        except Exception as exc:
            self.message = f"抬起失败：{exc}"
            return False
        found = self._plan_cartesian_tool_move(
            commanded,
            measured,
            current_tip,
            goal,
            solve_pose,
            must_move=True,
            stage_limited=False,
        )
        if self._return_ready.is_set():
            return False
        if found is None or not found.get("ok"):
            self.message = self._pbvs_search_failure("抬起", found)
            return False
        self._play_cartesian_samples(found["samples"])
        self._wait_until_tracked(found["q"])
        return True

    def _lower_right_after_sequence(self) -> None:
        if self.controller is None or self._return_ready.is_set() or self._stop.is_set():
            return
        self.message = "动作结束，放下右臂"
        self.log(f"[ARM] {self.message}")
        self.controller.ramp_arm("right", REST_DEG, seconds=6.0, dt=0.02)
        self._publish_ee()

    def _run_button_sequence(self) -> None:
        """Press the queued buttons. Lift clear of the panel between presses."""
        steps = list(self._button_sequence)
        failed = False
        for index, name in enumerate(steps):
            if self._return_ready.is_set() or self._stop.is_set():
                return
            goal = self._locked_buttons.get(name)
            if goal is None:
                self._pbvs_release = True
                self._finish_direct_pbvs(False, f"动作 {name} 没有锁定的胸口坐标")
                self.phase = "direct_pbvs_rejected"
                failed = True
                break
            self._goal_name = name
            self._ibvs_enabled = True
            self._running.set()
            self._control_stage = "DIRECT_PBVS"
            self.message = f"动作 {index + 1}/{len(steps)} {name}"
            self.log(f"[ARM] {self.message}")
            self._direct_pbvs_once(
                {"target": name, "target_torso_m": list(goal)},
                release=False,
            )
            if self._return_ready.is_set() or self._stop.is_set():
                return
            if self.phase == "direct_pbvs_rejected":
                failed = True
                break
            lift = sequence_lift_goal(goal)
            self.message = f"抬起夹爪，离开 {name}"
            self.log(f"[ARM] {self.message} -> {lift[0]:.3f} {lift[1]:.3f} {lift[2]:.3f}")
            if not self._move_tool_to(lift):
                self._pbvs_release = True
                self._finish_direct_pbvs(False, self.message or "抬起失败")
                self.phase = "direct_pbvs_rejected"
                failed = True
                break
        if self._return_ready.is_set() or self._stop.is_set():
            return
        try:
            self._lower_right_after_sequence()
        except Exception as exc:
            self.log(f"[ARM] 放下失败: {exc}")
        if failed:
            self._ibvs_enabled = False
            self._running.clear()
            self._control_stage = "HOLD"
            return
        if callable(self.on_screen_capture):
            try:
                self.on_screen_capture()
            except Exception as exc:
                self.log(f"[ARM][WARN] 屏幕截取失败: {exc}")
        self._pbvs_release = True
        self._finish_direct_pbvs(True, "动作组合完成，右臂已放下，屏幕在第二相机下方")

    def _tip_error_to(self, goal: tuple[float, float, float]) -> tuple[float, float, float] | None:
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        if not measured or not callable(tip_xyz):
            return None
        try:
            current = tuple(float(value) for value in tip_xyz(measured))
        except Exception:
            return None
        if len(current) != 3 or not all(math.isfinite(value) for value in current):
            return None
        return tuple(float(goal[index]) - current[index] for index in range(3))

    def _tip_distance_to(self, goal: tuple[float, float, float]) -> float | None:
        error = self._tip_error_to(goal)
        if error is None:
            return None
        return math.sqrt(sum(value * value for value in error))

    def _fresh_torso_target(self) -> tuple[float, float, float] | None:
        locked = self._locked_buttons.get(self._goal_name) if self._goal_name else None
        if locked is not None and len(locked) == 3:
            return tuple(float(value) for value in locked)
        telem = self.telemetry() or {}
        raw = telem.get("target_torso_m")
        if not raw or len(raw) != 3:
            return None
        try:
            target = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            return None
        return target

    def _service_coarse_target(self) -> tuple[float, float, float] | None:
        target = self._fresh_torso_target()
        if target is None:
            return None
        optical = (
            tuple(float(self.camera_R_torso[row][2]) for row in range(3))
            if self.camera_R_torso is not None
            else (1.0, 0.0, 0.0)
        )
        return tuple(target[index] - PBVS_STANDOFF_M * optical[index] for index in range(3))

    def _direct_pbvs_via_service(self, goal: tuple[float, float, float]) -> None:
        """Absolute shots. X starts 8 cm short and opens 2 cm at a time if Y/Z stalls."""
        rpy = tuple(float(value) for value in DIRECT_EE_RPY_DEG)
        x_clear = DIRECT_PBVS_X_CLEAR_M
        yz_before: float | None = None
        full_x_shots = 0
        shot = 0
        while shot < DIRECT_PBVS_MAX_SHOTS + int(math.ceil(DIRECT_PBVS_X_CLEAR_M / DIRECT_PBVS_X_RELAX_M)):
            if self._return_ready.is_set():
                return
            goal = self._fresh_torso_target() or goal
            error = self._tip_error_to(goal)
            distance = None if error is None else math.sqrt(sum(value * value for value in error))
            if distance is not None and distance <= DIRECT_PBVS_ARRIVE_TOL_M:
                self._finish_direct_pbvs(
                    True,
                    (
                        f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                        f"Y {error[1]*1000:.0f}mm Z {error[2]*1000:.0f}mm"
                    ),
                )
                return
            if distance is not None and distance > self.direct_pbvs_max_m:
                self._finish_direct_pbvs(
                    False,
                    (
                        f"Direct PBVS 拒绝：距离 {distance*1000:.0f}mm "
                        f"> {self.direct_pbvs_max_m*1000:.0f}mm"
                    ),
                )
                self.phase = "direct_pbvs_rejected"
                return
            current_tip = tuple(float(goal[index]) - float(error[index]) for index in range(3)) if error is not None else goal
            yz_now = 0.0 if error is None else math.hypot(error[1], error[2])
            if yz_before is not None:
                x_clear = relax_x_clearance(x_clear, yz_before, yz_now)
            yz_aligned = yz_now <= DIRECT_PBVS_ARRIVE_TOL_M
            if yz_aligned or x_clear <= 1e-9:
                full_x_shots += 1
                if full_x_shots > DIRECT_PBVS_MAX_SHOTS:
                    break
            motion_goal = direct_pbvs_motion_goal(current_tip, goal, x_clear_m=x_clear)
            closing_x = abs(motion_goal[0] - float(goal[0])) <= 1e-6
            seconds = 2.0 if distance is None else max(2.0, min(6.0, distance / 0.15))
            self.phase = "direct_pbvs"
            self.message = (
                f"Direct PBVS 服务 第 {shot + 1} 段 "
                f"{'沿 X 接近' if closing_x else f'对齐 Y/Z，X 留 {x_clear*100:.0f}cm'} "
                f"XYZ {motion_goal[0]:.3f} {motion_goal[1]:.3f} {motion_goal[2]:.3f}"
            )
            yz_before = yz_now
            shot += 1
            self.log(f"[ARM] {self.message}")
            ok, detail = self.controller.move_cartesian("right", motion_goal, rpy, seconds=seconds)
            if not ok:
                self._finish_direct_pbvs(False, f"Direct PBVS 服务失败：{detail}")
                self.phase = "direct_pbvs_rejected"
                return
            time.sleep(0.35)
        if self._return_ready.is_set():
            return
        goal = self._fresh_torso_target() or goal
        error = self._tip_error_to(goal)
        if error is None:
            self._finish_direct_pbvs(False, "Direct PBVS 实测未到位：末端坐标丢失")
            return
        distance = math.sqrt(sum(value * value for value in error))
        if distance <= DIRECT_PBVS_ARRIVE_TOL_M:
            self._finish_direct_pbvs(
                True,
                (
                    f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                    f"Y {error[1]*1000:.0f}mm Z {error[2]*1000:.0f}mm"
                ),
            )
            return
        self._finish_direct_pbvs(
            False,
            (
                "Direct PBVS 实测未到位，还差 "
                f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
            ),
        )

    def _direct_pbvs_once(self, telem: dict[str, Any], *, release: bool = True) -> None:
        """Move the measured fingertip onto the button, then correct gravity sag."""
        self._pbvs_release = bool(release)
        target = self._goal_from_telem(telem)
        if target is None:
            self._pbvs_release = True
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "direct_pbvs_rejected"
            self.message = "Direct PBVS 拒绝：按键 RGB-D 三维坐标丢失"
            return
        if callable(getattr(self.controller, "move_cartesian", None)):
            self._direct_pbvs_via_service(target)
            return
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        q = commanded or measured
        if not q or not measured:
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "direct_pbvs_rejected"
            self.message = "Direct PBVS 拒绝：没有右臂关节状态"
            return
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        if not callable(tip_xyz):
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "direct_pbvs_rejected"
            self.message = "Direct PBVS 拒绝：FK 指尖模型不可用"
            return
        try:
            current_tip = tuple(float(value) for value in tip_xyz(measured))
        except Exception as exc:
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "direct_pbvs_rejected"
            self.message = f"Direct PBVS 拒绝：FK 指尖失败 {exc}"
            return
        goal = target
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not commanded or not callable(solve_pose):
            self._finish_direct_pbvs(False, "Direct PBVS 拒绝：运控末端 IK 未更新")
            self.phase = "direct_pbvs_rejected"
            return

        error = tuple(goal[index] - current_tip[index] for index in range(3))
        distance = math.sqrt(sum(value * value for value in error))
        if distance > self.direct_pbvs_max_m:
            self._finish_direct_pbvs(
                False,
                (
                    f"Direct PBVS 拒绝：距离 {distance*1000:.0f}mm "
                    f"> {self.direct_pbvs_max_m*1000:.0f}mm"
                ),
            )
            self.phase = "direct_pbvs_rejected"
            return
        track_rad = 0.0
        x_clear = DIRECT_PBVS_X_CLEAR_M
        yz_before: float | None = None
        full_x_shots = 0
        shot = 0
        shot_limit = DIRECT_PBVS_MAX_SHOTS + int(math.ceil(DIRECT_PBVS_X_CLEAR_M / DIRECT_PBVS_X_RELAX_M))
        while shot < shot_limit:
            if self._return_ready.is_set():
                return
            error = tuple(goal[index] - current_tip[index] for index in range(3))
            distance = math.sqrt(sum(value * value for value in error))
            if distance <= DIRECT_PBVS_ARRIVE_TOL_M:
                self._finish_direct_pbvs(
                    True,
                    (
                        f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                        f"Y {error[1]*1000:.0f}mm Z {error[2]*1000:.0f}mm"
                    ),
                )
                return
            yz_now = math.hypot(error[1], error[2])
            if yz_before is not None:
                x_clear = relax_x_clearance(x_clear, yz_before, yz_now)
            if yz_now <= DIRECT_PBVS_ARRIVE_TOL_M or x_clear <= 1e-9:
                full_x_shots += 1
                if full_x_shots > DIRECT_PBVS_MAX_SHOTS:
                    break
            motion_goal = direct_pbvs_motion_goal(current_tip, goal, x_clear_m=x_clear)
            closing_x = abs(motion_goal[0] - float(goal[0])) <= 1e-6
            self.message = (
                f"Direct PBVS 第 {shot + 1} 段，"
                f"{'沿 X 接近' if closing_x else f'对齐 Y/Z，X 留 {x_clear*100:.0f}cm'}，还差 "
                f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
            )
            yz_before = yz_now
            shot += 1
            found = self._plan_cartesian_tool_move(
                commanded,
                measured,
                current_tip,
                motion_goal,
                solve_pose,
                must_move=True,
                stage_limited=False,
            )
            if self._return_ready.is_set():
                return
            if found is None or not found["ok"]:
                self._finish_direct_pbvs(
                    False,
                    (
                        f"{self._pbvs_search_failure('Direct PBVS', found)}；还差 "
                        f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
                    ),
                )
                self.phase = "direct_pbvs_rejected"
                return
            self._last_visible_q = [float(value) for value in q]
            self._play_cartesian_samples(found["samples"])
            track_rad = self._wait_until_tracked(found["q"])
            commanded = self.controller.commanded_arm_rad("right")
            measured = self.controller.measured_arm_rad("right")
            q = commanded or measured
            if not q or not measured:
                break
            try:
                current_tip = tuple(float(value) for value in tip_xyz(measured))
            except Exception:
                break
        if self._return_ready.is_set():
            return
        error = tuple(goal[index] - current_tip[index] for index in range(3))
        distance = math.sqrt(sum(value * value for value in error))
        if distance <= DIRECT_PBVS_ARRIVE_TOL_M:
            self._finish_direct_pbvs(
                True,
                (
                    f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                    f"Y {error[1]*1000:.0f}mm Z {error[2]*1000:.0f}mm"
                ),
            )
            return
        self._finish_direct_pbvs(
            False,
            (
                "Direct PBVS 实测未到位，还差 "
                f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm，"
                f"关节跟踪 {math.degrees(track_rad):.0f}°"
            ),
        )

    def _use_motion_gains(self) -> None:
        """Hold the reach with the same kp/kd as h2_arm_control."""
        controller = self.controller
        if controller is None or not hasattr(controller, "kp") or not hasattr(controller, "kd"):
            return
        controller.kp = MOTION_ARM_KP
        controller.kd = MOTION_ARM_KD

    def _tracked_joint_error_rad(self, target_rad: Sequence[float]) -> float:
        try:
            measured = self.controller.measured_arm_rad("right") or []
        except Exception:
            return math.pi
        if len(measured) < len(target_rad):
            return math.pi
        return max(abs(float(after) - float(before)) for before, after in zip(measured, target_rad))

    def _wait_until_tracked(self, target_rad: Sequence[float]) -> float:
        deadline = time.monotonic() + MOTION_TRACK_WAIT_S
        error = self._tracked_joint_error_rad(target_rad)
        limit = math.radians(MOTION_TRACK_TOL_DEG)
        while error > limit and time.monotonic() < deadline:
            time.sleep(0.05)
            error = self._tracked_joint_error_rad(target_rad)
        return error

    def _tracking_correction(self, sample: Sequence[float], phase: float, filtered: list[float]) -> list[float]:
        """Same bounded lead as H2ArmControl::Run(), faded in and out over 20%."""
        ramp = quintic_blend(min(1.0, phase / MOTION_TRACK_RAMP))
        fade_phase = (phase - (1.0 - MOTION_TRACK_FADE)) / MOTION_TRACK_FADE
        fade = quintic_blend(min(1.0, max(0.0, fade_phase)))
        weight = ramp * (1.0 - fade)
        corrected = [float(value) for value in sample]
        try:
            measured = self.controller.measured_arm_rad("right") or []
        except Exception:
            measured = []
        if len(measured) < 7:
            return corrected
        for joint in range(7):
            error = float(sample[joint]) - float(measured[joint])
            desired = min(MOTION_TRACK_MAX_RAD, max(-MOTION_TRACK_MAX_RAD, MOTION_TRACK_GAIN * error))
            blended = (1.0 - MOTION_TRACK_FILTER) * filtered[joint] + MOTION_TRACK_FILTER * desired
            delta = min(
                MOTION_TRACK_MAX_STEP_RAD,
                max(-MOTION_TRACK_MAX_STEP_RAD, blended - filtered[joint]),
            )
            filtered[joint] = min(
                MOTION_TRACK_MAX_RAD,
                max(-MOTION_TRACK_MAX_RAD, filtered[joint] + delta),
            )
            corrected[joint] += weight * filtered[joint]
        return corrected

    def _stream_arm_samples(self, samples: Sequence[Sequence[float]]) -> None:
        """Publish q and dq together. The 50 Hz hold keeps the latest dq until the next sample."""
        hold = dict(self.controller.hold_positions())
        filtered = [0.0] * 7
        previous = None
        count = len(samples)
        for index, sample in enumerate(samples):
            if self._return_ready.is_set():
                break
            phase = index / (count - 1) if count > 1 else 1.0
            corrected = self._tracking_correction(sample, phase, filtered)
            command = dict(hold)
            dq = {joint: 0.0 for joint in hold}
            for joint, value in zip(RIGHT_ARM_JOINTS, corrected):
                command[joint] = float(value)
            if previous is not None:
                for joint, prev, cur in zip(RIGHT_ARM_JOINTS, previous, corrected):
                    dq[joint] = (float(cur) - float(prev)) / MOTION_DT_S
            self.controller.write(command, weight=1.0, dq=dq)
            previous = list(corrected)
            time.sleep(MOTION_DT_S)
        if samples:
            final = dict(hold)
            for joint, value in zip(RIGHT_ARM_JOINTS, samples[-1]):
                final[joint] = float(value)
            self.controller.write(final, weight=1.0)

    def _play_cartesian_samples(self, samples: Sequence[Sequence[float]]) -> float:
        """Stream one Cartesian IK sequence. Tests without write() receive only the final pose."""
        self._use_motion_gains()
        if not samples:
            return 0.0
        seconds = max(MOTION_MIN_SECONDS, (len(samples) - 1) * MOTION_DT_S)
        writer = getattr(self.controller, "write", None)
        hold = getattr(self.controller, "hold_positions", None)
        if callable(writer) and callable(hold):
            self._stream_arm_samples(samples)
            return seconds
        self.controller.apply_arm_rad("right", samples[-1], seconds=seconds, dt=MOTION_DT_S)
        return seconds

    def _can_poke(self) -> bool:
        return (
            self._ibvs_enabled
            and self._poke_remaining_m > 1e-4
            and self._control_stage in {"PBVS_COARSE", "IBVS", "POKE"}
        )

    def _poke_once(self, reason: str) -> None:
        self._control_stage = "POKE"
        last = self._last_camera_delta
        if last is not None:
            dx, dy, dz = last
            dx = float(dx) + POKE_RIGHT_STEP_M
            dy = float(dy) - POKE_UP_STEP_M
            dz = max(float(dz), POKE_STEP_M)
        else:
            dx, dy, dz = POKE_RIGHT_STEP_M, -POKE_UP_STEP_M, POKE_STEP_M
        self._apply_camera_delta((dx, dy, dz), seconds=0.45, limit_m=POKE_STEP_M, phase="POKE")
        self._poke_remaining_m = max(0.0, self._poke_remaining_m - POKE_STEP_M)
        self.message = (
            f"点触推进 {reason} 剩余 {self._poke_remaining_m*1000:.0f}mm "
            "（近距丢面板不回退）"
        )

    def _apply_camera_delta(
        self,
        camera_delta: tuple[float, float, float],
        *,
        seconds: float,
        phase: str,
        limit_m: float | None = None,
    ) -> bool:
        self._last_camera_delta = (
            float(camera_delta[0]),
            float(camera_delta[1]),
            float(camera_delta[2]),
        )
        torso = clip_torso_delta(
            camera_delta_to_torso(*self._last_camera_delta, camera_R_torso=self.camera_R_torso),
            limit_m=POKE_STEP_M if limit_m is None else limit_m,
        )
        return self._apply_torso_delta(torso, seconds=seconds, phase=phase, limit_m=None)

    def _apply_torso_delta(
        self,
        torso_delta: tuple[float, float, float],
        *,
        seconds: float,
        phase: str,
        limit_m: float | None,
    ) -> bool:
        torso = torso_delta if limit_m is None else clip_torso_delta(torso_delta, limit_m=limit_m)
        move_delta = getattr(self.controller, "move_delta", None)
        if callable(move_delta):
            ok, detail = move_delta("right", torso, seconds=seconds)
            self.phase = phase
            self.message = (
                f"{phase} 服务 dXYZ_mm "
                f"{torso[0]*1000:.1f} {torso[1]*1000:.1f} {torso[2]*1000:.1f}"
                if ok
                else f"IBVS 服务失败：{detail}"
            )
            return ok
        # Never seed from loaded-arm sag. Feeding the measured lag back as the
        # next command ratchets the shoulder downward on every servo cycle.
        q = self.controller.commanded_arm_rad("right") or self.controller.measured_arm_rad("right")
        self._last_visible_q = [float(value) for value in q]
        # Bias toward the joints being held now. The Start pose is often a
        # hang or a raise; pulling a 6 mm image step back there swings a
        # joint about 20° and the tip never moves, reported as a few millimetres.
        apply_tip = getattr(self.ik, "apply_tip_delta", None)
        if apply_tip is not None:
            q_new, ok, pos_err, jump = apply_tip(q, torso, q)
        else:
            q_new, ok, pos_err, jump = self.ik.apply_delta(q, torso)
        if not ok:
            self.message = (
                f"IK 拒绝 pos={pos_err*1000:.1f}mm jump={math.degrees(float(jump)):.1f}deg"
            )
            return False
        shoulder_raised = float(q_new[0]) < float(q[0]) - math.radians(0.4)
        self.controller.apply_arm_rad("right", q_new, seconds=seconds, dt=0.02)
        self.phase = phase
        if phase != "POKE":
            self.message = (
                f"{phase} torso_mm="
                f"({torso[0]*1000:.1f},{torso[1]*1000:.1f},{torso[2]*1000:.1f})"
            )
        return shoulder_raised

    def _recover_last_visible_pose(self) -> bool:
        q = self._last_visible_q
        self._last_visible_q = None
        if q is None or self.controller is None:
            return False
        try:
            self.log("[ARM] 视觉丢失：退回上一可见姿态")
            self.controller.apply_arm_rad("right", q, seconds=0.8, dt=0.02)
            return True
        except Exception as exc:
            self.log(f"[ARM][WARN] 回退失败，保持当前姿态: {exc}")
            return False
