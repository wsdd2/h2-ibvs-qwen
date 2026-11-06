# -*- coding: utf-8 -*-
"""Chest PBVS coarse move, then right-wrist eye-in-hand IBVS for the poke."""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ibvs_pad_touch.config import load_ibvs_config
from ibvs_pad_touch.servo.interaction import eye_in_hand_close_align_step, eye_in_hand_ibvs_step
from ibvs_pad_touch.web.screen_read import blocks_differ, format_block, read_panel



HANDEYE_ROOTS = (
    Path("/home/unitree/MscapeTech/Hand_Eye_Calib"),
    Path(__file__).resolve().parents[3] / "Hand_Eye_Calib",
)
# Between the too-low -60° start and the -96° start that put the jaw above the buttons.
# Image feedback then raises or lowers until the real tip meets the button row.
# 130° folded the forearms down in front of the body. 46° was nearly straight
# and reached the wall. 65° keeps a horizontal raise with the hand short of the wall.
# Right shoulder roll is negative outward. -25° still swept the forearm up
# against the torso; -40° keeps a small gap beside the chest.
READY_DEG = (-76.0, -40.0, 0.0, 65.0, 0.0, 0.0, 0.0)
# From the natural hang, open only this roll. Pitch and elbow stay where they
# are so the forearm swings out beside the body instead of into the torso.
TORSO_CLEAR_ROLL_DEG = -48.0
# Do not command this pose. Every right-forearm "level" tried so far, including
# shoulder -90° / elbow 85° and the later forward pose, swung into the chest
# camera and destroyed the bracket. Right level and PBVS must not ramp toward it.
# pitch, roll, yaw, elbow, wrist roll, wrist pitch, wrist yaw.
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
# Coarse PBVS aims at the button itself. A camera-Z standoff was leaving
# the fingertip far short of the detected point.
PBVS_XY_TOL_M = 0.01
# Before the wrist approach, fingertip X must stay strictly short of this
# plane. 0.3 cm in front of the PBVS button; the coarse goal may neither
# cross it nor land on it. The wrist approach still closes the last gap.
COARSE_X_CLEARANCE_M = 0.003
# While Y or Z is still off, the fingertip stays at least this far short in X.
# X closes only after YZ is aligned, or after that YZ move cannot be solved.
# Off: that held-X pose folds the elbow backward. The button pose is used instead.
PBVS_YZ_FIRST = False
PBVS_YZ_X_CLEAR_M = 0.02
# Solver wrist error can look small while the fingertip is still far below
# the button. A 17 cm Z miss must not be played.
PBVS_TIP_MATCH_M = 0.02
PBVS_Z_TOL_M = 0.01
PBVS_MAX_STEP_M = 0.020
PBVS_STEP_SECONDS = 0.45
# Y and Z already this close means the remaining gap is forward. A full pose
# solve folds the elbow in place instead of moving the fingertip in X.
PBVS_FORWARD_YZ_TOL_M = 0.020
# Coarse PBVS is one 6-DoF move to the button, same pose path as
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
# Same order and URDF limits as the hand-eye sweep joint jog.
RIGHT_ARM_JOINT_NAMES = (
    "肩 pitch",
    "肩 roll",
    "肩 yaw",
    "右肘",
    "腕 roll",
    "腕 pitch",
    "腕 yaw",
)
RIGHT_ARM_JOINT_LIMITS_RAD = (
    (-2.618, 1.833),
    (-2.494, 0.517),
    (-2.618, 2.618),
    (-0.986, 3.071),
    (-2.618, 2.618),
    (-0.576, 0.576),
    (-1.22, 1.22),
)
RIGHT_ELBOW_INDEX = 3
RIGHT_ELBOW_LIMIT_RAD = RIGHT_ARM_JOINT_LIMITS_RAD[RIGHT_ELBOW_INDEX]
DIRECT_EE_RPY_DEG = (-88.8133, -3.98969, 3.20691)
# 5 mm left a Y miss of 5 mm marked arrived, so the arm stopped.
# The press has to keep moving until every axis is inside 2 mm.
DIRECT_PBVS_ARRIVE_TOL_M = 0.002
# A fingertip already deeper than the button is not arrival, even inside 2 mm.
DIRECT_PBVS_PAST_X_M = 0.001
# Combination: move off and retry the same press this many times, then the next step.
SEQUENCE_ACTION_RETRIES = 3
# Kept so older callers still import. The live path does not use an X gap.
DIRECT_PBVS_X_CLEAR_M = 0.08
DIRECT_PBVS_X_LEAD_M = 0.05
DIRECT_PBVS_X_RELAX_M = 0.02
DIRECT_PBVS_YZ_STALL_M = 0.002
# The service plans one open-loop Cartesian shot. From a distant pose its IK
# may stop inside the 25 mm gate, and the arm then holds that miss. Later
# shots start from the joints actually reached. A 2 cm Y miss is not done.
DIRECT_PBVS_MAX_SHOTS = 3
# One Start used to stop after a single burst. The arm undershoots, and the
# operator had to press Start again once or twice. Extra rounds continue from
# the pose actually reached. The right-wrist camera still ends a press.
DIRECT_PBVS_START_ROUNDS = 3
# A hang-to-level lift is about 130° on one joint. A low arm reaching a
# chest-height button is about 150°. The wrist often stops ~20° short of
# the exact motion-control roll.
# 15 mm rejected a pitch-8 solution at 16.0 mm / 9° / 99°.
# 25 mm / 150° then rejected the same kind of lift at 38.9 mm / 24.6° / 151°.
# 80 mm is still a bad solve and must not be sent. Arrival stays at 2 mm.
DIRECT_PBVS_POSE_POS_TOL_M = 0.045
DIRECT_PBVS_POSE_ROT_TOL_DEG = 25.0
DIRECT_PBVS_MIN_JOINT_DEG = 165.0
# One-shot endpoint IK can land on the other elbow branch. The last knot has
# to stay this close to the continuous chain, or that jump is not played.
PBVS_BRANCH_JOINT_RAD = math.radians(40.0)
# Wrist and chest button positions closer than this are averaged. Farther ones stay chest-only.
EYE_IN_HAND_AGREE_M = 0.02
EYE_IN_HAND_DWELL_S = 5.0
WRIST_RED_LIFT_STEP_M = 0.005
WRIST_RED_LIFT_MAX_M = 0.05
# Inside the D435 minimum range the depth image is empty. That means the
# panel is already closer than 5 cm. Align in the image, then walk torso +X
# to the farther of (current X + 3 cm) and the coarse PBVS target X.
# A nearer panel stops on the robot's own obstacle check.
WRIST_CLOSE_NOMINAL_Z_M = 0.04
WRIST_CLOSE_ADVANCE_M = 0.03
WRIST_CLOSE_ADVANCE_S = 1.0
_IBVS_CFG = load_ibvs_config()
WRIST_TARGET_MAX_AGE_S = float(_IBVS_CFG.get("wrist_target_max_age_s", 0.50))
WRIST_APPROACH_MAX_STEP_M = float(_IBVS_CFG.get("wrist_approach_max_step_m", 0.004))
WRIST_IBVS_LAMBDA_XY = float(_IBVS_CFG.get("wrist_ibvs_lambda_xy", 0.8))
WRIST_IBVS_LAMBDA_Z = float(_IBVS_CFG.get("wrist_ibvs_lambda_z", 0.6))
WRIST_IBVS_DT_S = float(_IBVS_CFG.get("wrist_ibvs_dt_s", 0.25))
WRIST_IBVS_PIXEL_TOL = float(_IBVS_CFG.get("wrist_ibvs_pixel_tol", 8.0))
WRIST_IBVS_DEPTH_TOL_M = float(_IBVS_CFG.get("wrist_ibvs_depth_tol_m", 0.004))
WRIST_IBVS_PIXEL_JUMP = float(_IBVS_CFG.get("wrist_ibvs_pixel_jump", 80.0))
WRIST_IBVS_FILTER_ALPHA = float(_IBVS_CFG.get("wrist_ibvs_filter_alpha", 0.55))
BUTTON_NAMES = ("button_1", "button_2", "button_3")
DIRECT_EE_PREALIGN_STANDOFF_M = 0.15
# Seeds and small RPY changes for one exact goal, or this search budget runs out.
PBVS_PATH_TIMEOUT_S = 8.0
PBVS_STAGE_TIMEOUT_S = 30.0
# /h2_arm/singlearmcartesian calls Plan(), not PlanJointSpace(). The hand
# follows one Cartesian quintic. feedrate 0.3 applies only to the unused
# Ruckig joint-space branch. Cartesian time uses the raw 80/140 deg/s limits
# and is at least the yaml move_duration of 2 s.
# MOTION PARAMETERS are defined in the yaml file, which are used to calculate the motion duration.
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
# TRAJECTORY_JOINT_INDEX is the index of the joints along the trajectory, which is used to calculate the motion duration.
# RIGHT_ARM_JOINTS is the joints of the right arm, which is used to calculate the motion duration.
# MOTION_TRACK_TOL_DEG is the tolerance of the motion tracking, which is used to calculate the motion duration.
# MOTION_TRACK_WAIT_S is the wait time of the motion tracking, which is used to calculate the motion duration.
# MOTION_MAX_VEL_DEG_S is the maximum velocity of the motion, which is used to calculate the motion duration.
# MOTION_MAX_ACC_DEG_S2 is the maximum acceleration of the motion, which is used to calculate the motion duration.
# TRAJECTORY_JOINT_INDEX is the index of the joints along the trajectory, which is used to calculate the motion duration.
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


CHEST_POSE_HEIGHT_TOL_M = 0.008
CHEST_POSE_HEIGHT_PX = 40.0
SEQUENCE_BUTTON_LABELS = {
    "button_1": "Button1",
    "button_2": "Button2",
    "button_3": "Button3",
}
# One press plus two repeats. A screen that never changes stops the step.
SCREEN_PRESS_ATTEMPTS = 3


def close_range_forward_x(
    current_x_m: float,
    pbvs_x_m: float | None,
    *,
    advance_m: float = 0.03,
) -> float:
    """Torso X after a no-depth alignment.

    Reach the button X, at most 3 cm this step when the button is farther.
    Never command an X deeper than the button.
    """
    current = float(current_x_m)
    goal = current + float(advance_m)
    if pbvs_x_m is not None and math.isfinite(float(pbvs_x_m)):
        button_x = float(pbvs_x_m)
        if current >= button_x:
            return button_x
        goal = min(goal, button_x)
    return goal


def chest_pose_height_z(
    button_z_m: float | None,
    ee_z_m: float | None,
    *,
    max_step_m: float = 0.004,
    tol_m: float = CHEST_POSE_HEIGHT_TOL_M,
) -> float | None:
    """Torso Z step from the chest-pose button toward the end effector.

    Positive raises. None means the chest pose does not currently see a
    button, so the wrist camera keeps the approach, including height.
    """
    if button_z_m is None or ee_z_m is None:
        return None
    if not math.isfinite(button_z_m) or not math.isfinite(ee_z_m):
        return None
    gap = float(button_z_m) - float(ee_z_m)
    if abs(gap) <= float(tol_m):
        return 0.0
    return max(-float(max_step_m), min(float(max_step_m), gap))


def _chest_pixel_gap(button_uv, tip_uv) -> float | None:
    """Image v of the tip minus the button. Negative means the tip is above the button."""
    if not button_uv or not tip_uv or len(button_uv) < 2 or len(tip_uv) < 2:
        return None
    try:
        gap_px = float(tip_uv[1]) - float(button_uv[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gap_px):
        return None
    return gap_px


def chest_height_command(
    button_z_m: float | None,
    ee_z_m: float | None,
    button_uv,
    tip_uv,
    *,
    max_step_m: float = 0.004,
    tol_m: float = CHEST_POSE_HEIGHT_TOL_M,
    image_tol_px: float = CHEST_POSE_HEIGHT_PX,
) -> tuple[float, str] | None:
    """Torso Z step from a live chest pose button.

    Measured button height decides both directions. The image is only used
    when that button has no depth, and then it can still raise or lower.
    """
    z_step = chest_pose_height_z(
        button_z_m,
        ee_z_m,
        max_step_m=max_step_m,
        tol_m=tol_m,
    )
    if z_step is not None and abs(z_step) >= 0.0005:
        gap_mm = 0.0 if button_z_m is None or ee_z_m is None else (float(button_z_m) - float(ee_z_m)) * 1000.0
        if gap_mm < 0.0:
            return z_step, f"胸口 pose 看到按键，末端偏高 {abs(gap_mm):.0f}mm，降低"
        return z_step, f"胸口 pose 看到按键，末端偏低 {gap_mm:.0f}mm，抬高"
    pixel_gap = _chest_pixel_gap(button_uv, tip_uv)
    if pixel_gap is not None and abs(pixel_gap) > float(image_tol_px):
        step = -float(max_step_m) if pixel_gap < 0.0 else float(max_step_m)
        if pixel_gap < 0.0:
            return step, f"胸口看到按键，画面里末端偏高 {abs(pixel_gap):.0f}px，降低"
        return step, f"胸口看到按键，画面里末端偏低 {pixel_gap:.0f}px，抬高"
    return None


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


def pbvs_forward_step(
    error_m: Sequence[float],
    *,
    yz_tol_m: float = PBVS_XY_TOL_M,
    arrive_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Step only in torso +X once height and lateral error are already small.

    Returns None when X is not the remaining gap. The step is capped so the
    elbow stays on the current branch.
    """
    if len(error_m) != 3:
        return None
    try:
        ex, ey, ez = (float(value) for value in error_m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ex, ey, ez)):
        return None
    if abs(ey) > float(yz_tol_m) or abs(ez) > float(yz_tol_m) or ex <= float(arrive_m):
        return None
    return (min(ex, float(max_step_m)), 0.0, 0.0)


def pbvs_retract_step(
    error_m: Sequence[float],
    *,
    past_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Pull torso -X when the fingertip is already deeper than the button.

    error_m is goal minus tip. A negative X means the jaw is past the key and
    into the wall. Raising from there cannot lift, so X comes back alone.
    """
    if len(error_m) != 3:
        return None
    try:
        ex = float(error_m[0])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ex) or ex >= -float(past_m):
        return None
    return (max(ex, -float(max_step_m)), 0.0, 0.0)


def coarse_x_ceiling(x_pbvs: float, clearance_m: float = COARSE_X_CLEARANCE_M) -> float:
    """Forbidden plane before the wrist approach: x_pbvs minus 0.3 cm."""
    return float(x_pbvs) - float(clearance_m)


def coarse_x_command(x_pbvs: float, clearance_m: float = COARSE_X_CLEARANCE_M) -> float:
    """Coarse command X. A tenth of a millimetre keeps it off the plane itself."""
    return coarse_x_ceiling(x_pbvs, clearance_m) - 1e-4


def clamp_coarse_goal(
    goal: Sequence[float],
    x_pbvs: float,
    clearance_m: float = COARSE_X_CLEARANCE_M,
) -> tuple[float, float, float]:
    """Keep a coarse Cartesian goal strictly before the pre-approach X plane."""
    command = coarse_x_command(x_pbvs, clearance_m)
    gx, gy, gz = (float(value) for value in goal)
    if gx > command:
        gx = command
    return (gx, gy, gz)


def service_rise_before_wall(
    current: Sequence[float],
    goal: Sequence[float],
    *,
    rise_m: float = 0.02,
    advance_m: float = 0.01,
) -> tuple[tuple[float, float, float], ...]:
    """Split a service line that would climb and move toward the wall together.

    /h2_arm/singlearmcartesian interpolates one straight XYZ segment. From
    below the panel that diagonal swings the forearm underneath and into the
    wall. Rise (and match Y) at the current X, then go forward.
    """
    try:
        gx, gy, gz = (float(value) for value in goal)
    except (TypeError, ValueError):
        return ((0.0, 0.0, 0.0),)
    goal_t = (gx, gy, gz)
    try:
        cx, cy, cz = (float(value) for value in current)
    except (TypeError, ValueError):
        return (goal_t,)
    if not all(math.isfinite(value) for value in (cx, cy, cz, gx, gy, gz)):
        return (goal_t,)
    if gz - cz <= float(rise_m) or gx - cx <= float(advance_m):
        return (goal_t,)
    lift = (min(cx, gx), gy, gz)
    if math.dist((cx, cy, cz), lift) < 0.008 or math.dist(lift, goal_t) < 0.008:
        return (goal_t,)
    return (lift, goal_t)


def coarse_x_retract(
    tip_x: float,
    x_pbvs: float,
    *,
    clearance_m: float = COARSE_X_CLEARANCE_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Pull -X when the fingertip is already on or past the pre-approach plane."""
    try:
        tip = float(tip_x)
        button = float(x_pbvs)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(tip) or not math.isfinite(button):
        return None
    if tip < coarse_x_ceiling(button, clearance_m):
        return None
    delta = coarse_x_command(button, clearance_m) - tip
    if delta >= -1e-6:
        return None
    return (max(delta, -float(max_step_m)), 0.0, 0.0)

def minimum_z_approach(
    error_m: Sequence[float],
    *,
    z_tol_m: float = PBVS_Z_TOL_M,
) -> tuple[float, float, float] | None:
    """Minimum Z approach."""
    if len(error_m) != 3:
        return None
    try:
        ez = float(error_m[2])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ez) or ez >= -float(z_tol_m):
        return None
    return (0.0, 0.0, max(-float(max_step_m), ez))

def ibvs_lateral_step(
    error_m: Sequence[float],
    *,
    x_tol_m: float = PBVS_FORWARD_YZ_TOL_M,
    arrive_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Step only in torso Y once X is already close and Y is still off."""
    if len(error_m) != 3:
        return None
    try:
        ex, ey, _ez = (float(value) for value in error_m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ex, ey)):
        return None
    if abs(ex) > float(x_tol_m) or abs(ey) <= float(arrive_m):
        return None
    return (0.0, max(-float(max_step_m), min(float(max_step_m), ey)), 0.0)

class IBVS_HeartBeat:
    def __init__(self, iface: Interface, telemetry: Telemetry, log: Callable[[str], None]) -> None:
        self.iface = iface
        self.telemetry = telemetry
        self.log = log
        self.heartbeat_interval_s = 1.0
        self.last_heartbeat_time = time.time()
        self.heartbeat_count = 0
        self.heartbeat_log_interval_s = 10.0
        self.last_heartbeat_log_time = time.time()
        self.running = True
    def run(self) -> None:
        while True:
            time.sleep(self.heartbeat_interval_s)
            self.heartbeat_count += 1
            if self.heartbeat_count % self.heartbeat_log_interval_s == 0:
                self.log(f"Heartbeat: {self.heartbeat_count}")
            if time.time() - self.last_heartbeat_time > self.heartbeat_interval_s:
                self.log("Heartbeat timeout")
                break
    def stop(self) -> None:
        self.running = False


def pbvs_lateral_step(
    error_m: Sequence[float],
    *,
    x_tol_m: float = PBVS_FORWARD_YZ_TOL_M,
    arrive_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Step only in torso Y once X is already close and Y is still off.

    A 3 cm lateral miss puts the wrist camera beside the panel. A full pose
    solve from there folds the elbow instead of sliding sideways.
    """
    if len(error_m) != 3:
        return None
    try:
        ex, ey, _ez = (float(value) for value in error_m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ex, ey)):
        return None
    if abs(ex) > float(x_tol_m) or abs(ey) <= float(arrive_m):
        return None
    return (0.0, max(-float(max_step_m), min(float(max_step_m), ey)), 0.0)


def pbvs_descent_step(
    error_m: Sequence[float],
    *,
    height_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Step only in torso -Z when the tip is already well above the goal.

    error_m is goal minus tip. A large negative Z means the gripper is too
    high. A full pose solve from there lifts the elbow instead of lowering.
    """
    if len(error_m) != 3:
        return None
    try:
        ez = float(error_m[2])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ez) or ez >= -float(height_m):
        return None
    return (0.0, 0.0, max(-float(max_step_m), ez))


def pbvs_yz_aligned(
    error_m: Sequence[float],
    yz_tol_m: float = PBVS_XY_TOL_M,
) -> bool:
    """True when torso Y and Z are already inside the PBVS gate."""
    if len(error_m) != 3:
        return False
    try:
        ey, ez = float(error_m[1]), float(error_m[2])
    except (TypeError, ValueError):
        return False
    if not math.isfinite(ey) or not math.isfinite(ez):
        return False
    return abs(ey) <= float(yz_tol_m) and abs(ez) <= float(yz_tol_m)


def pbvs_yz_hold_goal(
    current: Sequence[float],
    button: Sequence[float],
    *,
    x_clear_m: float = PBVS_YZ_X_CLEAR_M,
    yz_tol_m: float = PBVS_XY_TOL_M,
) -> tuple[float, float, float] | None:
    """Button Y and Z, with X held at least x_clear_m short of the button.

    None means Y and Z are already aligned, so the caller may close X.
    A tip that is already farther back stays at its current X.
    """
    try:
        cx, _cy, _cz = (float(value) for value in current)
        bx, by, bz = (float(value) for value in button)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (cx, bx, by, bz)):
        return None
    error = (bx - cx, by - float(current[1]), bz - float(current[2]))
    if pbvs_yz_aligned(error, yz_tol_m):
        return None
    return (min(cx, bx - float(x_clear_m)), by, bz)


def pbvs_yz_align_step(
    error_m: Sequence[float],
    *,
    x_clear_m: float = PBVS_YZ_X_CLEAR_M,
    yz_tol_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Move Y and Z onto the button. X only backs up to restore x_clear_m.

    None when Y and Z are already inside tolerance. A forward X step is not
    included: that waits until this move is solved or YZ is aligned.
    """
    if len(error_m) != 3:
        return None
    try:
        ex, ey, ez = (float(value) for value in error_m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ex, ey, ez)):
        return None
    if abs(ey) <= float(yz_tol_m) and abs(ez) <= float(yz_tol_m):
        return None
    step = float(max_step_m)
    dx = 0.0
    if ex < float(x_clear_m):
        dx = max(ex - float(x_clear_m), -step)
    dy = 0.0 if abs(ey) <= float(yz_tol_m) else max(-step, min(step, ey))
    dz = 0.0 if abs(ez) <= float(yz_tol_m) else max(-step, min(step, ez))
    if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
        return None
    return (dx, dy, dz)


def pbvs_x_step(
    error_m: Sequence[float],
    *,
    arrive_m: float = PBVS_XY_TOL_M,
    max_step_m: float = PBVS_MAX_STEP_M,
) -> tuple[float, float, float] | None:
    """Close torso +X without waiting for Y or Z. None when X is not still short."""
    if len(error_m) != 3:
        return None
    try:
        ex = float(error_m[0])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ex) or ex <= float(arrive_m):
        return None
    return (min(ex, float(max_step_m)), 0.0, 0.0)


def pbvs_coarse_delta(
    error_cam_m: list[float] | tuple[float, float, float],
) -> tuple[tuple[float, float, float], bool]:
    """Drive the fingertip onto the button. Camera Z is not held back."""
    ex, ey, ez = (float(value) for value in error_cam_m)
    done = (
        abs(ex) <= PBVS_XY_TOL_M
        and abs(ey) <= PBVS_XY_TOL_M
        and abs(ez) <= PBVS_Z_TOL_M
    )
    return (ex, ey, ez), done


def should_jog_ibvs(telem: dict[str, Any], height_locked: bool) -> bool:
    """Jog from chest image error or a measured side-view approach. FK does not count."""
    if not height_locked:
        return False
    if telem.get("side_contact") or telem.get("phase") == "CONTACT":
        return False
    return ibvs_features_ready(telem)


def direct_pbvs_round_stalled(distance_m: float, round_start_m: float, track_rad: float) -> bool:
    """A round that gained under 2 mm is finished only after the joints have settled.

    Eight degrees of lag is about 70 mm at the fingertip. Quitting there leaves
    the press short of the 2 mm gate.
    """
    if float(distance_m) <= DIRECT_PBVS_ARRIVE_TOL_M:
        return True
    if float(track_rad) > math.radians(1.5):
        return False
    return float(round_start_m) - float(distance_m) < 0.002


def direct_pbvs_arrived(error_m: Sequence[float]) -> bool:
    """True only when the tip is within 2 mm and not past the button in X.

    error_m is goal minus tip. Negative X means the fingertip is already
    deeper than the key. X -3 mm must stay 未到位.
    """
    if len(error_m) != 3:
        return False
    try:
        ex, ey, ez = (float(value) for value in error_m)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (ex, ey, ez)):
        return False
    if math.sqrt(ex * ex + ey * ey + ez * ez) > DIRECT_PBVS_ARRIVE_TOL_M:
        return False
    if ex < -DIRECT_PBVS_PAST_X_M:
        return False
    return True


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


def cap_tool_x(point: Sequence[float], limit_x: float) -> tuple[float, float, float]:
    """Torso +X is into the panel. The fingertip must not be commanded past the button."""
    return (min(float(point[0]), float(limit_x)), float(point[1]), float(point[2]))


def direct_pbvs_motion_goal(
    current: Sequence[float],
    button: Sequence[float],
    yz_tol_m: float = DIRECT_PBVS_ARRIVE_TOL_M,
    x_clear_m: float = DIRECT_PBVS_X_CLEAR_M,
    x_lead_m: float = DIRECT_PBVS_X_LEAD_M,
) -> tuple[float, float, float]:
    """September 30 path: the fingertip goes to the detected button point.

    Do not hold X short, and do not back the hand up. Those offsets either
    parked the arm on the chest or threw the recovery behind the body.
    """
    _ = (current, yz_tol_m, x_clear_m, x_lead_m)
    return (float(button[0]), float(button[1]), float(button[2]))


def level_right_waypoints(current: Sequence[float] | None) -> list[tuple[float, float, float, float, float, float, float]]:
    """Retired. Do not send these joints: the motion hits the chest camera."""
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


def sequence_lift_goal(button: Sequence[float]) -> tuple[float, float, float]:
    """September 30 path: no side retreat between presses."""
    return (float(button[0]), float(button[1]), float(button[2]))

def load_yolo_models() -> None:
    """Load the YOLO models."""
    pass
# Torso +Y is left and +Z is up, so clearing the faceplate is -Y and -Z.
SEQUENCE_CLEAR_RIGHT_M = 0.12
SEQUENCE_CLEAR_DOWN_M = 0.10


def sequence_clear_goal(tip: Sequence[float]) -> tuple[float, float, float]:
    """Shift the fingertip down and to the robot's right, off the faceplate."""
    return (
        float(tip[0]),
        float(tip[1]) - SEQUENCE_CLEAR_RIGHT_M,
        float(tip[2]) - SEQUENCE_CLEAR_DOWN_M,
    )


class SequenceNext(Exception):
    """The operator asked to leave the current combination action."""


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


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _uv_pair(value: Any) -> tuple[float, float] | None:
    if not value or len(value) < 2:
        return None
    u = _finite_float(value[0])
    v = _finite_float(value[1])
    if u is None or v is None:
        return None
    return u, v


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


def max_joint_gap(left: Sequence[float], right: Sequence[float]) -> float:
    """Largest absolute joint difference. Used to reject an elbow-branch jump."""
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def quintic_blend(phase: float) -> float:
    phase = min(1.0, max(0.0, float(phase)))
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def expand_quintic_knots(
    knots: Sequence[Sequence[float]],
    total_s: float,
    dt: float,
) -> list[list[float]]:
    """50 Hz samples along the solved joint polyline.

    The knots already sit on one Cartesian quintic, and each one was accepted
    by IK. Walking each gap at constant joint speed follows that quintic and
    keeps every sample between two accepted knots. A quintic inside every gap
    stops the arm at each knot. A single two-knot segment still eases in and
    out, because it has no interior knot to carry the slow start.
    """
    if not knots:
        return []
    if len(knots) == 1:
        return [[float(value) for value in knots[0]]]
    intervals = len(knots) - 1
    steps = max(1, int(round(max(float(total_s), dt) / intervals / dt)))
    single_segment = intervals == 1
    samples: list[list[float]] = []
    for start, end in zip(knots, knots[1:]):
        for step in range(steps):
            phase = step / steps
            blend = quintic_blend(phase) if single_segment else phase
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

def quintic_blend(phase: float) -> float:
    phase = min(1.0, max(0.0, float(phase)))
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5

def logger_info(message: str) -> None:
    message = f"ArmMover: {message}" 
    print(message)

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
        motion_backend: str = "arm_sdk",
        right_ee: str = "dex1",
    ) -> None:
        self.iface = iface
        self.telemetry = telemetry
        self.confirm = bool(confirm)
        self.log = log
        self.on_ee_pose = on_ee_pose
        self.wrist_watch = None
        self.prepare_right_hand = None
        self.wrist_hold = None
        self.wrist_view = None
        self.cam2hand_matrix = None
        self.on_screen_capture = None
        self.on_screen_record = None
        self.on_task_begin = None
        self.on_task_end = None
        self.read_screen = None
        self.screen_settle_s = 0.6
        self._screen_progress = False
        self._screen_report = ""
        self._screen_history: list[str] = []
        self._action_pose_history: list[str] = []
        self._action_pose_pending = False
        self._action_label = ""
        self._action_goal: tuple[float, float, float] | None = None
        self._screen_block: dict[str, Any] | None = None
        self._screen_before = None
        self._screen_before_info: dict[str, Any] | None = None
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
        self.right_ee = str(right_ee or "dex1").strip().lower()
        if self.right_ee == "revo2":
            from ibvs_pad_touch.web.revo2_index_pad import index_tip_in_wrist, index_tip_tool_rpy_deg

            self.tool_offset_wrist_m = index_tip_in_wrist()
            self.tool_rpy_deg = index_tip_tool_rpy_deg()
            self.log(
                "[ARM] 右手末端=Revo2 食指指尖 "
                f"xyz={tuple(round(value, 4) for value in self.tool_offset_wrist_m)} m "
                f"rpy={tuple(round(value, 2) for value in self.tool_rpy_deg)} deg"
            )
        elif self.right_ee == "dex1":
            self.tool_offset_wrist_m = RIGHT_EE_OFFSET_WRIST_M
            self.tool_rpy_deg = DIRECT_EE_RPY_DEG
        else:
            raise ValueError("right_ee 只能是 dex1 或 revo2")
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
        self._start_arm_sdk = None
        self._arm_sdk_thread: Optional[threading.Thread] = None
        self._drop_arm_sdk = None
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
        self._wrist_target_filtered: tuple[float, float, float] | None = None
        self._wrist_ibvs_step: tuple[float, float, float] | None = None
        self._wrist_ibvs_uv: tuple[float, float] | None = None
        self._wrist_ibvs_jump_frames = 0
        self._wrist_yellow_id = 0
        self._wrist_missing_frames = 0
        self._wrist_latched_torso: dict[str, tuple[float, float, float]] = {}
        self._wrist_red_lift_m = 0.0
        self._wrist_close_advanced = False
        self._wrist_close_advance_m = 0.0
        self._chest_height_after_pbvs = False
        self._chest_height_hold = False
        self._forward_hold = False
        self._button_sequence: list[str] = []
        self._sequence_mode = "hybrid"
        self._sequence_run = False
        self._sequence_stage = ""
        self._sequence_index = -1
        self._retry_count = 0
        self._sequence_lock = threading.Lock()
        self._sequence_skip = 0
        self._sequence_inside = False
        self._sequence_resume = False
        self._sequence_resume_index = 0
        self._sequence_finish_only = False
        self._motion_backend = self._normalize_backend(motion_backend, default="arm_sdk")
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
            "message": self._message_for_page(),
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
            "right_ee": self.right_ee,
            "waist_yaw_error_deg": (
                None
                if self.controller is None
                else getattr(self.controller, "waist_yaw_error_deg", None)
            ),
            "waist_yaw_tau": (
                None if self.controller is None else getattr(self.controller, "waist_yaw_tau", None)
            ),
            "body_yaw_error_deg": (
                None
                if self.controller is None
                else getattr(self.controller, "body_yaw_error_deg", None)
            ),
            "body_yaw_rate": (
                None if self.controller is None else getattr(self.controller, "body_yaw_rate", None)
            ),
            "tool_offset_wrist_m": [round(float(value), 4) for value in self.tool_offset_wrist_m],
            "tool_rpy_deg": [round(float(value), 2) for value in self.tool_rpy_deg],
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
            "sequence_mode": self._sequence_mode,
            "screen_progress": self._screen_progress,
            "screen_report": self._screen_report,
            "screen_history": list(self._screen_history),
            "action_pose_history": list(self._action_pose_history),
            "screen_block": None if self._screen_block is None else dict(self._screen_block),
            "sequence_run": self._sequence_run,
            "sequence_stage": self._sequence_stage,
            "locked_buttons": {name: list(value) for name, value in self._locked_buttons.items()},
            "pbvs_source": self._pbvs_source,
            "wrist_command_step_m": (
                None
                if self._wrist_ibvs_step is None
                else [float(value) for value in self._wrist_ibvs_step]
            ),
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
            self.message = "模式 Hybrid VS：胸口 PBVS 粗定位，右腕图像 IBVS 完成末端进近"
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

    @staticmethod
    def _normalize_backend(backend: str, default: str = "") -> str:
        chosen = str(backend or "").strip().lower().replace("-", "_")
        if chosen in {"sdk", "arm"}:
            chosen = "arm_sdk"
        if chosen in {"", "arm_sdk", "service"}:
            return chosen
        return default

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

    def _arm_sdk_bringup(self) -> None:
        """Stop a competing service node, then connect. Runs off the web request."""
        try:
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
                    return
            fn = self._start_arm_sdk
            if fn is not None:
                ok = bool(fn())
                if not ok:
                    self.message = "arm_sdk 没有起来。请再点一次 arm_sdk。"
                    self.log(f"[ARM][WARN] {self.message}")
                    return
            else:
                self._connect()
            if fn is None and self.controller is None:
                self.message = "arm_sdk 没有起来。请再点一次 arm_sdk。"
                self.log(f"[ARM][WARN] {self.message}")
                return
            self._motion_backend = "arm_sdk"
            self.message = "已拉起 arm_sdk。控制链路：arm_sdk。"
            self.log(f"[ARM] {self.message}")
        except Exception as exc:
            self._drop_failed_sdk()
            self.message = f"arm_sdk 没有起来: {exc}"
            self.log(f"[ARM][WARN] {self.message}")

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
            chosen == "arm_sdk"
            and self._sdk_holding()
            and not node_up
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
            worker = self._arm_sdk_thread
            if worker is not None and worker.is_alive():
                self._motion_backend = "arm_sdk"
                self.message = "正在拉起 arm_sdk"
                return self.message
            self._motion_backend = "arm_sdk"
            self.message = "正在拉起 arm_sdk"
            self.log(f"[ARM] {self.message}")
            self._arm_sdk_thread = threading.Thread(
                target=self._arm_sdk_bringup,
                name="arm-sdk-start",
                daemon=True,
            )
            self._arm_sdk_thread.start()
            return self.message
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
        """Refuse. Right-forearm leveling hits the chest camera."""
        self._level_right.clear()
        self.message = "右臂平举已关闭：该姿态会撞胸口相机，手臂保持不动"
        self.log(f"[ARM] {self.message}")
        return self.message

    def set_screen_progress(self, enabled: bool) -> str:
        """Watch the panel after each press. Off leaves the arm path unchanged."""
        if self._ibvs_enabled or self._running.is_set() or self._sequence_run:
            self.message = "先 Pause 或等这一轮结束，再切换读取屏幕"
            self.log(f"[ARM] {self.message}")
            return self.message
        self._screen_progress = bool(enabled)
        self.message = "读取屏幕判断进度：开" if self._screen_progress else "读取屏幕判断进度：关"
        self.log(f"[ARM] {self.message}")
        return self.message

    def set_sequence_mode(self, mode: str) -> str:
        """Choose Direct PBVS or Hybrid for the next combination. Does not move the arm."""
        if self._ibvs_enabled or self._running.is_set() or self._return_ready.is_set() or self._hang_down.is_set() or self._level_right.is_set():
            self.message = "先 Pause 或等 Reset 完成，再切换组合模式"
            self.log(f"[ARM] {self.message}")
            return self.message
        chosen = str(mode or "").strip().lower()
        if chosen == "direct_pbvs":
            self._sequence_mode = "direct_pbvs"
            self.message = "动作组合：Direct PBVS，每颗一次到位，不做右腕进近"
        elif chosen == "hybrid":
            self._sequence_mode = "hybrid"
            self.message = "动作组合：Hybrid，胸口 PBVS 后右腕进近"
        else:
            self.message = "未知组合模式"
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

    def advance_sequence(self) -> str:
        """Skip the current combination action, or finish and release on the last one."""
        steps = list(self._button_sequence)
        if not steps:
            self.message = "动作组合为空，请先添加 Button"
            self.log(f"[ARM] {self.message}")
            return self.message
        with self._sequence_lock:
            inside = self._sequence_inside
            if inside:
                self._sequence_skip += 1
        if inside:
            last = self._sequence_index >= len(steps) - 1
            self.message = (
                "Next Step：最后一个动作，放下右臂并释放"
                if last
                else "Next Step：移开手臂，进入下一步"
            )
            self.log(f"[ARM] {self.message}")
            return self.message
        if self._sequence_index < 0:
            self.message = "请先点组合启动"
            self.log(f"[ARM] {self.message}")
            return self.message
        if self.controller is None or self._thread is None or not self._thread.is_alive():
            self.message = "Next Step 需要手臂仍由本页控制，请先组合启动"
            self.log(f"[ARM] {self.message}")
            return self.message
        nxt = self._sequence_index + 1
        self._stop.clear()
        self._sequence_run = True
        self._ibvs_enabled = True
        self._running.set()
        self._control_stage = "DIRECT_PBVS"
        if nxt >= len(steps):
            self._sequence_resume = False
            self._sequence_finish_only = True
            self._sequence_stage = "结束放下"
            self.message = "Next Step：最后一个动作，放下右臂并释放"
        else:
            self._sequence_finish_only = False
            self._sequence_resume_index = nxt
            self._sequence_resume = True
            label = SEQUENCE_BUTTON_LABELS.get(steps[nxt], steps[nxt])
            self.message = f"Next Step：移开手臂，进入 {label}"
        self.log(f"[ARM] {self.message}")
        return self.message

    def _raise_if_sequence_next(self) -> None:
        with self._sequence_lock:
            if self._sequence_skip <= 0:
                return
            self._sequence_skip -= 1
        raise SequenceNext()

    def start(self, *, sequence: bool = False) -> str:
        self._sequence_run = False
        self._sequence_stage = ""
        self._sequence_index = -1
        self._sequence_resume = False
        self._sequence_finish_only = False
        with self._sequence_lock:
            self._sequence_skip = 0
            self._sequence_inside = False
        if callable(self.wrist_watch):
            self.wrist_watch(False)
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
        self._wrist_target_filtered = None
        self._wrist_ibvs_step = None
        self._wrist_ibvs_uv = None
        self._wrist_ibvs_jump_frames = 0
        self._wrist_missing_frames = 0
        self._wrist_latched_torso = {}
        self._wrist_red_lift_m = 0.0
        self._wrist_close_advanced = False
        self._wrist_close_advance_m = 0.0
        self._chest_height_after_pbvs = False
        self._sequence_run = bool(sequence)
        self._reset_left_seek()
        self._running.set()
        if sequence:
            labels = {"button_1": "Button 1", "button_2": "Button 2", "button_3": "Button 3"}
            names = " → ".join(labels.get(name, name) for name in self._button_sequence)
            self._control_stage = "DIRECT_PBVS"
            kind = "Direct PBVS" if self._sequence_mode == "direct_pbvs" else "Hybrid"
            self.message = f"组合启动 {kind}：{names}"
            self._emit_task_begin()
            return self.message
        if self.direct_ee_poke:
            self._control_stage = "DIRECT_EE_ALIGN_YZ"
            self.message = "Start：先在面板前 15cm 对齐 Y/Z，再沿 X 点触"
        elif self.direct_pbvs_poke:
            self._control_stage = "DIRECT_PBVS"
            self.message = "Start：Direct PBVS 交给 /h2_arm/singlearmcartesian"
        elif self.pbvs_first:
            self._control_stage = "PBVS_COARSE"
            self.message = "Start：胸口 PBVS 粗定位，右腕图像 IBVS 接管末端进近"
        else:
            self._control_stage = "DUAL_IBVS"
            self.message = "Start：兼容胸视/侧视模式；建议选择 Hybrid VS 使用右腕 Eye-in-Hand"
        return self.message

    def _prepare_right_hand(self) -> str:
        """Index out, every other finger closed, before the arm moves."""
        fn = self.prepare_right_hand
        if not callable(fn):
            self.message = "右手 Revo2 没有接上，不能进入食指预备姿势"
            self.log(f"[ARM][WARN] {self.message}")
            return self.message
        message = str(fn() or "")
        if "食指" not in message:
            self.message = message or "右手没有进入食指预备姿势"
            self.log(f"[ARM][WARN] {self.message}")
            return self.message
        self.message = f"{message}，外展后手指到位，继续上举"
        self.log(f"[ARM] {self.message}")
        return ""

    def pause(self) -> str:
        self._stop_playback()
        if callable(self.wrist_watch):
            self.wrist_watch(False)
        self._ibvs_enabled = False
        self._control_stage = "MANUAL"
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._pbvs_deadline = None
        self._reset_left_seek()
        self.phase = "manual"
        self.message = "Pause：冻结当前位置，已进入右臂细调"
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
        self._play_joint_target(q_new, 0.35)
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
        return self.manual_joint(RIGHT_ELBOW_INDEX, delta_deg)

    def manual_joint(self, index: int, delta_deg: float) -> dict[str, Any]:
        """Nudge one right-arm joint: shoulder, elbow, then wrist."""
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
            joint = int(index)
            delta = float(delta_deg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "关节角度无效"}
        if joint < 0 or joint >= len(RIGHT_ARM_JOINT_NAMES):
            return {"ok": False, "error": "关节序号无效"}
        label = RIGHT_ARM_JOINT_NAMES[joint]
        if not math.isfinite(delta) or abs(delta) > 5.0 or abs(delta) < 1e-6:
            return {"ok": False, "error": f"单次{label}调节不得超过 5deg"}
        commanded = self.controller.commanded_arm_rad("right")
        if not commanded:
            return {"ok": False, "error": "没有右臂关节状态"}
        q_new = [float(value) for value in commanded]
        low, high = RIGHT_ARM_JOINT_LIMITS_RAD[joint]
        q_new[joint] = min(high, max(low, q_new[joint] + math.radians(delta)))
        if abs(q_new[joint] - float(commanded[joint])) < 1e-5:
            return {"ok": False, "error": f"{label}已到限位"}
        self._play_joint_target(q_new, 0.35)
        joint_deg = math.degrees(q_new[joint])
        self.phase = "manual"
        self.message = f"{label} {joint_deg:.1f}°"
        self._publish_ee()
        result = {"ok": True, "message": self.message, "joint": joint, "joint_deg": round(joint_deg, 2)}
        if joint == RIGHT_ELBOW_INDEX:
            result["elbow_deg"] = result["joint_deg"]
        return result

    def release_for_operator(self) -> str:
        """Drop the current arm_sdk holder. Do not blend weight or start another publisher."""
        self._record_action_pose("已释放")
        self._stop_playback()
        if self.controller is None and (self._thread is None or not self._thread.is_alive()):
            self.phase = "released"
            self._control_stage = "RELEASED"
            self._left_side_ready = False
            self.message = "双臂已释放。arm_sdk 已关闭。"
            self._emit_task_end("stopped")
            return self.message
        self._ibvs_enabled = False
        self._control_stage = "RELEASED"
        self._left_side_ready = False
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._return_ready.clear()
        self._hang_down.clear()
        self._level_right.clear()
        self._height_locked = False
        self._seek_done = False
        self._stop.set()
        controller = self.controller
        if controller is not None:
            abort = getattr(controller, "abort", None)
            try:
                if callable(abort):
                    self.log("[ARM] 释放：只关闭当前 arm_sdk，不拉起运控服务")
                    abort()
                else:
                    self.log("[ARM] 释放：当前控制器没有 abort，改发 weight=0")
                    controller.release(seconds=0.0)
            except Exception as exc:
                self.log(f"[ARM] 松权失败: {exc}")
                try:
                    controller.release(seconds=0.0)
                except Exception as release_exc:
                    self.log(f"[ARM] release: {release_exc}")
        self.controller = None
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self.phase = "released"
        self.message = "已释放双臂。arm_sdk 已关闭，没有拉起新的控制。"
        self.log(f"[ARM] {self.message}")
        self._emit_task_end("stopped")
        return self.message

    def emergency_stop(self) -> str:
        """Drop both arms now and release SDK and the service node. No pose plan."""
        self._record_action_pose("已释放")
        self._stop_playback()
        self._ibvs_enabled = False
        self._control_stage = "ESTOP"
        self._scanning.clear()
        self._scan_needs_prepare = False
        self._running.clear()
        self._return_ready.clear()
        self._hang_down.clear()
        self._level_right.clear()
        self._sequence_run = False
        self._pbvs_release = True
        self._height_locked = False
        self._seek_done = False
        self._left_side_ready = False
        self._stop.set()
        self.phase = "estop"
        controller = self.controller
        self.controller = None
        self._kill_service_now()
        sdk_released = False
        if controller is not None:
            abort = getattr(controller, "abort", None)
            try:
                if callable(abort):
                    abort()
                    sdk_released = bool(getattr(controller, "_aborted", False))
                else:
                    controller.release(seconds=0.0)
            except Exception as exc:
                self.log(f"[ARM] 急停释放失败: {exc}")
        if not sdk_released:
            self._drop_sdk_hold()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self.message = "急停：双臂已立即松开，控制已释放"
        self.log(f"[ARM] {self.message}")
        self._emit_task_end("stopped")
        return self.message

    def _kill_service_now(self) -> None:
        if self._stop_service_node is not None:
            try:
                self._stop_service_node()
            except Exception as exc:
                self.log(f"[ARM] 急停停服务失败: {exc}")
            return
        try:
            from ibvs_pad_touch.web.arm_service import stop_service_node

            stop_service_node(immediate=True)
        except Exception as exc:
            self.log(f"[ARM] 急停停服务失败: {exc}")

    def _drop_failed_sdk(self) -> None:
        """A failed Raise must not leave this page publishing rt/arm_sdk."""
        controller = self.controller
        self.controller = None
        if controller is None or callable(getattr(controller, "move_cartesian", None)):
            return
        try:
            controller.abort()
        except Exception as exc:
            self.log(f"[ARM] 抬臂失败后释放 arm_sdk 失败: {exc}")

    def _drop_sdk_hold(self) -> None:
        fn = self._drop_arm_sdk
        if fn is None:
            from ibvs_pad_touch.web.arm_service import drop_arm_sdk_now

            fn = drop_arm_sdk_now
        try:
            fn(self.iface)
        except TypeError:
            try:
                fn()
            except Exception as exc:
                self.log(f"[ARM] 急停松 arm_sdk 失败: {exc}")
        except Exception as exc:
            self.log(f"[ARM] 急停松 arm_sdk 失败: {exc}")

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
                self.controller = service
                self.controller.ramp_arm("right", RIGHT_HANG_DEG, seconds=8.0, dt=0.02)
                self.phase = "released"
                self.message = "已松开 arm_sdk，并由运控服务下垂"
                self.log("[ARM] 已由运控服务下垂")
            except Exception as exc:
                self.log(f"[ARM] 服务下垂失败: {exc}")
            return
        try:
            self.log("[ARM] 退出：放下左右臂并交回 arm_sdk")
            self._ramp_arms(
                (
                    ("left", LEFT_REST_DEG, 6.0),
                    ("right", REST_DEG, 6.0),
                )
            )
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
        self.controller.hold_waist_yaw = True
        self.controller.wait_lowstate()
        self._ensure_gravity()
        self.controller.ensure_ai_mode(select_ai=True)
        self.controller.enable_arm_sdk()
        self.ik = H2CameraIK(arm="right")
        self.log("[ARM] arm_sdk 已接，重力补偿开，腰 yaw 补偿开")
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
            self._drop_failed_sdk()
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

    def _swing_right_arm_clear(self) -> None:
        """Abduct a low, torso-hugging right arm before any raise.

        Opening the elbow while shoulder roll is still near the body swings the
        gripper into the chest. Only the outward roll changes here.
        """
        if self._motion_backend == "service" or self.controller is None:
            return
        try:
            measured = self.controller.measured_arm_deg("right")
        except Exception:
            return
        if not measured or len(measured) < 7:
            return
        pitch = float(measured[0])
        roll = float(measured[1])
        if pitch <= -25.0 or roll <= -35.0:
            return
        target = [float(value) for value in measured[:7]]
        target[1] = TORSO_CLEAR_ROLL_DEG
        self.message = "右臂先外展离开躯干，再上举"
        self.log(f"[ARM] {self.message}")
        self.controller.ramp_arm("right", target, seconds=3.0, dt=0.02)

    def _parallel_arm_ramps(self) -> bool:
        """arm_sdk can blend each arm into the same command. The service yaml cannot."""
        controller = self.controller
        return (
            self._motion_backend != "service"
            and controller is not None
            and callable(getattr(controller, "write_joints", None))
        )

    def _ramp_arms(self, moves: Sequence[tuple[str, Sequence[float], float]]) -> None:
        """Run the listed arm ramps together when the controller can merge joints."""
        controller = self.controller
        if controller is None:
            return
        if not self._parallel_arm_ramps() or len(moves) < 2:
            for arm, pose, seconds in moves:
                controller.ramp_arm(arm, pose, seconds=float(seconds), dt=0.02)
            return
        errors: list[BaseException] = []

        def run(arm: str, pose: Sequence[float], seconds: float) -> None:
            try:
                controller.ramp_arm(arm, pose, seconds=float(seconds), dt=0.02)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=move, name=f"ramp-{move[0]}", daemon=True)
            for move in moves
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    def _raise_right_to_ready(self) -> None:
        if self._motion_backend != "service":
            self._swing_right_arm_clear()
            self.controller.ramp_arm("right", READY_DEG, seconds=6.0, dt=0.02)
            return
        self.controller.ramp_arm("right", READY_DEG, seconds=8.0, dt=0.02)

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
        if self._parallel_arm_ramps():
            errors: list[BaseException] = []

            def lift_left() -> None:
                try:
                    self.controller.ramp_arm("left", LEFT_SIDE_VIEW_DEG, seconds=8.0, dt=0.02)
                    self._left_side_ready = True
                except BaseException as exc:
                    errors.append(exc)

            def lift_right() -> None:
                try:
                    self._raise_right_to_ready()
                except BaseException as exc:
                    errors.append(exc)

            workers = (
                threading.Thread(target=lift_left, name="ramp-left", daemon=True),
                threading.Thread(target=lift_right, name="ramp-right", daemon=True),
            )
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            if errors:
                raise errors[0]
        else:
            self.controller.ramp_arm("left", LEFT_SIDE_VIEW_DEG, seconds=8.0, dt=0.02)
            self._left_side_ready = True
            self._raise_right_to_ready()
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
        self._ramp_arms(
            (
                ("left", LEFT_REST_DEG, 6.0),
                ("right", REST_DEG, 6.0),
            )
        )
        self._height_locked = False
        self._seek_done = False
        self._control_stage = "HOLD"
        self.phase = "hold_for_start"
        self.message = "已回到放下位，运控服务仍保持" if via_service else "已回到放下位，arm_sdk 仍保持"
        self.log(f"[ARM] {self.message}")

    def _ramp_right_level(self) -> None:
        """Never send the level joints. The pose hits the chest camera."""
        self._level_right.clear()
        self.log("[ARM] 右臂平举已关闭，不发送关节指令")

    def _execute_level_right(self) -> None:
        """Right-forearm leveling is disabled. The arm stays where it is."""
        self._level_right.clear()
        self._running.clear()
        self._ibvs_enabled = False
        self.message = "右臂平举已关闭：该姿态会撞胸口相机，手臂保持不动"
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
        self._ramp_arms(
            (
                ("left", LEFT_HANG_DEG, 8.0),
                ("right", RIGHT_HANG_DEG, 8.0),
            )
        )
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
                tip_fn(q_rad, self.tool_offset_wrist_m)
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
        if self._sequence_finish_only:
            self._sequence_finish_only = False
            self._finish_sequence_and_release()
            return
        if self._sequence_resume:
            start = int(self._sequence_resume_index)
            self._sequence_resume = False
            self._run_button_sequence(start_index=start)
            return
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
        if self._control_stage == "DIRECT_PBVS" and not self._pbvs_wrist_checked:
            self._pbvs_wrist_checked = True
            if (
                self._sequence_run
                and self._button_sequence
                and not self._return_ready.is_set()
                and not self._stop.is_set()
            ):
                self._run_button_sequence()
                return
            if self._screen_progress and not self._return_ready.is_set() and not self._stop.is_set():
                self._run_single_press_with_screen(telem, direct=True)
                return
            chest = self._snapshot_chest_buttons(telem)
            if chest:
                self._locked_buttons = dict(chest)
            self._pbvs_source = "chest"
            if callable(self.wrist_watch):
                self.wrist_watch(False)
            self.log("[ARM] Direct PBVS 只用胸口坐标，不等待右腕")
        if self._control_stage == "DIRECT_PBVS":
            self._direct_pbvs_once(telem)
            return
        if self._control_stage == "PBVS_COARSE":
            if self._screen_progress and not self._sequence_run and not self._return_ready.is_set() and not self._stop.is_set():
                self._run_single_press_with_screen(telem, direct=False)
                return
            self._pbvs_once(telem)
            return
        if self._control_stage == "WRIST_APPROACH":
            self._wrist_approach_once(telem)
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

    def _tool_xyz(self, q_rad) -> tuple[float, float, float]:
        """Tool point in torso_link. Older Hand_Eye_Calib builds have no tip_xyz."""
        offset = self.tool_offset_wrist_m
        tip_fn = getattr(self.ik, "tip_xyz", None)
        if callable(tip_fn):
            try:
                point = tip_fn(q_rad, offset)
            except TypeError:
                point = tip_fn(q_rad)
            return tuple(float(value) for value in point)
        matrix_fn = getattr(self.ik, "fk_matrix", None)
        if callable(matrix_fn):
            pose = matrix_fn(q_rad)
            tip = [float(pose[row][3]) for row in range(3)]
            for row in range(3):
                tip[row] += sum(float(pose[row][col]) * float(offset[col]) for col in range(3))
            return (tip[0], tip[1], tip[2])
        fk = getattr(self.ik, "fk_xyz_rpy", None)
        if not callable(fk):
            raise RuntimeError("FK 指尖模型不可用")
        from ibvs_pad_touch.servo.virtual_tip import apply_ee_offset

        xyz, rpy = fk(q_rad)
        return apply_ee_offset(xyz, rpy, offset)

    def _nominal_tool_rpy_deg(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.tool_rpy_deg)

    def _nominal_tool_rpy(self) -> tuple[float, float, float]:
        return tuple(math.radians(value) for value in self._nominal_tool_rpy_deg())

    def _pbvs_path_candidates(
        self,
        _current_tip: tuple[float, float, float],
        goal: tuple[float, float, float],
        _optical: tuple[float, float, float],
    ):
        """Joint seeds and small RPY changes for this exact goal."""
        nominal = self._nominal_tool_rpy()
        ready = [math.radians(value) for value in READY_DEG]
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
            "tool_offset_wrist_m": self.tool_offset_wrist_m,
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

    def _tip_miss(self, q_rad, goal: tuple[float, float, float]):
        """Tool point minus the button. Uses the active pad or gripper offset."""
        if not q_rad:
            return None
        try:
            tip = self._tool_xyz(q_rad)
        except Exception:
            return None
        if len(tip) != 3 or not all(math.isfinite(value) for value in tip):
            return None
        return tuple(float(goal[index]) - tip[index] for index in range(3))

    def _solved_tip_reaches(self, q_rad, goal: tuple[float, float, float]) -> bool:
        miss = self._tip_miss(q_rad, goal)
        if miss is None:
            return True
        return all(abs(value) <= PBVS_TIP_MATCH_M for value in miss)

    def _solve_tool_sample(self, solve_pose, commanded, measured, xyz, rpy, seed):
        detail = getattr(self.ik, "solve_loaded_tool_pose", None)
        if callable(detail):
            kwargs = {
                "target_tool_xyz_m": xyz,
                "target_tool_rpy_rad": rpy,
                "tool_offset_wrist_m": self.tool_offset_wrist_m,
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

    def _finish_on_the_same_branch(
        self,
        solve_pose,
        commanded,
        measured,
        goal,
        goal_rpy,
        solved_seed,
        endpoint,
        previous,
    ):
        """Goal joints for the last knot, or None when that would flip the elbow.

        Prefer a fresh solve seeded from the chain. The one-shot endpoint is
        kept only when it is already on that same branch.
        """
        finish = self._solve_tool_sample(
            solve_pose, commanded, measured, goal, goal_rpy, solved_seed
        )
        endpoint_q = [float(value) for value in endpoint["q_command"]]
        if finish is not None and self._solved_tip_reaches(finish.get("q_solved"), goal):
            candidate = [float(value) for value in finish["q_command"]]
            # Agreeing with the one-shot means this seed found the same basin,
            # including a long last step. A seed that stays on the walked chain
            # is also kept, and that is what drops an elbow-up one-shot.
            if (
                max_joint_gap(candidate, endpoint_q) <= PBVS_BRANCH_JOINT_RAD
                or max_joint_gap(candidate, previous) <= PBVS_BRANCH_JOINT_RAD
            ):
                return candidate
        if max_joint_gap(endpoint_q, previous) <= PBVS_BRANCH_JOINT_RAD:
            return endpoint_q
        return None

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
        """Endpoint IK first, then one Cartesian knot chain on that same elbow branch.

        The one-shot endpoint is only a gate. The knot that is actually played
        is solved again from the chain's last seed, so a shoulder-yaw flip in
        the one-shot solution cannot become the last segment.
        """
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
            try:
                solved_tip = self._tool_xyz(endpoint.get("q_solved") or endpoint["q_command"])
            except Exception:
                solved_tip = None
            if solved_tip is not None and len(solved_tip) == 3:
                if float(solved_tip[0]) > float(goal[0]) + 1e-4:
                    self.log(
                        f"[ARM] PBVS {label} 指尖 X {solved_tip[0]*1000:.0f}mm "
                        f"超过目标 {float(goal[0])*1000:.0f}mm，换下一条"
                    )
                    continue
                requested_z = float(goal[2]) - float(current_tip[2])
                solved_z = float(solved_tip[2]) - float(current_tip[2])
                if requested_z * solved_z < 0.0 and abs(requested_z) > 0.01 and abs(solved_z) > 0.01:
                    self.log(
                        f"[ARM] PBVS {label} 竖直方向反了，"
                        f"要 {requested_z*1000:.0f}mm 解出 {solved_z*1000:.0f}mm，换下一条"
                    )
                    continue
            if pose_ok and not self._solved_tip_reaches(endpoint.get("q_solved"), goal):
                miss = self._tip_miss(endpoint.get("q_solved"), goal)
                if miss is not None:
                    last["tip_miss_m"] = miss
                    self.log(
                        f"[ARM] PBVS {label} 指尖未到按钮 "
                        f"dXYZ_mm {miss[0]*1000:.0f} {miss[1]*1000:.0f} {miss[2]*1000:.0f}"
                    )
                pose_ok = False
            if not pose_ok:
                continue
            seconds = quintic_joint_seconds(
                baseline, endpoint["q_command"], feedrate=1.0
            )
            knot_count = max(2, int(math.ceil(seconds / MOTION_KNOT_S)) + 1)
            knots = []
            solved_seed = seed
            blocked_by_x = False
            for step in range(knot_count - 1):
                if self._return_ready.is_set():
                    return None
                blend = quintic_blend(step / (knot_count - 1))
                xyz = cap_tool_x(
                    tuple(
                        current_tip[index] * (1.0 - blend) + goal[index] * blend
                        for index in range(3)
                    ),
                    float(goal[0]),
                )
                rpy = slerp_rpy(start_rpy, goal_rpy, blend)
                sample_seed = seed if step == 0 else solved_seed
                result = self._solve_tool_sample(
                    solve_pose, commanded, measured, xyz, rpy, sample_seed
                )
                if result is None:
                    break
                try:
                    knot_tip = self._tool_xyz(result.get("q_solved") or result["q_command"])
                except Exception:
                    knot_tip = None
                if knot_tip is not None and float(knot_tip[0]) > float(goal[0]) + 1e-4:
                    self.log(
                        f"[ARM] PBVS {label} 路径点 X {float(knot_tip[0])*1000:.0f}mm "
                        f"超过目标 {float(goal[0])*1000:.0f}mm，换下一条"
                    )
                    knots = []
                    blocked_by_x = True
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
            if blocked_by_x:
                continue
            if not knots:
                # Midpoint solves can fail while the one-shot goal is fine.
                # With no chain there is no elbow branch to keep.
                knots = [list(baseline), [float(value) for value in endpoint["q_command"]]]
            else:
                finish_q = self._finish_on_the_same_branch(
                    solve_pose, commanded, measured, goal, goal_rpy, solved_seed, endpoint, knots[-1]
                )
                if finish_q is not None:
                    knots.append(finish_q)
                elif max_joint_gap(knots[-1], baseline) <= math.radians(5.0):
                    self.log(f"[ARM] PBVS {label} 终点换了肘支，连续路径还没走开，换下一条")
                    continue
                else:
                    self.log(f"[ARM] PBVS {label} 终点换了肘支，停在连续路径上")
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
        text = (
            f"{prefix} IK 未通过，试了 {found['tried']} 条路径，"
            f"末次 {found['label']} pos={found['pos_err']*1000:.1f}mm "
            f"rot={math.degrees(found['ori_err']):.1f}deg "
            f"jump={math.degrees(found['jump']):.0f}deg"
        )
        miss = found.get("tip_miss_m")
        if miss and len(miss) == 3:
            text += (
                "，指腹还差 "
                f"X {float(miss[0])*1000:.0f} Y {float(miss[1])*1000:.0f} "
                f"Z {float(miss[2])*1000:.0f} mm"
            )
        return text

    def _correct_height_from_chest(self, ee_xyz, *, phase: str) -> bool:
        """Step torso Z from a live chest pose button. Locked buttons do not count.

        The comparison uses the FK tip in the image. A pose tip sitting on the
        button does not hide a gripper that is still above the panel.
        """
        live = self.telemetry() or {}
        pose_xyz = live.get("chest_pose_button_torso_m")
        pose_z = _finite_float(pose_xyz[2]) if pose_xyz and len(pose_xyz) == 3 else None
        ee_z = None
        if ee_xyz is not None and len(ee_xyz) >= 3:
            ee_z = _finite_float(ee_xyz[2])
        if ee_z is None and self._right_ee_xyz_m and len(self._right_ee_xyz_m) == 3:
            ee_z = _finite_float(self._right_ee_xyz_m[2])
        command = chest_height_command(
            pose_z,
            ee_z,
            live.get("chest_pose_button_uv"),
            live.get("virtual_tip_uv") or live.get("tip_uv"),
            max_step_m=WRIST_APPROACH_MAX_STEP_M,
        )
        if command is None:
            self._chest_height_hold = False
            return False
        step, text = command
        self._apply_torso_delta(
            (0.0, 0.0, step),
            seconds=max(0.20, WRIST_IBVS_DT_S),
            phase=phase,
            limit_m=WRIST_APPROACH_MAX_STEP_M,
        )
        # The coarse planner still aims at the locked Z. Leave that lock
        # behind and the next pose solve lifts whatever this step just lowered.
        locked = self._locked_buttons.get(self._goal_name) if self._goal_name else None
        if locked is not None and ee_z is not None:
            self._locked_buttons[self._goal_name] = (
                float(locked[0]),
                float(locked[1]),
                float(ee_z) + float(step),
            )
        self._chest_height_hold = True
        if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self.message = text
        return True

    def _begin_wrist_approach(self, telem: dict[str, Any]) -> None:
        """At the chest button, validate wrist PBVS and hand over the final approach."""
        self._pbvs_wrist_checked = True
        if not self._fuse_wrist_pbvs(telem, keep_watching=True):
            live_xyz = self._live_chest_xyz()
            live_err = None if live_xyz is None else self._tip_error_to(live_xyz)
            if "没有看到" in self.message and live_err is not None and self._step_lateral_if_aligned(
                live_err, "PBVS_COARSE"
            ):
                self._ibvs_enabled = True
                self._running.set()
                self._pbvs_wrist_checked = False
                self._control_stage = "PBVS_COARSE"
                self.phase = "pbvs_coarse"
                return
            if "没有看到" in self.message and self._correct_height_from_chest(
                self._right_ee_xyz_m, phase="PBVS_COARSE"
            ):
                self._ibvs_enabled = True
                self._running.set()
                self._pbvs_wrist_checked = False
                self._control_stage = "PBVS_COARSE"
                self.phase = "pbvs_coarse"
                return
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self._running.clear()
            self.phase = "wrist_pbvs_unavailable"
            if not self.message:
                self.message = "右腕 PBVS 不可用，粗定位后保持"
            return
        self._wrist_target_filtered = None
        self._wrist_ibvs_step = None
        self._wrist_ibvs_uv = None
        self._wrist_ibvs_jump_frames = 0
        self._wrist_missing_frames = 0
        self._wrist_close_advanced = False
        self._wrist_close_advance_m = 0.0
        self._chest_height_after_pbvs = True
        self._control_stage = "WRIST_APPROACH"
        self.phase = "wrist_approach"
        if self._sequence_run and self._goal_name:
            self._sequence_stage = f"{SEQUENCE_BUTTON_LABELS.get(self._goal_name, self._goal_name)} 进近"
        self.message = "胸口 PBVS 已核对，右腕图像 IBVS 接管末端进近"
        self.log("[ARM] chest PBVS ready -> wrist IBVS")

    def _wrist_R_torso_from_cam(self):
        """Rotation that maps a wrist-camera displacement into the torso frame.

        Only the hand-eye rotation is used. The calibrated translation does not
        enter the image-error step.
        """
        matrix_fn = self.cam2hand_matrix
        cam2hand = matrix_fn() if callable(matrix_fn) else None
        fk_matrix = getattr(self.ik, "fk_matrix", None)
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        if cam2hand is None or not callable(fk_matrix) or not measured:
            return None
        try:
            torso_T_hand = fk_matrix(measured)
            return [
                [
                    sum(float(torso_T_hand[row][col]) * float(cam2hand[col][index]) for col in range(3))
                    for index in range(3)
                ]
                for row in range(3)
            ]
        except Exception as exc:
            self.log(f"[ARM][WARN] 右腕相机旋转失败: {exc}")
            return None

    def _wrist_goal_name(self, telem: dict[str, Any]) -> str:
        return self._goal_name or str(telem.get("target") or "")

    def _step_toward_latched_button(self, goal: tuple[float, float, float]) -> None:
        """Keep closing on the torso point recorded when the button was visible."""
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        if not measured or not callable(tip_xyz):
            self.phase = "wrist_wait_fk"
            self.message = "右腕已记录按键，等待指尖坐标"
            return
        try:
            current = tuple(float(value) for value in tip_xyz(measured))
        except Exception as exc:
            self.phase = "wrist_wait_fk"
            self.message = f"右腕已记录按键，FK 失败: {exc}"
            return
        delta = tuple(goal[index] - current[index] for index in range(3))
        distance = math.sqrt(sum(value * value for value in delta))
        if distance <= DIRECT_PBVS_ARRIVE_TOL_M:
            self.phase = "wrist_wait_contact"
            self.message = f"右腕已按记录到位，剩余 {distance*1000:.0f}mm，等待接触"
            return
        self._apply_torso_delta(
            delta,
            seconds=max(0.20, WRIST_IBVS_DT_S),
            phase="WRIST_APPROACH",
            limit_m=WRIST_APPROACH_MAX_STEP_M,
        )
        if not self.message.startswith(("IK 拒绝", "IBVS 服务失败")):
            self.message = f"右腕按记录进近，还差 {distance*1000:.0f}mm"

    def _lift_for_red_bar(self) -> None:
        """No buttons in view, but the long red key is. Raise in small steps."""
        remaining = WRIST_RED_LIFT_MAX_M - float(self._wrist_red_lift_m)
        if remaining <= 1e-6:
            self.phase = "wrist_red_lift_limit"
            self.message = "红条已抬高 50mm，仍没看到按键"
            return
        step = min(WRIST_RED_LIFT_STEP_M, remaining)
        self._wrist_red_lift_m += step
        self._apply_torso_delta(
            (0.0, 0.0, step),
            seconds=max(0.20, WRIST_IBVS_DT_S),
            phase="WRIST_RED_LIFT",
            limit_m=step,
        )
        if not self.message.startswith(("IK 拒绝", "IBVS 服务失败")):
            self.message = f"看到红条，抬高夹爪 {self._wrist_red_lift_m*1000:.0f}/50mm"

    def _apply_wrist_camera_step(
        self,
        camera_step: tuple[float, float, float],
        *,
        limit_m: float,
        seconds: float,
    ) -> bool:
        rotation = self._wrist_R_torso_from_cam()
        if rotation is None:
            self.phase = "wrist_wait_fk"
            self.message = "右腕 IBVS 等待手眼旋转"
            return False
        torso = tuple(
            sum(rotation[row][col] * float(camera_step[col]) for col in range(3)) for row in range(3)
        )
        return self._apply_torso_delta(
            torso,
            seconds=seconds,
            phase="WRIST_APPROACH",
            limit_m=limit_m,
        )

    def _pbvs_target_x(self, telem: dict[str, Any], goal_name: str) -> float | None:
        locked = self._locked_buttons.get(goal_name)
        if locked is not None and len(locked) == 3:
            value = _finite_float(locked[0])
            if value is not None:
                return value
        raw = telem.get("target_torso_m")
        if raw and len(raw) == 3:
            return _finite_float(raw[0])
        return None

    def _current_ee_x(self) -> float | None:
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        if measured and callable(tip_xyz):
            try:
                point = tip_xyz(measured, self.tool_offset_wrist_m)
                value = _finite_float(point[0])
                if value is not None:
                    return value
            except Exception:
                pass
        ee = self._right_ee_xyz_m
        if ee and len(ee) == 3:
            return _finite_float(ee[0])
        return None

    def _wrist_close_once(
        self,
        button_uv: tuple[float, float],
        tip_uv: tuple[float, float],
        intrinsics: dict[str, Any],
        *,
        pbvs_x: float | None,
    ) -> None:
        """No wrist depth means the panel is already inside 5 cm.

        Translate until the tip pixel meets the button, then walk torso +X
        up to the button X, at most 3 cm when the button is farther. Do not
        command an X deeper than the button.
        """
        if self._wrist_close_advanced:
            self.phase = "wrist_close_hold"
            sent = float(getattr(self, "_wrist_close_advance_m", WRIST_CLOSE_ADVANCE_M))
            self.message = f"近距已向前 {sent * 1000:.0f}mm，等待接触"
            return
        if self._wrist_ibvs_uv is not None:
            jump_px = math.hypot(button_uv[0] - self._wrist_ibvs_uv[0], button_uv[1] - self._wrist_ibvs_uv[1])
            if jump_px > WRIST_IBVS_PIXEL_JUMP:
                self._wrist_ibvs_jump_frames += 1
                if self._wrist_ibvs_jump_frames < 3:
                    self.phase = "wrist_pixel_jump"
                    self.message = f"右腕按钮像素跳变 {jump_px:.0f}px，拒绝本帧"
                    return
        self._wrist_ibvs_jump_frames = 0
        self._wrist_ibvs_uv = button_uv
        pixel_err = math.hypot(button_uv[0] - tip_uv[0], button_uv[1] - tip_uv[1])
        if pixel_err > WRIST_IBVS_PIXEL_TOL:
            try:
                command = eye_in_hand_close_align_step(
                    button_uv,
                    tip_uv,
                    fx=float(intrinsics.get("fx") or 615.0),
                    fy=float(intrinsics.get("fy") or 615.0),
                    cx=float(intrinsics.get("cx") or 320.0),
                    cy=float(intrinsics.get("cy") or 240.0),
                    nominal_z_m=WRIST_CLOSE_NOMINAL_Z_M,
                    lambda_xy=WRIST_IBVS_LAMBDA_XY,
                    dt_s=WRIST_IBVS_DT_S,
                    max_step_m=WRIST_APPROACH_MAX_STEP_M,
                )
            except (TypeError, ValueError) as exc:
                self.phase = "wrist_wait_target"
                self.message = f"右腕近距对齐无效：{exc}"
                return
            camera_step = tuple(float(value) for value in command["camera_step_m"])
            self._apply_wrist_camera_step(
                camera_step,
                limit_m=WRIST_APPROACH_MAX_STEP_M,
                seconds=max(0.20, WRIST_IBVS_DT_S),
            )
            if not self.message.startswith(("IK 拒绝", "IBVS 服务失败")):
                self.message = f"近距无深度，先平移对齐，像素差 {pixel_err:.0f}px"
            return
        current_x = self._current_ee_x()
        if current_x is None:
            self.phase = "wrist_wait_fk"
            self.message = "近距已对齐，没有末端 X，不能向前"
            return
        goal_x = close_range_forward_x(current_x, pbvs_x, advance_m=WRIST_CLOSE_ADVANCE_M)
        forward_m = round(max(0.0, goal_x - current_x), 4)
        if forward_m < 0.0005:
            self._wrist_close_advanced = True
            self._wrist_close_advance_m = 0.0
            self.phase = "wrist_close_hold"
            self.message = "近距已在 PBVS 的 X 上，等待接触"
            return
        seconds = max(WRIST_CLOSE_ADVANCE_S, WRIST_CLOSE_ADVANCE_S * forward_m / WRIST_CLOSE_ADVANCE_M)
        ok = self._apply_torso_delta(
            (forward_m, 0.0, 0.0),
            seconds=seconds,
            phase="WRIST_APPROACH",
            limit_m=forward_m,
        )
        if ok:
            self._wrist_close_advanced = True
            self._wrist_close_advance_m = forward_m
            self.message = f"近距已对齐，向前 {forward_m * 1000:.0f}mm 到 X {goal_x:.3f}"

    def _retract_if_deeper_than_button(self, error, phase: str, past_m: float = PBVS_XY_TOL_M) -> bool:
        """Back out in X before any raise when the jaw is already past the key."""
        retract = pbvs_retract_step(error, past_m=past_m)
        if retract is None:
            return False
        self._forward_hold = True
        self._apply_torso_delta(
            retract,
            seconds=PBVS_STEP_SECONDS,
            phase=phase,
            limit_m=abs(retract[0]),
        )
        if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self.message = (
                f"末端比按键深 {abs(float(error[0])) * 1000:.0f}mm，"
                f"先后退 {abs(retract[0]) * 1000:.0f}mm，不上举"
            )
        return True

    def _step_toward_chest_while_wrist_misses(self) -> bool:
        """Keep a 4 mm chest step while the wrist pose score stays under 0.3.

        The yellow keys can fill the wrist frame and still score about 0.04.
        Those points do not drive. The chest pose does, until a confident
        wrist button appears.
        """
        if not self._chest_height_after_pbvs:
            return False
        raw = (self.telemetry() or {}).get("chest_pose_button_torso_m")
        if not raw or len(raw) != 3:
            return False
        try:
            live = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return False
        if len(live) != 3 or not all(math.isfinite(value) for value in live):
            return False
        err = self._tip_error_to(live)
        if err is None or all(abs(value) <= PBVS_XY_TOL_M for value in err):
            return False
        if self._retract_if_deeper_than_button(err, "WRIST_APPROACH"):
            return True
        axis = max(range(3), key=lambda index: abs(err[index]))
        if abs(err[axis]) <= CHEST_POSE_HEIGHT_TOL_M:
            return False
        step = [0.0, 0.0, 0.0]
        step[axis] = max(-WRIST_APPROACH_MAX_STEP_M, min(WRIST_APPROACH_MAX_STEP_M, err[axis]))
        self._apply_torso_delta(
            (step[0], step[1], step[2]),
            seconds=max(0.20, WRIST_IBVS_DT_S),
            phase="WRIST_APPROACH",
            limit_m=WRIST_APPROACH_MAX_STEP_M,
        )
        if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self.message = (
                "右腕按键分数低于 0.3，胸口还差 "
                + " ".join(f"{name} {err[index] * 1000:+.0f}mm" for index, name in enumerate("XYZ"))
            )
        return True

    def _wrist_approach_once(self, telem: dict[str, Any]) -> None:
        """Servo the wrist-camera button onto the gripper tip, then wait for contact.

        Chest height is armed only after coarse PBVS has arrived and this
        wrist approach has started. A pose button then means the height gap
        is still large, so step torso Z from that button toward the end
        effector. Before that handover, and once the chest pose loses the
        button, the wrist camera owns the approach, height included.
        """
        view = self.wrist_view() if callable(self.wrist_view) else {}
        if view.get("buttons_fresh"):
            self._latch_wrist_buttons(view)
        goal_name = self._wrist_goal_name(telem)
        latched = self._wrist_latched_torso.get(goal_name)
        stamp = float(view.get("updated_at") or 0.0)
        fresh = stamp > 0.0 and time.time() - stamp <= WRIST_TARGET_MAX_AGE_S
        button_uv = _uv_pair(view.get("target_uv")) if fresh else None
        tip_uv = _uv_pair(view.get("tip_uv")) if fresh else None
        button_z = _finite_float(view.get("target_z_m")) if fresh else None
        # A close-range frame has already dropped the button depth. Do not
        # refill it from an older camera point taken farther away.
        if button_z is None and fresh and not view.get("close_range"):
            raw = (view.get("buttons_cam_m") or {}).get(goal_name)
            if raw and len(raw) >= 3:
                button_z = _finite_float(raw[2])
        tip_z = _finite_float(view.get("tip_z_m")) if fresh else None
        image_ready = button_uv is not None and tip_uv is not None and button_z is not None
        close_range = (
            bool(view.get("close_range"))
            and fresh
            and button_uv is not None
            and tip_uv is not None
            and button_z is None
        )
        searching_red = (
            (not image_ready)
            and latched is None
            and bool(view.get("red_bar"))
            and not close_range
        )
        if not searching_red and (bool(telem.get("wrist_contact")) or bool(view.get("contact"))):
            if callable(self.wrist_watch):
                self.wrist_watch(False)
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self._running.clear()
            self.phase = "contact"
            self.message = "右腕相机确认接触，末端进近停止"
            self.log("[ARM] right-wrist contact, stop")
            return
        live_button = self._live_chest_xyz()
        past = self._tip_error_to(live_button) if live_button is not None else None
        if past is None:
            raw_button = telem.get("chest_pose_button_torso_m") or telem.get("target_torso_m")
            if raw_button and len(raw_button) == 3:
                try:
                    past = self._tip_error_to(tuple(float(value) for value in raw_button))
                except (TypeError, ValueError):
                    past = None
        if past is not None and self._retract_if_deeper_than_button(past, "WRIST_APPROACH"):
            return
        clicked = bool(view.get("yellow_lock")) and fresh and button_uv is not None
        if clicked and int(view.get("yellow_id") or 0) != self._wrist_yellow_id:
            self._wrist_yellow_id = int(view.get("yellow_id") or 0)
            self._wrist_ibvs_uv = button_uv
            self._wrist_ibvs_jump_frames = 0
        if clicked and tip_uv is None:
            self.phase = "wrist_wait_target"
            self.message = "已点选黄色区域，还没看到夹爪尖"
            return
        if clicked and button_z is None:
            self._wrist_close_once(
                button_uv,
                tip_uv,
                view.get("intrinsics") or {},
                pbvs_x=self._pbvs_target_x(telem, goal_name),
            )
            if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                self.message = "跟踪点选黄色区域，" + self.message
            return
        pose_xyz = telem.get("chest_pose_button_torso_m") if self._chest_height_after_pbvs else None
        pose_button_z = _finite_float(pose_xyz[2]) if pose_xyz and len(pose_xyz) == 3 else None
        ee_xyz = self._right_ee_xyz_m
        ee_z = _finite_float(ee_xyz[2]) if ee_xyz and len(ee_xyz) == 3 else None
        # The wrist sample is not a fresh button. The pose tip often sits on
        # the keys while the jaw is still above the panel, so use the gripper
        # mask in the chest image before waiting on the wrist.
        wrist_has_target = image_ready or close_range
        if (not clicked) and self._chest_height_after_pbvs and not wrist_has_target:
            jaw_gap = _chest_pixel_gap(telem.get("chest_pose_button_uv"), telem.get("chest_gripper_uv"))
            if jaw_gap is not None and abs(jaw_gap) > CHEST_POSE_HEIGHT_PX:
                step = -WRIST_APPROACH_MAX_STEP_M if jaw_gap < 0.0 else WRIST_APPROACH_MAX_STEP_M
                self._apply_torso_delta(
                    (0.0, 0.0, step),
                    seconds=max(0.20, WRIST_IBVS_DT_S),
                    phase="WRIST_APPROACH",
                    limit_m=WRIST_APPROACH_MAX_STEP_M,
                )
                if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                    if jaw_gap < 0.0:
                        self.message = f"胸口看到夹爪在按键上方 {abs(jaw_gap):.0f}px，降低"
                    else:
                        self.message = f"胸口看到夹爪在按键下方 {jaw_gap:.0f}px，抬高"
                return
        chest_tip_uv = telem.get("virtual_tip_uv") or telem.get("chest_gripper_uv") or telem.get("tip_uv")
        height = (
            chest_height_command(
                pose_button_z,
                ee_z,
                telem.get("chest_pose_button_uv"),
                chest_tip_uv,
                max_step_m=WRIST_APPROACH_MAX_STEP_M,
            )
            if self._chest_height_after_pbvs
            else None
        )
        if (not clicked) and not wrist_has_target and height is not None:
            height_z, height_message = height
            self._apply_torso_delta(
                (0.0, 0.0, height_z),
                seconds=max(0.20, WRIST_IBVS_DT_S),
                phase="WRIST_APPROACH",
                limit_m=WRIST_APPROACH_MAX_STEP_M,
            )
            if not self.message.startswith(("IK 拒绝", "IBVS 服务失败")):
                self.message = height_message
            return
        if close_range:
            self._wrist_latched_torso.pop(goal_name, None)
            self._wrist_close_once(
                button_uv,
                tip_uv,
                view.get("intrinsics") or {},
                pbvs_x=self._pbvs_target_x(telem, goal_name),
            )
            if (
                str(view.get("target_source") or "") == "yellow_bar"
                and not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败"))
                and "黄色长条" not in self.message
            ):
                self.message = "按键未看到，对齐最近黄色长条，" + self.message
            return
        if bool(view.get("occluded")) and not image_ready and latched is None:
            self.phase = "wrist_occluded"
            self.message = "夹爪遮挡面板，末端 IBVS 保持"
            return
        if image_ready:
            pass
        elif latched is not None:
            self._step_toward_latched_button(latched)
            return
        elif view.get("red_bar"):
            self._lift_for_red_bar()
            return
        elif not fresh:
            if self._step_toward_chest_while_wrist_misses():
                return
            self.phase = "wrist_wait_target"
            self.message = "右腕还没看到当前按键"
            return
        else:
            self.phase = "wrist_wait_target"
            self.message = "右腕 IBVS 等待按钮、夹爪尖和深度"
            return
        if self._wrist_ibvs_uv is not None and not clicked:
            jump_px = math.hypot(button_uv[0] - self._wrist_ibvs_uv[0], button_uv[1] - self._wrist_ibvs_uv[1])
            if jump_px > WRIST_IBVS_PIXEL_JUMP:
                self._wrist_ibvs_jump_frames += 1
                if self._wrist_ibvs_jump_frames < 3:
                    self.phase = "wrist_pixel_jump"
                    self.message = f"右腕按钮像素跳变 {jump_px:.0f}px，拒绝本帧"
                    return
        self._wrist_ibvs_jump_frames = 0
        self._wrist_ibvs_uv = button_uv
        intrinsics = view.get("intrinsics") or {}
        try:
            command = eye_in_hand_ibvs_step(
                button_uv,
                tip_uv,
                button_z,
                tip_z,
                fx=float(intrinsics.get("fx") or 615.0),
                fy=float(intrinsics.get("fy") or 615.0),
                cx=float(intrinsics.get("cx") or 320.0),
                cy=float(intrinsics.get("cy") or 240.0),
                lambda_xy=WRIST_IBVS_LAMBDA_XY,
                lambda_z=WRIST_IBVS_LAMBDA_Z,
                dt_s=WRIST_IBVS_DT_S,
                max_step_m=WRIST_APPROACH_MAX_STEP_M,
                pixel_tol=WRIST_IBVS_PIXEL_TOL,
                depth_tol_m=WRIST_IBVS_DEPTH_TOL_M,
            )
        except (TypeError, ValueError) as exc:
            self.phase = "wrist_wait_target"
            self.message = f"右腕 IBVS 特征无效：{exc}"
            return
        camera_step = tuple(float(value) for value in command["camera_step_m"])
        previous = self._wrist_ibvs_step
        alpha = WRIST_IBVS_FILTER_ALPHA
        if previous is None:
            filtered = camera_step
        else:
            filtered = tuple(
                (1.0 - alpha) * previous[index] + alpha * camera_step[index] for index in range(3)
            )
        self._wrist_ibvs_step = filtered
        rotation = self._wrist_R_torso_from_cam()
        if rotation is None:
            self.phase = "wrist_wait_fk"
            self.message = "右腕 IBVS 等待手眼旋转"
            return
        torso = tuple(
            sum(rotation[row][col] * filtered[col] for col in range(3)) for row in range(3)
        )
        pixel_err = float(command["pixel_err"])
        depth_err = command["depth_err_m"]
        yellow_bar = str(view.get("target_source") or "") == "yellow_bar"
        aim = "最近黄色长条" if yellow_bar else "按键"
        if bool(command["arrived"]):
            self.phase = "wrist_wait_contact"
            depth_mm = 0.0 if depth_err is None else abs(float(depth_err)) * 1000.0
            self.message = (
                f"右腕 IBVS 指尖已对齐{aim}，像素差 {pixel_err:.0f}px，"
                f"深度差 {depth_mm:.1f}mm，等待接触"
            )
            if clicked:
                self.message = "跟踪点选黄色区域，" + self.message
            return
        self._apply_torso_delta(
            torso,
            seconds=max(0.20, WRIST_IBVS_DT_S),
            phase="WRIST_APPROACH",
            limit_m=WRIST_APPROACH_MAX_STEP_M,
        )
        if not self.message.startswith(("IK 拒绝", "IBVS 服务失败")):
            if pixel_err > WRIST_IBVS_PIXEL_TOL:
                self.message = (
                    f"先对齐指尖和{aim}，像素差 {pixel_err:.0f}px，先不深入"
                )
            elif depth_err is None:
                self.message = f"指尖已对齐{aim}，像素差 {pixel_err:.0f}px，沿前方深入"
            else:
                self.message = (
                    f"指尖已对齐{aim}，像素差 {pixel_err:.0f}px，"
                    f"深入还差 {float(depth_err)*1000:.1f}mm"
                )
            if clicked:
                self.message = "跟踪点选黄色区域，" + self.message

    def _play_service_cartesian(self, current_tip, goal, rpy, label: str) -> tuple[bool, str]:
        """Send a service Cartesian goal, lifting clear of the wall before +X."""
        goal_t = tuple(float(value) for value in goal)
        points = (goal_t,) if current_tip is None else service_rise_before_wall(current_tip, goal_t)
        rpy_t = tuple(float(value) for value in rpy)
        lifted = False
        for index, point in enumerate(points):
            if self._return_ready.is_set() or self._stop.is_set():
                return False, "已停止"
            origin = current_tip if index == 0 else points[index - 1]
            if origin is None:
                seconds = 2.0
            else:
                gap = math.dist(tuple(origin), point)
                seconds = max(2.0, min(6.0, gap / 0.15))
            if len(points) > 1 and index == 0:
                lifted = True
                self.message = (
                    f"{label} 先在 X {point[0]:.3f} 抬到 {point[2]:.3f}，"
                    "不从下方绕进墙"
                )
                self.log(f"[ARM] {self.message}")
            ok, detail = self.controller.move_cartesian("right", point, rpy_t, seconds=seconds)
            if not ok:
                return False, str(detail)
            if index + 1 < len(points):
                time.sleep(0.2)
        if lifted:
            return True, f"{label} 已先抬高再向前"
        return True, ""

    def _service_followup_cartesian(self, goal, rpy, label: str) -> tuple[bool, str]:
        """Send another Cartesian shot while the fingertip is still short of the button."""
        after = self._tip_distance_to(goal)
        if after is not None and after <= PBVS_XY_TOL_M:
            return True, f"{label} 已到 {after*1000:.0f}mm"
        ok, detail = self._play_service_cartesian(self._measured_tip(), goal, rpy, label)
        if not ok:
            return False, f"{label} 修正失败：{detail}"
        final = self._tip_distance_to(goal)
        err = self._tip_error_to(goal)
        ztxt = "" if err is None else f"，Z 还差 {err[2]*1000:.0f}mm"
        prefix = f"{detail}；" if detail else ""
        if final is None:
            return True, f"{prefix}{label} 已发送修正"
        return True, f"{prefix}{label} 修正后还差 {final*1000:.0f}mm{ztxt}"

    def _can_axis_step(self) -> bool:
        """True when a pure fingertip translation can be solved without a 6-DoF pose."""
        if callable(getattr(self.controller, "move_delta", None)):
            return True
        ik = self.ik
        return callable(getattr(ik, "apply_tip_delta", None)) or callable(getattr(ik, "apply_delta", None))

    def _yz_align_text(self, error, step) -> str:
        """Status line for a YZ alignment step that keeps the X gap."""
        dx, dy, dz = (float(value) for value in step)
        if abs(dy) < 1e-6 and abs(dx) < 1e-6 and dz < -1e-6:
            return (
                f"末端高于目标 {abs(float(error[2]))*1000:.0f}mm，"
                f"下降 {abs(dz)*1000:.0f}mm"
            )
        parts = [f"先对齐 YZ，X 保持 {PBVS_YZ_X_CLEAR_M*1000:.0f}mm"]
        if abs(dy) >= 1e-6:
            parts.append(f"Y 方向 {dy*1000:.0f}mm")
        if dz < -1e-6:
            parts.append(f"下降 {abs(dz)*1000:.0f}mm")
        elif dz > 1e-6:
            parts.append(f"抬高 {dz*1000:.0f}mm")
        if dx < -1e-6:
            parts.append(f"X 退 {abs(dx)*1000:.0f}mm")
        return "，".join(parts)

    def _yz_attempt_stalled(self, goal_key, yz_now: float) -> bool:
        """True when the previous YZ attempt on this button did not get closer."""
        previous = getattr(self, "_pbvs_yz_before", None)
        if previous is None or previous[0] != goal_key:
            return False
        return float(yz_now) > float(previous[1]) - DIRECT_PBVS_YZ_STALL_M

    def _step_x_after_yz(
        self,
        error,
        phase: str,
        *,
        max_tip_x: float | None = None,
        tip_x: float | None = None,
    ) -> bool:
        """Move +X after YZ is aligned or the YZ-only step was rejected."""
        forward = pbvs_x_step(error)
        if forward is None:
            return False
        step_x = float(forward[0])
        if max_tip_x is not None and tip_x is not None:
            room = float(max_tip_x) - float(tip_x)
            if room <= 1e-4:
                return False
            step_x = min(step_x, room)
        forward = (step_x, 0.0, 0.0)
        before = self._tool_xyz_measured()
        self._apply_torso_delta(
            forward,
            seconds=PBVS_STEP_SECONDS,
            phase=phase,
            limit_m=forward[0],
        )
        if str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self._forward_hold = False
            return False
        after = self._tool_xyz_measured()
        moved_x = 0.0 if before is None or after is None else float(after[0]) - float(before[0])
        if moved_x < 0.002:
            self._forward_hold = False
            return False
        self._forward_hold = True
        self._pbvs_yz_before = None
        self.message = (
            f"YZ 解不出，尝试 X，向前 {forward[0]*1000:.0f}mm，"
            f"X 还差 {float(error[0])*1000:.0f}mm"
        )
        return True

    def _step_forward_if_aligned(
        self,
        error,
        phase: str,
        *,
        max_tip_x: float | None = None,
        tip_x: float | None = None,
    ) -> bool:
        """Advance X without a 6-DoF re-solve when Y and Z are already close."""
        forward = pbvs_forward_step(error)
        if forward is None:
            return False
        step_x = float(forward[0])
        if max_tip_x is not None and tip_x is not None:
            room = float(max_tip_x) - float(tip_x)
            if room <= 1e-4:
                return False
            step_x = min(step_x, room)
        forward = (step_x, 0.0, 0.0)
        before = self._tool_xyz_measured()
        self._apply_torso_delta(
            forward,
            seconds=PBVS_STEP_SECONDS,
            phase=phase,
            limit_m=forward[0],
        )
        if str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self._forward_hold = False
            return False
        after = self._tool_xyz_measured()
        moved_x = 0.0 if before is None or after is None else float(after[0]) - float(before[0])
        if moved_x < 0.002:
            self._forward_hold = False
            return False
        self._forward_hold = True
        self.message = (
            f"左右和高度已靠近，向前 {forward[0]*1000:.0f}mm，"
            f"X 还差 {float(error[0])*1000:.0f}mm"
        )
        return True

    def _live_chest_xyz(self) -> tuple[float, float, float] | None:
        """Button the chest pose sees now. The frozen coarse goal does not count."""
        telem = self.telemetry() or {}
        raw = telem.get("chest_pose_button_torso_m") or telem.get("target_torso_m")
        if not raw or len(raw) != 3:
            return None
        try:
            point = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        return point

    def _step_lateral_if_aligned(self, error, phase: str) -> bool:
        """Slide in Y without a 6-DoF re-solve when X is already close."""
        lateral = pbvs_lateral_step(error)
        if lateral is None:
            return False
        self._forward_hold = True
        self._apply_torso_delta(
            lateral,
            seconds=PBVS_STEP_SECONDS,
            phase=phase,
            limit_m=abs(lateral[1]),
        )
        if str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
            self._forward_hold = False
            return False
        self.message = (
            f"X 已靠近，Y 方向移动 {lateral[1]*1000:.0f}mm，"
            f"还差 {float(error[1])*1000:.0f}mm"
        )
        return True

    def _pbvs_once(self, telem: dict[str, Any]) -> None:
        self._swing_right_arm_clear()
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
            current_tip = self._tool_xyz(measured)
        except Exception as exc:
            self.phase = "pbvs_wait_fk"
            self.message = f"PBVS FK 失败: {exc}"
            return
        coarse_target = target
        error_torso = tuple(
            coarse_target[index] - current_tip[index]
            for index in range(3)
        )
        # Same 6-DoF target as /h2_arm/singlearmcartesian: button XYZ plus the
        # right_ee orientation already stored in simple_cartesian.yaml.
        done = all(abs(value) <= PBVS_XY_TOL_M for value in error_torso)
        close_xy = (
            abs(error_torso[0]) <= PBVS_FORWARD_YZ_TOL_M
            and abs(error_torso[1]) <= PBVS_FORWARD_YZ_TOL_M
        )
        live_xyz = self._live_chest_xyz()
        live_error = None
        live_arrived = False
        if live_xyz is not None:
            live_error = tuple(live_xyz[index] - current_tip[index] for index in range(3))
            # The lock can stay centimetres off to the side after the chest pose
            # has already moved. Arrival follows the button that is visible now.
            live_arrived = all(abs(value) <= PBVS_XY_TOL_M for value in live_error)
            if live_arrived:
                done = True
                close_xy = True
            else:
                if abs(live_error[0]) > PBVS_XY_TOL_M or abs(live_error[1]) > PBVS_XY_TOL_M:
                    done = False
                if (
                    abs(live_error[0]) > PBVS_FORWARD_YZ_TOL_M
                    or abs(live_error[1]) > PBVS_FORWARD_YZ_TOL_M
                ):
                    close_xy = False
            # The fingertip can sit on the frozen lock while the chest pose has
            # moved a few centimetres. Keep driving at the button that is visible now.
            if not live_arrived and all(abs(value) <= PBVS_XY_TOL_M for value in error_torso):
                coarse_target = live_xyz
                error_torso = live_error
                done = False
                close_xy = (
                    abs(error_torso[0]) <= PBVS_FORWARD_YZ_TOL_M
                    and abs(error_torso[1]) <= PBVS_FORWARD_YZ_TOL_M
                )
                if self._goal_name:
                    self._locked_buttons[self._goal_name] = tuple(live_xyz)
        x_pbvs = float(coarse_target[0])
        # The wrist approach may still reach the button. Coarse motion may not
        # arrive at, or pass, x_pbvs - 0.3 cm.
        limit_step = coarse_x_retract(current_tip[0], x_pbvs)
        if limit_step is not None:
            self._forward_hold = True
            self._apply_torso_delta(
                limit_step,
                seconds=PBVS_STEP_SECONDS,
                phase="PBVS_COARSE",
                limit_m=abs(limit_step[0]),
            )
            if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                self.message = (
                    f"进近前 X 不得超过 {coarse_x_ceiling(x_pbvs):.4f}（按键前 3mm），"
                    f"先后退 {abs(limit_step[0]) * 1000:.0f}mm，不上举"
                )
            self.phase = "pbvs_coarse"
            self._control_stage = "PBVS_COARSE"
            return
        coarse_target = clamp_coarse_goal(coarse_target, x_pbvs)
        retract_error = live_error if live_error is not None else error_torso
        if self._retract_if_deeper_than_button(retract_error, "PBVS_COARSE"):
            self.phase = "pbvs_coarse"
            self._control_stage = "PBVS_COARSE"
            return
        # Align Y and Z first. X stays at least 2 cm short until that alignment
        # is done, or until the YZ-only step cannot be solved.
        button_xyz = (x_pbvs, float(coarse_target[1]), float(coarse_target[2]))
        yz_now = math.hypot(float(error_torso[1]), float(error_torso[2]))
        goal_key = tuple(round(float(value), 3) for value in button_xyz)
        yz_ready = live_arrived or pbvs_yz_aligned(error_torso)
        yz_unsolved = (not yz_ready) and self._yz_attempt_stalled(goal_key, yz_now)
        raised_first = False
        service_move = callable(getattr(self.controller, "move_cartesian", None))
        if PBVS_YZ_FIRST and not yz_ready and not yz_unsolved and not service_move and self._can_axis_step():
            step = pbvs_yz_align_step(error_torso)
            if step is not None:
                self._forward_hold = True
                self._apply_torso_delta(
                    step,
                    seconds=PBVS_STEP_SECONDS,
                    phase="PBVS_COARSE",
                    limit_m=max(abs(float(value)) for value in step),
                )
                if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                    self._pbvs_yz_before = (goal_key, yz_now)
                    self.message = self._yz_align_text(error_torso, step)
                    self.phase = "pbvs_coarse"
                    self._control_stage = "PBVS_COARSE"
                    return
                # The translation did not move. Keep the YZ pose; do not jog X
                # while height or lateral error is still open.
        if PBVS_YZ_FIRST and not yz_ready and not yz_unsolved and service_move:
            hold = pbvs_yz_hold_goal(current_tip, button_xyz)
            if hold is not None:
                rpy_hold = tuple(float(value) for value in self._nominal_tool_rpy_deg())
                raised_first = float(hold[2]) > float(current_tip[2]) + 0.01
                ok_hold, _detail_hold = self._play_service_cartesian(
                    current_tip, hold, rpy_hold, "PBVS"
                )
                if not ok_hold:
                    yz_unsolved = True
                else:
                    try:
                        current_tip = self._tool_xyz(self.controller.measured_arm_rad("right"))
                    except Exception:
                        current_tip = None
                    if current_tip is None:
                        yz_unsolved = True
                    else:
                        error_torso = tuple(
                            button_xyz[index] - current_tip[index] for index in range(3)
                        )
                        yz_after = math.hypot(float(error_torso[1]), float(error_torso[2]))
                        done = all(abs(value) <= PBVS_XY_TOL_M for value in error_torso)
                        close_xy = (
                            abs(error_torso[0]) <= PBVS_FORWARD_YZ_TOL_M
                            and abs(error_torso[1]) <= PBVS_FORWARD_YZ_TOL_M
                        )
                        if pbvs_yz_aligned(error_torso):
                            yz_ready = True
                            self._pbvs_yz_before = None
                        elif yz_after > yz_now - DIRECT_PBVS_YZ_STALL_M:
                            yz_unsolved = True
                            self._pbvs_yz_before = None
                        else:
                            self._pbvs_yz_before = (goal_key, yz_now)
                            self.phase = "pbvs_coarse"
                            self._control_stage = "PBVS_COARSE"
                            self.message = "先对齐 YZ，X 保持 20mm"
                            if raised_first:
                                self.message = "先抬高，" + self.message
                            return
        # Image height is only a trim once X/Y are already close.
        if (yz_ready if PBVS_YZ_FIRST else (done or close_xy)) and self._correct_height_from_chest(
            current_tip, phase="PBVS_COARSE"
        ):
            self.phase = "pbvs_coarse"
            self._control_stage = "PBVS_COARSE"
            return
        if not PBVS_YZ_FIRST and not live_arrived and not service_move:
            descent = pbvs_descent_step(error_torso)
            if descent is not None:
                self._forward_hold = True
                self._apply_torso_delta(
                    descent,
                    seconds=PBVS_STEP_SECONDS,
                    phase="PBVS_COARSE",
                    limit_m=abs(descent[2]),
                )
                if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                    self.message = (
                        f"末端高于目标 {abs(error_torso[2])*1000:.0f}mm，"
                        f"下降 {abs(descent[2])*1000:.0f}mm"
                    )
                self.phase = "pbvs_coarse"
                self._control_stage = "PBVS_COARSE"
                return
        if done:
            self._begin_wrist_approach(self.telemetry() or telem)
            return
        if (yz_ready or not PBVS_YZ_FIRST) and not service_move and self._step_forward_if_aligned(
            error_torso,
            "PBVS_COARSE",
            max_tip_x=coarse_x_command(x_pbvs),
            tip_x=current_tip[0],
        ):
            self._pbvs_yz_before = None
            return
        if not PBVS_YZ_FIRST and not service_move and self._step_lateral_if_aligned(
            live_error if live_error is not None else error_torso, "PBVS_COARSE"
        ):
            return
        distance = math.sqrt(sum(value * value for value in error_torso))
        if callable(getattr(self.controller, "move_cartesian", None)):
            rpy = tuple(float(value) for value in self._nominal_tool_rpy_deg())
            ok, detail = self._play_service_cartesian(current_tip, coarse_target, rpy, "PBVS")
            if not ok:
                self._ibvs_enabled = False
                self._control_stage = "HOLD"
                self._running.clear()
                self.phase = "pbvs_timeout"
                self.message = f"PBVS 服务失败：{detail}"
                self.log(f"[ARM] {self.message}")
                return
            follow_raw = self._service_coarse_target()
            if follow_raw is None:
                follow_goal = coarse_target
            else:
                follow_goal = clamp_coarse_goal(follow_raw, float(follow_raw[0]))
            ok, message = self._service_followup_cartesian(follow_goal, rpy, "PBVS")
            if not ok:
                self._ibvs_enabled = False
                self._control_stage = "HOLD"
                self._running.clear()
                self.phase = "pbvs_timeout"
                self.message = message
                self.log(f"[ARM] {self.message}")
                return
            if detail and "先抬高" not in message:
                message = f"{detail}；{message}"
            if raised_first and "先抬高" not in message:
                message = f"先抬高；{message}"
            self._service_pbvs_sent = True
            self.phase = "pbvs_coarse"
            self.message = message
            return
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not commanded or not callable(solve_pose):
            self.phase = "pbvs_ik_rejected"
            self.message = "PBVS 拒绝：运控末端 IK 未更新"
            return
        pose_goal = coarse_target
        if PBVS_YZ_FIRST and not yz_ready:
            hold = pbvs_yz_hold_goal(current_tip, button_xyz)
            if hold is not None:
                pose_goal = hold
        found = self._plan_cartesian_tool_move(
            commanded,
            measured,
            current_tip,
            pose_goal,
            solve_pose,
            must_move=False,
            stage_limited=True,
        )
        if (found is None or not found["ok"]) and pose_goal is not coarse_target:
            self._pbvs_yz_before = None
            found = self._plan_cartesian_tool_move(
                commanded,
                measured,
                current_tip,
                coarse_target,
                solve_pose,
                must_move=False,
                stage_limited=True,
            )
        elif found is not None and found["ok"] and pose_goal is not coarse_target:
            self._pbvs_yz_before = (goal_key, yz_now)
        if found is None or not found["ok"]:
            # The stage path budget expires with tried=0 even when the jaw is
            # already on the panel. That used to hold instead of starting the
            # wrist approach.
            near_live = live_error is not None and all(
                abs(value) <= PBVS_FORWARD_YZ_TOL_M for value in live_error
            )
            timed_out = found is None or int(found.get("tried") or 0) == 0
            if near_live and timed_out:
                self._pbvs_deadline = None
                self._begin_wrist_approach(self.telemetry() or telem)
                return
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
        current = tuple(float(value) for value in tip_xyz(measured, self.tool_offset_wrist_m))
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
                offset_m=self.tool_offset_wrist_m,
            )
        else:
            rpy = tuple(math.radians(value) for value in self._nominal_tool_rpy_deg())
            q_new, ok, pos_err, ori_err = solve_pose(
                commanded,
                measured,
                target_tool_xyz_m=motion_target,
                target_tool_rpy_rad=rpy,
                tool_offset_wrist_m=self.tool_offset_wrist_m,
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
        self._play_joint_target(q_new, 5.0 if (aligning or orienting) else 3.0)
        measured_after = self.controller.measured_arm_rad("right")
        actual_after = tuple(
            float(value) for value in tip_xyz(measured_after, self.tool_offset_wrist_m)
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
            desired_rpy = tuple(math.radians(value) for value in self._nominal_tool_rpy_deg())
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
        self._record_action_pose("到位" if arrived else "未到位")

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

    def _latch_wrist_buttons(self, sample: dict[str, Any]) -> None:
        """Keep the torso fix from the frame that actually saw the buttons."""
        fresh = self._wrist_buttons_torso(sample.get("buttons_cam_m") or {})
        if fresh:
            self._wrist_latched_torso.update(fresh)

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

    def _fuse_wrist_pbvs(
        self,
        telem: dict[str, Any],
        *,
        keep_watching: bool = False,
    ) -> bool:
        """Compare chest PBVS with a settled right-wrist Eye-in-Hand observation."""
        chest = self._snapshot_chest_buttons(telem)
        self._locked_buttons = dict(chest)
        self._pbvs_source = "chest"
        watch = self.wrist_watch
        view = self.wrist_view
        matrix_fn = self.cam2hand_matrix
        ready = callable(watch) and callable(view) and callable(matrix_fn) and matrix_fn() is not None
        if not ready or self.controller is None:
            self.log("[ARM] 眼在手上标定不可用，三个按键只用胸口坐标")
            return False
        success = False
        watch(True)
        try:
            self.phase = "wrist_pbvs"
            dwell_s = max(0.0, float(self.wrist_dwell_s))
            self.message = f"胸口已到位，等待右腕图像 {dwell_s:.0f} 秒后再进近"
            self.log(f"[ARM] {self.message}")
            self._publish_ee()
            if self._return_ready.is_set() or self._stop.is_set():
                return False
            if callable(self.wrist_hold):
                self.wrist_hold()
            self._wrist_latched_torso = {}
            self._wrist_red_lift_m = 0.0
            self._wrist_close_advanced = False
            deadline = time.monotonic() + dwell_s
            close_range = False
            while True:
                self._raise_if_sequence_next()
                if self._return_ready.is_set() or self._stop.is_set():
                    return False
                sample = view() or {}
                if sample.get("buttons_fresh"):
                    self._latch_wrist_buttons(sample)
                if self._wrist_latched_torso:
                    break
                if (
                    sample.get("close_range")
                    and _uv_pair(sample.get("target_uv")) is not None
                    and _uv_pair(sample.get("tip_uv")) is not None
                ):
                    close_range = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.1, remaining))
            if self._wrist_latched_torso:
                self._pbvs_source = "wrist"
                self.message = "右腕已记录按键，开始进近"
                self.log(f"[ARM] {self.message}")
                success = True
                return True
            if close_range:
                self._pbvs_source = "wrist"
                self.message = "右腕近距无深度，开始图像对齐"
                self.log(f"[ARM] {self.message}")
                success = True
                return True
            sample = view() or {}
            if sample.get("red_bar") and not sample.get("buttons_fresh"):
                self._pbvs_source = "wrist"
                self.message = "右腕没看到按键，看到红条，抬高后再看"
                self.log(f"[ARM] {self.message}")
                success = True
                return True
            if not bool(sample.get("seen")):
                self.message = "右腕没有看到面板和按键，粗定位后保持"
                self.log(f"[ARM] {self.message}")
                return False
            wrist = self._wrist_buttons_torso(sample.get("buttons_cam_m") or {})
            if not wrist:
                self._pbvs_source = "chest"
                self.message = "右腕没有算出按键三维坐标，粗定位后保持"
                self.log(f"[ARM] {self.message}")
                return False
            fused, source = fuse_pbvs_buttons(chest, wrist)
            self._locked_buttons = fused
            self._pbvs_source = source
            if source == "average":
                self.message = "胸口和右腕 PBVS 误差小于 2cm，粗定位采用平均坐标"
                success = True
            else:
                self.message = "胸口和右腕 PBVS 相差不小于 2cm，拒绝末端进近"
            self.log(f"[ARM] {self.message}")
            return success
        except SequenceNext:
            raise
        except Exception as exc:
            self._pbvs_source = "chest"
            self.message = f"右腕 PBVS 核对失败：{exc}"
            self.log(f"[ARM][WARN] {self.message}")
            return False
        finally:
            if not (keep_watching and success):
                try:
                    watch(False)
                except Exception:
                    pass

    def _move_tool_to(self, goal: tuple[float, float, float]) -> bool:
        """One Cartesian move to an absolute torso point, without the X-gap policy."""
        if self.controller is None:
            return False
        rpy = tuple(float(value) for value in self._nominal_tool_rpy_deg())
        if callable(getattr(self.controller, "move_cartesian", None)):
            ok, detail = self.controller.move_cartesian("right", goal, rpy, seconds=2.0)
            if not ok:
                self.message = f"抬起失败：{detail}"
                return False
            time.sleep(0.35)
            return True
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        solve_pose = None if self.ik is None else getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not commanded or not measured or not callable(solve_pose):
            self.message = "抬起失败：末端 IK 不可用"
            return False
        try:
            current_tip = self._measured_tip()
        except Exception as exc:
            self.message = f"抬起失败：{exc}"
            return False
        if current_tip is None:
            self.message = "抬起失败：没有指尖坐标"
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

    def _lower_right_arm(self, reason: str) -> None:
        if self.controller is None or self._return_ready.is_set() or self._stop.is_set():
            return
        self.message = reason
        self.log(f"[ARM] {self.message}")
        self.controller.ramp_arm("right", REST_DEG, seconds=6.0, dt=0.02)
        self._publish_ee()

    def _lower_right_after_sequence(self) -> None:
        self._lower_right_arm("动作结束，放下右臂")

    def _button_xyz_from_telem(self, telem: dict[str, Any], name: str) -> tuple[float, float, float] | None:
        raw = (telem.get("buttons_torso_m") or {}).get(name)
        if not raw or len(raw) != 3:
            return None
        try:
            goal = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(goal) != 3 or not all(math.isfinite(value) for value in goal):
            return None
        return goal

    def _recognize_sequence_button(self, name: str, timeout_s: float = 4.0) -> tuple[float, float, float] | None:
        """Chest fix after the arm is down. The previous press lock is not reused."""
        self._locked_buttons.pop(name, None)
        forget = getattr(self, "on_reidentify", None)
        if callable(forget):
            try:
                forget()
            except Exception as exc:
                self.log(f"[ARM][WARN] 清除按键锁定失败: {exc}")
        select = getattr(self, "on_select_target", None)
        if callable(select):
            try:
                select(name)
            except Exception as exc:
                self.log(f"[ARM][WARN] 切换识别目标失败: {exc}")
        self.message = f"重新识别 {name}"
        self.log(f"[ARM] {self.message}")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            self._raise_if_sequence_next()
            if self._return_ready.is_set() or self._stop.is_set():
                return None
            telem = self.telemetry() or {}
            goal = self._button_xyz_from_telem(telem, name)
            fresh = (
                goal is not None
                and bool(telem.get("panel_visible"))
                and bool(telem.get("depth_available"))
                and not bool(telem.get("using_last_target"))
            )
            if fresh:
                return goal
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _sequence_pbvs_once(self, name: str, goal: tuple[float, float, float]) -> bool:
        """One Cartesian move onto the button. No standoff and no second correction."""
        self._swing_right_arm_clear()
        self._goal_name = name
        self._locked_buttons = {name: goal}
        self._pbvs_release = False
        self._ibvs_enabled = True
        self._running.set()
        self._control_stage = "DIRECT_PBVS"
        self.phase = "direct_pbvs"
        self.message = (
            f"{name} PBVS 一次到位 "
            f"{goal[0]:.3f} {goal[1]:.3f} {goal[2]:.3f}"
        )
        self.log(f"[ARM] {self.message}")
        distance = self._tip_distance_to(goal)
        if distance is not None and distance > self.direct_pbvs_max_m:
            self.phase = "direct_pbvs_rejected"
            self.message = (
                f"PBVS 拒绝：距离 {distance*1000:.0f}mm "
                f"> {self.direct_pbvs_max_m*1000:.0f}mm"
            )
            self.log(f"[ARM] {self.message}")
            return False
        if callable(getattr(self.controller, "move_cartesian", None)):
            ok, detail = self._play_service_cartesian(
                self._measured_tip(),
                goal,
                tuple(float(value) for value in self._nominal_tool_rpy_deg()),
                f"{name} PBVS",
            )
            if not ok:
                self.phase = "direct_pbvs_rejected"
                self.message = f"PBVS 失败：{detail}"
                self.log(f"[ARM] {self.message}")
                return False
            self.phase = "direct_pbvs_complete"
            if detail:
                self.message = detail
            return True
        commanded = self.controller.commanded_arm_rad("right")
        measured = self.controller.measured_arm_rad("right")
        if not commanded or not measured:
            self.phase = "direct_pbvs_rejected"
            self.message = "PBVS 拒绝：没有右臂关节状态"
            self.log(f"[ARM] {self.message}")
            return False
        tip_xyz = getattr(self.ik, "tip_xyz", None)
        solve_pose = getattr(self.ik, "apply_loaded_tool_pose_target", None)
        if not callable(tip_xyz) or not callable(solve_pose):
            self.phase = "direct_pbvs_rejected"
            self.message = "PBVS 拒绝：运控末端 IK 未更新"
            self.log(f"[ARM] {self.message}")
            return False
        try:
            current_tip = self._measured_tip()
        except Exception as exc:
            self.phase = "direct_pbvs_rejected"
            self.message = f"PBVS 拒绝：FK 指尖失败 {exc}"
            self.log(f"[ARM] {self.message}")
            return False
        if current_tip is None:
            self.phase = "direct_pbvs_rejected"
            self.message = "PBVS 拒绝：没有指尖坐标"
            self.log(f"[ARM] {self.message}")
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
            self.phase = "direct_pbvs_rejected"
            self.message = self._pbvs_search_failure("PBVS", found)
            self.log(f"[ARM] {self.message}")
            return False
        self._play_cartesian_samples(found["samples"])
        self._wait_until_tracked(found["q"])
        self.phase = "direct_pbvs_complete"
        return True

    def _run_hybrid_button(self, name: str, goal: tuple[float, float, float]) -> bool:
        label = SEQUENCE_BUTTON_LABELS.get(name, name)
        self._begin_action_pose(label, goal)
        arrived = False
        try:
            arrived = bool(self._run_hybrid_button_run(name, goal))
            return arrived
        finally:
            if self._action_pose_pending:
                outcome = "已释放" if self._stop.is_set() else ("到位" if arrived else "未到位")
                self._record_action_pose(outcome)

    def _run_hybrid_button_run(self, name: str, goal: tuple[float, float, float]) -> bool:
        """Same path as one Hybrid Start: chest PBVS until the fingertip arrives, then wrist IBVS."""
        self._goal_name = name
        self._locked_buttons = {name: goal}
        self._pbvs_release = False
        self._service_pbvs_sent = False
        self._pbvs_deadline = None
        self._pbvs_wrist_checked = False
        self._wrist_close_advanced = False
        self._chest_height_after_pbvs = False
        self._wrist_latched_torso = {}
        label = SEQUENCE_BUTTON_LABELS.get(name, name)
        if self._sequence_run:
            self._sequence_stage = f"{label} PBVS"
            self.message = f"{label} PBVS"
        self._wrist_red_lift_m = 0.0
        self._wrist_ibvs_step = None
        self._wrist_ibvs_uv = None
        self._wrist_ibvs_jump_frames = 0
        self._control_stage = "PBVS_COARSE"
        self._ibvs_enabled = True
        self._running.set()
        telem = {
            "target": name,
            "target_torso_m": [float(value) for value in goal],
            "buttons_torso_m": {name: [float(value) for value in goal]},
            "panel_visible": True,
            "depth_available": True,
        }
        moves = 0
        drops = 0
        nudges = 0
        while True:
            self._raise_if_sequence_next()
            if self._return_ready.is_set() or self._stop.is_set():
                return False
            err = self._tip_error_to(goal)
            live_xyz = self._live_chest_xyz()
            live_err = None if live_xyz is None else self._tip_error_to(live_xyz)
            motion_err = live_err if live_err is not None else err
            can_forward = motion_err is not None and pbvs_forward_step(motion_err) is not None
            can_lateral = motion_err is not None and pbvs_lateral_step(motion_err) is not None
            can_descend = motion_err is not None and pbvs_descent_step(motion_err) is not None
            x_ref = None
            if live_xyz is not None:
                x_ref = float(live_xyz[0])
            elif goal is not None:
                x_ref = float(goal[0])
            past_plane = False
            if x_ref is not None and motion_err is not None:
                past_plane = coarse_x_retract(float(x_ref) - float(motion_err[0]), x_ref) is not None
            can_retract = (
                motion_err is not None and pbvs_retract_step(motion_err) is not None
            ) or past_plane
            if moves >= 8 and not ((can_forward or can_lateral or can_descend or can_retract) and nudges < 15):
                break
            fresh = self.telemetry() or {}
            telem["wrist_contact"] = bool(fresh.get("wrist_contact"))
            self._chest_height_hold = False
            self._forward_hold = False
            self._pbvs_once(telem)
            if str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                return False
            if self._control_stage == "WRIST_APPROACH":
                break
            if self._control_stage == "HOLD" or self.phase in {
                "pbvs_timeout",
                "pbvs_ik_rejected",
                "wrist_pbvs_unavailable",
            }:
                return False
            if self._chest_height_hold:
                drops += 1
                if drops >= 25 and moves >= 8:
                    break
                continue
            if self._forward_hold:
                nudges += 1
                continue
            if moves >= 8:
                break
            moves += 1
        if self._control_stage != "WRIST_APPROACH":
            live_xyz = self._live_chest_xyz()
            live_err = None if live_xyz is None else self._tip_error_to(live_xyz)
            err = live_err if live_err is not None else self._tip_error_to(goal)
            if err is None:
                detail = ""
            else:
                detail = "，目标−末端 " + " ".join(
                    f"{axis} {err[index] * 1000:+.0f}mm" for index, axis in enumerate("XYZ")
                )
            self.phase = "pbvs_timeout"
            self.message = f"{name} 粗定位未到位{detail}"
            self.log(f"[ARM] {self.message}")
            return False
        self._chest_height_after_pbvs = True
        if self._sequence_run:
            self._sequence_stage = f"{label} 进近"
            self.message = f"{label} 进近"
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            self._raise_if_sequence_next()
            if self._return_ready.is_set() or self._stop.is_set():
                return False
            if self.phase == "contact" or self._wrist_contact():
                return True
            fresh = self.telemetry() or {}
            self._wrist_approach_once(fresh)
            if str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                return False
            if self.phase == "contact":
                return True
            if self._control_stage == "HOLD":
                return False
            time.sleep(0.05)
        self.phase = "direct_pbvs_short"
        self.message = f"{name} 右腕进近未碰到按键"
        self.log(f"[ARM] {self.message}")
        return False

    def _snapshot_right_joints(self) -> list[float] | None:
        if self.controller is None:
            return None
        try:
            measured = self.controller.measured_arm_rad("right")
        except Exception:
            return None
        if not measured or len(measured) < 7:
            return None
        return [float(value) for value in measured[:7]]

    def _restore_right_joints(self, q_rad: Sequence[float]) -> bool:
        """Joint-space return. A failed press must not be the next press's start pose."""
        if self._stop.is_set() or getattr(self.controller, "_aborted", False):
            return False
        if self.controller is None or q_rad is None or len(q_rad) < 7:
            return False
        apply = getattr(self.controller, "apply_arm_rad", None)
        if not callable(apply):
            return False
        target = [float(value) for value in q_rad[:7]]
        try:
            apply("right", target, seconds=2.5, dt=MOTION_DT_S)
        except Exception as exc:
            self.log(f"[ARM] 回到动作前关节失败: {exc}")
            return False
        self._wait_until_tracked(target)
        return True

    def _retreat_clear_of_panel(self, label: str) -> None:
        """Move down and right so the chest can see the faceplate again.

        One large tip delta is rejected by the 0.25 rad IK jump limit, so this
        is a planned Cartesian move. The planner still plays sag-compensated
        q_command.
        """
        self._sequence_stage = f"{label} 移开"
        self.message = f"{label} 移开：向下、向右离开面板"
        self.log(f"[ARM] {self.message}")
        tip = self._measured_tip()
        if tip is None:
            self.message = f"{label} 移开失败：没有指尖坐标"
            self.log(f"[ARM] {self.message}")
            return
        goal = sequence_clear_goal(tip)
        moved = self._move_tool_to(goal)
        if moved:
            self.message = f"{label} 移开：向下、向右离开面板"
        elif str(self.message).startswith("抬起失败"):
            self.message = "移开失败" + str(self.message)[len("抬起失败") :]
            self.log(f"[ARM] {self.message}")
        self._publish_ee()

    def _drop_sequence_controller(self) -> None:
        """Release whichever backend is holding the arms. Do not join this thread."""
        controller = self.controller
        self.controller = None
        if self._motion_backend == "service":
            self._kill_service_now()
        aborted = False
        abort = getattr(controller, "abort", None) if controller is not None else None
        if callable(abort):
            try:
                abort()
                aborted = bool(getattr(controller, "_aborted", False))
            except Exception as exc:
                self.log(f"[ARM] 松权失败: {exc}")
        if not aborted and controller is not None:
            release = getattr(controller, "release", None)
            if callable(release):
                try:
                    release(seconds=0.0)
                except Exception as exc:
                    self.log(f"[ARM] 松权失败: {exc}")
        if self._motion_backend != "service" and not aborted:
            self._drop_sdk_hold()
        self._stop.set()

    def _finish_sequence_and_release(self) -> None:
        self._sequence_stage = "结束放下"
        try:
            self._lower_right_arm("结束放下")
        except Exception as exc:
            self.log(f"[ARM] 放下失败: {exc}")
        if callable(self.on_screen_capture):
            try:
                self.on_screen_capture()
            except Exception as exc:
                self.log(f"[ARM][WARN] 屏幕截取失败: {exc}")
        self._pbvs_release = True
        self._ibvs_enabled = False
        self._running.clear()
        self._sequence_run = False
        self._control_stage = "RELEASED"
        self.phase = "released"
        self.message = "动作组合完成，右臂已放下，已释放控制"
        self.log(f"[ARM] {self.message}")
        self._emit_task_end("done")
        self._drop_sequence_controller()

    def _run_direct_sequence_button(self, name: str, goal: tuple[float, float, float]) -> bool:
        """Same path as one Direct PBVS start. The combination releases only at the end."""
        self._raise_if_sequence_next()
        label = SEQUENCE_BUTTON_LABELS.get(name, name)
        self._sequence_stage = f"{label} PBVS"
        self.message = f"{label} Direct PBVS"
        telem = {
            "target": name,
            "target_torso_m": [float(value) for value in goal],
            "buttons_torso_m": {name: [float(value) for value in goal]},
            "panel_visible": True,
            "depth_available": True,
        }
        self._direct_pbvs_once(telem, release=False)
        self._raise_if_sequence_next()
        if self._return_ready.is_set() or self._stop.is_set():
            return False
        return self.phase == "direct_pbvs_complete"

    def _reset_screen_watch(self) -> None:
        self._screen_before = None
        self._screen_before_info = None
        self._screen_block = None
        self._screen_report = ""
        self._screen_history = []

    def _format_xyz_mm(self, point: tuple[float, float, float] | None) -> str:
        if point is None or len(point) != 3:
            return "未读到"
        try:
            values = tuple(float(value) for value in point)
        except (TypeError, ValueError):
            return "未读到"
        if not all(math.isfinite(value) for value in values):
            return "未读到"
        return " ".join(
            f"{axis} {values[index] * 1000:.0f}" for index, axis in enumerate("XYZ")
        ) + " mm"

    def _begin_action_pose(self, label: str, goal: tuple[float, float, float] | None) -> None:
        if self._action_pose_pending:
            self._record_action_pose("未完成")
        self._action_label = str(label or "动作")
        parsed = None
        if goal is not None and len(goal) == 3:
            try:
                parsed = tuple(float(value) for value in goal)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and not all(math.isfinite(value) for value in parsed):
                parsed = None
        self._action_goal = parsed
        self._action_pose_pending = True

    def _record_action_pose(self, outcome: str) -> None:
        """Keep the fingertip and the button target when an action ends."""
        if not self._action_pose_pending:
            return
        self._action_pose_pending = False
        try:
            tip = self._measured_tip()
        except Exception:
            tip = None
        goal = self._action_goal or self._pose_target_xyz() or self._fresh_torso_target()
        line = (
            f"{self._action_label} {outcome}  "
            f"到位 {self._format_xyz_mm(tip)}  目标 {self._format_xyz_mm(goal)}"
        )
        history = list(self._action_pose_history)
        history.append(line)
        self._action_pose_history = history[-12:]
        self.log(f"[ARM] {line}")

    def _retry_text(self, count: int) -> str:
        return f"【重试】第{int(count)}次"

    def _mark_retry(self, count: int) -> None:
        """0 is the first try. A later count stays on the page until this try ends."""
        self._retry_count = max(0, int(count))

    def _message_for_page(self) -> str:
        text = str(self.message or "")
        count = int(getattr(self, "_retry_count", 0) or 0)
        if count <= 0:
            return text
        banner = self._retry_text(count)
        if banner in text:
            return text
        return f"{banner} {text}".strip()

    def _push_screen_line(self, line: str) -> None:
        self._screen_report = line
        history = list(self._screen_history)
        history.append(line)
        self._screen_history = history[-12:]
        self.message = line
        self.log(f"[ARM] {line}")

    def _emit_screen_record(self, image, label: str, info: dict[str, Any] | None) -> None:
        callback = self.on_screen_record
        if not callable(callback):
            return
        try:
            callback(image, label, None if info is None else dict(info))
        except Exception as exc:
            self.log(f"[ARM][WARN] 屏幕记录失败: {exc}")

    def _emit_task_begin(self) -> None:
        callback = self.on_task_begin
        if not callable(callback):
            return
        try:
            callback()
        except Exception as exc:
            self.log(f"[ARM][WARN] 任务记录失败: {exc}")

    def _emit_task_end(self, reason: str) -> None:
        callback = self.on_task_end
        if not callable(callback):
            return
        try:
            callback(reason)
        except Exception as exc:
            self.log(f"[ARM][WARN] 任务记录失败: {exc}")

    def _sleep_for_screen(self) -> None:
        remaining = max(0.0, float(self.screen_settle_s))
        while remaining > 0:
            self._raise_if_sequence_next()
            if self._return_ready.is_set() or self._stop.is_set():
                return
            step = min(0.05, remaining)
            time.sleep(step)
            remaining -= step

    def _capture_progress_frame(self):
        reader = self.read_screen
        if not callable(reader):
            return None
        frame = None
        tries = 1 if self.screen_settle_s <= 0 else 3
        for index in range(tries):
            if index:
                self._sleep_for_screen()
                if self._return_ready.is_set() or self._stop.is_set():
                    return frame
            try:
                shot = reader()
            except Exception as exc:
                self.log(f"[ARM][WARN] 屏幕截取失败: {exc}")
                shot = None
            if shot is not None and getattr(shot, "size", 0):
                frame = shot
        return frame

    def _publish_screen_jpeg(self) -> None:
        if not callable(self.on_screen_capture):
            return
        try:
            self.on_screen_capture()
        except Exception as exc:
            self.log(f"[ARM][WARN] 屏幕截取失败: {exc}")

    def _read_progress_screen(self, label: str) -> bool | None:
        """Capture one screen. True means it changed, False means the same, None means unreadable."""
        self._sleep_for_screen()
        if self._return_ready.is_set() or self._stop.is_set():
            return None
        image = self._capture_progress_frame()
        self._publish_screen_jpeg()
        if image is not None and hasattr(image, "copy"):
            image = image.copy()
        if image is None:
            self._push_screen_line(f"{label} 没有读到屏幕")
            self._emit_screen_record(None, label, None)
            return None
        info = read_panel(image)
        self._screen_block = dict(info)
        desc = format_block(info)
        previous = self._screen_before
        previous_info = self._screen_before_info
        self._screen_before = image
        self._screen_before_info = dict(info)
        self._emit_screen_record(image, label, info)
        if previous is None or previous_info is None:
            if label == "动作前":
                self._push_screen_line(f"{label}。{desc}")
                return True
            self._push_screen_line(f"{label} 补上动作前屏幕。{desc}")
            return None
        changed = blocks_differ(previous, image, previous_info, info)
        if changed:
            # Any visible change counts as a press. A required screen can reject it later.
            self._push_screen_line(f"{label} 按到了。{desc}")
        else:
            self._push_screen_line(f"{label} 屏幕没有变化。{desc}")
        return changed

    def _screen_attempt_failed(self, label: str, changed: bool | None, attempt: int) -> bool:
        if changed is True:
            return False
        if attempt + 1 >= SCREEN_PRESS_ATTEMPTS:
            done = self._retry_text(attempt) if attempt else "这次"
            if changed is None:
                self._push_screen_line(f"{label} 没有读到屏幕，{done}后仍失败，停止")
            else:
                self._push_screen_line(f"{label} 屏幕仍无变化，{done}后仍失败，停止")
            return True
        nxt = attempt + 1
        if changed is None:
            self._push_screen_line(f"{label} 没有读到屏幕，{self._retry_text(nxt)}")
        else:
            self._push_screen_line(f"{label} 屏幕没有变化，{self._retry_text(nxt)}")
        return False

    def _hold_after_screen(self, *, arrived: bool) -> None:
        self._ibvs_enabled = False
        self._running.clear()
        self._control_stage = "HOLD"
        self.phase = "direct_pbvs_complete" if arrived else "direct_pbvs_short"

    def _run_single_press_with_screen(self, telem: dict[str, Any], *, direct: bool) -> None:
        """One selected button. Look at the screen before it, and again after each try.
        If the button is pressed, the screen will change.
        If the button is not pressed, the screen will not change.
        If the screen is not changed, the button is not pressed.
        If the screen is changed, the button is pressed.
        If the button is pressed, the screen will change.
        If the button is not pressed, the screen will not change.
        If the screen is not changed, the button is not pressed.
        If the screen is changed, the button is pressed.
        """
        name = str(telem.get("target") or "")
        if name not in BUTTON_NAMES:
            self.message = "读取屏幕需要先选择 Button"
            self.log(f"[ARM] {self.message}")
            self._hold_after_screen(arrived=False)
            return
        label = SEQUENCE_BUTTON_LABELS.get(name, name)
        self._reset_screen_watch()
        self._read_progress_screen("动作前")
        if self._return_ready.is_set() or self._stop.is_set():
            return
        attempt = 0
        self._mark_retry(0)
        while True:
            if self._return_ready.is_set() or self._stop.is_set():
                self._mark_retry(0)
                return
            self._mark_retry(attempt)
            goal = self._recognize_sequence_button(name)
            if goal is None:
                self.phase = "direct_pbvs_rejected"
                self.message = f"重新识别 {name} 失败，胸口没有看到该按键"
                self.log(f"[ARM] {self.message}")
                self._mark_retry(0)
                self._hold_after_screen(arrived=False)
                return
            kind = "Direct PBVS" if direct else "Hybrid"
            self.message = f"{label} {kind}" if not attempt else f"{self._retry_text(attempt)} {label} {kind}"
            self.log(f"[ARM] {self.message}")
            ran = (
                self._run_direct_sequence_button(name, goal)
                if direct
                else self._run_hybrid_button(name, goal) 
            )
            if self._return_ready.is_set() or self._stop.is_set():
                self._mark_retry(0)
                return
            if not ran:
                self._mark_retry(0)
                self._hold_after_screen(arrived=False)
                return
            self._retreat_clear_of_panel(label)
            if self._return_ready.is_set() or self._stop.is_set():
                self._mark_retry(0)
                return
            changed = self._read_progress_screen(label)
            if self._return_ready.is_set() or self._stop.is_set():
                self._mark_retry(0)
                return
            if changed is True:
                self._mark_retry(0)
                self._hold_after_screen(arrived=True)
                return
            if self._screen_attempt_failed(label, changed, attempt):
                self.message = f"{label} {self._retry_text(attempt)}后仍失败，停止"
                self.log(f"[ARM] {self.message}")
                self._mark_retry(0)
                self._hold_after_screen(arrived=False)
                return
            attempt += 1
            self._mark_retry(attempt)

    def _run_button_sequence(self, start_index: int = 0) -> None:
        """Each button is Direct PBVS or Hybrid. Between buttons the arm clears down and right."""
        steps = list(self._button_sequence)
        direct = self._sequence_mode == "direct_pbvs"
        self._locked_buttons = {}
        with self._sequence_lock:
            self._sequence_skip = 0
            self._sequence_inside = True
        cleared = False
        try:
            if self._screen_progress and start_index <= 0:
                self._reset_screen_watch()
                self._read_progress_screen("动作前")
                if self._return_ready.is_set() or self._stop.is_set():
                    return
            for index, name in enumerate(steps):
                if index < start_index:
                    continue
                if self._return_ready.is_set() or self._stop.is_set():
                    return
                self._sequence_index = index
                cleared = False
                label = SEQUENCE_BUTTON_LABELS.get(name, name)
                attempt = 0
                goal_fixed = None
                anchor = None
                self._mark_retry(0)
                try:
                    while True:
                        if self._return_ready.is_set() or self._stop.is_set():
                            return
                        self._mark_retry(attempt)
                        if index > 0 and not cleared:
                            previous = SEQUENCE_BUTTON_LABELS.get(steps[index - 1], steps[index - 1])
                            self._retreat_clear_of_panel(previous)
                            if self._return_ready.is_set() or self._stop.is_set():
                                return
                            cleared = True
                            anchor = None
                        if anchor is None:
                            anchor = self._snapshot_right_joints()
                        elif attempt:
                            self._restore_right_joints(anchor)
                        cleared = True
                        self._sequence_stage = f"{label} PBVS"
                        if goal_fixed is None:
                            goal_fixed = self._recognize_sequence_button(name)
                        goal = goal_fixed
                        kind = "Direct PBVS" if direct else "Hybrid"
                        self.message = f"动作 {index + 1}/{len(steps)} {name} {kind}"
                        if attempt:
                            self.message = f"{self._retry_text(attempt)} {label} {kind}"
                        self.log(f"[ARM] {self.message}")
                        if goal is None:
                            self.phase = "direct_pbvs_rejected"
                            self.message = f"重新识别 {name} 失败，胸口没有看到该按键"
                            self.log(f"[ARM] {self.message}")
                            ran = False
                        else:
                            ran = (
                                self._run_direct_sequence_button(name, goal)
                                if direct
                                else self._run_hybrid_button(name, goal)
                            )
                        if not ran:
                            if self._return_ready.is_set() or self._stop.is_set():
                                return
                            if attempt >= SEQUENCE_ACTION_RETRIES:
                                self.message = (
                                    f"{label} {self._retry_text(attempt)}后仍失败，进行下一步"
                                )
                                self.log(f"[ARM] {self.message}")
                                self._mark_retry(0)
                                self._restore_right_joints(anchor)
                                break
                            self._mark_retry(attempt + 1)
                            self.message = f"{self._retry_text(attempt + 1)} {label} {kind}"
                            self.log(f"[ARM] {self.message}")
                            if self._return_ready.is_set() or self._stop.is_set():
                                return
                            attempt += 1
                            continue
                        if self._return_ready.is_set() or self._stop.is_set():
                            return
                        if not self._screen_progress:
                            self._mark_retry(0)
                            self.message = (
                                f"Direct PBVS 完成，{name} 完成"
                                if direct
                                else f"右腕进近完成，{name} 完成"
                            )
                            self.log(f"[ARM] {self.message}")
                            break
                        self._restore_right_joints(anchor)
                        if self._return_ready.is_set() or self._stop.is_set():
                            return
                        changed = self._read_progress_screen(label)
                        if self._return_ready.is_set() or self._stop.is_set():
                            return
                        if changed is True:
                            self._mark_retry(0)
                            break
                        if self._screen_attempt_failed(label, changed, attempt):
                            self.message = f"{label} {self._retry_text(attempt)}后仍失败，进行下一步"
                            self.log(f"[ARM] {self.message}")
                            self._mark_retry(0)
                            break
                        self._mark_retry(attempt + 1)
                        attempt += 1
                except SequenceNext:
                    if index + 1 >= len(steps):
                        break
                    continue
            if self._return_ready.is_set() or self._stop.is_set():
                return
            self._finish_sequence_and_release()
        finally:
            self._mark_retry(0)
            with self._sequence_lock:
                self._sequence_inside = False
                self._sequence_skip = 0

    def _wrist_contact(self) -> bool:
        telem = self.telemetry() or {}
        return bool(telem.get("wrist_contact"))

    def _wait_wrist_contact(self, timeout_s: float = 0.4) -> bool:
        """The wrist image is debounced. A touch already in the frame returns at once."""
        if self._wrist_contact():
            return True
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if self._return_ready.is_set() or self._stop.is_set():
                return False
            time.sleep(0.05)
            if self._wrist_contact():
                return True
        return self._wrist_contact()

    def _tool_xyz_measured(self) -> tuple[float, float, float] | None:
        measured = self.controller.measured_arm_rad("right") if self.controller is not None else None
        if not measured:
            return None
        try:
            point = self._tool_xyz(measured)
        except Exception:
            return None
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        return point

    def _measured_tip(self) -> tuple[float, float, float] | None:
        if self.controller is None:
            return None
        try:
            measured = self.controller.measured_arm_rad("right")
        except Exception:
            return None
        if not measured:
            return None
        try:
            current = self._tool_xyz(measured)
        except Exception:
            return None
        if len(current) != 3 or not all(math.isfinite(value) for value in current):
            return None
        return current

    def _tip_error_to(self, goal: tuple[float, float, float]) -> tuple[float, float, float] | None:
        current = self._measured_tip()
        if current is None:
            return None
        return tuple(float(goal[index]) - current[index] for index in range(3))

    def _tip_distance_to(self, goal: tuple[float, float, float]) -> float | None:
        error = self._tip_error_to(goal)
        if error is None:
            return None
        return math.sqrt(sum(value * value for value in error))

    def _pose_target_xyz(self) -> tuple[float, float, float] | None:
        """Current chest-pose button. A lock from the start of the move does not count."""
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

    def _service_coarse_target(self) -> tuple[float, float, float] | None: # pylint: disable=unused-argument
        return self._fresh_torso_target()

    def _direct_pbvs_via_service(self, goal: tuple[float, float, float]) -> None: # this function is used to move the arm to the button
        """Service Cartesian to the button. A low start rises before it advances."""
        rpy = tuple(float(value) for value in self._nominal_tool_rpy_deg())
        distance = self._tip_distance_to(goal)
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
        self.phase = "direct_pbvs"
        tip = self._measured_tip()
        hold = pbvs_yz_hold_goal(tip, goal) if PBVS_YZ_FIRST and tip is not None else None
        first = hold if hold is not None else goal
        self.message = (
            f"Direct PBVS 服务 /h2_arm/singlearmcartesian "
            f"XYZ {first[0]:.3f} {first[1]:.3f} {first[2]:.3f}"
        )
        self.log(f"[ARM] {self.message}")
        ok, detail = self._play_service_cartesian(tip, first, rpy, "Direct PBVS")
        if not ok and first is not goal:
            ok, detail = self._play_service_cartesian(self._measured_tip(), goal, rpy, "Direct PBVS")
        if not ok:
            self._finish_direct_pbvs(False, f"Direct PBVS 服务失败：{detail}")
            self.phase = "direct_pbvs_rejected"
            return
        if self._return_ready.is_set():
            return
        time.sleep(0.35)
        follow = self._pose_target_xyz() or goal
        after = self._tip_distance_to(follow)
        after_error = self._tip_error_to(follow)
        if after_error is not None and direct_pbvs_arrived(after_error):
            note = f"{detail}；" if detail else ""
            after = math.sqrt(sum(value * value for value in after_error))
            self._finish_direct_pbvs(True, f"{note}Direct PBVS 运控一次到位，剩余 {after*1000:.0f}mm")
            return
        if after is not None and after > self.direct_pbvs_max_m:
            self._finish_direct_pbvs(
                False,
                (
                    f"Direct PBVS 首段已执行，修正距离 {after*1000:.0f}mm "
                    f"> {self.direct_pbvs_max_m*1000:.0f}mm，未再发"
                ),
            )
            return
        self.message = "Direct PBVS 修正 /h2_arm/singlearmcartesian"
        self.log(f"[ARM] {self.message}")
        ok, detail = self._play_service_cartesian(self._measured_tip(), follow, rpy, "Direct PBVS")
        if not ok:
            self._finish_direct_pbvs(False, f"Direct PBVS 修正失败：{detail}")
            self.phase = "direct_pbvs_rejected"
            return
        time.sleep(0.25)
        follow = self._pose_target_xyz() or follow
        error = self._tip_error_to(follow)
        if error is None:
            self._finish_direct_pbvs(False, "Direct PBVS 运控两段已执行，末端坐标丢失")
            return
        final = math.sqrt(sum(value * value for value in error))
        detail = f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
        if direct_pbvs_arrived(error):
            self._finish_direct_pbvs(True, f"Direct PBVS 运控两段已执行，剩余 {final*1000:.0f}mm，{detail}")
            return
        self._finish_direct_pbvs(False, f"Direct PBVS 运控两段已执行，未到位，还差 {detail}")

    def _direct_pbvs_once(self, telem: dict[str, Any], *, release: bool = True) -> None:
        name = str(telem.get("target") or "")
        label = SEQUENCE_BUTTON_LABELS.get(name, name or "Direct PBVS")
        raw = telem.get("target_torso_m")
        goal = None
        if raw and len(raw) == 3:
            try:
                goal = tuple(float(value) for value in raw)
            except (TypeError, ValueError):
                goal = None
        self._begin_action_pose(label, goal)
        try:
            self._direct_pbvs_run(telem, release=release)
        finally:
            if self._action_pose_pending:
                self._record_action_pose("已释放" if self._stop.is_set() else "未到位")

    def _direct_pbvs_run(self, telem: dict[str, Any], *, release: bool = True) -> None:
        """Move the measured fingertip onto the button, then correct gravity sag."""
        self._swing_right_arm_clear()
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
            self._direct_pbvs_via_service(self._pose_target_xyz() or target)
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
        try:
            current_tip = self._tool_xyz(measured)
        except Exception as exc:
            self._ibvs_enabled = False
            self._control_stage = "HOLD"
            self.phase = "direct_pbvs_rejected"
            self.message = f"Direct PBVS 拒绝：FK 指尖失败 {exc}"
            return
        goal = self._pose_target_xyz() or target
        self._action_goal = tuple(float(value) for value in goal)
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
        for _round in range(DIRECT_PBVS_START_ROUNDS):
            error = tuple(goal[index] - current_tip[index] for index in range(3))
            round_start = math.sqrt(sum(value * value for value in error))
            for shot in range(DIRECT_PBVS_MAX_SHOTS):
                if self._return_ready.is_set():
                    return
                error = tuple(goal[index] - current_tip[index] for index in range(3))
                distance = math.sqrt(sum(value * value for value in error))
                if direct_pbvs_arrived(error):
                    self._finish_direct_pbvs(
                        True,
                        (
                            f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                            f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
                        ),
                    )
                    return
                if float(current_tip[0]) > float(goal[0]) + 1e-3:
                    wall = (float(goal[0]) - float(current_tip[0]), error[1], error[2])
                    if self._retract_if_deeper_than_button(wall, "DIRECT_PBVS", past_m=0.0):
                        measured = self.controller.measured_arm_rad("right")
                        commanded = self.controller.commanded_arm_rad("right") or measured
                        q = commanded or measured
                        if not q or not measured:
                            break
                        try:
                            current_tip = self._tool_xyz(measured)
                        except Exception:
                            break
                        continue
                    self._finish_direct_pbvs(
                        False,
                        (
                            "Direct PBVS 末端 X 已超过目标 "
                            f"{(float(current_tip[0]) - float(goal[0])) * 1000:.0f}mm，停止前推"
                        ),
                    )
                    return
                self.message = (
                    f"Direct PBVS 第 {_round + 1} 轮 {shot + 1} 段，去按键，还差 "
                    f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
                )
                yz_now = math.hypot(float(error[1]), float(error[2]))
                goal_key = tuple(round(float(value), 3) for value in goal)
                hold = pbvs_yz_hold_goal(current_tip, goal) if PBVS_YZ_FIRST else None
                stalled = hold is not None and self._yz_attempt_stalled(goal_key, yz_now)
                axis_moved = False
                if hold is not None and not stalled and self._can_axis_step():
                    step = pbvs_yz_align_step(error)
                    if step is not None:
                        self._apply_torso_delta(
                            step,
                            seconds=PBVS_STEP_SECONDS,
                            phase="DIRECT_PBVS",
                            limit_m=max(abs(float(value)) for value in step),
                        )
                        if not str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                            self._pbvs_yz_before = (goal_key, yz_now)
                            self.message = self._yz_align_text(error, step)
                            axis_moved = True
                        elif self._sequence_run:
                            self._finish_direct_pbvs(False, self.message)
                            return
                elif hold is None and self._can_axis_step() and self._step_forward_if_aligned(
                    error, "DIRECT_PBVS"
                ):
                    self._pbvs_yz_before = None
                    axis_moved = True
                elif self._sequence_run and str(self.message).startswith(("IK 拒绝", "IBVS 服务失败")):
                    self._finish_direct_pbvs(False, self.message)
                    return
                if axis_moved:
                    measured = self.controller.measured_arm_rad("right")
                    commanded = self.controller.commanded_arm_rad("right") or measured
                    q = commanded or measured
                    if not q or not measured:
                        break
                    try:
                        current_tip = self._tool_xyz(measured)
                    except Exception:
                        break
                    continue
                # A failed YZ translation still aims the pose at button Y/Z with
                # X held. Closing X is only the fallback if that pose cannot be solved.
                plan_goal = goal if stalled or hold is None else hold
                found = self._plan_cartesian_tool_move(
                    commanded,
                    measured,
                    current_tip,
                    plan_goal,
                    solve_pose,
                    must_move=True,
                    stage_limited=False,
                )
                if (found is None or not found["ok"]) and plan_goal is not goal:
                    self._pbvs_yz_before = None
                    found = self._plan_cartesian_tool_move(
                        commanded,
                        measured,
                        current_tip,
                        goal,
                        solve_pose,
                        must_move=True,
                        stage_limited=False,
                    )
                    plan_goal = goal
                elif found is not None and found["ok"] and plan_goal is not goal:
                    self._pbvs_yz_before = (goal_key, yz_now)
                else:
                    self._pbvs_yz_before = None
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
                shot_q = [float(value) for value in measured[:7]] if measured and len(measured) >= 7 else []
                shot_distance = distance
                self._play_cartesian_samples(found["samples"])
                track_rad = self._hold_until_settled(found["q"])
                commanded = self.controller.commanded_arm_rad("right")
                measured = self.controller.measured_arm_rad("right")
                q = commanded or measured
                if not q or not measured:
                    break
                try:
                    current_tip = self._tool_xyz(measured)
                except Exception:
                    break
                moved = tuple(goal[index] - current_tip[index] for index in range(3))
                moved_distance = math.sqrt(sum(value * value for value in moved))
                if self._stop.is_set() or getattr(self.controller, "_aborted", False):
                    self._finish_direct_pbvs(False, "Direct PBVS 已释放，停在当前位置")
                    return
                if shot_q and (
                    moved_distance > shot_distance + 0.03
                    or float(current_tip[0]) > float(goal[0]) + 1e-3
                ):
                    self._restore_right_joints(shot_q)
                    try:
                        measured = self.controller.measured_arm_rad("right")
                        current_tip = self._tool_xyz(measured)
                    except Exception:
                        pass
                    self._finish_direct_pbvs(
                        False,
                        (
                            "Direct PBVS 末端 X 不能超过目标，已回到动作前，还差 "
                            f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
                        ),
                    )
                    return
                if direct_pbvs_arrived(moved):
                    self._finish_direct_pbvs(
                        True,
                        (
                            f"Direct PBVS 实测到位，剩余 {moved_distance*1000:.0f}mm，"
                            f"X {moved[0]*1000:.0f} Y {moved[1]*1000:.0f} Z {moved[2]*1000:.0f} mm"
                        ),
                    )
                    return
            if self._return_ready.is_set():
                return
            error = tuple(goal[index] - current_tip[index] for index in range(3))
            distance = math.sqrt(sum(value * value for value in error))
            if direct_pbvs_round_stalled(distance, round_start, track_rad):
                break
        error = tuple(goal[index] - current_tip[index] for index in range(3))
        distance = math.sqrt(sum(value * value for value in error))
        if direct_pbvs_arrived(error):
            self._finish_direct_pbvs(
                True,
                (
                    f"Direct PBVS 实测到位，剩余 {distance*1000:.0f}mm，"
                    f"X {error[0]*1000:.0f} Y {error[1]*1000:.0f} Z {error[2]*1000:.0f} mm"
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

    def _hold_until_settled(self, target_rad: Sequence[float]) -> float:
        """Republish the final joints until the arm is within 1.5° or two seconds pass."""
        error = self._wait_until_tracked(target_rad)
        writer = getattr(self.controller, "write", None)
        hold_fn = getattr(self.controller, "hold_positions", None)
        if not callable(writer) or not callable(hold_fn):
            return error
        tight = math.radians(1.5)
        deadline = time.monotonic() + 2.0
        while error > tight and time.monotonic() < deadline:
            if self._return_ready.is_set() or self._stop.is_set():
                break
            final = dict(hold_fn())
            for joint, value in zip(RIGHT_ARM_JOINTS, target_rad):
                final[joint] = float(value)
            writer(final, weight=1.0)
            time.sleep(0.05)
            error = self._tracked_joint_error_rad(target_rad)
        return error

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
        if (
            self._stop.is_set()
            or self.controller is None
            or getattr(self.controller, "_aborted", False)
        ):
            return
        hold = dict(self.controller.hold_positions())
        filtered = [0.0] * 7
        previous = None
        count = len(samples)
        for index, sample in enumerate(samples):
            if self._return_ready.is_set() or self._stop.is_set():
                break
            controller = self.controller
            if controller is None or getattr(controller, "_aborted", False):
                break
            # Coarse PBVS has to finish the raise. A wrist contact flag during
            # that move was leaving Z short of the measured button.
            if self._control_stage not in {"PBVS_COARSE", "DIRECT_PBVS"} and self._wrist_contact():
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
            controller.write(command, weight=1.0, dq=dq)
            previous = list(corrected)
            time.sleep(MOTION_DT_S)
        if self._return_ready.is_set() or self._stop.is_set():
            return
        controller = self.controller
        if samples and controller is not None and not getattr(controller, "_aborted", False):
            final = dict(hold)
            for joint, value in zip(RIGHT_ARM_JOINTS, samples[-1]):
                final[joint] = float(value)
            controller.write(final, weight=1.0)

    def _play_joint_target(self, target_rad: Sequence[float], seconds: float) -> None:
        """One solved joint segment. Quintic timing, with dq, when the SDK can stream.

        The line between the held joints and this IK solution is unchanged.
        apply_arm_rad leaves dq at 0, so kd brakes the whole ramp.
        """
        target = [float(value) for value in target_rad]
        writer = getattr(self.controller, "write", None)
        hold_fn = getattr(self.controller, "hold_positions", None)
        if callable(writer) and callable(hold_fn) and len(target) >= 7:
            hold = hold_fn()
            try:
                start = [float(hold[joint]) for joint in RIGHT_ARM_JOINTS]
            except (KeyError, TypeError):
                start = []
            if len(start) == 7:
                self._use_motion_gains()
                steps = max(1, int(round(max(float(seconds), MOTION_DT_S) / MOTION_DT_S)))
                samples = []
                for step in range(steps):
                    blend = quintic_blend(step / steps)
                    samples.append(
                        [
                            start[index] * (1.0 - blend) + target[index] * blend
                            for index in range(7)
                        ]
                    )
                samples.append(target[:7])
                self._stream_arm_samples(samples)
                return
        self.controller.apply_arm_rad("right", target, seconds=seconds, dt=MOTION_DT_S)

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
            try:
                q_new, ok, pos_err, jump = apply_tip(
                    q, torso, q, offset_m=tuple(self.tool_offset_wrist_m)
                )
            except TypeError:
                q_new, ok, pos_err, jump = apply_tip(q, torso, q)
        else:
            q_new, ok, pos_err, jump = self.ik.apply_delta(q, torso)
        if not ok:
            self.message = (
                f"IK 拒绝 这一步 {torso[0]*1000:.1f} {torso[1]*1000:.1f} {torso[2]*1000:.1f} mm，"
                f"解算剩余 {pos_err*1000:.1f} mm，关节 {math.degrees(float(jump)):.1f}°"
            )
            return False
        shoulder_raised = float(q_new[0]) < float(q[0]) - math.radians(0.4)
        self._play_joint_target(q_new, seconds)
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
            self._play_joint_target(q, 0.8)
            return True
        except Exception as exc:
            self.log(f"[ARM][WARN] 回退失败，保持当前姿态: {exc}")
            return False
