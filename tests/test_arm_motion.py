'''
test_arm_motion.py

This file contains the tests for the arm motion module.
It is used to test the arm motion module and ensure that it is working correctly.   
'''

import math
import sys
import time

from ibvs_pad_touch.web.arm_motion import (
    ArmMover,
    coarse_x_ceiling,
    coarse_x_command,
    service_rise_before_wall,
    pbvs_descent_step,
    pbvs_forward_step,
    pbvs_lateral_step,
    quintic_joint_seconds,
    slerp_rpy,
    RIGHT_ARM_JOINTS,
    DIRECT_EE_RPY_DEG,
    JOG_STEP_M,
    POKE_EXTRA_M,
    LEFT_HANG_DEG,
    RIGHT_HANG_DEG,
    LEFT_REST_DEG,
    LEFT_SIDE_VIEW_DEG,
    READY_DEG,
    TORSO_CLEAR_ROLL_DEG,
    REST_DEG,
    SWEEP_HIGH_ELBOW_DEG,
    SWEEP_HIGH_SHOULDER_DEG,
    SWEEP_LOW_ELBOW_DEG,
    SWEEP_LOW_SHOULDER_DEG,
    WRIST_VIEW_DEG,
    camera_delta_to_torso,
    direct_pbvs_motion_goal,
    fuse_pbvs_buttons,
    relax_x_clearance,
    sequence_clear_goal,
    sequence_lift_goal,
    chest_panel_seen,
    clip_torso_delta,
    ibvs_features_ready,
    LEFT_SEEK_SPAN_DEG,
    next_left_gripper_seek,
    next_sweep_joints_deg,
    panel_seen,
    pbvs_coarse_delta,
    pbvs_features_ready,
    raise_direction,
    scan_lock_ready,
    should_jog_ibvs,
    target_features_ready,
    wrist_panel_seen,
)
from ibvs_pad_touch.web.dashboard import RealRobotDashboard
from ibvs_pad_touch.web import dds_bind


def test_direct_pbvs_goes_to_the_button_point():
    button = (0.60, -0.12, 0.44)
    assert direct_pbvs_motion_goal((0.20, -0.02, 0.20), button) == button
    assert direct_pbvs_motion_goal((0.58, -0.10, 0.44), button) == button
    assert direct_pbvs_motion_goal((0.40, -0.12, 0.44), button) == button
    assert abs(relax_x_clearance(0.08, 0.04, 0.04) - 0.06) < 1e-9


def test_quintic_duration_follows_the_motion_control_joint_limits():
    assert quintic_joint_seconds([0.0] * 7, [0.01] * 7) == 2.0
    target = [0.0] * 7
    target[0] = math.radians(130.0)
    assert quintic_joint_seconds([0.0] * 7, target) > 8.0
    cartesian = quintic_joint_seconds([0.0] * 7, target, feedrate=1.0)
    assert 2.0 <= cartesian < 4.0


def test_slerp_rpy_holds_the_endpoints():
    start = (0.2, -0.4, 0.1)
    goal = tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
    assert slerp_rpy(start, goal, 0.0) == start
    end = slerp_rpy(start, goal, 1.0)
    assert all(abs(left - right) < 1e-9 for left, right in zip(end, goal))
    same = slerp_rpy(goal, goal, 0.4)
    assert all(abs(left - right) < 1e-9 for left, right in zip(same, goal))


def test_direct_pbvs_streams_cartesian_samples_with_joint_velocity(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    written = []

    class _Controller:
        def __init__(self):
            self.command = [0.2] * 7
            self.measured = [0.1] * 7
            self.kp = 80.0
            self.kd = 1.5

        def commanded_arm_rad(self, arm):
            return list(self.command)

        def measured_arm_rad(self, arm):
            return list(self.measured)

        def hold_positions(self):
            return {joint: 0.0 for joint in range(22, 29)}

        def write(self, command, weight=1.0, dq=None):
            written.append((dict(command), dict(dq or {})))
            self.command = [command[joint] for joint in RIGHT_ARM_JOINTS]
            self.measured = list(self.command)

        def apply_arm_rad(self, arm, q, seconds, dt):
            raise AssertionError("cartesian playback must stream samples")

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] >= 0.5:
                return [0.23, -0.11, 0.32]
            return [0.20, -0.10, 0.30]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] = 0.8
            return q, True, 0.001, 0.01

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.23, -0.11, 0.32]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(written) > 10
    moving = [item[1] for item in written if any(abs(value) > 1e-6 for value in item[1].values())]
    assert moving
    assert all(abs(value) < 1e-9 for value in written[-1][1].values())
    assert "实测到位" in mover.message


def test_camera_right_is_torso_minus_y():
    # camera +X right, +Y down, +Z forward
    assert camera_delta_to_torso(0.01, 0.0, 0.0) == (0.0, -0.01, 0.0)
    assert camera_delta_to_torso(0.0, 0.02, 0.0) == (0.0, 0.0, -0.02)
    assert camera_delta_to_torso(0.0, 0.0, 0.03) == (0.03, 0.0, 0.0)


def test_height_seek_uses_chest_panel():
    assert chest_panel_seen({"panel_visible": True})
    assert panel_seen({"panel_partial": True})
    assert not wrist_panel_seen({"wrist_panel_partial": True})
    assert not panel_seen({"wrist_panel_visible": True})


def test_ibvs_jog_uses_chest_visual_tip_not_fk():
    ready = {
        "panel_visible": True,
        "target_uv": [320.0, 240.0],
        "virtual_tip_xyz_m": [0.02, 0.01, 0.55],
        "virtual_tip_uv": [410.0, 180.0],
        "tip_source": "chest_visual",
        "tip_uv": [400.0, 200.0],
        "tip_visible": False,
    }
    fk_only = {
        "panel_visible": True,
        "target_uv": [320.0, 240.0],
        "virtual_tip_xyz_m": [0.02, 0.01, 0.55],
        "tip_source": "virtual_fk",
    }
    assert target_features_ready(ready)
    assert ibvs_features_ready(ready)
    assert not ibvs_features_ready(fk_only)
    assert not should_jog_ibvs(ready, height_locked=False)
    assert not should_jog_ibvs(fk_only, height_locked=True)
    assert should_jog_ibvs(ready, height_locked=True)
    assert not should_jog_ibvs({**ready, "side_contact": True}, height_locked=True)
    clipped = clip_torso_delta((0.05, -0.08, 0.01))
    assert max(abs(v) for v in clipped) <= JOG_STEP_M + 1e-9


def test_camera_delta_uses_calibrated_eye_to_hand_rotation():
    rotation = [
        [0.0440311, 0.0188880, 0.9988516],
        [-0.9964719, 0.0723367, 0.0425583],
        [-0.0714498, -0.9972014, 0.0220064],
    ]
    torso = camera_delta_to_torso(0.0, 0.0, 0.01, rotation)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(torso, (0.009988516, 0.000425583, 0.000220064))
    )


def test_tip_above_the_buttons_lowers_instead_of_raising():
    above = {
        "tip_source": "chest_visual",
        "target_uv": [428.0, 251.0],
        "tip_uv": [205.0, 50.0],
    }
    assert raise_direction(above) == 1
    assert not scan_lock_ready(above)
    below = {
        "tip_source": "chest_visual",
        "target_uv": [428.0, 251.0],
        "tip_uv": [400.0, 400.0],
    }
    assert raise_direction(below) == -1
    assert raise_direction({"panel_visible": True}) == -1
    assert -90.0 < READY_DEG[0] < -70.0


def test_scan_lock_waits_for_the_real_tip_not_the_panel():
    assert not scan_lock_ready({"panel_visible": True, "tip_visible": False, "tip_source": "virtual_fk"})
    assert not scan_lock_ready(
        {
            "panel_visible": True,
            "tip_source": "chest_visual",
            "target_uv": [400.0, 200.0],
            "tip_uv": [410.0, 360.0],
        }
    )
    assert scan_lock_ready(
        {
            "panel_visible": True,
            "tip_source": "chest_visual",
            "target_uv": [400.0, 250.0],
            "tip_uv": [430.0, 280.0],
        }
    )
    assert not scan_lock_ready(
        {
            "wrist_panel_visible": True,
            "wrist_panel_partial": True,
            "tip_visible": True,
        }
    )


def test_pbvs_coarse_aims_at_the_button():
    delta, done = pbvs_coarse_delta((0.01, -0.02, 0.45))
    assert not done
    assert delta == (0.01, -0.02, 0.45)
    delta, done = pbvs_coarse_delta((0.01, -0.01, 0.005))
    assert done
    assert abs(delta[2]) <= 0.01


def test_seek_sweeps_up_then_comes_back_down():
    planned = next_sweep_joints_deg(list(WRIST_VIEW_DEG), -1)
    assert planned is not None
    up, direction, turned = planned
    assert up[0] < WRIST_VIEW_DEG[0]
    assert up[3] <= WRIST_VIEW_DEG[3]
    assert SWEEP_HIGH_ELBOW_DEG <= up[3] <= SWEEP_LOW_ELBOW_DEG
    assert up[4] == 0.0
    assert direction == -1
    assert not turned

    at_high = [SWEEP_HIGH_SHOULDER_DEG, -25, 0, SWEEP_HIGH_ELBOW_DEG, 0, 0, 0]
    planned = next_sweep_joints_deg(at_high, -1)
    assert planned is not None
    high, direction, turned = planned
    assert high[0] == SWEEP_HIGH_SHOULDER_DEG
    assert direction == 1
    assert turned

    down, direction, turned = next_sweep_joints_deg(high, direction)
    assert down[0] > high[0]
    assert direction == 1
    assert not turned
    assert down[0] <= SWEEP_LOW_SHOULDER_DEG


def test_seek_clamps_command_above_command_limit():
    over_head = [-120.0, -25, 0, 42, 0, 0, 0]
    planned = next_sweep_joints_deg(over_head, -1)
    assert planned is not None
    recovered, direction, turned = planned
    assert recovered[0] == SWEEP_HIGH_SHOULDER_DEG
    assert direction == 1
    assert turned


def test_sweep_reverses_from_measured_high_limit():
    targets = []

    class _Controller:
        def measured_arm_deg(self, arm):
            return [-89.0, -18.0, 0.0, 50.0, 0.0, 0.0, 0.0]

        def commanded_arm_rad(self, arm):
            return [math.radians(v) for v in (-100.0, -25.0, 0.0, 44.0, 0.0, 0.0, 0.0)]

        def ramp_arm(self, arm, target, seconds, dt):
            targets.append(list(target))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._sweep_direction = -1
    assert mover._sweep_a_bit()
    assert mover._sweep_direction == 1
    assert targets[0][0] > -100.0


def test_rest_pose_is_lower_than_ready():
    assert REST_DEG[0] > READY_DEG[0]
    assert REST_DEG[3] < READY_DEG[3]


def test_wrist_stays_at_original_roll_after_raise():
    assert READY_DEG[1] <= -35.0
    assert TORSO_CLEAR_ROLL_DEG < READY_DEG[1]
    assert READY_DEG[4] == 0.0
    assert WRIST_VIEW_DEG == READY_DEG
    assert WRIST_VIEW_DEG[1] == READY_DEG[1]
    assert WRIST_VIEW_DEG[4] == 0.0
    assert WRIST_VIEW_DEG[5:] == READY_DEG[5:]


def test_left_side_view_is_horizontal_look_right():
    assert LEFT_SIDE_VIEW_DEG[0] < 0.0
    assert LEFT_SIDE_VIEW_DEG[1] > 0.0
    assert LEFT_SIDE_VIEW_DEG[6] < 0.0
    assert LEFT_REST_DEG[1] > 0.0


def test_raise_keeps_both_elbows_bent():
    assert 55.0 <= READY_DEG[3] <= 90.0
    assert 55.0 <= LEFT_SIDE_VIEW_DEG[3] <= 90.0


def test_raise_moves_left_side_view_then_right_ready():
    calls = []

    class _Controller:
        def ramp_arm(self, arm, target, seconds, dt):
            calls.append((arm, list(target)))

        def commanded_arm_rad(self, arm):
            return [0.0] * 7

        def measured_arm_deg(self, arm):
            return [0.0] * 7

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._scanning.set()
    mover._scan_needs_prepare = True
    mover._prepare_scan_pose()
    assert calls[0] == ("left", list(LEFT_SIDE_VIEW_DEG))
    right = [item[1] for item in calls if item[0] == "right"]
    assert right[0][1] == TORSO_CLEAR_ROLL_DEG
    assert right[0][0] == 0.0
    assert right[0][3] == 0.0
    assert right[1] == list(READY_DEG)
    assert mover._scanning.is_set() is False
    assert mover._height_locked is True
    assert mover.phase == "hold_for_start"
    assert mover.status()["left_side_ready"] is True


def test_release_for_operator_drops_sdk_and_keeps_idle_preview():
    released = []

    class _Controller:
        def release(self, seconds):
            released.append(seconds)

    class _Thread:
        def is_alive(self):
            return False

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = object()
    mover._thread = _Thread()
    mover._ibvs_enabled = True
    mover._left_side_ready = True
    mover._running.set()
    mover._scanning.set()
    msg = mover.release_for_operator()
    assert "释放" in msg
    assert "运控" in msg
    assert released == [1.2]
    assert mover.controller is None
    assert mover.ik is not None
    status = mover.status()
    assert status["sdk_active"] is False
    assert status["ibvs_enabled"] is False
    assert status["scanning"] is False
    assert status["phase"] == "released"
    assert status["left_side_ready"] is False
    assert mover.start() == "请先点 Raise/Scan 举臂"


def test_release_when_already_idle_is_safe():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    msg = mover.release_for_operator()
    assert "释放" in msg
    assert mover.controller is None
    assert mover.status()["phase"] == "released"


def test_read_only_joint_feedback_refreshes_right_ee_after_release():
    class _IK:
        def fk_xyz_rpy(self, q):
            return [0.30, -0.10, 0.20], [0.1, 0.2, 0.3]

        def tip_xyz(self, q, offset):
            assert offset == (0.205, 0.0, 0.0)
            return [0.50, -0.08, 0.38]

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.ik = _IK()
    mover.phase = "released"
    mover.update_joint_readback([0.1] * 7)
    status = mover.status()
    assert status["right_ee_xyz_m"] == [0.5, -0.08, 0.38]
    assert status["ee_xyz_m"] == [0.3, -0.1, 0.2]


def test_saved_joint_trajectory_replays_faster_and_holds_last_pose():
    writes = []

    class _Controller:
        def hold_positions(self):
            return {index: 0.0 for index in range(12, 29)}

        def write(self, command, weight=1.0):
            writes.append(dict(command))

    samples = [
        {
            "t": 10.0,
            "joints": [
                {"name": "right_shoulder_pitch_joint", "q": -0.4},
                {"name": "right_elbow_joint", "q": 0.5},
            ],
        },
        {
            "t": 10.08,
            "joints": [
                {"name": "right_shoulder_pitch_joint", "q": -0.6},
                {"name": "right_elbow_joint", "q": 0.7},
            ],
        },
    ]
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._thread = _AliveThread()
    result = mover.start_playback("take_1", samples, speed=4.0)
    assert result["ok"] is True and result["speed"] == 4.0
    mover._playback_thread.join(timeout=2.0)
    assert writes
    assert math.isclose(writes[-1][22], -0.6)
    assert math.isclose(writes[-1][25], 0.7)
    assert mover.phase == "playback_complete"


def test_shutdown_without_controller_is_safe():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.shutdown()
    assert mover.controller is None


def test_start_without_confirm_does_not_enable_arm():
    mover = ArmMover(telemetry=lambda: {}, confirm=False, log=lambda _m: None)
    msg = mover.start()
    assert "confirm-arm-motion" in msg
    assert mover.status()["enabled"] is False
    assert mover.controller is None


def test_start_requires_raise_scan_first():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    msg = mover.start()
    assert "Raise/Scan" in msg
    assert mover.status()["ibvs_enabled"] is False
    assert mover.controller is None


class _AliveThread:
    def is_alive(self):
        return True


def test_start_uses_dual_view_and_does_not_blind_poke():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = object()
    mover._thread = _AliveThread()
    mover._running.set()
    mover._scanning.set()
    msg = mover.start()
    status = mover.status()
    assert status["scanning"] is False
    assert status["ibvs_enabled"] is True
    assert status["control_stage"] == "DUAL_IBVS"
    assert mover._poke_remaining_m == 0.0
    assert "胸视" in msg
    assert "侧视" in msg


def test_start_enables_pbvs_with_virtual_tip_and_no_visual_tip():
    ready = {
        "wrist_panel_visible": True,
        "panel_visible": True,
        "target_uv": [320.0, 240.0],
        "tip_visible": False,
        "tip_tracker": {"mode": "auto_stable", "confirmed": False},
        "depth_available": True,
        "target_xyz_m": [0.1, 0.0, 0.7],
        "virtual_tip_xyz_m": [0.0, 0.04, 0.50],
        "pbvs_error_cam_m": [0.1, -0.04, 0.20],
    }
    mover = ArmMover(telemetry=lambda: ready, confirm=True, log=lambda _m: None)
    mover.controller = object()
    mover._thread = _AliveThread()
    mover._running.set()
    mover._scanning.set()
    mover.start()
    status = mover.status()
    assert status["scanning"] is False
    assert status["ibvs_enabled"] is True
    assert status["control_stage"] == "DUAL_IBVS"


def test_start_enters_calibrated_pbvs_before_dual_ibvs():
    ready = {"target_torso_m": [0.50, -0.10, 0.35]}
    mover = ArmMover(
        telemetry=lambda: ready,
        confirm=True,
        pbvs_first=True,
        log=lambda _m: None,
    )
    mover.controller = object()
    mover._thread = _AliveThread()
    mover._running.set()
    mover.start()
    assert mover.status()["control_stage"] == "PBVS_COARSE"
    assert "PBVS" in mover.message


def test_web_mode_buttons_select_direct_pbvs_or_hybrid_for_the_next_start():
    ready = {"target_torso_m": [0.50, -0.10, 0.35]}
    mover = ArmMover(telemetry=lambda: ready, confirm=True, log=lambda _m: None)
    mover.set_servo_mode("direct_pbvs")
    assert mover.servo_mode() == "direct_pbvs"
    mover.controller = object()
    mover._thread = _AliveThread()
    mover.start()
    assert mover.status()["control_stage"] == "DIRECT_PBVS"
    mover._ibvs_enabled = False
    mover._running.clear()
    mover.set_servo_mode("hybrid")
    assert mover.servo_mode() == "hybrid"
    mover.start()
    assert mover.status()["control_stage"] == "PBVS_COARSE"
    mover._running.set()
    kept = mover.set_servo_mode("direct_pbvs")
    assert "Pause" in kept


def test_motion_stays_idle_until_a_backend_is_clicked():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None, motion_backend="")
    mover._service_node_probe = lambda: False
    assert mover.status()["motion_backend"] == ""
    refused = mover.start()
    assert "请先点" in refused
    assert mover.controller is None
    failed = []
    mover._start_arm_sdk = lambda: failed.append("sdk") or False
    missed = mover.set_motion_backend("arm_sdk")
    assert failed == ["sdk"]
    assert "没有起来" in missed
    assert mover.status()["motion_backend"] == ""
    mover._start_arm_sdk = lambda: True
    picked = mover.set_motion_backend("arm_sdk")
    assert "拉起 arm_sdk" in picked
    assert mover.status()["motion_backend"] == "arm_sdk"


def test_motion_backend_switches_from_idle_and_refuses_a_running_move():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: True
    accepted = mover.set_motion_backend("service")
    assert "拒绝" not in accepted
    assert mover.status()["motion_backend"] == "service"
    same = mover.set_motion_backend("service")
    assert "已是运控服务" in same
    mover._ibvs_enabled = True
    blocked = mover.set_motion_backend("arm_sdk")
    assert "拒绝" in blocked
    assert mover.status()["motion_backend"] == "service"


def test_arm_sdk_click_stops_the_service_node_even_when_already_selected():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: True

    def _stop() -> bool:
        mover._service_node_probe = lambda: False
        mover._service_node_cached = None
        return True

    mover._stop_service_node = _stop
    started = []
    mover._start_arm_sdk = lambda: started.append(True) or True
    message = mover.set_motion_backend("arm_sdk")
    assert started == [True]
    assert "已停止运控服务" in message
    assert "拉起 arm_sdk" in message
    assert "拒绝" not in message
    assert mover.status()["motion_backend"] == "arm_sdk"


def test_service_click_drops_an_sdk_hold_without_a_weight_ramp():
    class _Controller:
        def __init__(self):
            self.released = False
            self.stopped = False

        def release(self, seconds):
            self.released = True

        def stop_servo(self):
            self.stopped = True

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: True
    controller = _Controller()
    mover.controller = controller
    mover.phase = "hold_for_start"
    message = mover.set_motion_backend("service")
    assert "拒绝" not in message
    assert mover.status()["motion_backend"] == "service"
    assert controller.stopped is True
    assert controller.released is False


def test_service_hang_drops_both_forearms_beside_the_body():
    class _Controller:
        def __init__(self):
            self.ramps = []
            self.released = False

        def move_cartesian(self, *args, **kwargs):
            return True, "ok"

        def ramp_arm(self, arm, degrees, seconds, dt):
            self.ramps.append((arm, tuple(degrees), seconds))

        def measured_arm_deg(self, arm):
            return [-60.0, -10.0, 0.0, 40.0, 0.0, 0.0, 0.0]

        def release(self, seconds):
            self.released = True

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: True
    mover._motion_backend = "service"
    controller = _Controller()
    mover.controller = controller
    mover._thread = _AliveThread()
    message = mover.hang_arms()
    assert "下垂" in message
    assert "arm_sdk" in message
    assert mover._hang_down.is_set()
    mover._execute_hang()
    assert controller.ramps[0] == ("left", LEFT_HANG_DEG, 8.0)
    right = [item for item in controller.ramps if item[0] == "right"]
    assert right == [("right", RIGHT_HANG_DEG, 8.0)]
    assert LEFT_HANG_DEG[3] == 90.0
    assert RIGHT_HANG_DEG[3] == 90.0
    assert LEFT_HANG_DEG[1] > 0.0
    assert RIGHT_HANG_DEG[1] < 0.0
    assert controller.released is False
    assert mover.controller is controller
    assert "自然下垂" in mover.message
    assert mover._can_switch_backend()


def test_right_level_does_not_move_the_arm():
    class _Controller:
        def __init__(self):
            self.ramps = []
            self.released = False

        def ramp_arm(self, arm, degrees, seconds, dt):
            self.ramps.append((arm, tuple(degrees), seconds))

        def release(self, seconds):
            self.released = True

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: False
    controller = _Controller()
    mover.controller = controller
    mover._thread = _AliveThread()
    message = mover.level_right_arm()
    assert "胸口" in message
    assert "不动" in message
    assert not mover._level_right.is_set()
    mover._execute_level_right()
    mover._ramp_right_level()
    assert controller.ramps == []
    assert controller.released is False


def test_right_level_refuses_without_confirm():
    mover = ArmMover(telemetry=lambda: {}, confirm=False, log=lambda _m: None)
    mover._service_node_probe = lambda: False
    message = mover.level_right_arm()
    assert "胸口" in message
    assert not mover._level_right.is_set()


def test_emergency_stop_releases_sdk_and_service_without_a_pose():
    class _Sdk:
        def __init__(self):
            self.ramps = []
            self._aborted = False

        def abort(self):
            self._aborted = True

        def ramp_arm(self, arm, degrees, seconds, dt):
            self.ramps.append((arm, tuple(degrees)))

    killed = []
    dropped = []
    mover = ArmMover(telemetry=lambda: {}, confirm=False, log=lambda _m: None)
    controller = _Sdk()
    mover.controller = controller
    mover._running.set()
    mover._ibvs_enabled = True
    mover._stop_service_node = lambda: killed.append("node") or True
    mover._drop_arm_sdk = lambda iface="eth0": dropped.append(iface)
    message = mover.emergency_stop()
    assert "急停" in message
    assert "释放" in message
    assert controller._aborted is True
    assert controller.ramps == []
    assert killed == ["node"]
    assert dropped == []
    assert mover.controller is None
    assert mover._running.is_set() is False
    assert mover.phase == "estop"

    class _Service:
        def __init__(self):
            self.ramps = []

        def abort(self):
            self.ramps.append("abort")

    mover = ArmMover(telemetry=lambda: {}, confirm=False, log=lambda _m: None)
    service = _Service()
    mover.controller = service
    mover.iface = "eth0"
    mover._motion_backend = "service"
    mover._stop_service_node = lambda: killed.append("node2") or True
    mover._drop_arm_sdk = lambda iface="eth0": dropped.append(iface)
    mover.emergency_stop()
    assert service.ramps == ["abort"]
    assert dropped == ["eth0"]
    assert "node2" in killed


def test_hang_refuses_on_the_arm_sdk_link():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: False
    message = mover.hang_arms()
    assert "已拒绝" in message
    assert not mover._hang_down.is_set()


def test_arm_sdk_motion_refuses_while_the_service_node_is_up():
    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, -0.10, 0.40]},
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover._service_node_probe = lambda: True
    reset = mover.return_to_ready()
    assert "冲突" in reset
    assert "已拒绝" in reset
    assert not mover._return_ready.is_set()
    started = mover.start()
    assert "冲突" in started
    mover.release_for_operator()
    mover._motion_backend = "service"
    mover._service_node_cached = None
    mover._service_node_probe = lambda: True

    def _stop_fails() -> bool:
        return False

    mover._stop_service_node = _stop_fails
    blocked = mover.set_motion_backend("arm_sdk")
    assert "冲突" in blocked
    assert "拒绝" in blocked
    assert mover.status()["motion_backend"] == "service"

    def _stop_clears() -> bool:
        mover._service_node_probe = lambda: False
        mover._service_node_cached = None
        return True

    mover._stop_service_node = _stop_clears
    mover._service_node_cached = None
    mover._start_arm_sdk = lambda: True
    switched = mover.set_motion_backend("arm_sdk")
    assert "已停止运控服务" in switched
    assert "拒绝" not in switched
    assert mover.status()["motion_backend"] == "arm_sdk"


def test_service_call_refuses_while_this_page_holds_arm_sdk():
    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, -0.10, 0.40]},
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover._service_node_probe = lambda: False
    mover._motion_backend = "service"
    mover.controller = object()
    mover._thread = _AliveThread()
    message = mover.start()
    assert "冲突" in message
    assert "已拒绝" in message
    assert mover.status()["control_stage"] != "DIRECT_PBVS"


def test_reset_returns_to_the_ready_pose_without_releasing_sdk():
    class _Controller:
        def __init__(self):
            self.ramps = []
            self.released = False

        def ramp_arm(self, arm, degrees, seconds, dt):
            self.ramps.append((arm, tuple(degrees)))

        def release(self, seconds):
            self.released = True

        def commanded_arm_rad(self, arm):
            return [0.0] * 7

        def measured_arm_deg(self, arm):
            return [0.0] * 7

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover._service_node_probe = lambda: False
    controller = _Controller()
    mover.controller = controller
    mover._thread = _AliveThread()
    mover._ibvs_enabled = True
    mover._running.set()
    message = mover.return_to_ready()
    assert "arm_sdk" in message
    assert mover._return_ready.is_set()
    assert mover.controller is controller
    assert controller.released is False
    mover._execute_return_ready()
    assert controller.ramps[0] == ("left", LEFT_REST_DEG)
    right = [pose for arm, pose in controller.ramps if arm == "right"]
    assert right == [REST_DEG]
    assert mover.status()["left_side_ready"] is False
    assert controller.released is False
    assert mover.controller is controller
    assert mover.phase == "hold_for_start"
    assert mover.status()["sdk_active"] is True
    assert not mover._return_ready.is_set()


def test_direct_pbvs_starts_without_a_prior_raise(monkeypatch):
    import ibvs_pad_touch.web.arm_motion as arm_motion_module

    holder = {}

    class _Thread:
        def __init__(self, *args, **kwargs):
            self._alive = False

        def start(self):
            self._alive = True
            holder["mover"].controller = object()

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(arm_motion_module.threading, "Thread", _Thread)
    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, -0.10, 0.40]},
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    holder["mover"] = mover
    msg = mover.start()
    assert "Raise" not in msg
    assert mover.status()["control_stage"] == "DIRECT_PBVS"


def test_direct_pbvs_requires_an_rgbd_button_target():
    mover = ArmMover(
        telemetry=lambda: {
            "target_torso_m": None,
        },
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = object()
    mover._thread = _AliveThread()
    msg = mover.start()
    assert "按键三维坐标" in msg
    assert mover.status()["ibvs_enabled"] is False


def test_direct_pbvs_applies_one_loaded_pose_move_then_holds():
    targets = []
    sent = []
    telem = {
        "target_torso_m": [0.23, -0.11, 0.32],
    }

    class _Controller:
        def __init__(self):
            self.command = [0.2] * 7
            self.measured = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.command)

        def measured_arm_rad(self, arm):
            return list(self.measured)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append((list(q), seconds, dt))
            self.command = list(q)
            self.measured = list(q)

    class _IK: 
        def tip_xyz(self, measured):
            if measured[0] >= 0.2:
                return [0.23, -0.11, 0.32]
            return [0.20, -0.10, 0.30]

        def apply_loaded_tool_pose_target(
            self,
            commanded,
            measured,
            target_tool_xyz_m,
            target_tool_rpy_rad,
            tool_offset_wrist_m=(0.205, 0.0, 0.0),
        ):
            targets.append(
                (
                    tuple(target_tool_xyz_m),
                    tuple(target_tool_rpy_rad),
                    tuple(tool_offset_wrist_m),
                )
            )
            return [0.21] * 7, True, 0.001, 0.01

        def fk_xyz_rpy(self, q):
            return [0.4, -0.2, 0.3], [0.0, 0.0, 0.0]

    mover = ArmMover(
        telemetry=lambda: telem,
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=0.10,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover.start()
    assert mover.status()["control_stage"] == "DIRECT_PBVS"
    mover._servo_once()
    xyzs = [item[0] for item in targets]
    button = (0.23, -0.11, 0.32)
    assert abs(xyzs[-1][0] - button[0]) < 0.005
    assert abs(xyzs[-1][1] - button[1]) < 0.005
    assert abs(xyzs[-1][2] - button[2]) < 0.005
    assert all(item[0] + 1e-9 >= 0.20 for item in xyzs)
    assert len(targets) < 40
    assert sent[0][0] == [0.21] * 7
    assert sent and sent[0][1] >= 2.0
    assert mover.status()["control_stage"] == "HOLD"
    assert mover.status()["ibvs_enabled"] is False
    assert "实测到位" in mover.message
    mover._servo_once()
    assert len(sent) == 1


def test_direct_pbvs_keeps_going_when_the_live_pose_is_still_short():
    telem = {"target": "button_1", "target_torso_m": [0.6186, 0.0166, 0.4313]}
    shots = []

    class _Controller:
        def __init__(self):
            self.command = [0.2] * 7
            self.measured = [0.2] * 7

        def commanded_arm_rad(self, arm):
            return list(self.command)

        def measured_arm_rad(self, arm):
            return list(self.measured)

        def apply_arm_rad(self, arm, q, seconds, dt):
            self.command = list(q)
            self.measured = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if shots and shots[-1][0] > 0.60:
                return [0.6186, 0.0166, 0.4313]
            return [0.5805, 0.0033, 0.4564]

        def apply_loaded_tool_pose_target(self, commanded, measured, target_tool_xyz_m, **kwargs):
            shots.append(tuple(float(value) for value in target_tool_xyz_m))
            return [0.4] * 7, True, 0.001, 0.01

    mover = ArmMover(
        telemetry=lambda: telem,
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._locked_buttons = {"button_1": (0.5819, 0.0023, 0.4574)}
    mover._direct_pbvs_once(telem)
    assert shots
    assert abs(shots[-1][0] - 0.6186) < 0.002
    assert abs(shots[-1][1] - 0.0166) < 0.002
    assert abs(shots[-1][2] - 0.4313) < 0.002
    assert shots[-1][0] > 0.61
    assert "X " in mover.message
    assert "实测到位" in mover.message


def test_direct_pbvs_accepts_a_hang_lift_with_twenty_degree_orientation_error():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] > 1.0:
                return [0.23, -0.11, 0.32]
            return [0.05, -0.20, -0.30]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] += math.radians(128.0)
            return q, False, 0.0021, math.radians(19.0)

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.23, -0.11, 0.32]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        direct_pbvs_max_joint_deg=120.0,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert sent
    assert "拒绝" not in mover.message


def test_direct_pbvs_sends_a_low_start_lift_at_thirty_nine_millimetres_and_151_degrees():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] > 1.0:
                return [0.50, 0.053, 0.47]
            return [0.05, -0.10, -0.30]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] += math.radians(151.0)
            return q, False, 0.0389, math.radians(24.6)

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, 0.053, 0.47]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        direct_pbvs_max_joint_deg=120.0,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert sent
    assert mover.phase != "direct_pbvs_rejected"
    assert "IK 未通过" not in mover.message


def test_direct_pbvs_still_rejects_an_eighty_millimetre_ik_miss():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))

    class _IK:
        def tip_xyz(self, measured):
            return [0.05, -0.10, -0.30]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] += math.radians(20.0)
            return q, False, 0.080, math.radians(5.0)

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, 0.053, 0.47]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        direct_pbvs_max_joint_deg=120.0,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert sent == []
    assert mover.phase == "direct_pbvs_rejected"
    assert "IK 未通过" in mover.message


def test_direct_pbvs_sends_a_pitch_solution_sixteen_millimetres_from_the_button():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] > 1.0:
                return [0.60, -0.02, 0.46]
            return [0.05, -0.02, 0.46]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            nominal = tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
            at_button = tuple(kwargs["target_tool_xyz_m"]) == (0.60, -0.02, 0.46)
            pitch = kwargs["target_tool_rpy_rad"][1] - nominal[1]
            if not at_button or abs(pitch - math.radians(-8.0)) > 1e-6:
                if at_button:
                    return list(commanded), False, 0.08, 1.2
                q = list(commanded)
                q[0] += math.radians(5.0)
                return q, True, 0.001, 0.01
            q = list(commanded)
            q[0] += math.radians(99.0)
            return q, False, 0.016, math.radians(9.0)

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.60, -0.02, 0.46]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        direct_pbvs_max_joint_deg=120.0,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert sent
    assert "IK 未通过" not in mover.message
    assert "超时" not in mover.message


def test_direct_pbvs_still_moves_when_a_midpoint_ik_fails():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append((list(q), seconds))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] > 0.5:
                return [0.55, -0.10, 0.46]
            return [0.05, -0.10, 0.46]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            if tuple(kwargs["target_tool_xyz_m"]) != (0.55, -0.10, 0.46):
                return list(commanded), False, 0.2, 1.2
            q = list(commanded)
            q[0] = 0.8
            return q, True, 0.001, 0.01

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.55, -0.10, 0.46]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert sent
    assert sent[0][0][0] == 0.8
    assert sent[0][1] >= 2.0
    assert "实测到位" in mover.message


def test_direct_pbvs_tries_another_seed_when_the_measured_basin_fails():
    sent = []
    seeds = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7
            self.moved = False

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)
            self.moved = True

    class _IK:
        def tip_xyz(self, measured):
            if controller.moved:
                return [0.55, -0.10, 0.46]
            return [0.05, -0.20, -0.30]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            seeds.append(kwargs.get("seed_q_rad"))
            if kwargs.get("seed_q_rad") is None:
                return [0.1] * 7, False, 0.08, 1.2
            q = list(commanded)
            q[0] = 1.2
            return q, True, 0.002, 0.02

    controller = _Controller()
    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.55, -0.10, 0.46]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = controller
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert seeds[0] is None
    assert seeds[1] is not None
    assert sent
    assert "实测到位" in mover.message
    assert "超时" not in mover.message


def test_pbvs_search_stops_when_the_stage_budget_is_already_spent():
    applied = []

    class _IK:
        def tip_xyz(self, q):
            return [0.0, 0.0, 0.0]

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("search should stop before another solve")

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.10, 0.0, 0.0]},
        confirm=True,
        pbvs_first=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "PBVS_COARSE"
    mover._pbvs_deadline = time.monotonic() - 1.0
    mover._servo_once()
    assert applied == []
    assert mover.phase == "pbvs_timeout"
    assert mover.status()["control_stage"] == "HOLD"
    assert "超时" in mover.message


def test_direct_pbvs_does_not_execute_a_partial_waypoint_when_the_pose_fails():
    requested = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            raise AssertionError("an unsolved pose must not be executed")

    class _IK:
        def tip_xyz(self, measured):
            return [0.05, -0.11, 0.32]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            requested.append(tuple(kwargs["target_tool_xyz_m"]))
            return list(commanded), False, 0.2, 1.2

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.23, -0.11, 0.32]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert requested
    assert set(requested) == {(0.23, -0.11, 0.32)}
    assert mover.phase == "direct_pbvs_rejected"
    assert "实测未到位" not in mover.message


def test_direct_pbvs_approaches_in_x_even_after_the_stage_clock_expires():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] > 0.5:
                return [0.23, -0.11, 0.32]
            return [0.05, -0.11, 0.32]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] = 0.8
            return q, True, 0.001, 0.01

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.23, -0.11, 0.32]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._pbvs_deadline = time.monotonic() - 5.0
    mover._servo_once()
    assert sent
    assert "实测到位" in mover.message
    assert "超时" not in mover.message


def test_direct_pbvs_keeps_moving_until_the_stage_budget_when_the_tip_stays_short():
    sent = []

    class _Controller:
        def __init__(self):
            self.q = [0.1] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)

    class _IK:
        def tip_xyz(self, measured):
            return [0.05, -0.11, 0.32]

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[0] += 0.2
            return q, True, 0.001, 0.01

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.23, -0.11, 0.32]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(sent) == 3
    assert "实测未到位" in mover.message
    assert "X 180" in mover.message
    assert "关节跟踪" in mover.message


def test_direct_pbvs_still_moves_when_only_two_centimetres_of_y_remain():
    sent = []
    tip = [0.6029, -0.0953, -0.4244]

    class _Controller:
        def __init__(self):
            self.q = [0.2] * 7

        def commanded_arm_rad(self, arm):
            return list(self.q)

        def measured_arm_rad(self, arm):
            return list(self.q)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.q = list(q)
            tip[:] = [0.6029, -0.1150, -0.4244]

    class _IK:
        def tip_xyz(self, measured):
            return list(tip)

        def apply_loaded_tool_pose_target(self, commanded, measured, **kwargs):
            q = list(commanded)
            q[1] += 0.05
            return q, True, 0.001, 0.01

        def fk_xyz_rpy(self, q):
            return list(tip), [0.0, 0.0, 0.0]

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.6029, -0.1150, -0.4244]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(sent) == 1
    assert "实测到位" in mover.message
    assert "Y " in mover.message


def test_service_direct_pbvs_keeps_correcting_a_two_centimetre_y_miss():
    calls = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.2] * 7

        def measured_arm_rad(self, arm):
            return [0.2] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            calls.append((arm, tuple(goal), seconds))
            return True, "ok"

    class _IK:
        def tip_xyz(self, measured):
            return [0.6029, -0.0953, -0.4244]

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.6029, -0.1150, -0.4244]},
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover._motion_backend = "service"
    mover._service_node_probe = lambda: False
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(calls) == 2
    assert calls[0][1] == (0.6029, -0.1150, -0.4244)
    assert calls[1][1] == (0.6029, -0.1150, -0.4244)
    assert "两段" in mover.message
    assert "Y -20" in mover.message


def test_direct_pbvs_commands_the_button_when_the_tip_starts_low():
    sent = []
    sent_targets = []
    telem = {"target_torso_m": [0.55, -0.10, 0.46]}

    class _Controller:
        def __init__(self):
            self.command = [0.2] * 7
            self.measured = [0.05] * 7

        def commanded_arm_rad(self, arm):
            return list(self.command)

        def measured_arm_rad(self, arm):
            return list(self.measured)

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))
            self.command = list(q)
            self.measured = [value + 0.05 for value in q]

    class _IK:
        def tip_xyz(self, measured):
            if measured[0] >= 0.4:
                return [0.55, -0.10, 0.46]
            return [0.05, -0.20, -0.30]

        def apply_loaded_tool_pose_target(
            self,
            commanded,
            measured,
            target_tool_xyz_m,
            target_tool_rpy_rad,
            tool_offset_wrist_m=(0.205, 0.0, 0.0),
        ):
            sent_targets.append(tuple(target_tool_xyz_m))
            lifted = list(commanded)
            lifted[0] = 0.8
            return lifted, True, 0.001, 0.0

    mover = ArmMover(
        telemetry=lambda: telem,
        confirm=True,
        direct_pbvs_poke=True,
        direct_pbvs_max_m=1.20,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover._height_locked = True
    mover._seek_done = True
    mover._ibvs_enabled = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(sent) == 1
    last = sent_targets[-1]
    assert abs(last[0] - 0.55) < 0.01
    assert abs(last[1] + 0.10) < 0.01
    assert abs(last[2] - 0.46) < 0.01
    assert last[0] > 0.10
    assert "实测到位" in mover.message


def test_direct_ee_uses_motion_control_tool_frame_and_given_rpy():
    calls = []
    align_targets = []
    sent = []
    actual_tip = [0.40, -0.10, 0.30]
    pending_tip = list(actual_tip)
    actual_rpy = [0.0, 0.0, 0.0]
    pending_rpy = list(actual_rpy)

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.2] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append((list(q), seconds, dt))
            actual_tip[:] = pending_tip
            actual_rpy[:] = pending_rpy

    class _IK:
        def tip_xyz(self, q, offset):
            assert offset == (0.205, 0.0, 0.0)
            return list(actual_tip)

        def apply_loaded_tip_target(
            self,
            commanded,
            measured,
            target_tip_xyz_m,
            offset_m,
        ):
            align_targets.append((tuple(target_tip_xyz_m), tuple(offset_m)))
            pending_tip[:] = target_tip_xyz_m
            return [0.21] * 7, True, 0.001, 0.0

        def apply_loaded_tool_pose_target(
            self,
            commanded,
            measured,
            target_tool_xyz_m,
            target_tool_rpy_rad,
            tool_offset_wrist_m,
        ):
            pending_tip[:] = target_tool_xyz_m
            pending_rpy[:] = target_tool_rpy_rad
            calls.append(
                (
                    tuple(target_tool_xyz_m),
                    tuple(math.degrees(value) for value in target_tool_rpy_rad),
                    tuple(tool_offset_wrist_m),
                )
            )
            q = [0.21] * 7
            q[4] = 0.2 + math.radians(102.0)
            return q, True, 0.001, math.radians(0.2)

        def fk_xyz_rpy(self, q):
            return [0.4, -0.1, 0.3], list(actual_rpy)

    mover = ArmMover(
        # First segment is 0.75m from the mocked hanging tip. The old 0.70m
        # Direct EE gate must not reject the staged Y/Z -> X workflow.
        telemetry=lambda: {"target_torso_m": [1.30, -0.10, 0.30]},
        confirm=True,
        direct_ee_poke=True,
        direct_ee_poke_depth_m=0.005,
        camera_R_torso=[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover.start()
    assert mover.status()["control_stage"] == "DIRECT_EE_ALIGN_YZ"
    assert mover.status()["right_ee_xyz_m"] == [0.4, -0.1, 0.3]
    mover._servo_once()
    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(align_targets[0][0], (1.15, -0.10, 0.30))
    )
    assert align_targets[0][1] == (0.205, 0.0, 0.0)
    assert mover.status()["control_stage"] == "DIRECT_EE_ORIENT_SAFE"
    assert sent and sent[0][1] == 5.0
    mover._servo_once()
    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(calls[0][0], (1.15, -0.10, 0.30))
    )
    assert calls[0][2] == (0.205, 0.0, 0.0)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(calls[0][1], (-88.8133, -3.98969, 3.20691))
    )
    assert sent[1][1] == 5.0
    assert mover.status()["control_stage"] == "DIRECT_EE_APPROACH_X"
    mover._servo_once()
    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(calls[1][0], (1.305, -0.10, 0.30))
    )
    assert sent[2][1] == 3.0
    assert mover.phase == "direct_ee_complete"


def test_direct_ee_does_not_claim_completion_when_measured_tip_does_not_follow():
    applied = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.2] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    class _IK:
        def tip_xyz(self, q, offset):
            return [0.0, 0.0, 0.0]

        def apply_loaded_tip_target(self, commanded, measured, target_tip_xyz_m, offset_m):
            return [0.21] * 7, True, 0.001, 0.0

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            return [0.21] * 7, True, 0.001, 0.0

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.50, -0.10, 0.30]},
        confirm=True,
        direct_ee_poke=True,
        camera_R_torso=[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DIRECT_EE_ALIGN_YZ"
    for _ in range(6):
        mover._servo_once()
    assert len(applied) == 6
    assert mover.phase == "direct_ee_rejected"
    assert "未收敛" in mover.message


def test_pbvs_stage_waits_for_a_calibrated_target():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            applied.append(tuple(torso))
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    mover = ArmMover(
        telemetry=lambda: {"pbvs_error_cam_m": [0.1, -0.04, 0.20], "dx": 0.0, "dy": 0.0, "dz": 0.0},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "PBVS_COARSE"
    mover._servo_once()
    assert mover._control_stage == "PBVS_COARSE"
    assert mover.phase == "pbvs_wait_depth"
    assert applied == []


def test_calibrated_pbvs_goes_direct_then_switches_to_right_wrist_approach():
    targets = []
    applied = []
    raised = []
    tip = [0.0, 0.05, -0.05]
    old_axis_map = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]

    class _IK:
        def tip_xyz(self, q):
            if q and abs(float(q[0]) - 0.1) > 1e-6 and targets:
                return list(targets[-1][0])
            return list(tip)

        def fk_matrix(self, q):
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        def apply_loaded_tool_pose_target(
            self,
            commanded,
            measured,
            target_tool_xyz_m,
            target_tool_rpy_rad,
            tool_offset_wrist_m=(0.205, 0.0, 0.0),
        ):
            targets.append(
                (
                    tuple(target_tool_xyz_m),
                    tuple(target_tool_rpy_rad),
                    tuple(tool_offset_wrist_m),
                )
            )
            q = list(commanded)
            q[0] += math.radians(15.8)
            return q, True, 0.001, 0.01

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append((list(q), seconds))

        def ramp_arm(self, arm, target, seconds, dt):
            raised.append((arm, list(target)))

    telem = {
        "target": "button_1",
        "target_torso_m": [0.10, 0.0, 0.0],
        "buttons_torso_m": {"button_1": [0.10, 0.0, 0.0]},
    }
    mover = ArmMover(
        telemetry=lambda: telem,
        confirm=True,
        pbvs_first=True,
        camera_R_torso=old_axis_map,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.wrist_dwell_s = 0.0
    mover.wrist_watch = lambda _enabled: None
    mover.wrist_hold = lambda: None
    mover.wrist_view = lambda: {
        "seen": True,
        "buttons_cam_m": {"button_1": [0.10, 0.0, 0.0]},
        "updated_at": time.time(),
    }
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "PBVS_COARSE"
    mover._servo_once()
    assert targets[0][0] == (coarse_x_command(0.10), 0.0, 0.0)
    assert targets[0][0][0] < coarse_x_ceiling(0.10)
    assert (0.0, 0.05, -0.05) in [item[0] for item in targets]
    assert targets[0][1] == tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
    assert targets[0][2] == (0.205, 0.0, 0.0)
    assert applied
    assert applied[0][1] >= 1.2
    assert "运控位姿一次到位" in mover.message
    assert mover._control_stage == "PBVS_COARSE"
    tip[:] = [coarse_x_command(0.10), 0.0, 0.0]
    mover._servo_once()
    assert mover._control_stage == "WRIST_APPROACH"
    assert mover.phase == "wrist_approach"
    assert mover._pbvs_source == "average"
    assert raised == []


def test_side_contact_stops_instead_of_poking():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(
        telemetry=lambda: {"side_contact": True, "phase": "CONTACT", "dx": 0.0, "dy": 0.0, "dz": 0.01},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    mover._servo_once()
    assert applied == []
    assert mover._ibvs_enabled is False
    assert mover.phase == "contact"


def test_dual_approach_uses_chest_forward_step():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            applied.append(tuple(torso))
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    mover = ArmMover(
        telemetry=lambda: {"phase": "APPROACH", "dx": 0.0, "dy": 0.0, "dz": 0.004, "side_measured": True},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    mover._servo_once()
    assert applied
    assert applied[0][0] > 0.0
    assert mover._ibvs_enabled is True


def test_missing_gripper_holds_the_right_arm_and_steps_the_left():
    applied = []
    left = []

    class _IK:
        def apply_delta(self, q, torso):
            applied.append(tuple(torso))
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def ramp_arm(self, arm, target_deg, seconds, dt):
            left.append((arm, list(target_deg)))

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(("right", arm))

    mover = ArmMover(
        telemetry=lambda: {
            "phase": "SEEK_GRIPPER",
            "dx": 0.0,
            "dy": 0.0,
            "dz": -0.004,
            "tip_uv": None,
            "side_message": "侧视未看到右夹爪",
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    mover._servo_once()
    assert applied == []
    assert left and left[0][0] == "left"
    assert left[0][1][5] > left[0][1][0]
    assert "右臂保持" in mover.message
    mover._servo_once()
    assert left[1][1][5] > left[0][1][5]


def test_missing_gripper_closes_the_3d_gap_instead_of_holding():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            applied.append(tuple(torso))
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def ramp_arm(self, arm, target_deg, seconds, dt):
            raise AssertionError("left seek must not run while the tip is still short")

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(("moved", arm))

    mover = ArmMover(
        telemetry=lambda: {
            "phase": "SEEK_GRIPPER",
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "tip_uv": None,
            "pbvs_error_torso_m": [0.12, -0.02, 0.01],
            "side_message": "侧视未看到右夹爪",
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    mover._servo_once()
    assert applied[0] == (0.04, -0.02, 0.01)
    assert "尖端未到按键" in mover.message
    assert "122mm" in mover.message


def test_left_gripper_seek_reverses_inside_one_joint_span():
    origin = [0.0] * 7
    joint = 0
    direction = 1
    offset = 0.0
    seen = []
    for _ in range(20):
        pose, joint, direction, offset = next_left_gripper_seek(origin, joint, direction, offset)
        seen.append(pose[5])
    assert max(seen) <= LEFT_SEEK_SPAN_DEG + 1e-6
    assert min(seen) >= -LEFT_SEEK_SPAN_DEG - 1e-6
    assert seen[0] == 4.0
    assert any(value < 0.0 for value in seen)


def test_start_allows_a_step_that_drops_the_shoulder():
    sent = []

    class _IK:
        def apply_tip_delta(self, q, torso, posture):
            dropped = list(q)
            dropped[0] = math.radians(-8.0)
            dropped[3] = math.radians(10.0)
            return dropped, True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return None

        def measured_arm_rad(self, arm):
            return [math.radians(v) for v in (-85.0, -18.0, 0.0, 69.0, 0.0, 0.0, 0.0)]

        def measured_arm_deg(self, arm):
            return [-85.0, -18.0, 0.0, 69.0, 0.0, 0.0, 0.0]

        def apply_arm_rad(self, arm, q, seconds, dt):
            sent.append(list(q))

    mover = ArmMover(
        telemetry=lambda: {"phase": "BACKOFF", "dx": 0.0, "dy": 0.0, "dz": -0.004},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover.start()
    mover._servo_once()
    assert sent
    assert abs(sent[0][0] - math.radians(-8.0)) < 1e-9
    assert abs(sent[0][3] - math.radians(10.0)) < 1e-9
    assert "放下" not in mover.message


def test_servo_uses_the_held_command_instead_of_loaded_arm_sag():
    seeds = []
    commanded = [math.radians(v) for v in (-112.0, -25.0, 0.0, 61.0, 0.0, 0.0, 0.0)]
    sagged = [math.radians(v) for v in (-70.0, -20.0, 0.0, 55.0, 0.0, 0.0, 0.0)]

    class _IK:
        def apply_tip_delta(self, q, torso, posture):
            seeds.append((list(q), list(posture)))
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return list(commanded)

        def measured_arm_rad(self, arm):
            return list(sagged)

        def measured_arm_deg(self, arm):
            return [math.degrees(value) for value in sagged]

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    mover = ArmMover(
        telemetry=lambda: {"phase": "APPROACH", "dx": 0.0, "dy": 0.0, "dz": 0.004},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _AliveThread()
    mover.start()
    mover._servo_once()
    assert seeds == [(commanded, commanded)]


def test_pause_freezes_ibvs_and_enables_six_axis_manual_jog():
    applied = []
    calls = []

    class _IK:
        def apply_delta(self, q, dxyz, drpy, rotation_frame):
            calls.append((list(q), tuple(dxyz), tuple(drpy), rotation_frame))
            moved = list(q)
            moved[0] += 0.01
            return moved, True, 0.001, math.radians(0.2)

        def fk_xyz_rpy(self, q):
            return [0.50, -0.20, 0.35], [0.1, -0.2, 0.3]

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.09] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    assert "细调" in mover.pause()
    assert mover.status()["manual_enabled"] is True
    result = mover.manual_jog((0.002, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert result["ok"] is True
    assert calls[0][0] == [0.1] * 7
    assert calls[0][1] == (0.002, 0.0, 0.0)
    assert calls[0][3] == "base"
    assert applied
    status = mover.status()
    assert status["ee_xyz_m"] == [0.5, -0.2, 0.35]
    assert status["ee_rpy_deg"] is not None


def test_manual_jog_uses_measured_pose_without_replacing_the_hold_command():
    calls = []
    commanded = [0.2] * 7
    measured = [0.1] * 7

    class _IK:
        def apply_loaded_delta(self, command, actual, dxyz, drpy, rotation_frame):
            calls.append((list(command), list(actual), tuple(dxyz), tuple(drpy), rotation_frame))
            return list(command), True, 0.0, 0.0

        def fk_xyz_rpy(self, q):
            return [0.4, -0.2, 0.3], [0.0, 0.0, 0.0]

    class _Controller:
        def commanded_arm_rad(self, arm):
            return list(commanded)

        def measured_arm_rad(self, arm):
            return list(measured)

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.pause()
    assert mover.manual_jog((0.002, 0.0, 0.0), (0.0, 0.0, 0.0))["ok"]
    assert calls == [(commanded, measured, (0.002, 0.0, 0.0), (0.0, 0.0, 0.0), "base")]


def test_manual_elbow_moves_only_that_joint():
    applied = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = object()
    assert mover.manual_elbow(1.0)["ok"] is False
    mover.pause()
    result = mover.manual_elbow(2.0)
    assert result["ok"] is True
    assert applied
    assert abs(applied[0][3] - (0.4 + math.radians(2.0))) < 1e-9
    assert applied[0][:3] == [0.1, 0.2, 0.3]
    assert applied[0][4:] == [0.5, 0.6, 0.7]
    assert "右肘" in result["message"]


def test_manual_joint_moves_only_the_selected_axis():
    applied = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = object()
    mover.pause()
    shoulder = mover.manual_joint(2, 1.0)
    wrist = mover.manual_joint(6, -0.5)
    assert shoulder["ok"] is True and wrist["ok"] is True
    assert abs(applied[0][2] - (0.3 + math.radians(1.0))) < 1e-9
    assert applied[0][0:2] == [0.1, 0.2]
    assert applied[0][3:] == [0.4, 0.5, 0.6, 0.7]
    assert abs(applied[1][6] - (0.7 - math.radians(0.5))) < 1e-9
    assert applied[1][:6] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert "肩 yaw" in shoulder["message"]
    assert "腕 yaw" in wrist["message"]


def test_manual_joint_stops_at_the_urdf_limit():
    applied = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.576, 0.0]

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.pause()
    result = mover.manual_joint(5, 1.0)
    assert result["ok"] is False
    assert "限位" in result["error"]
    assert applied == []


def test_upward_servo_stops_when_the_tip_stays_below_the_button():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            applied.append(tuple(torso))
            raised = list(q)
            raised[0] -= 0.02
            return raised, True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_deg(self, arm):
            return [-70.0, 0.0, 0.0, 65.0, 0.0, 0.0, 0.0]

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    mover = ArmMover(
        telemetry=lambda: {"phase": "ALIGN_XY", "dx": 0.0, "dy": -0.004, "dz": 0.0, "tip_uv": [200.0, 400.0]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    for _ in range(5):
        mover._servo_once()
    assert len(applied) == 3
    assert "停止上举" in mover.message


def test_rejected_raise_does_not_trip_the_upward_guard():
    class _IK:
        def apply_delta(self, q, torso):
            return list(q), False, 0.02, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_deg(self, arm):
            return [-70.0, 0.0, 0.0, 65.0, 0.0, 0.0, 0.0]

        def apply_arm_rad(self, arm, q, seconds, dt):
            raise AssertionError("rejected IK must not move the arm")

    mover = ArmMover(
        telemetry=lambda: {"phase": "ALIGN_XY", "dx": 0.0, "dy": -0.004, "dz": 0.0, "tip_uv": [200.0, 400.0]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DUAL_IBVS"
    for _ in range(5):
        mover._servo_once()
    assert mover._up_stuck == 0
    assert "停止上举" not in mover.message
    assert "IK 拒绝" in mover.message


def test_panel_loss_pokes_forward_instead_of_retreating():
    applied = []

    class _IK:
        def apply_delta(self, q, torso):
            return list(q), True, 0.0, 0.0

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            applied.append(list(q))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "IBVS"
    mover._poke_remaining_m = POKE_EXTRA_M
    mover._servo_once()
    assert mover._ibvs_enabled is True
    assert mover._control_stage == "POKE"
    assert applied
    assert mover._poke_remaining_m < POKE_EXTRA_M
    assert "回退" not in mover.message or "不回退" in mover.message


def test_tip_occlusion_does_not_disable_enabled_servo():
    telem = {
        "panel_visible": True,
        "target_uv": [320.0, 240.0],
        "virtual_tip_xyz_m": [0.0, 0.0, 0.5],
        "virtual_tip_uv": [400.0, 200.0],
        "tip_visible": False,
    }
    mover = ArmMover(telemetry=lambda: telem, confirm=True, log=lambda _m: None)
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._servo_once()
    assert mover._ibvs_enabled is True
    assert mover.phase != "hold"


def test_visual_loss_recovers_previous_visible_pose_once():
    calls = []

    class _Controller:
        def apply_arm_rad(self, arm, q, seconds, dt):
            calls.append((arm, list(q), seconds, dt))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._last_visible_q = [0.1] * 7
    assert mover._recover_last_visible_pose()
    assert calls and calls[0][0] == "right"
    assert mover._last_visible_q is None
    assert not mover._recover_last_visible_pose()


def test_cyclonedds_search_finds_clayer_dir(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    pkg = site / "cyclonedds"
    pkg.mkdir(parents=True)
    (pkg / "_clayer.abi3.so").write_bytes(b"x")
    monkeypatch.setattr(dds_bind, "_candidate_site_dirs", lambda: [site])
    assert site in dds_bind.cyclonedds_search_dirs()


def test_ensure_cyclonedds_inserts_clayer_dir(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    pkg = site / "cyclonedds"
    pkg.mkdir(parents=True)
    (pkg / "_clayer.abi3.so").write_bytes(b"x")
    monkeypatch.setattr(dds_bind, "cyclonedds_search_dirs", lambda: [site])
    monkeypatch.setattr(dds_bind, "_try_import_clayer", lambda: str(site) in sys.path)
    if str(site) in sys.path:
        sys.path.remove(str(site))
    dds_bind.ensure_cyclonedds_clayer()
    assert sys.path[0] == str(site)


def test_usable_realsense_restores_user_site():
    import site

    from ibvs_pad_touch.camera.realsense_bind import usable_realsense

    user = site.getusersitepackages()
    usable_realsense()
    if user:
        assert user in sys.path


def test_fuse_averages_buttons_inside_two_centimetres_and_keeps_chest_otherwise():
    chest = {"button_1": (0.50, 0.10, 0.40), "button_2": (0.52, 0.00, 0.41)}
    close = {"button_1": (0.51, 0.10, 0.40), "button_2": (0.52, 0.01, 0.41)}
    fused, source = fuse_pbvs_buttons(chest, close)
    assert source == "average"
    assert abs(fused["button_1"][0] - 0.505) < 1e-9
    assert fused["button_2"][1] == 0.005
    far = {"button_1": (0.50, 0.13, 0.40)}
    kept, source = fuse_pbvs_buttons(chest, far)
    assert source == "chest"
    assert kept["button_1"] == (0.50, 0.10, 0.40)
    assert kept["button_2"] == (0.52, 0.00, 0.41)


def test_sequence_lift_stays_on_the_button():
    assert sequence_lift_goal((0.40, -0.02, 0.25)) == (0.40, -0.02, 0.25)


def test_hybrid_records_a_seen_wrist_button_and_starts_the_approach(monkeypatch):
    from ibvs_pad_touch.web.arm_motion import EYE_IN_HAND_DWELL_S

    sleeps = []
    monkeypatch.setattr("ibvs_pad_touch.web.arm_motion.time.sleep", lambda seconds: sleeps.append(seconds))

    class _Controller:
        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def fk_matrix(self, q):
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    mover = ArmMover(telemetry=lambda: {}, confirm=True, pbvs_first=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.wrist_watch = lambda _enabled: None
    mover.wrist_hold = lambda: None
    mover.wrist_view = lambda: {
        "seen": True,
        "buttons_fresh": True,
        "buttons_cam_m": {"button_1": [0.50, 0.10, 0.40]},
    }
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert EYE_IN_HAND_DWELL_S == 5.0
    assert mover.wrist_dwell_s == 5.0
    ok = mover._fuse_wrist_pbvs(
        {"target": "button_1", "buttons_torso_m": {"button_1": [0.50, 0.10, 0.40]}}
    )
    assert ok is True
    assert sleeps == []
    assert mover._pbvs_source == "wrist"
    assert mover._wrist_latched_torso["button_1"] == (0.50, 0.10, 0.40)
    assert "记录" in mover.message


def test_hidden_panel_skips_the_wrist_average_after_the_level_pose():
    ramps = []

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            ramps.append(tuple(deg))

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    mover = ArmMover(telemetry=lambda: {}, confirm=True, direct_pbvs_poke=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.wrist_dwell_s = 0.0
    mover.wrist_watch = lambda _enabled: None
    mover.wrist_view = lambda: {"seen": False, "buttons_cam_m": {}}
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    mover._fuse_wrist_pbvs(
        {"target": "button_1", "buttons_torso_m": {"button_1": [0.50, 0.10, 0.40], "button_2": [0.52, 0.0, 0.41]}}
    )
    assert ramps == []
    assert mover._pbvs_source == "chest"
    assert mover._locked_buttons["button_1"] == (0.50, 0.10, 0.40)
    assert mover._locked_buttons["button_2"] == (0.52, 0.0, 0.41)
    assert "保持" in mover.message


def test_level_hold_averages_a_wrist_fix_within_two_centimetres():
    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            return None

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def fk_matrix(self, q):
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    mover = ArmMover(telemetry=lambda: {}, confirm=True, direct_pbvs_poke=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.wrist_dwell_s = 0.0
    mover.wrist_watch = lambda _enabled: None
    mover.wrist_view = lambda: {"seen": True, "buttons_cam_m": {"button_1": [0.51, 0.10, 0.40]}}
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    mover._fuse_wrist_pbvs({"target": "button_1", "buttons_torso_m": {"button_1": [0.50, 0.10, 0.40]}})
    assert mover._pbvs_source == "average"
    assert abs(mover._locked_buttons["button_1"][0] - 0.505) < 1e-9


def _wrist_ibvs_mover(applied, view):
    class _Controller:
        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def fk_matrix(self, q):
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    mover = ArmMover(telemetry=lambda: {}, confirm=True, pbvs_first=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._goal_name = "button_1"
    mover._ibvs_enabled = True
    mover._running.set()
    mover.wrist_view = lambda: view
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    mover._apply_torso_delta = (
        lambda delta, **kwargs: applied.append((delta, kwargs))
        or setattr(mover, "phase", kwargs["phase"])
        or True
    )
    return mover


def _wrist_ibvs_view(**overrides):
    view = {
        "seen": True,
        "occluded": False,
        "target_uv": [382.0, 240.0],
        "tip_uv": [320.0, 240.0],
        "target_z_m": 0.20,
        "tip_z_m": 0.20,
        "intrinsics": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0},
        "updated_at": time.time(),
        "contact": False,
        "buttons_cam_m": {"button_1": [0.02, 0.0, 0.20]},
    }
    view.update(overrides)
    return view


def test_stale_wrist_lowers_from_the_chest_gripper_instead_of_holding():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_cam_m={}, buttons_fresh=False, red_bar=False)
    mover = _wrist_ibvs_mover(applied, view)
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.57, -0.07, 0.44]
    mover._wrist_approach_once(
        {
            "chest_pose_button_uv": [400.0, 280.0],
            "chest_pose_button_torso_m": [0.57, -0.07, 0.44],
            "chest_gripper_uv": [520.0, 90.0],
            "tip_uv": [410.0, 270.0],
        }
    )
    assert applied
    assert applied[0][0][2] < 0.0
    assert "新鲜" not in mover.message
    assert "降低" in mover.message


def test_tip_past_the_button_backs_out_in_x_before_raising():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_cam_m={}, buttons_fresh=False, red_bar=False)
    mover = _wrist_ibvs_mover(applied, view)
    mover._chest_height_after_pbvs = True
    mover.ik.tip_xyz = lambda q, offset=(0.205, 0.0, 0.0): (0.5628, 0.0167, 0.3432)
    mover.telemetry = lambda: {"chest_pose_button_torso_m": [0.5417, 0.0134, 0.4205]}
    mover._wrist_approach_once(
        {
            "chest_pose_button_torso_m": [0.5417, 0.0134, 0.4205],
            "chest_pose_button_uv": [400.0, 120.0],
            "chest_gripper_uv": [400.0, 200.0],
        }
    )
    assert applied
    assert abs(applied[0][0][0] + 0.02) < 1e-9
    assert abs(applied[0][0][1]) < 1e-9
    assert abs(applied[0][0][2]) < 1e-9
    assert "后退" in mover.message
    assert "不上举" in mover.message


def test_low_wrist_button_score_keeps_closing_on_the_chest_pose():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_cam_m={}, buttons_fresh=False, red_bar=False)
    mover = _wrist_ibvs_mover(applied, view)
    mover._chest_height_after_pbvs = True
    mover.ik.tip_xyz = lambda q, offset=(0.205, 0.0, 0.0): (0.5317, -0.0608, 0.4579)
    mover.telemetry = lambda: {"chest_pose_button_torso_m": [0.5637, -0.0420, 0.4348]}
    mover._wrist_approach_once({})
    assert applied
    assert applied[0][0][0] > 0.0
    assert abs(applied[0][0][0]) <= 0.004 + 1e-9
    assert "0.3" in mover.message
    assert "还没看到" not in mover.message


def test_recorded_wrist_button_keeps_the_approach_after_the_image_loses_it():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_fresh=False, buttons_cam_m={})
    mover = _wrist_ibvs_mover(applied, view)
    mover.ik.tip_xyz = lambda q: (0.0, 0.0, 0.20)
    mover._wrist_latched_torso = {"button_1": (0.02, 0.0, 0.20)}
    mover._wrist_approach_once({})
    assert applied
    step = applied[0][0]
    assert abs(step[0] - 0.02) < 1e-9
    assert abs(step[1]) < 1e-9
    assert abs(step[2]) < 1e-9
    assert applied[0][1]["limit_m"] == 0.004
    assert "记录" in mover.message


def test_red_bar_raises_the_gripper_in_small_steps():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_cam_m={}, red_bar=True, buttons_fresh=False)
    mover = _wrist_ibvs_mover(applied, view)
    mover._wrist_approach_once({})
    mover._wrist_approach_once({})
    assert [item[0][2] for item in applied] == [0.005, 0.005]
    assert all(abs(item[0][0]) < 1e-9 and abs(item[0][1]) < 1e-9 for item in applied)
    assert mover._wrist_red_lift_m == 0.01
    mover._wrist_red_lift_m = 0.05
    mover._wrist_approach_once({})
    assert len(applied) == 2
    assert "50mm" in mover.message


def test_chest_pose_lowers_when_the_end_effector_is_above_the_button():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.40, 0.0, 0.47]
    mover._wrist_approach_once({"chest_pose_button_torso_m": [0.48, -0.02, 0.30]})
    assert applied
    step = applied[0][0]
    assert abs(step[0]) < 1e-9
    assert abs(step[1]) < 1e-9
    assert step[2] < 0.0
    assert abs(step[2]) <= 0.004 + 1e-9
    assert "偏高" in mover.message
    assert "降低" in mover.message


def test_chest_pose_raises_when_the_end_effector_is_below_the_button():
    applied = []
    view = _wrist_ibvs_view(updated_at=0.0, target_uv=None, buttons_cam_m={}, red_bar=True)
    mover = _wrist_ibvs_mover(applied, view)
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.40, 0.0, 0.30]
    mover._wrist_approach_once({"chest_pose_button_torso_m": [0.48, -0.02, 0.47]})
    assert applied[0][0][2] > 0.0
    assert "偏低" in mover.message
    assert mover._wrist_red_lift_m == 0.0


def test_missing_chest_pose_button_leaves_height_to_the_wrist():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._right_ee_xyz_m = [0.40, 0.0, 0.47]
    mover._wrist_approach_once(
        {
            "target_uv": [320.0, 220.0],
            "tip_uv": [340.0, 80.0],
            "target_torso_m": [0.48, -0.02, 0.30],
            "chest_pose_button_torso_m": None,
        }
    )
    assert applied[0][0][0] > 0.002
    assert "像素差" in mover.message


def test_chest_pose_stays_out_until_pbvs_hands_over_to_the_wrist():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._chest_height_after_pbvs = False
    mover._right_ee_xyz_m = [0.40, 0.0, 0.47]
    mover._wrist_approach_once({"chest_pose_button_torso_m": [0.48, -0.02, 0.30]})
    assert applied[0][0][0] > 0.002
    assert "像素差" in mover.message


def test_chest_pose_raises_when_the_button_is_above_the_tip_even_if_the_image_looks_high():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.4848, -0.0813, 0.358]
    mover._wrist_approach_once(
        {
            "chest_pose_button_torso_m": [0.5739, -0.0733, 0.440],
            "chest_pose_button_uv": [400.0, 280.0],
            "tip_uv": [220.0, 90.0],
        }
    )
    assert applied
    assert applied[0][0][2] > 0.0
    assert abs(applied[0][0][2]) <= 0.004 + 1e-9
    assert abs(applied[0][0][0]) < 1e-9
    assert "偏低" in mover.message
    assert "抬高" in mover.message


def test_chest_image_lowers_without_button_depth():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.4848, -0.0813, 0.4328]
    mover._wrist_approach_once(
        {
            "chest_pose_button_uv": [400.0, 280.0],
            "virtual_tip_uv": [220.0, 80.0],
        }
    )
    assert applied[0][0][2] < 0.0
    assert "画面" in mover.message


def test_small_chest_pose_z_gap_lets_the_wrist_keep_approaching():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._chest_height_after_pbvs = True
    mover._right_ee_xyz_m = [0.40, 0.0, 0.400]
    mover._wrist_approach_once({"chest_pose_button_torso_m": [0.48, -0.02, 0.404]})
    assert applied[0][0][0] > 0.002
    assert "像素差" in mover.message


def test_right_wrist_approach_servos_the_image_error():
    applied = []
    mover = _wrist_ibvs_mover(applied, _wrist_ibvs_view())
    mover._wrist_approach_once({})
    assert applied
    step = applied[0][0]
    assert step[0] > 0.002
    assert abs(step[1]) < 1e-4
    assert abs(step[2]) < 1e-3
    assert math.sqrt(sum(value * value for value in step)) <= 0.004 + 1e-9
    assert applied[0][1]["limit_m"] == 0.004
    assert mover.phase == "WRIST_APPROACH"
    assert "像素差" in mover.message


def test_right_wrist_approach_closes_a_depth_gap_in_the_camera():
    applied = []
    mover = _wrist_ibvs_mover(
        applied,
        _wrist_ibvs_view(target_uv=[320.0, 240.0], tip_uv=[320.0, 240.0], target_z_m=0.25, tip_z_m=0.10),
    )
    mover._wrist_approach_once({})
    step = applied[0][0]
    assert step[2] > 0.002
    assert abs(step[0]) < 1e-4
    assert abs(step[1]) < 1e-4


def test_close_range_translates_before_moving_forward():
    applied = []
    view = _wrist_ibvs_view(
        target_z_m=None,
        tip_z_m=None,
        buttons_cam_m={"button_1": [0.02, 0.0, 0.20]},
        close_range=True,
        target_uv=[382.0, 240.0],
        tip_uv=[320.0, 240.0],
    )
    mover = _wrist_ibvs_mover(applied, view)
    mover._wrist_latched_torso = {"button_1": (0.05, 0.0, 0.20)}
    mover._wrist_approach_once({})
    assert applied
    step = applied[0][0]
    assert step[0] > 0.0
    assert abs(step[1]) < 1e-6
    assert abs(step[2]) < 1e-9
    assert applied[0][1]["limit_m"] == 0.004
    assert mover._wrist_close_advanced is False
    assert "button_1" not in mover._wrist_latched_torso
    assert "对齐" in mover.message


def test_close_range_advances_30mm_once_the_tip_meets_the_button():
    applied = []
    view = _wrist_ibvs_view(
        target_z_m=None,
        tip_z_m=None,
        buttons_cam_m={},
        close_range=True,
        target_uv=[322.0, 241.0],
        tip_uv=[320.0, 240.0],
    )
    mover = _wrist_ibvs_mover(applied, view)
    mover._right_ee_xyz_m = [0.6088, 0.0134, 0.4277]
    mover._wrist_approach_once({"target_torso_m": [0.6284, -0.0279, 0.4269]})
    assert len(applied) == 1
    step = applied[0][0]
    assert abs(step[0] - (0.6284 - 0.6088)) < 1e-9
    assert step[0] < 0.03
    assert abs(step[1]) < 1e-9
    assert abs(step[2]) < 1e-9
    assert abs(applied[0][1]["limit_m"] - step[0]) < 1e-9
    assert applied[0][1]["seconds"] == 1.0
    assert mover._wrist_close_advanced is True
    assert "向前" in mover.message
    mover._wrist_approach_once({"target_torso_m": [0.6284, -0.0279, 0.4269]})
    assert len(applied) == 1
    assert "等待接触" in mover.message


def test_close_range_reaches_the_pbvs_x_when_it_is_farther_than_3cm():
    applied = []
    view = _wrist_ibvs_view(
        target_z_m=None,
        tip_z_m=None,
        buttons_cam_m={},
        close_range=True,
        target_uv=[322.0, 241.0],
        tip_uv=[320.0, 240.0],
    )
    mover = _wrist_ibvs_mover(applied, view)
    mover._goal_name = "button_1"
    mover._locked_buttons = {"button_1": (0.6284, -0.0279, 0.4269)}
    mover._right_ee_xyz_m = [0.50, 0.0, 0.42]
    mover._wrist_approach_once({"target_torso_m": [0.40, 0.0, 0.42]})
    step = applied[0][0]
    assert abs(step[0] - (0.6284 - 0.50)) < 1e-9
    assert abs(step[1]) < 1e-9
    assert abs(step[2]) < 1e-9
    assert applied[0][1]["limit_m"] == step[0]


def test_hybrid_starts_when_the_wrist_is_too_close_for_depth(monkeypatch):
    sleeps = []
    monkeypatch.setattr("ibvs_pad_touch.web.arm_motion.time.sleep", lambda seconds: sleeps.append(seconds))

    class _Controller:
        def measured_arm_rad(self, arm):
            return [0.1] * 7

    mover = ArmMover(telemetry=lambda: {}, confirm=True, pbvs_first=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.wrist_watch = lambda _enabled: None
    mover.wrist_hold = lambda: None
    mover.wrist_view = lambda: {
        "seen": True,
        "buttons_fresh": False,
        "buttons_cam_m": {},
        "close_range": True,
        "target_uv": [400.0, 240.0],
        "tip_uv": [320.0, 240.0],
    }
    mover.cam2hand_matrix = lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ok = mover._fuse_wrist_pbvs(
        {"target": "button_1", "buttons_torso_m": {"button_1": [0.50, 0.10, 0.40]}}
    )
    assert ok is True
    assert sleeps == []
    assert mover._pbvs_source == "wrist"
    assert mover._wrist_latched_torso == {}
    assert "近距" in mover.message


def test_right_wrist_approach_holds_when_the_jaw_covers_the_panel():
    applied = []
    covered = _wrist_ibvs_view(occluded=True, updated_at=0.0, target_uv=None, target_z_m=None, buttons_cam_m={})
    mover = _wrist_ibvs_mover(applied, covered)
    mover._wrist_approach_once({})
    assert applied == []
    assert mover.phase == "wrist_occluded"
    assert "遮挡" in mover.message


def test_clicked_yellow_region_is_tracked_past_the_pixel_jump_hold():
    applied = []
    view = _wrist_ibvs_view(yellow_lock=True, yellow_id=3, target_uv=[520.0, 240.0])
    mover = _wrist_ibvs_mover(applied, view)
    mover._wrist_ibvs_uv = (320.0, 240.0)
    mover._wrist_approach_once({})
    assert applied
    assert "跟踪点选" in mover.message
    assert "跳变" not in mover.message


def test_right_wrist_approach_rejects_one_pixel_jump():
    applied = []
    view = _wrist_ibvs_view()
    mover = _wrist_ibvs_mover(applied, view)
    mover._wrist_approach_once({})
    view["target_uv"] = [582.0, 240.0]
    mover._wrist_approach_once({})
    assert len(applied) == 1
    assert "跳变" in mover.message


def test_right_wrist_contact_stops_final_approach():
    watch = []
    mover = ArmMover(telemetry=lambda: {}, confirm=True, pbvs_first=True, log=lambda _m: None)
    mover._ibvs_enabled = True
    mover._running.set()
    mover._control_stage = "WRIST_APPROACH"
    mover.wrist_watch = lambda enabled: watch.append(enabled)
    mover.wrist_view = lambda: {"contact": True}
    mover._wrist_approach_once({})
    assert watch == [False]
    assert mover._ibvs_enabled is False
    assert mover._running.is_set() is False
    assert mover._control_stage == "HOLD"
    assert mover.phase == "contact"


def test_sequence_start_refuses_an_empty_queue_and_single_start_ignores_one():
    class _Alive:
        def is_alive(self):
            return True

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.2, 0.0, 0.3]},
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = object()
    mover._thread = _Alive()
    refused = mover.start_sequence()
    assert "为空" in refused
    assert mover._sequence_run is False
    assert mover._ibvs_enabled is False
    mover.set_button_sequence(["button_2", "button_1"])
    single = mover.start()
    assert mover._sequence_run is False
    assert "组合启动" not in single
    queued = mover.start_sequence()
    assert mover._sequence_run is True
    assert queued == "组合启动 Hybrid：Button 2 → Button 1"
    assert mover._control_stage == "DIRECT_PBVS"
    mover._ibvs_enabled = False
    mover._running.clear()
    chosen = mover.set_sequence_mode("direct_pbvs")
    assert "Direct PBVS" in chosen
    assert mover.status()["sequence_mode"] == "direct_pbvs"
    again = mover.start_sequence()
    assert again == "组合启动 Direct PBVS：Button 2 → Button 1"
    blocked = mover.set_sequence_mode("hybrid")
    assert "Pause" in blocked
    assert mover.status()["sequence_mode"] == "direct_pbvs"


def _sequence_chest(buttons, *, contact=True, held=False):
    return {
        "wrist_contact": contact,
        "panel_visible": True,
        "depth_available": True,
        "using_last_target": held,
        "buttons_torso_m": buttons,
    }


def test_direct_sequence_presses_each_button_without_the_wrist_approach():
    calls = []
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", tuple(deg)))

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            calls.append(("clear", tuple(float(value) for value in goal)))
            return True, "ok"

        def release(self, seconds=1.2):
            calls.append(("release", seconds))

    class _IK:
        def tip_xyz(self, q):
            return (0.50, 0.08, 0.42)

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons),
        confirm=True,
        pbvs_first=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append(("drop",))
    mover.set_sequence_mode("direct_pbvs")
    mover.set_button_sequence(["button_1", "button_2"])

    def direct(telem, release=True):
        calls.append(("direct", telem["target"], release))
        mover.phase = "direct_pbvs_complete"

    def hybrid(name, goal):
        raise AssertionError(name)

    mover._direct_pbvs_once = direct
    mover._run_hybrid_button = hybrid
    mover._run_button_sequence()
    assert [item for item in calls if item[0] == "direct"] == [
        ("direct", "button_1", False),
        ("direct", "button_2", False),
    ]
    assert [item[0] for item in calls].count("clear") == 1
    assert ("ramp", REST_DEG) in calls
    assert ("release", 1.2) in calls
    assert mover.phase == "released"
    assert mover._control_stage == "RELEASED"


def test_button_sequence_lowers_and_reidentifies_between_one_shot_presses():
    calls = []
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", arm, tuple(deg)))

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            calls.append(("clear", arm, tuple(float(value) for value in goal)))
            return True, "ok"

        def release(self, seconds=1.2):
            calls.append(("release", seconds))

    class _IK:
        def tip_xyz(self, q):
            return (0.50, 0.08, 0.42)

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append(("drop",))
    mover._locked_buttons = {"button_1": (9.0, 9.0, 9.0), "button_2": (8.0, 8.0, 8.0)}
    assert mover.set_button_sequence(["button_1", "button_2", "button_1", "nope"]) == [
        "button_1",
        "button_2",
        "button_1",
    ]

    def shoot(name, goal):
        calls.append(("shoot", name, tuple(goal), mover._sequence_stage))
        mover.phase = "contact"
        return True

    captured = []
    selected = []
    mover._run_hybrid_button = shoot
    mover.on_select_target = selected.append
    mover.on_screen_capture = lambda: captured.append(True)
    mover._run_button_sequence()
    shots = [item for item in calls if item[0] == "shoot"]
    assert [item[1] for item in shots] == ["button_1", "button_2", "button_1"]
    assert shots[0][2] == (0.40, 0.10, 0.20)
    assert shots[1][2] == (0.42, 0.00, 0.20)
    assert [item[3] for item in shots] == ["Button1 PBVS", "Button2 PBVS", "Button1 PBVS"]
    assert mover._sequence_stage == "结束放下"
    assert [item[2] for item in calls if item[0] == "ramp"] == [REST_DEG]
    assert [item[0] for item in calls] == [
        "shoot",
        "clear",
        "shoot",
        "clear",
        "shoot",
        "ramp",
        "release",
        "drop",
    ]
    clear = sequence_clear_goal((0.50, 0.08, 0.42))
    assert [item[2] for item in calls if item[0] == "clear"] == [clear, clear]
    assert clear[1] < 0.08 and clear[2] < 0.42 and clear[0] == 0.50
    assert selected == ["button_1", "button_2", "button_1"]
    assert captured == [True]
    assert mover.phase == "released"
    assert mover.controller is None


def test_arm_sdk_swings_a_hanging_arm_out_before_the_raise():
    calls = []

    class _Controller:
        def measured_arm_deg(self, arm):
            return [0.0, -10.0, 0.0, 90.0, 0.0, 0.0, 0.0]

        def ramp_arm(self, arm, target, seconds, dt):
            calls.append((arm, list(target)))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._motion_backend = "arm_sdk"
    mover._swing_right_arm_clear()
    assert calls == [("right", [0.0, TORSO_CLEAR_ROLL_DEG, 0.0, 90.0, 0.0, 0.0, 0.0])]
    mover._motion_backend = "service"
    mover._swing_right_arm_clear()
    assert len(calls) == 1
    mover._motion_backend = "arm_sdk"
    mover.controller.measured_arm_deg = lambda arm: [-76.0, -10.0, 0.0, 65.0, 0.0, 0.0, 0.0]
    mover._swing_right_arm_clear()
    assert len(calls) == 1


def test_service_rise_keeps_x_until_the_arm_is_up():
    low = (0.50, 0.02, 0.30)
    button = (0.60, -0.04, 0.45)
    lift, forward = service_rise_before_wall(low, button)
    assert lift[0] == low[0]
    assert lift[0] < button[0]
    assert lift[1:] == button[1:]
    assert forward == button
    assert service_rise_before_wall((0.50, 0.0, 0.45), button) == (button,)
    assert service_rise_before_wall((0.62, 0.0, 0.30), button) == (button,)


def test_forward_step_is_only_x_when_height_and_lateral_are_close():
    step = pbvs_forward_step((0.0576, 0.0072, -0.0110))
    assert step is not None
    assert abs(step[0] - 0.02) < 1e-9
    assert step[1] == 0.0 and step[2] == 0.0
    assert pbvs_forward_step((0.0576, 0.05, -0.0110)) is None
    assert pbvs_forward_step((0.005, 0.0072, -0.0110)) is None
    assert pbvs_descent_step((0.05, 0.0, -0.103)) == (0.0, 0.0, -0.02)
    assert pbvs_descent_step((0.0576, 0.0072, -0.0110)) is None
    assert pbvs_lateral_step((0.0086, 0.0323, -0.0117)) == (0.0, 0.02, 0.0)
    assert pbvs_lateral_step((0.0086, 0.005, -0.0117)) is None
    assert pbvs_lateral_step((0.05, 0.0323, -0.0117)) is None


def test_coarse_pbvs_descends_when_the_tip_is_already_above_the_goal():
    deltas = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.50, -0.07, 0.55)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a tip above the goal must not re-solve the pose")

    mover = ArmMover(
        telemetry=lambda: {
            "chest_pose_button_uv": [400.0, 120.0],
            "virtual_tip_uv": [420.0, 280.0],
            "chest_pose_button_torso_m": [0.55, -0.07, 0.65],
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.55, -0.07, 0.447]})
    assert deltas == [(0.0, 0.0, -0.02)]
    assert "下降" in mover.message
    assert "103" in mover.message


def test_close_pbvs_raises_when_the_chest_button_is_above_the_tip():
    deltas = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.590, -0.067, 0.358)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("chest height must raise without re-solving the pose")

    mover = ArmMover(
        telemetry=lambda: {
            "chest_pose_button_torso_m": [0.594, -0.070, 0.440],
            "chest_pose_button_uv": [400.0, 280.0],
            "virtual_tip_uv": [420.0, 80.0],
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.594, -0.070, 0.440]})
    assert deltas == [(0.0, 0.0, 0.004)]
    assert "偏低" in mover.message
    assert "抬高" in mover.message


def test_chest_height_step_moves_the_locked_goal_with_the_tip():
    mover = ArmMover(
        telemetry=lambda: {"chest_pose_button_torso_m": [0.579, -0.015, 0.434]},
        confirm=True,
        log=lambda _m: None,
    )

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def apply_tip_delta(self, q, dxyz, posture):
            return list(q), True, 0.0, 0.01

    mover.controller = _Controller()
    mover.ik = _IK()
    mover._goal_name = "button_1"
    mover._locked_buttons = {"button_1": (0.5794, -0.0149, 0.4532)}
    assert mover._correct_height_from_chest((0.569, -0.017, 0.4966), phase="PBVS_COARSE") is True
    assert abs(mover._locked_buttons["button_1"][2] - (0.4966 - 0.004)) < 1e-9
    assert mover._locked_buttons["button_1"][0] == 0.5794


def test_hybrid_keeps_walking_forward_after_the_pose_budget():
    seen = {"n": 0}
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)

    def tip_error(_goal):
        if seen["n"] >= 8:
            return (0.012, 0.0, 0.001)
        return (0.08, 0.04, 0.0)

    def once(_telem):
        seen["n"] += 1
        if seen["n"] <= 8:
            return
        if seen["n"] == 9:
            mover._forward_hold = True
            return
        mover._control_stage = "WRIST_APPROACH"

    mover._tip_error_to = tip_error
    mover._pbvs_once = once
    mover._wrist_approach_once = lambda _telem: setattr(mover, "phase", "contact")
    assert mover._run_hybrid_button("button_1", (0.55, -0.03, 0.45)) is True
    assert seen["n"] == 10
    assert mover._sequence_stage == ""


def test_coarse_pbvs_slides_in_y_when_the_chest_button_is_still_beside_the_tip():
    deltas = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.5557, 0.0231, 0.4482)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a 3 cm Y gap must not re-solve the pose")

    mover = ArmMover(
        telemetry=lambda: {"chest_pose_button_torso_m": [0.5643, 0.0554, 0.4365]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.wrist_watch = lambda enabled: (_ for _ in ()).throw(AssertionError("wrist handover"))
    mover._control_stage = "PBVS_COARSE"
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.5557, 0.0231, 0.4482]})
    assert deltas == [(0.0, 0.02, 0.0)]
    assert mover._control_stage == "PBVS_COARSE"
    assert "Y 方向" in mover.message


def test_coarse_pbvs_hands_the_live_button_to_the_wrist_when_the_lock_is_still_aside():
    """The fingertip can sit on the chest pose while the frozen lock is 4 cm off in Y."""

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            # Short of x_pbvs - 0.3 cm, and still within 1 cm of the live button.
            return (0.626, -0.01987, 0.43963)

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a live arrival must not chase the stale lock")

    mover = ArmMover(
        telemetry=lambda: {"chest_pose_button_torso_m": [0.631, -0.0206, 0.439]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_deadline = time.monotonic() - 1.0
    handed = []

    def handoff(_telem):
        handed.append(True)
        mover._control_stage = "WRIST_APPROACH"
        mover.phase = "wrist_approach"

    mover._begin_wrist_approach = handoff
    mover._pbvs_once(
        {"target": "button_1", "target_torso_m": [0.641, -0.0586, 0.4381]}
    )
    assert handed == [True]
    assert mover._control_stage == "WRIST_APPROACH"
    assert "超时" not in mover.message


def test_coarse_pbvs_leaves_a_reached_lock_for_the_live_chest_button():
    """Dashboard pose can be 3 cm away after the frozen goal is already under the tip."""
    deltas = []
    tip = (0.5317, -0.0608, 0.4579)
    lock = (0.5317, -0.0578, 0.4599)
    live = (0.5637, -0.0420, 0.4348)

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return tip

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a live gap must not re-solve from the stale lock")

    mover = ArmMover(
        telemetry=lambda: {"chest_pose_button_torso_m": list(live)},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._locked_buttons = {"button_1": lock}
    mover._pbvs_once({"target": "button_1", "target_torso_m": list(lock)})
    assert deltas == [(0.0, 0.0, -0.02)]
    assert mover._locked_buttons["button_1"] == live
    assert "下降" in mover.message
    assert mover._control_stage == "PBVS_COARSE"


def test_coarse_pbvs_backs_out_before_the_preapproach_plane():
    """A tip already past x_pbvs - 0.3 cm backs up in X and does not rise into the wall."""
    deltas = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.5628, 0.0167, 0.3432)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("past the pre-approach plane must not solve a pose")

    mover = ArmMover(
        telemetry=lambda: {
            "chest_pose_button_torso_m": [0.5417, 0.0134, 0.4205],
            "chest_pose_button_uv": [400.0, 280.0],
            "virtual_tip_uv": [420.0, 80.0],
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    handed = []
    mover._begin_wrist_approach = lambda _telem: handed.append(True)
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.5417, 0.0134, 0.4205]})
    assert handed == []
    assert deltas == [(-0.02, 0.0, 0.0)]
    assert mover._control_stage == "PBVS_COARSE"
    ceiling = coarse_x_ceiling(0.5417)
    assert abs(ceiling - (0.5417 - 0.003)) < 1e-9
    assert 0.5628 >= ceiling
    assert "不得超过" in mover.message
    assert "不上举" in mover.message


def test_coarse_forward_stops_short_of_the_preapproach_plane():
    deltas = []
    button_x = 0.5417
    tip_x = 0.5300

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (tip_x, 0.0134, 0.4205)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a short forward gap must not re-solve the pose")

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": [button_x, 0.0134, 0.4205]})
    assert len(deltas) == 1
    step = deltas[0]
    assert step[1] == 0.0 and step[2] == 0.0
    assert step[0] > 0.0
    assert tip_x + step[0] < coarse_x_ceiling(button_x)
    assert abs((tip_x + step[0]) - coarse_x_command(button_x)) < 1e-9


def test_coarse_pose_goal_stays_short_of_the_preapproach_plane():
    asked = []
    button = (0.60, 0.08, 0.40)

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            if asked and abs(float(q[0]) - 0.2) < 1e-9:
                return asked[-1]
            return (0.50, 0.0, 0.40)

        def solve_loaded_tool_pose(self, commanded, measured, **kwargs):
            asked.append(tuple(float(value) for value in kwargs["target_tool_xyz_m"]))
            return {
                "q_command": [0.2] * 7,
                "q_solved": [0.2] * 7,
                "pos_err": 0.001,
                "ori_err": 0.01,
            }

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("coarse planning should use solve_loaded_tool_pose")

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": list(button)})
    assert asked
    ceiling = coarse_x_ceiling(button[0])
    assert all(goal[0] < ceiling for goal in asked)
    assert abs(asked[0][0] - coarse_x_command(button[0])) < 1e-9
    assert abs(asked[0][1] - button[1]) < 1e-9
    assert abs(asked[0][2] - button[2]) < 1e-9


def test_coarse_pbvs_walks_forward_instead_of_resolving_the_pose():
    deltas = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.5120, -0.0825, 0.4479)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("aligned X gap must not re-solve the full pose")

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.5696, -0.0753, 0.4369]})
    assert deltas == [(0.02, 0.0, 0.0)]
    assert mover._forward_hold is True
    assert "向前" in mover.message
    assert "58" in mover.message or "57" in mover.message


def test_pbvs_lowers_when_the_chest_still_sees_the_button_above_the_gripper():
    deltas = []
    watched = []

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def apply_arm_rad(self, arm, q, seconds, dt):
            return None

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return (0.565, -0.07, 0.44)

        def apply_tip_delta(self, q, dxyz, posture):
            deltas.append(tuple(float(value) for value in dxyz))
            return list(q), True, 0.0, 0.01

        def apply_loaded_tool_pose_target(self, *args, **kwargs):
            raise AssertionError("a visible height gap must not re-solve the pose")

    mover = ArmMover(
        telemetry=lambda: {
            "chest_pose_button_uv": [420.0, 300.0],
            "virtual_tip_uv": [500.0, 80.0],
            "tip_uv": [420.0, 290.0],
            "chest_pose_button_torso_m": [0.57, -0.07, 0.44],
        },
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover.wrist_watch = lambda enabled: watched.append(enabled)
    mover._control_stage = "PBVS_COARSE"
    mover._sequence_stage = "Button1 PBVS"
    mover._pbvs_once({"target": "button_1", "target_torso_m": [0.57, -0.07, 0.44]})
    assert deltas == [(0.0, 0.0, -0.004)]
    assert mover._control_stage == "PBVS_COARSE"
    assert watched == []
    assert "偏高" in mover.message
    assert "降低" in mover.message
    assert mover.phase != "wrist_pbvs_unavailable"


def test_pbvs_does_not_play_an_elbow_branch_flip_on_the_last_knot():
    goal = (0.40, 0.0, 0.46)
    straight = [0.30] * 7
    flipped = [0.30] * 7
    flipped[2] = 2.0

    class _IK:
        def tip_xyz(self, q):
            return goal

        def fk_xyz_rpy(self, q):
            return [0.4, 0.0, 0.46], [0.0, 0.0, 0.0]

        def solve_loaded_tool_pose(self, commanded, measured, **kwargs):
            xyz = kwargs["target_tool_xyz_m"]
            at_goal = abs(float(xyz[0]) - goal[0]) < 1e-9 and abs(float(xyz[2]) - goal[2]) < 1e-9
            # The one-shot goal solve has no chain seed and lands elbow-up.
            # The same goal, seeded from the walked chain, stays elbow-down.
            if at_goal and kwargs.get("seed_q_rad") is None:
                q = list(flipped)
            else:
                q = list(straight)
            return {
                "q_command": q,
                "q_solved": [value - 0.05 for value in q],
                "ok": True,
                "pos_err": 0.001,
                "ori_err": 0.01,
            }

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.ik = _IK()
    found = mover._plan_cartesian_tool_move(
        [0.1] * 7,
        [0.1] * 7,
        (0.30, 0.0, 0.30),
        goal,
        None,
        must_move=True,
        stage_limited=False,
    )
    assert found["ok"] is True
    assert abs(found["samples"][-1][2] - 0.30) < 1e-9
    assert all(abs(sample[2] - 2.0) > 0.5 for sample in found["samples"])
    assert all(abs(sample[0] - 0.25) > 1e-6 for sample in found["samples"])


def test_pbvs_keeps_sag_compensation_through_the_endpoint():
    goal = (0.40, 0.0, 0.46)

    class _IK:
        def tip_xyz(self, q):
            return goal

        def fk_xyz_rpy(self, q):
            return [0.4, 0.0, 0.46], [0.0, 0.0, 0.0]

        def solve_loaded_tool_pose(self, commanded, measured, **kwargs):
            return {
                "q_command": [0.25] * 7,
                "q_solved": [0.20] * 7,
                "ok": True,
                "pos_err": 0.001,
                "ori_err": 0.01,
            }

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.ik = _IK()
    found = mover._plan_cartesian_tool_move(
        [0.1] * 7,
        [0.1] * 7,
        (0.30, 0.0, 0.30),
        goal,
        None,
        must_move=True,
        stage_limited=False,
    )
    assert found["ok"] is True
    assert abs(found["samples"][-1][0] - 0.25) < 1e-9
    assert all(abs(sample[0] - 0.20) > 1e-6 for sample in found["samples"])


def test_coarse_pbvs_rejects_a_fingertip_that_stops_short_in_z():
    class _IK:
        def tip_xyz(self, q):
            return (0.40, 0.0, 0.30)

        def solve_loaded_tool_pose(self, commanded, measured, **kwargs):
            return {
                "q_command": [0.2] * 7,
                "q_solved": [0.2] * 7,
                "pos_err": 0.001,
                "ori_err": 0.01,
            }

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    mover = ArmMover(telemetry=lambda: {}, confirm=True, pbvs_first=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    found = mover._plan_cartesian_tool_move(
        [0.1] * 7,
        [0.1] * 7,
        (0.20, 0.0, 0.20),
        (0.40, 0.0, 0.47),
        lambda *args, **kwargs: None,
        must_move=False,
        stage_limited=False,
    )
    assert found["ok"] is False
    assert abs(found["tip_miss_m"][2] - 0.17) < 1e-9
    assert "Z" in mover._pbvs_search_failure("PBVS", found)


def test_service_coarse_lifts_before_advancing_into_the_wall(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)
    goals = []
    tip = [0.50, 0.0, 0.30]
    button = (0.60, 0.0, 0.45)

    class _Controller:
        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            goals.append(tuple(float(value) for value in goal))
            tip[:] = [float(value) for value in goal]
            return True, "ok"

    class _IK:
        def tip_xyz(self, q, offset=(0.205, 0.0, 0.0)):
            return tuple(tip)

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": list(button)},
        confirm=True,
        log=lambda _m: None,
    )
    mover._motion_backend = "service"
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._pbvs_once({"target": "button_1", "target_torso_m": list(button)})
    command_x = coarse_x_command(button[0])
    assert goals == [(0.50, 0.0, 0.45), (command_x, 0.0, 0.45)]
    assert goals[0][0] < goals[1][0] < coarse_x_ceiling(button[0])
    assert "先抬高" in mover.message

    goals.clear()
    tip[:] = [0.50, 0.0, 0.45]
    mover._pbvs_once({"target": "button_1", "target_torso_m": list(button)})
    assert goals == [(command_x, 0.0, 0.45)]


def test_sequence_pbvs_is_one_cartesian_shot_onto_the_button():
    moves = []

    class _Controller:
        def move_cartesian(self, arm, goal, rpy, seconds):
            moves.append((arm, tuple(goal)))
            return True, "ok"

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def tip_xyz(self, q):
            return (0.20, 0.0, 0.20)

    mover = ArmMover(telemetry=lambda: {}, confirm=True, direct_pbvs_poke=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    assert mover._sequence_pbvs_once("button_2", (0.42, 0.00, 0.20)) is True
    assert moves == [("right", (0.42, 0.00, 0.20))]
    assert mover.phase == "direct_pbvs_complete"


def test_sequence_service_lifts_before_the_forward_shot(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)
    moves = []

    class _Controller:
        def move_cartesian(self, arm, goal, rpy, seconds):
            moves.append(tuple(float(value) for value in goal))
            return True, "ok"

        def measured_arm_rad(self, arm):
            return [0.1] * 7

    class _IK:
        def tip_xyz(self, q):
            return (0.20, 0.0, 0.05)

    mover = ArmMover(telemetry=lambda: {}, confirm=True, direct_pbvs_poke=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover.ik = _IK()
    assert mover._sequence_pbvs_once("button_2", (0.42, 0.00, 0.20)) is True
    assert moves == [(0.20, 0.0, 0.20), (0.42, 0.00, 0.20)]
    assert "先抬高" in mover.message


def test_sequence_next_step_skips_the_current_button_and_clears_the_panel():
    calls = []
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", tuple(deg)))

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            calls.append(("clear", tuple(float(value) for value in goal)))
            return True, "ok"

        def release(self, seconds=1.2):
            calls.append("release")

    class _IK:
        def tip_xyz(self, q):
            return (0.50, 0.08, 0.42)

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append("drop")
    mover.set_button_sequence(["button_1", "button_2"])

    def shoot(name, goal):
        calls.append(("shoot", name))
        if name == "button_1":
            assert "下一步" in mover.advance_sequence()
            mover._raise_if_sequence_next()
        mover.phase = "contact"
        return True

    mover._run_hybrid_button = shoot
    mover._run_button_sequence()
    assert [item for item in calls if isinstance(item, tuple) and item[0] == "shoot"] == [
        ("shoot", "button_1"),
        ("shoot", "button_2"),
    ]
    assert [item for item in calls if isinstance(item, tuple) and item[0] == "clear"] == [
        ("clear", sequence_clear_goal((0.50, 0.08, 0.42)))
    ]
    assert ("ramp", REST_DEG) in calls
    assert mover.phase == "released"
    assert "释放" in mover.message


def test_sequence_next_on_the_last_button_lowers_and_releases():
    calls = []
    buttons = {"button_1": [0.40, 0.10, 0.20]}

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", tuple(deg)))

        def release(self, seconds=1.2):
            calls.append("release")

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append("drop")
    mover.set_button_sequence(["button_1"])

    def shoot(name, goal):
        calls.append(name)
        assert "释放" in mover.advance_sequence()
        mover._raise_if_sequence_next()
        return True

    mover._run_hybrid_button = shoot
    mover._run_button_sequence()
    assert calls[0] == "button_1"
    assert ("ramp", REST_DEG) in calls
    assert "release" in calls
    assert "drop" in calls
    assert "clear" not in calls
    assert mover.phase == "released"
    assert mover._control_stage == "RELEASED"


def test_sequence_next_after_a_missed_contact_resumes_the_following_button():
    calls = []
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", tuple(deg)))

        def measured_arm_rad(self, arm):
            return [0.1] * 7

        def commanded_arm_rad(self, arm):
            return [0.1] * 7

        def move_cartesian(self, arm, goal, rpy, seconds):
            calls.append(("clear", tuple(float(value) for value in goal)))
            return True, "ok"

        def release(self, seconds=1.2):
            calls.append("release")

    class _IK:
        def tip_xyz(self, q):
            return (0.50, 0.08, 0.42)

    class _Thread:
        def is_alive(self):
            return True

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons, contact=False),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._thread = _Thread()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append("drop")
    mover.set_button_sequence(["button_1", "button_2"])
    missed = {"button_1": True}

    def shoot(name, goal):
        calls.append(("shoot", name))
        if missed.pop(name, False):
            mover.message = "button_1 右腕进近未碰到按键"
            return False
        mover.phase = "contact"
        return True

    mover._run_hybrid_button = shoot
    mover._run_button_sequence()
    assert [item for item in calls if isinstance(item, tuple) and item[0] == "shoot"] == [
        ("shoot", "button_1")
    ]
    assert mover._control_stage == "HOLD"
    assert "Button2" in mover.advance_sequence()
    mover._servo_once()
    assert [item for item in calls if isinstance(item, tuple) and item[0] == "shoot"] == [
        ("shoot", "button_1"),
        ("shoot", "button_2"),
    ]
    assert ("clear", sequence_clear_goal((0.50, 0.08, 0.42))) in calls
    assert mover.phase == "released"


def test_sequence_finish_stops_the_service_when_that_backend_is_active():
    stopped = []

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            return None

        def release(self, seconds=1.2):
            stopped.append("release")

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._motion_backend = "service"
    mover._stop_service_node = lambda: stopped.append("service")
    mover._drop_arm_sdk = lambda *_args, **_kwargs: stopped.append("sdk")
    mover._finish_sequence_and_release()
    assert stopped == ["service", "release"]
    assert mover.phase == "released"
    assert mover.controller is None
    assert mover._stop.is_set()


def test_button_sequence_stops_when_the_wrist_camera_does_not_see_contact():
    calls = []
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", arm, tuple(deg)))

    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons, contact=False),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.set_button_sequence(["button_1", "button_2"])
    mover._run_hybrid_button = (
        lambda name, goal: calls.append(name) or setattr(mover, "message", "button_1 右腕进近未碰到按键") or False
    )
    captured = []
    mover.on_screen_capture = lambda: captured.append(True)
    mover._run_button_sequence()
    assert calls == ["button_1"]
    assert captured == []
    assert "右腕" in mover.message


def test_dashboard_start_message_mentions_flag_in_demo():
    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    client = dash.app.test_client()
    posted = client.post("/command", json={"command": "start"}).get_json()
    assert posted["ok"]
    state = client.get("/state").get_json()
    assert state["arm"]["enabled"] is False
    assert "confirm-arm-motion" in (state["message"] or "")
