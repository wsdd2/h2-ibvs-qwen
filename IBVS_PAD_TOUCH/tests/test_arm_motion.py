import math
import sys
import time

from ibvs_pad_touch.web.arm_motion import (
    ArmMover,
    quintic_joint_seconds,
    slerp_rpy,
    RIGHT_ARM_JOINTS,
    DIRECT_EE_RPY_DEG,
    JOG_STEP_M,
    PBVS_STANDOFF_M,
    POKE_EXTRA_M,
    LEFT_HANG_DEG,
    RIGHT_HANG_DEG,
    LEFT_REST_DEG,
    LEFT_SIDE_VIEW_DEG,
    FOREARM_LEVEL_DEG,
    READY_DEG,
    level_right_waypoints,
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


def test_direct_pbvs_relaxes_x_one_step_when_yz_stops_closing():
    button = (0.60, -0.12, 0.44)
    far = direct_pbvs_motion_goal((0.20, -0.02, 0.20), button)
    assert far == (0.25, -0.12, 0.44)
    close = direct_pbvs_motion_goal((0.58, -0.10, 0.44), button)
    assert abs(close[0] - (button[0] - 0.08)) < 1e-9
    assert close[1:] == button[1:]
    relaxed = direct_pbvs_motion_goal((0.58, -0.10, 0.44), button, x_clear_m=0.06)
    assert abs(relaxed[0] - (button[0] - 0.06)) < 1e-9
    assert abs(relax_x_clearance(0.08, 0.04, 0.04) - 0.06) < 1e-9
    assert relax_x_clearance(0.08, 0.04, 0.03) == 0.08
    assert relax_x_clearance(0.02, 0.04, 0.04) == 0.0
    approach = direct_pbvs_motion_goal((0.40, -0.12, 0.44), button)
    assert approach == button


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


def test_pbvs_coarse_preserves_standoff_before_ibvs():
    delta, done = pbvs_coarse_delta((0.01, -0.02, 0.45))
    assert not done
    assert delta == (0.01, -0.02, 0.45 - PBVS_STANDOFF_M)
    delta, done = pbvs_coarse_delta((0.01, -0.01, PBVS_STANDOFF_M + 0.005))
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
    assert READY_DEG[1] == -25.0
    assert READY_DEG[4] == 0.0
    assert WRIST_VIEW_DEG == READY_DEG
    assert WRIST_VIEW_DEG[1] == -25.0
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
    assert [item[0] for item in calls] == ["left", "right"]
    assert calls[0][1] == list(LEFT_SIDE_VIEW_DEG)
    assert calls[1][1] == list(READY_DEG)
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
    message = mover.set_motion_backend("arm_sdk")
    assert "已停止运控服务" in message
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
    assert controller.ramps == [
        ("left", LEFT_HANG_DEG, 8.0),
        ("right", RIGHT_HANG_DEG, 8.0),
    ]
    assert LEFT_HANG_DEG[3] == 90.0
    assert RIGHT_HANG_DEG[3] == 90.0
    assert LEFT_HANG_DEG[1] > 0.0
    assert RIGHT_HANG_DEG[1] < 0.0
    assert controller.released is False
    assert mover.controller is controller
    assert "自然下垂" in mover.message
    assert mover._can_switch_backend()


def test_right_level_raises_only_the_right_forearm():
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
    assert "右臂" in message
    assert "左臂" in message
    assert mover._level_right.is_set()
    mover._execute_level_right()
    assert controller.ramps == [("right", FOREARM_LEVEL_DEG, 8.0)]
    assert FOREARM_LEVEL_DEG[0] < 0.0
    assert FOREARM_LEVEL_DEG[3] > 0.0
    assert controller.released is False
    assert "左臂未动" in mover.message


def test_level_pose_stays_in_front_of_the_chest_camera():
    assert FOREARM_LEVEL_DEG[0] > -75.0
    assert FOREARM_LEVEL_DEG[1] <= -40.0
    assert FOREARM_LEVEL_DEG[3] < 60.0
    tucked = level_right_waypoints((-90.0, -15.0, 0.0, 85.0, 0.0, 0.0, 0.0))
    assert tucked == [FOREARM_LEVEL_DEG]
    rest_clear, goal = level_right_waypoints((10.0, -8.0, 0.0, 15.0, 0.0, 0.0, 0.0))
    assert rest_clear[0] == 10.0
    assert rest_clear[1] <= -40.0
    assert rest_clear[3] == 15.0
    assert goal == FOREARM_LEVEL_DEG


def test_right_level_refuses_without_confirm():
    mover = ArmMover(telemetry=lambda: {}, confirm=False, log=lambda _m: None)
    mover._service_node_probe = lambda: False
    message = mover.level_right_arm()
    assert "confirm-arm-motion" in message
    assert not mover._level_right.is_set()


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
    assert controller.ramps == [
        ("left", LEFT_REST_DEG),
        ("right", REST_DEG),
    ]
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
    assert targets[0][0] == direct_pbvs_motion_goal((0.20, -0.10, 0.30), (0.23, -0.11, 0.32))
    assert (0.20, -0.10, 0.30) in xyzs
    assert any(0.15 < item[0] < 0.20 for item in xyzs)
    assert all(item[0] <= 0.20 + 1e-9 for item in xyzs)
    assert len(targets) < 40
    assert sent[0][0] == [0.21] * 7
    assert sent and sent[0][1] >= 2.0
    assert mover.status()["control_stage"] == "HOLD"
    assert mover.status()["ibvs_enabled"] is False
    assert "实测到位" in mover.message
    mover._servo_once()
    assert len(sent) == 1


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
    xs = [call[1][0] for call in calls]
    assert xs[0] < 0.6029
    assert abs(xs[1] - xs[0] - 0.02) < 1e-6
    assert abs(xs[2] - xs[1] - 0.02) < 1e-6
    assert xs[-1] == 0.6029
    assert "实测未到位" in mover.message
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
    assert sent_targets[0] == (0.10, -0.10, 0.46)
    assert (0.05, -0.20, -0.30) in sent_targets
    assert all(item[0] <= 0.10 + 1e-6 for item in sent_targets)
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


def test_calibrated_pbvs_goes_direct_then_switches_to_dual_ibvs():
    targets = []
    applied = []
    raised = []
    tip = [0.0, 0.05, -0.05]
    old_axis_map = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]

    class _IK:
        def tip_xyz(self, q):
            return list(tip)

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

    telem = {"target_torso_m": [0.10, 0.0, 0.0]}
    mover = ArmMover(
        telemetry=lambda: telem,
        confirm=True,
        pbvs_first=True,
        camera_R_torso=old_axis_map,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "PBVS_COARSE"
    mover._servo_once()
    assert targets[0][0] == (0.07, 0.0, 0.0)
    assert (0.0, 0.05, -0.05) in [item[0] for item in targets]
    assert targets[0][1] == tuple(math.radians(value) for value in DIRECT_EE_RPY_DEG)
    assert targets[0][2] == (0.205, 0.0, 0.0)
    assert applied
    assert applied[0][1] >= 1.2
    assert "运控位姿一次到位" in mover.message
    assert mover._control_stage == "PBVS_COARSE"
    tip[:] = [0.07, 0.0, 0.0]
    mover._servo_once()
    assert mover._control_stage == "RAISE_LEFT_FOR_IBVS"
    assert mover.phase == "pbvs_ready_raise_left"
    mover._servo_once()
    assert mover._control_stage == "DUAL_IBVS"
    assert mover.phase == "dual_ibvs_ready"
    assert mover.status()["left_side_ready"] is True
    assert raised == [("left", list(LEFT_SIDE_VIEW_DEG))]


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


def test_sequence_lift_backs_away_and_raises_the_jaw():
    goal = sequence_lift_goal((0.40, -0.02, 0.25))
    assert abs(goal[0] - 0.30) < 1e-9
    assert abs(goal[1] + 0.02) < 1e-9
    assert abs(goal[2] - 0.33) < 1e-9


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
    assert ramps == [FOREARM_LEVEL_DEG]
    assert mover._pbvs_source == "chest"
    assert mover._locked_buttons["button_1"] == (0.50, 0.10, 0.40)
    assert mover._locked_buttons["button_2"] == (0.52, 0.0, 0.41)
    assert "跳过" in mover.message


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
    assert queued == "组合启动：Button 2 → Button 1"
    assert mover._control_stage == "DIRECT_PBVS"


def test_button_sequence_lifts_between_presses_then_lowers_and_captures():
    calls = []

    class _Controller:
        def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
            calls.append(("ramp", arm, tuple(deg)))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, direct_pbvs_poke=True, log=lambda _m: None)
    mover.controller = _Controller()
    mover._locked_buttons = {
        "button_1": (0.40, 0.10, 0.20),
        "button_2": (0.42, 0.00, 0.20),
    }
    assert mover.set_button_sequence(["button_1", "button_2", "button_1", "nope"]) == [
        "button_1",
        "button_2",
        "button_1",
    ]

    def poke(telem, release=True):
        calls.append(("poke", telem["target"], release))
        mover.phase = "direct_pbvs_complete"

    def lift(goal):
        calls.append(("lift", tuple(goal)))
        return True

    captured = []
    mover._direct_pbvs_once = poke
    mover._move_tool_to = lift
    mover.on_screen_capture = lambda: captured.append(True)
    mover._run_button_sequence()
    assert [item[1] for item in calls if item[0] == "poke"] == ["button_1", "button_2", "button_1"]
    assert all(item[2] is False for item in calls if item[0] == "poke")
    lifts = [item[1] for item in calls if item[0] == "lift"]
    assert abs(lifts[0][2] - 0.28) < 1e-9
    assert abs(lifts[0][0] - 0.30) < 1e-9
    assert calls[-1] == ("ramp", "right", REST_DEG)
    assert captured == [True]
    assert mover.phase == "direct_pbvs_complete"


def test_dashboard_start_message_mentions_flag_in_demo():
    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    client = dash.app.test_client()
    posted = client.post("/command", json={"command": "start"}).get_json()
    assert posted["ok"]
    state = client.get("/state").get_json()
    assert state["arm"]["enabled"] is False
    assert "confirm-arm-motion" in (state["message"] or "")
