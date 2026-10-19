import numpy as np

from ibvs_pad_touch.servo.controller import IBVSController
from ibvs_pad_touch.perception.features import sample_depth
from ibvs_pad_touch.servo.delta import gate_xy_then_depth, image_xy_delta, tip_to_button_delta
from ibvs_pad_touch.servo.interaction import eye_in_hand_close_align_step, eye_in_hand_ibvs_step
from ibvs_pad_touch.sim.camera_view import ChestScene
from ibvs_pad_touch.types import CameraPoint, Intrinsics


def _K() -> Intrinsics:
    return Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)


def test_eye_in_hand_error_moves_the_camera_toward_the_button():
    right = eye_in_hand_ibvs_step(
        (382.0, 240.0),
        (320.0, 240.0),
        0.20,
        0.20,
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    assert right["camera_step_m"][0] > 0.0
    assert abs(right["camera_step_m"][1]) < 1e-6
    far = eye_in_hand_ibvs_step(
        (320.0, 240.0),
        (320.0, 240.0),
        0.25,
        0.10,
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    assert far["camera_step_m"][2] > 0.0
    assert abs(far["camera_step_m"][0]) < 1e-6
    assert far["arrived"] is False
    arrived = eye_in_hand_ibvs_step(
        (322.0, 241.0),
        (320.0, 240.0),
        0.200,
        0.201,
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    assert arrived["arrived"] is True


def test_close_range_alignment_stays_in_the_image_plane():
    aligned = eye_in_hand_close_align_step(
        (382.0, 240.0),
        (320.0, 240.0),
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    step = aligned["camera_step_m"]
    assert step[0] > 0.0
    assert abs(step[1]) < 1e-9
    assert step[2] == 0.0
    assert aligned["pixel_err"] > 8.0


def test_tip_right_of_button_moves_left():
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.30).with_depth(K, 0.30)
    tip = CameraPoint(u=400, v=240, z_m=0.30).with_depth(K, 0.30)
    delta, reason = tip_to_button_delta(button, tip, K)
    assert reason == "ok"
    assert delta[0] < 0


def test_tip_depth_prefers_near_foreground_over_panel_background():
    depth = np.full((25, 25), 0.80, dtype=np.float32)
    depth[9:16, 9:16] = 0.45
    depth[11:14, 11:14] = 0.0
    z = sample_depth(depth, 12, 12, radius=10, percentile=5)
    assert z is not None
    assert abs(z - 0.45) < 1e-5


def test_tip_too_far_moves_forward():
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.40).with_depth(K, 0.40)
    tip = CameraPoint(u=320, v=240, z_m=0.20).with_depth(K, 0.20)
    delta, _ = tip_to_button_delta(button, tip, K)
    assert delta[2] > 0


def test_localize_without_tip_does_not_invent_delta():
    ctrl = IBVSController.from_configs()
    K = _K()
    scene = ChestScene(ctrl.panel, K)
    cmd = ctrl.step(
        scene.render_bgr(1, draw_tip=False),
        None,
        K,
        target=1,
        observation=scene.observe(1, include_tip=False),
        detect_tip=False,
    )
    assert cmd.valid
    assert cmd.phase == "LOCALIZED"
    assert cmd.delta_xyz == (0.0, 0.0, 0.0)
    assert cmd.reached is False


def test_tip_without_depth_uses_xy_only_ibvs():
    ctrl = IBVSController.from_configs()
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.30).with_depth(K, 0.30)
    tip = CameraPoint(u=400, v=240, source="dex1_tip")
    cmd = ctrl.command_from_points(button, tip, K)
    assert cmd.valid
    assert cmd.phase == "ALIGN_XY"
    assert cmd.dx < 0.0
    assert cmd.dy == 0.0
    assert cmd.dz == 0.0
    assert cmd.tip_uv == (400.0, 240.0)
    assert cmd.message == "align_switch_2d_no_depth"


def test_image_xy_delta_never_moves_forward():
    K = _K()
    button = CameraPoint(u=320, v=180)
    tip = CameraPoint(u=400, v=300)
    delta = image_xy_delta(button, tip, K, nominal_depth_m=0.60)
    assert delta[0] < 0
    assert delta[1] < 0
    assert delta[2] == 0.0


def test_xy_then_depth():
    dx, dy, dz, phase = gate_xy_then_depth(0.01, -0.02, 0.006, xy_err=0.08, xy_tol_m=0.015)
    assert phase == "ALIGN_XY"
    assert (dx, dy) == (0.01, -0.02)
    assert dz == 0.0
    dx, dy, dz, phase = gate_xy_then_depth(0.01, -0.02, 0.006, xy_err=0.005, xy_tol_m=0.015)
    assert phase == "APPROACH_Z"
    assert (dx, dy, dz) == (0.0, 0.0, 0.006)


def test_controller_aligns_xy_before_depth():
    ctrl = IBVSController.from_configs()
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.40).with_depth(K, 0.40)
    tip = CameraPoint(u=400, v=200, z_m=0.20).with_depth(K, 0.20)
    cmd = ctrl.command_from_points(button, tip, K)
    assert cmd.phase == "ALIGN_XY"
    assert cmd.dz == 0.0
    assert cmd.dx != 0.0 or cmd.dy != 0.0


def test_controller_approaches_panel_after_xy_aligned():
    ctrl = IBVSController.from_configs()
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.40).with_depth(K, 0.40)
    tip = CameraPoint(u=320, v=240, z_m=0.20).with_depth(K, 0.20)
    cmd = ctrl.command_from_points(button, tip, K)
    assert cmd.phase == "APPROACH_Z"
    assert cmd.dx == 0.0 and cmd.dy == 0.0
    assert cmd.dz > 0.0


def test_command_zeros_when_tip_on_button():
    ctrl = IBVSController.from_configs()
    K = _K()
    button = CameraPoint(u=320, v=240, z_m=0.30).with_depth(K, 0.30)
    last = None
    for _ in range(int(ctrl.cfg["stable_frames"])):
        last = ctrl.command_from_points(button, button, K)
    assert last is not None
    assert last.reached
    assert last.delta_xyz == (0.0, 0.0, 0.0)
    assert last.phase == "HOLD"


def test_chest_scene_tip_reaches_each_button():
    K = _K()
    for target in (1, 2, 3):
        ctrl = IBVSController.from_configs()
        scene = ChestScene(ctrl.panel, K)
        reached = False
        for _ in range(400):
            obs = scene.observe(target, include_tip=True)
            cmd = ctrl.step(np.zeros((K.height, K.width, 3), np.uint8), None, K, target=target, observation=obs)
            scene.apply_delta(cmd.dx, cmd.dy, cmd.dz)
            if cmd.reached:
                reached = True
                break
        assert reached, f"tip failed to reach button {target}"
