import math

import numpy as np

from ibvs_pad_touch.servo.virtual_tip import (
    DEFAULT_TIP_OFFSET_EE_M,
    apply_ee_offset,
    chest_camera_to_torso,
    choose_control_tip,
    project_camera_xyz,
    snap_visual_to_virtual_depth,
    torso_to_chest_camera,
    virtual_tip_from_ee,
    visual_bias_cam_m,
    wrist_cam_delta_to_torso,
)
from ibvs_pad_touch.types import CameraPoint, Intrinsics
from ibvs_pad_touch.web.live_ibvs import camera_xyz_to_torso


def _K() -> Intrinsics:
    return Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)


def test_torso_camera_round_trip_matches_live_overlay():
    cam = (0.10, 0.20, 0.70)
    torso = chest_camera_to_torso(cam, camera_height_m=0.40)
    assert torso == camera_xyz_to_torso(cam, camera_height_m=0.40)
    assert torso == (0.725, -0.10, 0.20)
    back = torso_to_chest_camera(torso, camera_height_m=0.40)
    assert all(abs(back[i] - cam[i]) < 1e-9 for i in range(3))


def test_identity_pose_puts_tip_along_torso_x():
    tip = apply_ee_offset((0.20, 0.05, 0.90), (0.0, 0.0, 0.0), DEFAULT_TIP_OFFSET_EE_M)
    assert abs(tip[0] - (0.20 + DEFAULT_TIP_OFFSET_EE_M[0])) < 1e-9
    assert abs(tip[1] - 0.05) < 1e-9


def test_virtual_tip_projects_in_front_of_chest_camera():
    K = _K()
    tip = virtual_tip_from_ee((0.55, 0.08, 0.35), (0.0, 0.0, 0.0), K)
    assert tip.source == "virtual_fk"
    assert tip.xyz_m is not None
    assert tip.xyz_m[2] > 0.4
    uv = project_camera_xyz(tip.xyz_m, K)
    assert uv is not None
    assert abs(uv[0] - tip.u) < 1e-6


def test_click_on_panel_is_ignored_for_control():
    K = _K()
    virtual = CameraPoint(u=100, v=400, z_m=0.5, xyz_m=(0.0, 0.0, 0.5), source="virtual_fk")
    on_panel = CameraPoint(u=320, v=240, source="dex1_tip_tracked")
    panel = [(200.0, 160.0), (440.0, 160.0), (440.0, 320.0), (200.0, 320.0)]
    chosen = choose_control_tip(
        virtual=virtual,
        manual=on_panel,
        K=K,
        tracker_manual=True,
        panel_corners=panel,
    )
    assert chosen is virtual


def test_manual_click_wins_and_auto_is_ignored():
    K = _K()
    virtual = CameraPoint(u=100, v=100, z_m=0.5, xyz_m=(0.0, 0.0, 0.5), source="virtual_fk")
    manual = CameraPoint(u=200, v=180, source="dex1_tip_tracked")
    chosen = choose_control_tip(virtual=virtual, manual=manual, K=K, tracker_manual=True)
    assert chosen is not None
    assert chosen.u == 200
    assert chosen.xyz_m is not None
    ignored = choose_control_tip(virtual=virtual, manual=manual, K=K, tracker_manual=False)
    assert ignored is virtual


def test_visual_bias_is_camera_frame_difference():
    virtual = CameraPoint(u=10, v=10, xyz_m=(0.0, 0.0, 0.50), source="virtual_fk")
    visual = CameraPoint(u=20, v=20, xyz_m=(0.02, -0.01, 0.48), source="manual")
    bias = visual_bias_cam_m(visual, virtual)
    assert bias is not None
    assert all(abs(a - b) < 1e-9 for a, b in zip(bias, (0.02, -0.01, -0.02)))


def test_snap_keeps_click_pixel_and_fk_depth():
    K = _K()
    virtual = CameraPoint(u=100, v=100, z_m=0.60, xyz_m=K.backproject(100, 100, 0.60), source="virtual_fk")
    click = CameraPoint(u=140, v=90, source="manual")
    snapped = snap_visual_to_virtual_depth(click, virtual, K)
    assert snapped.uv == (140.0, 90.0)
    assert snapped.z_m == 0.60


def test_wrist_delta_follows_hand_rotation():
    # EE aligned with torso, camera = hand. +X cam should stay +X hand/torso.
    dx, dy, dz = wrist_cam_delta_to_torso((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), np.eye(3))
    assert math.isclose(dx, 0.01, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dz, 0.0, abs_tol=1e-9)
