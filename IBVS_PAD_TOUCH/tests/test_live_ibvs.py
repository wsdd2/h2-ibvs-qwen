import json

import numpy as np

from ibvs_pad_touch.perception.tip_tracker import StableTipTracker
from ibvs_pad_touch.perception.yolo_touch import YoloMask, YoloPoint, YoloTouch
from ibvs_pad_touch.types import CameraPoint
from ibvs_pad_touch.web.live_ibvs import (
    LiveIbvsAnnotator,
    _debounce_side_contact,
    _model_status_lines,
    pose_button_from_memory,
    remember_pose_buttons,
    _preview_line,
    camera_xyz_to_torso,
    discover_handeye_json,
    load_eye_to_hand_transform,
    hold_last_target,
    snapshot_good_target,
)


def test_side_contact_requires_a_continuous_confirmation_window():
    confirmed, since = _debounce_side_contact(True, True, None, 10.0)
    assert confirmed is False and since == 10.0
    confirmed, since = _debounce_side_contact(True, True, since, 10.20)
    assert confirmed is False
    confirmed, since = _debounce_side_contact(True, True, since, 10.31)
    assert confirmed is True
    confirmed, since = _debounce_side_contact(True, False, since, 10.32)
    assert confirmed is False and since is None


def test_eye_to_hand_transform_maps_camera_optical_axis_into_torso(tmp_path):
    transform = [
        [0.0440311, 0.0188880, 0.9988516, 0.1153953],
        [-0.9964719, 0.0723367, 0.0425583, 0.0310793],
        [-0.0714498, -0.9972014, 0.0220064, 0.3731118],
        [0.0, 0.0, 0.0, 1.0],
    ]
    path = tmp_path / "eye_to_hand_20260924_101819.json"
    path.write_text(
        json.dumps(
            {
                "calibration_method": "eye_to_hand",
                "reference_frame": "torso_link",
                "context": {"camera_info": {"serial": "347622074029"}},
                "cam2reference": {"transform": transform},
                "quality": {
                    "pass": True,
                    "translation_residual_norm_mean_m": 0.008,
                    "rotation_error_deg_mean": 0.82,
                },
            }
        ),
        encoding="utf-8",
    )
    loaded, status = load_eye_to_hand_transform(path)
    assert loaded is not None and status["pass"] is True
    origin = camera_xyz_to_torso((0.0, 0.0, 0.0), torso_T_camera=loaded)
    ahead = camera_xyz_to_torso((0.0, 0.0, 1.0), torso_T_camera=loaded)
    assert np.allclose(origin, [0.1153953, 0.0310793, 0.3731118], atol=1e-7)
    assert np.allclose(np.subtract(ahead, origin), [0.9988516, 0.0425583, 0.0220064], atol=1e-7)
    annotator = LiveIbvsAnnotator(
        eye_to_hand_json=path,
        eye_to_hand_x_bias_m=0.03,
        eye_to_hand_y_bias_m=-0.015,
        eye_to_hand_z_bias_m=0.025,
    )
    corrected = camera_xyz_to_torso(
        (0.0, 0.0, 0.0),
        torso_T_camera=annotator.torso_T_camera,
    )
    assert np.allclose(corrected, [0.1453953, 0.0160793, 0.3981118], atol=1e-7)
    assert annotator.eye_to_hand["torso_x_bias_m"] == 0.03
    assert annotator.eye_to_hand["torso_y_bias_m"] == -0.015
    assert annotator.eye_to_hand["torso_z_bias_m"] == 0.025


def test_seg_and_pose_status_have_separate_hud_rows():
    touch = YoloTouch(
        panel_corners=[(1.0, 1.0), (10.0, 1.0), (10.0, 6.0), (1.0, 6.0)],
        panel_conf=0.91,
        masks=[YoloMask("gripper", 0.82, [(1.0, 1.0), (3.0, 1.0), (2.0, 3.0)])],
        points={"gripper_tip": YoloPoint("gripper_tip", 2.0, 2.0, 0.77)},
    )
    lines = _model_status_lines(touch)
    assert len(lines) == 2
    assert lines[0].startswith("SEG ") and "panel:0.91" in lines[0]
    assert lines[1].startswith("POSE ") and "tip:0.77" in lines[1]


def test_ibvs_after_pbvs_keeps_the_pose_button_when_detection_jumps():
    original = CameraPoint(u=360.0, v=200.0, z_m=0.62, xyz_m=(0.04, -0.02, 0.62), source="pose")
    memory = remember_pose_buttons({}, {"button_1": original, "button_2": original}, freeze=False)
    jumped = CameraPoint(u=80.0, v=40.0, z_m=0.21, xyz_m=(0.2, 0.2, 0.21), source="pose")
    frozen = remember_pose_buttons(memory, {"button_1": jumped}, freeze=True)
    locked = pose_button_from_memory(frozen, "button_1")
    assert locked is not None
    assert (locked.u, locked.v) == (360.0, 200.0)
    assert locked.xyz_m == (0.04, -0.02, 0.62)
    assert locked.source == "pose_locked"
    scale = CameraPoint(u=10.0, v=10.0, source="scale")
    kept = remember_pose_buttons(memory, {"button_1": scale}, freeze=False)
    assert pose_button_from_memory(kept, "button_1").u == 360.0
    moved = remember_pose_buttons(memory, {"button_1": jumped}, freeze=False)
    assert pose_button_from_memory(moved, "button_1").u == 360.0
    drifted = CameraPoint(u=368.0, v=204.0, z_m=0.62, xyz_m=(0.041, -0.019, 0.62), source="pose")
    followed = remember_pose_buttons(memory, {"button_1": drifted}, freeze=False)
    assert pose_button_from_memory(followed, "button_1").u == 368.0


def test_hold_last_target_keeps_button_after_panel_loss():
    good = {
        "panel_visible": True,
        "target_uv": [360.0, 200.0],
        "target_xyz_m": [0.05, 0.02, 0.70],
        "pbvs_error_cam_m": [0.04, 0.03, 0.12],
        "dx": 0.002,
        "dy": 0.001,
        "dz": 0.0,
    }
    last = snapshot_good_target(good, 10.0)
    lost = {
        "panel_visible": False,
        "target_uv": None,
        "target_xyz_m": None,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "message": "panel_not_found",
    }
    held = hold_last_target(lost, last, 11.5)
    assert held["using_last_target"] is True
    assert held["target_uv"] == [360.0, 200.0]
    assert held["dz"] == 0.0
    assert held["dx"] == 0.002
    expired = hold_last_target(lost, last, 14.0)
    assert expired["using_last_target"] is False


def test_annotator_uses_fk_virtual_tip_without_visual_gripper():
    annotator = LiveIbvsAnnotator()
    annotator.set_ee_pose([0.55, 0.08, 0.35], [0.0, 0.0, 0.0])
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["virtual_tip_xyz_m"] is not None
    assert telem["tip_source"] in {"", "virtual_fk"}
    assert telem["tip_visible"] is False


def test_annotator_empty_chest_still_has_deltas():
    annotator = LiveIbvsAnnotator()
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    out = annotator.annotate("chest", frame)
    assert out.shape == frame.shape
    telem = annotator.telemetry()
    assert telem["chest_serial"] == "347622074029"
    assert telem["left_serial"] == "233722073797"
    assert telem["wrist_serial"] == "233722073797"
    for key in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"):
        assert key in telem


def test_side_range_uses_the_panel_edge_when_the_jaw_is_a_depth_hole():
    from ibvs_pad_touch.web.live_ibvs import _side_range_m

    depth = np.zeros((48, 48), dtype=np.float32)
    depth[30:40, 30:40] = 0.42
    assert _side_range_m(None, (8.0, 8.0), (34.0, 34.0)) is None
    assert abs(_side_range_m(depth, (8.0, 8.0), (34.0, 34.0)) - 0.42) < 1e-6


def test_preview_line_does_not_turn_chinese_into_question_marks():
    text = _preview_line("胸视没有真实尖端，FK 不驱动")
    assert "?" not in text
    assert text.isascii()
    assert "visual tip" in text
    assert _preview_line("侧视伺服 间隙 36px") == "side IBVS gap 36px"


def test_side_view_draws_gripper_mask_when_the_panel_is_absent():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask, YoloPoint, YoloTouch

    class _GripperOnly:
        def predict(self, bgr):
            return YoloTouch(
                masks=[
                    YoloMask(
                        "gripper",
                        0.86,
                        [(400.0, 40.0), (520.0, 50.0), (500.0, 220.0), (390.0, 200.0)],
                    )
                ],
                points={"gripper_tip": YoloPoint("gripper_tip", 460.0, 48.0, 0.22)},
            )

    annotator = LiveIbvsAnnotator(detector=_GripperOnly())
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    out = annotator.annotate("d435i", frame)
    assert annotator.telemetry()["side_measured"] is False
    assert annotator.telemetry()["left_contact"] is False
    assert int(out.sum()) != int(frame.sum())


def test_side_view_does_not_place_buttons():
    from ibvs_pad_touch.perception.yolo_touch import YoloPoint, YoloTouch

    class _PanelAndButtons:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(120.0, 140.0), (500.0, 150.0), (490.0, 320.0), (130.0, 310.0)],
                panel_conf=0.9,
                points={
                    "button_1": YoloPoint("button_1", 430.0, 180.0, 0.95),
                    "button_2": YoloPoint("button_2", 450.0, 180.0, 0.95),
                    "button_3": YoloPoint("button_3", 440.0, 210.0, 0.95),
                },
            )

    annotator = LiveIbvsAnnotator(detector=_PanelAndButtons())
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    annotator.annotate("d435i", frame)
    assert annotator.ctrl.last_localization is None or not annotator.ctrl.last_localization.buttons


def test_side_view_draws_yolo_before_raise_without_contact():
    from ibvs_pad_touch.perception.yolo_touch import YoloPoint, YoloTouch

    class _SideYolo:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(120.0, 140.0), (500.0, 150.0), (490.0, 320.0), (130.0, 310.0)],
                panel_conf=0.9,
                points={"gripper_tip": YoloPoint("gripper_tip", 80.0, 200.0, 0.8)},
            )

    annotator = LiveIbvsAnnotator(detector=_SideYolo())
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    out = annotator.annotate("d435i", frame)
    assert annotator.telemetry()["side_measured"] is False
    assert annotator.telemetry()["left_contact"] is False
    assert annotator.telemetry()["wrist_panel_visible"] is True
    assert int(out.sum()) != int(frame.sum())


def test_side_view_ignores_gripper_until_raise():
    import cv2

    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (160, 180), (460, 310), (208, 218, 214), -1)
    cv2.rectangle(image, (195, 208), (370, 280), (32, 36, 38), -1)
    cv2.rectangle(image, (520, 210), (610, 250), (16, 16, 16), -1)
    annotator = LiveIbvsAnnotator()
    annotator.annotate("d435i", image)
    assert annotator.telemetry()["side_measured"] is False
    assert annotator.telemetry()["left_contact"] is False
    annotator.set_side_ready(lambda: True)
    annotator._last_wrist_ts = 0.0
    annotator.annotate("d435i", image)
    assert annotator.telemetry()["side_measured"] is True


def test_left_overlay_is_side_view_not_ibvs():
    annotator = LiveIbvsAnnotator()
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    out = annotator.annotate("d435i", frame)
    assert out.shape == frame.shape
    telem = annotator.telemetry()
    assert telem["wrist_panel_visible"] is False
    assert telem["wrist_panel_partial"] is False
    assert telem["left_contact"] is False
    assert telem["handeye"] == {}


def test_discover_handeye_returns_path_or_none():
    path = discover_handeye_json("20260920")
    assert path is None or path.name.startswith("eye_in_hand_20260920")


def test_chest_camera_point_maps_to_torso_with_40cm_height():
    torso = camera_xyz_to_torso((0.10, 0.20, 0.70), camera_height_m=0.40)
    assert torso == (0.725, -0.10, 0.20)
    origin = camera_xyz_to_torso((0.0, 0.0, 0.0))
    assert origin == (0.025, 0.0, 0.40)
    ahead = camera_xyz_to_torso((0.0, 0.0, 1.0))
    assert ahead[2] == origin[2]
    assert ahead[0] == origin[0] + 1.0


def test_tip_tracker_requires_stable_auto_frames_and_rejects_jump():
    tracker = StableTipTracker(confirm_frames=3, max_jump_px=30)
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    assert tracker.update(frame, CameraPoint(u=80, v=70)) is None
    assert tracker.update(frame, CameraPoint(u=82, v=69)) is None
    confirmed = tracker.update(frame, CameraPoint(u=81, v=71))
    assert confirmed is not None
    assert tracker.update(frame, CameraPoint(u=145, v=20)) is None
    assert tracker.status()["reason"] == "auto_jump_rejected"


def test_manual_tip_seed_is_immediately_available_then_clearable():
    tracker = StableTipTracker()
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    tracker.seed(75, 65)
    tip = tracker.update(frame, None)
    assert tip is not None
    assert tip.uv == (75.0, 65.0)
    assert tracker.status()["mode"] == "manual_flow"
    tracker.clear()
    assert tracker.update(frame, None) is None
