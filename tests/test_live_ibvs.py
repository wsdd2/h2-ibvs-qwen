import json

import cv2
import numpy as np

from ibvs_pad_touch.perception.overlay import draw_pose_buttons
from ibvs_pad_touch.perception.yellow_buttons import (
    YELLOW_TRACK_MISS_FRAMES,
    detect_red_bar,
    nearest_long_yellow_bar,
    yellow_blob_at,
)
from ibvs_pad_touch.perception.tip_tracker import StableTipTracker
from ibvs_pad_touch.perception.yolo_touch import YoloMask, YoloPoint, YoloTouch
from ibvs_pad_touch.types import CameraPoint, Intrinsics
from ibvs_pad_touch.types import Localization
from ibvs_pad_touch.web.live_ibvs import (
    LiveIbvsAnnotator,
    latch_wrist_finger_tip,
    fresh_chest_pose,
    fresh_chest_pose_torso,
    _debounce_side_contact,
    _model_status_lines,
    chest_control_under_occlusion,
    gripper_occludes_panel,
    wrist_depth_confirms_contact,
    hold_last_pose,
    hold_buttons_under_occlusion,
    pose_button_from_memory,
    remember_pose_buttons,
    _preview_line,
    camera_xyz_to_torso,
    discover_eye_in_hand_json,
    discover_handeye_json,
    load_eye_in_hand_transform,
    load_eye_to_hand_transform,
    hold_last_target,
    snapshot_good_target,
)


def test_fresh_chest_pose_uses_the_button_seen_this_frame():
    depth = np.full((30, 30), 0.50, dtype=np.float32)
    K = Intrinsics(fx=100.0, fy=100.0, cx=10.0, cy=10.0, width=30, height=30)
    touch = YoloTouch(
        points={
            "button_1": YoloPoint("button_1", 10.0, 10.0, 0.91),
            "button_2": YoloPoint("button_2", 10.0, 20.0, 0.88),
        }
    )
    seen = fresh_chest_pose_torso(touch, depth, K, None, target_name="button_1", conf_min=0.3)
    other = fresh_chest_pose_torso(
        YoloTouch(points={"button_2": YoloPoint("button_2", 10.0, 20.0, 0.88)}),
        depth,
        K,
        None,
        target_name="button_1",
        conf_min=0.3,
    )
    assert seen is not None
    assert other is None
    borrowed = fresh_chest_pose(
        YoloTouch(
            points={
                "button_1": YoloPoint("button_1", -20.0, -20.0, 0.9),
                "button_3": YoloPoint("button_3", 10.0, 20.0, 0.9),
            }
        ),
        depth,
        K,
        None,
        target_name="button_1",
    )
    assert borrowed is not None
    assert borrowed["uv"] == [-20.0, -20.0]
    assert borrowed["torso_m"] is None
    assert fresh_chest_pose_torso(
        YoloTouch(points={"button_1": YoloPoint("button_1", 10.0, 10.0, 0.1)}),
        depth,
        K,
        None,
        target_name="button_1",
        conf_min=0.3,
    ) is None
    assert fresh_chest_pose_torso(YoloTouch(), depth, K, None, target_name="button_1") is None
    seen_pixel = fresh_chest_pose(
        YoloTouch(points={"button_1": YoloPoint("button_1", 12.0, 20.0, 0.8)}),
        None,
        None,
        None,
        target_name="button_1",
    )
    assert seen_pixel["uv"] == [12.0, 20.0]
    assert seen_pixel["torso_m"] is None


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
    ok, detail = annotator.set_eye_to_hand_bias(0.03, -0.015, 0.03)
    assert ok and detail == ""
    replaced = camera_xyz_to_torso((0.0, 0.0, 0.0), torso_T_camera=annotator.torso_T_camera)
    assert np.allclose(replaced, [0.1453953, 0.0160793, 0.4031118], atol=1e-7)
    assert annotator.eye_to_hand["torso_z_bias_m"] == 0.03
    again, _detail = annotator.set_eye_to_hand_bias(0.03, -0.015, 0.03)
    assert again
    stacked = camera_xyz_to_torso((0.0, 0.0, 0.0), torso_T_camera=annotator.torso_T_camera)
    assert np.allclose(stacked, replaced, atol=1e-7)


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


def test_wrist_partial_panel_still_tracks_pose_buttons(monkeypatch):
    depth = np.full((480, 640), 0.22, dtype=np.float32)
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: depth)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.ctrl.set_target("button_1")
    panel = [(80.0, 80.0), (420.0, 80.0), (420.0, 320.0), (80.0, 320.0)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clear = YoloTouch(
        panel_corners=panel,
        panel_conf=0.9,
        points={"button_1": YoloPoint("button_1", 250.0, 180.0, 0.9)},
        masks=[YoloMask("gripper", 0.9, [(500.0, 20.0), (560.0, 20.0), (560.0, 80.0)])],
    )
    annotator._note_wrist_pbvs(frame, clear, CameraPoint(u=200.0, v=180.0))
    view = annotator.wrist_pbvs_view()
    assert view["occluded"] is False
    assert view["target_uv"] == [250.0, 180.0]
    assert abs(view["target_z_m"] - 0.22) < 1e-6
    assert view["tip_uv"] == [200.0, 180.0]
    assert abs(view["tip_z_m"] - 0.22) < 1e-6
    assert view["intrinsics"]["fx"] > 0.0
    covered = YoloTouch(
        panel_corners=panel,
        panel_conf=0.9,
        points={"button_1": YoloPoint("button_1", 40.0, 40.0, 0.9)},
        masks=[YoloMask("gripper", 0.9, [(100.0, 100.0), (360.0, 100.0), (360.0, 280.0), (100.0, 280.0)])],
    )
    annotator._note_wrist_pbvs(frame, covered, CameraPoint(u=300.0, v=220.0))
    followed = annotator.wrist_pbvs_view()
    assert followed["occluded"] is False
    assert followed["target_uv"] == [40.0, 40.0]
    assert abs(followed["target_z_m"] - 0.22) < 1e-6
    assert followed["tip_uv"] == [300.0, 220.0]
    partial = YoloTouch(
        panel_corners=None,
        points={"button_1": YoloPoint("button_1", 180.0, 90.0, 0.95)},
        masks=[YoloMask("gripper", 0.9, [(100.0, 100.0), (360.0, 100.0), (360.0, 280.0), (100.0, 280.0)])],
    )
    annotator._note_wrist_pbvs(frame, partial, CameraPoint(u=120.0, v=80.0))
    again = annotator.wrist_pbvs_view()
    assert again["occluded"] is False
    assert again["buttons_fresh"] is True
    assert again["target_uv"] == [180.0, 90.0]
    lost = YoloTouch(points={})
    annotator._note_wrist_pbvs(frame, lost, None)
    kept = annotator.wrist_pbvs_view()
    assert kept["buttons_fresh"] is False
    assert kept["seen"] is True
    assert kept["target_uv"] == [180.0, 90.0]
    assert "button_1" in kept["buttons_cam_m"]


def test_scale_points_never_replace_the_last_pose():
    panel = [(40.0, 80.0), (520.0, 80.0), (520.0, 360.0), (40.0, 360.0)]
    pose = {
        "button_1": CameraPoint(u=430.0, v=150.0, z_m=0.41, xyz_m=(0.08, -0.04, 0.41), source="pose"),
        "button_2": CameraPoint(u=430.0, v=190.0, z_m=0.41, xyz_m=(0.08, -0.01, 0.41), source="pose"),
        "button_3": CameraPoint(u=430.0, v=230.0, z_m=0.41, xyz_m=(0.08, 0.02, 0.41), source="pose"),
    }
    clear = Localization(visible=True, corners_px=panel, buttons=pose)
    held_loc, held, frozen = hold_last_pose(clear, {}, None)
    assert frozen is False
    assert held_loc.button("button_1").u == 430.0
    assert set(held_loc.buttons) == {"button_1", "button_2", "button_3"}
    false_panel = [(300.0, 180.0), (430.0, 180.0), (430.0, 250.0), (300.0, 250.0)]
    jumped = Localization(
        visible=True,
        corners_px=false_panel,
        buttons={
            "button_1": CameraPoint(u=360.0, v=210.0, z_m=1.20, xyz_m=(0.2, 0.1, 1.20), source="scale+pnp"),
            "button_2": CameraPoint(u=360.0, v=230.0, z_m=1.20, xyz_m=(0.2, 0.12, 1.20), source="scale+pnp"),
            "button_3": CameraPoint(u=360.0, v=250.0, z_m=1.20, xyz_m=(0.2, 0.14, 1.20), source="scale+pnp"),
        },
    )
    kept, still, frozen = hold_last_pose(jumped, held, panel)
    assert frozen is True
    assert still["button_1"].u == 430.0
    assert kept.button("button_1").source == "pose"
    assert kept.button("button_1").xyz_m == (0.08, -0.04, 0.41)
    assert kept.corners_px == panel
    assert all(point.source == "pose" for point in kept.buttons.values())
    empty = hold_last_pose(
        Localization(visible=True, corners_px=false_panel, buttons=jumped.buttons),
        {},
        None,
    )
    assert empty[0].buttons == {}
    assert empty[2] is False
    nearby = Localization(
        visible=True,
        corners_px=panel,
        buttons={
            "button_1": CameraPoint(u=434.0, v=152.0, z_m=0.41, xyz_m=(0.081, -0.04, 0.41), source="pose"),
        },
    )
    updated, stored, frozen = hold_last_pose(nearby, still, panel)
    assert frozen is False
    assert updated.button("button_1").u == 434.0
    assert stored["button_2"].u == 430.0
    assert stored["button_3"].source == "pose"


def test_pixel_contact_is_not_a_press_while_the_jaw_is_still_far():
    assert wrist_depth_confirms_contact(True, 0.18) is False
    assert wrist_depth_confirms_contact(True, None) is False
    assert wrist_depth_confirms_contact(False, 0.01) is False
    assert wrist_depth_confirms_contact(True, 0.012) is True


def test_red_bar_is_the_long_key_not_a_dot():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(image, (20, 50), (120, 62), (0, 0, 220), -1)
    assert detect_red_bar(image) is True
    dot = np.zeros((80, 160, 3), dtype=np.uint8)
    cv2.circle(dot, (40, 40), 3, (0, 0, 220), -1)
    assert detect_red_bar(dot) is False
    yellow = np.zeros((80, 160, 3), dtype=np.uint8)
    cv2.rectangle(yellow, (20, 50), (120, 62), (0, 220, 255), -1)
    assert detect_red_bar(yellow) is False


def test_detected_buttons_are_drawn_even_when_motion_keeps_another_point():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    points = {"button_1": YoloPoint("button_1", 30.0, 40.0, 0.93)}
    marked = draw_pose_buttons(image, points, skip={"button_1": (90.0, 20.0)})
    assert marked[40, 30].tolist() != [0, 0, 0]
    same = draw_pose_buttons(image, points, skip={"button_1": (30.0, 40.0)})
    assert same[40, 30].tolist() == [0, 0, 0]


def test_chest_occlusion_keeps_the_clear_buttons_for_motion():
    panel = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
    clear_button = CameraPoint(u=40.0, v=40.0, source="pose")
    clear = Localization(visible=True, corners_px=panel, buttons={"button_1": clear_button})
    touch = YoloTouch(
        points={"button_1": YoloPoint("button_1", 40.0, 40.0, 0.9)},
        masks=[YoloMask("gripper", 0.9, [(210.0, 0.0), (260.0, 0.0), (260.0, 40.0)])],
    )
    updated, held, frozen = chest_control_under_occlusion(clear, touch, {}, None)
    assert frozen is False
    assert updated.button("button_1").u == 40.0
    covered_touch = YoloTouch(
        points={"button_1": YoloPoint("button_1", 160.0, 40.0, 0.96)},
        masks=[YoloMask("gripper", 0.9, [(150.0, 20.0), (220.0, 20.0), (220.0, 90.0), (150.0, 90.0)])],
    )
    jumped = Localization(
        visible=True,
        corners_px=panel,
        buttons={"button_1": CameraPoint(u=160.0, v=40.0, source="pose")},
    )
    kept, still, frozen = chest_control_under_occlusion(jumped, covered_touch, held, panel)
    assert frozen is True
    assert kept.button("button_1").u == 40.0
    assert still["button_1"].u == 40.0


def test_wrist_ignores_button_points_below_0_3(monkeypatch):
    depth = np.full((80, 120), 0.22, dtype=np.float32)
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: depth)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    weak = YoloTouch(
        panel_corners=[(10.0, 10.0), (100.0, 10.0), (100.0, 70.0), (10.0, 70.0)],
        panel_conf=0.9,
        points={"button_1": YoloPoint("button_1", 40.0, 30.0, 0.02)},
    )
    vis, _corners, _tip = annotator._paint_side(frame, weak)
    assert vis[30, 40].tolist() == [0, 0, 0]
    annotator._note_wrist_pbvs(frame, weak, None)
    view = annotator.wrist_pbvs_view()
    assert view["buttons_fresh"] is False
    assert view["buttons_cam_m"] == {}
    assert view["close_range"] is False


def test_wrist_without_depth_keeps_the_button_pixels(monkeypatch):
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: None)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.ctrl.set_target("button_1")
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 40.0, 30.0, 0.92)})
    annotator._note_wrist_pbvs(frame, touch, CameraPoint(u=20.0, v=28.0))
    view = annotator.wrist_pbvs_view()
    assert view["close_range"] is True
    assert view["buttons_fresh"] is False
    assert view["buttons_cam_m"] == {}
    assert view["target_uv"] == [40.0, 30.0]
    assert view["tip_uv"] == [20.0, 28.0]
    assert view["target_z_m"] is None
    assert view["updated_at"] > 0.0


def test_wrist_button_stands_in_for_the_panel(monkeypatch):
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: None)
    annotator = LiveIbvsAnnotator()
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    touch = YoloTouch(
        panel_corners=[(10.0, 10.0), (100.0, 10.0), (100.0, 70.0), (10.0, 70.0)],
        panel_conf=0.18,
        points={"button_1": YoloPoint("button_1", 40.0, 30.0, 0.92)},
    )
    message = annotator._note_wrist_contact(frame, touch, CameraPoint(u=70.0, v=30.0), None, 10.0)
    assert "按键" in message
    assert "面板" not in message
    assert annotator._side["wrist_contact"] is False
    assert annotator._side["wrist_gap_px"] == 30.0
    assert _preview_line(message) == "wrist: button seen, tip gap 30px"
    missing = annotator._note_wrist_contact(frame, YoloTouch(points={}), CameraPoint(u=70.0, v=30.0), None, 11.0)
    assert missing == "右腕相机还没看到按键"
    assert _preview_line(missing) == "wrist: button not in view"


def test_wrist_view_marks_detected_buttons():
    annotator = LiveIbvsAnnotator()
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    touch = YoloTouch(points={"button_2": YoloPoint("button_2", 50.0, 30.0, 0.9)})
    vis, _corners, _tip = annotator._paint_side(frame, touch)
    assert vis[30, 50].tolist() != [0, 0, 0]


def test_gripper_cover_freezes_button_positions():
    panel = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
    clear = [(210.0, 0.0), (260.0, 0.0), (260.0, 40.0), (210.0, 40.0)]
    covering = [(150.0, 20.0), (220.0, 20.0), (220.0, 90.0), (150.0, 90.0)]
    assert gripper_occludes_panel(clear, panel, [(40.0, 40.0)]) is False
    assert gripper_occludes_panel(covering, panel) is True
    assert gripper_occludes_panel(clear, panel, [(230.0, 20.0)]) is True
    original = CameraPoint(u=170.0, v=40.0, z_m=0.62, xyz_m=(0.50, -0.02, 0.44), source="pose")
    jumped = CameraPoint(u=40.0, v=80.0, z_m=0.21, xyz_m=(0.12, 0.20, 0.21), source="scale")
    held_loc = Localization(visible=True, corners_px=panel, buttons={"button_1": original})
    _, held, frozen = hold_buttons_under_occlusion(held_loc, {}, covered=False)
    assert frozen is False
    assert held["button_1"].xyz_m == (0.50, -0.02, 0.44)
    occluded = Localization(visible=True, corners_px=panel, buttons={"button_1": jumped})
    kept, still, frozen = hold_buttons_under_occlusion(occluded, held, covered=True)
    assert frozen is True
    assert kept.button("button_1").u == 170.0
    assert kept.button("button_1").xyz_m == (0.50, -0.02, 0.44)
    assert still["button_1"].xyz_m == (0.50, -0.02, 0.44)
    dropped, empty, frozen = hold_buttons_under_occlusion(occluded, {}, covered=True)
    assert frozen is True and empty == {} and dropped.buttons == {}


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


def test_preferred_right_wrist_eye_in_hand_is_loaded():
    path = discover_eye_in_hand_json()
    assert path is not None
    assert path.name == "eye_in_hand_20261029_160039.json"
    matrix, info = load_eye_in_hand_transform(path)
    assert matrix is not None
    assert info["npy_file"].endswith("T_cam2hand.npy")
    assert "eye_in_hand_20261029_160039_npy" in info["npy_file"]
    assert info["serial"] == "233722073797"
    assert info["hand_frame"] == "right_wrist_yaw_link"
    assert info["pass"] is True


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


def _two_yellow_keys(shift: int = 0) -> np.ndarray:
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30 + shift, 40), (78 + shift, 58), (0, 220, 255), -1)
    cv2.rectangle(frame, (150, 40), (198, 58), (0, 220, 255), -1)
    return frame


def test_click_locks_one_yellow_key_and_follows_a_small_shift(monkeypatch):
    depth = np.full((120, 220), 0.18, dtype=np.float32)
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: depth)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.set_target("button_1")
    annotator.set_control_stage(lambda: "WRIST_APPROACH")
    frame = _two_yellow_keys()
    annotator._last_wrist_bgr = frame
    ok, detail = annotator.pick_wrist_yellow(50, 48)
    assert ok, detail
    weak = YoloTouch(points={"button_1": YoloPoint("button_1", 170.0, 48.0, 0.04)})
    annotator._note_wrist_pbvs(frame, weak, CameraPoint(u=20.0, v=48.0))
    view = annotator.wrist_pbvs_view()
    assert view["yellow_lock"] is True
    assert abs(view["target_uv"][0] - 54.0) < 8.0
    assert abs(view["target_uv"][1] - 49.0) < 8.0
    assert view["target_uv"][0] < 100.0
    assert abs(view["target_z_m"] - 0.18) < 1e-6
    shifted = _two_yellow_keys(12)
    annotator._note_wrist_pbvs(shifted, weak, CameraPoint(u=20.0, v=48.0))
    followed = annotator.wrist_pbvs_view()
    assert followed["yellow_lock"] is True
    assert abs(followed["target_uv"][0] - 66.0) < 8.0
    assert followed["target_uv"][0] < 120.0
    blank = np.zeros_like(frame)
    annotator._note_wrist_pbvs(blank, weak, CameraPoint(u=20.0, v=48.0))
    lost = annotator.wrist_pbvs_view()
    assert lost["yellow_lock"] is False
    assert "没有黄色" in annotator._wrist_yellow_message


def test_yellow_lock_drops_when_the_action_changes(monkeypatch):
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: None)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.set_target("button_1")
    stage = {"name": "WRIST_APPROACH"}
    annotator.set_control_stage(lambda: stage["name"])
    frame = _two_yellow_keys()
    annotator._last_wrist_bgr = frame
    assert annotator.pick_wrist_yellow(50, 48)[0]
    stage["name"] = "PBVS_COARSE"
    annotator._note_wrist_pbvs(frame, YoloTouch(points={}), None)
    assert annotator.wrist_pbvs_view()["yellow_lock"] is False
    stage["name"] = "WRIST_APPROACH"
    annotator._last_wrist_bgr = frame
    assert annotator.pick_wrist_yellow(50, 48)[0]
    annotator.set_target("button_2")
    annotator._note_wrist_pbvs(frame, YoloTouch(points={}), None)
    assert annotator.wrist_pbvs_view()["yellow_lock"] is False
    assert yellow_blob_at(np.zeros_like(frame), 50, 48) is None


def test_yellow_lock_gives_up_after_the_region_jumps_away(monkeypatch):
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: None)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.set_target("button_1")
    annotator.set_control_stage(lambda: "WRIST_APPROACH")
    frame = _two_yellow_keys()
    annotator._last_wrist_bgr = frame
    assert annotator.pick_wrist_yellow(50, 48)[0]
    far = np.zeros_like(frame)
    cv2.rectangle(far, (150, 40), (198, 58), (0, 220, 255), -1)
    touch = YoloTouch(points={})
    for _ in range(YELLOW_TRACK_MISS_FRAMES - 1):
        annotator._note_wrist_pbvs(far, touch, None)
        assert annotator.wrist_pbvs_view()["yellow_lock"] is True
    annotator._note_wrist_pbvs(far, touch, None)
    assert annotator.wrist_pbvs_view()["yellow_lock"] is False


def test_wrist_finger_lock_ignores_later_detections_until_recapture():
    first = CameraPoint(u=30.0, v=40.0, source="finger", confidence=0.9)
    later = CameraPoint(u=80.0, v=10.0, source="finger", confidence=0.95)
    tip, locked, _pending = latch_wrist_finger_tip(None, first, sdk_active=False, recapture=False)
    assert tip is first and locked is None
    tip, locked, pending = latch_wrist_finger_tip(None, first, sdk_active=True, recapture=False)
    assert (tip.u, tip.v) == (30.0, 40.0)
    assert tip.source == "finger_lock"
    assert pending is False
    tip, locked, pending = latch_wrist_finger_tip(locked, later, sdk_active=True, recapture=False)
    assert (tip.u, tip.v) == (30.0, 40.0)
    tip, locked, pending = latch_wrist_finger_tip(locked, later, sdk_active=False, recapture=False)
    assert (tip.u, tip.v) == (30.0, 40.0)
    tip, locked, pending = latch_wrist_finger_tip(locked, later, sdk_active=True, recapture=True)
    assert (tip.u, tip.v) == (80.0, 10.0)
    assert pending is False
    third = CameraPoint(u=1.0, v=1.0, source="finger")
    tip, locked, _pending = latch_wrist_finger_tip(locked, third, sdk_active=True, recapture=False)
    assert (tip.u, tip.v) == (80.0, 10.0)


def test_revo2_wrist_keeps_the_first_fingertip_when_the_task_changes():
    annotator = LiveIbvsAnnotator(ignore_gripper=True)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    first = YoloTouch(points={"FingerTip": YoloPoint("FingerTip", 30.0, 40.0, 0.9)})
    second = YoloTouch(points={"FingerTip": YoloPoint("FingerTip", 90.0, 10.0, 0.95)})
    _vis, _corners, tip = annotator._paint_side(frame, first)
    assert tip is not None and (tip.u, tip.v) == (30.0, 40.0)
    assert annotator._finger_lock is None
    _vis, _corners, tip = annotator._paint_side(frame, second)
    assert (tip.u, tip.v) == (90.0, 10.0)
    annotator.set_sdk_active(True)
    _vis, _corners, tip = annotator._paint_side(frame, first)
    assert tip.source == "finger_lock"
    assert (tip.u, tip.v) == (30.0, 40.0)
    annotator.set_target("button_2")
    annotator.forget_button_lock()
    _vis, _corners, tip = annotator._paint_side(frame, second)
    assert (tip.u, tip.v) == (30.0, 40.0)
    assert "等待下一帧" in     annotator.recapture_finger_tip()
    _vis, _corners, tip = annotator._paint_side(frame, second)
    assert (tip.u, tip.v) == (90.0, 10.0)
    assert annotator.finger_tip_status()["finger_tip_lock"]["u"] == 90.0
    assert "已关闭" in annotator.toggle_finger_lock()
    _vis, _corners, tip = annotator._paint_side(frame, first)
    assert tip.source == "finger"
    assert (tip.u, tip.v) == (30.0, 40.0)
    assert annotator._finger_lock is None
    assert annotator.finger_tip_status()["finger_tip_lock_enabled"] is False
    assert "已打开" in annotator.toggle_finger_lock()
    _vis, _corners, tip = annotator._paint_side(frame, second)
    assert tip.source == "finger_lock"
    assert (tip.u, tip.v) == (90.0, 10.0)


def test_nearest_yellow_bar_follows_the_fingertip():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[10:16, 10:80] = (0, 220, 255)
    image[50:56, 90:150] = (0, 220, 255)
    near_top = nearest_long_yellow_bar(image, (20.0, 30.0))
    near_bottom = nearest_long_yellow_bar(image, (130.0, 60.0))
    assert near_top is not None and near_bottom is not None
    assert near_top[1] < 30.0
    assert near_bottom[1] > 40.0
    assert nearest_long_yellow_bar(image, None) is None


def test_wrist_uses_the_nearest_yellow_bar_when_the_button_is_missing(monkeypatch):
    monkeypatch.setattr("ibvs_pad_touch.web.live_ibvs.read_depth", lambda _path: None)
    annotator = LiveIbvsAnnotator()
    annotator.set_wrist_pbvs_watch(True)
    annotator.ctrl.set_target("button_1")
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frame[10:16, 10:80] = (0, 220, 255)
    frame[50:56, 90:150] = (0, 220, 255)
    annotator._note_wrist_pbvs(frame, YoloTouch(points={}), CameraPoint(u=20.0, v=30.0, source="finger"))
    view = annotator.wrist_pbvs_view()
    assert view["target_source"] == "yellow_bar"
    assert view["close_range"] is True
    assert view["target_uv"][1] < 30.0
    assert view["tip_uv"] == [20.0, 30.0]
