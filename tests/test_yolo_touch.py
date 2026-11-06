import time

import numpy as np

from ibvs_pad_touch.perception.yolo_touch import (
    YoloMask,
    YoloPoint,
    YoloTouch,
    YoloTouchDetector,
    apply_pose_buttons,
    _pose_points,
    normalize_button_layout,
    pose_button_ok,
    pose_finger_tip,
    pose_tip,
    without_gripper,
)
from ibvs_pad_touch.types import CameraPoint, Localization
from ibvs_pad_touch.web.live_ibvs import LiveIbvsAnnotator


PANEL = [(80.0, 60.0), (520.0, 70.0), (510.0, 380.0), (90.0, 370.0)]


def _loc() -> Localization:
    return Localization(
        visible=True,
        corners_px=list(PANEL),
        buttons={
            "button_1": CameraPoint(u=200.0, v=140.0, source="scale"),
            "button_2": CameraPoint(u=240.0, v=140.0, source="scale"),
            "button_3": CameraPoint(u=220.0, v=180.0, source="scale"),
        },
        message="ok",
    )


def test_gripper_without_a_pose_tip_uses_the_narrow_end_of_the_long_axis():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask, gripper_axis_tip

    jaw = YoloMask(
        "gripper",
        0.84,
        [(200.0, 30.0), (150.0, 240.0), (250.0, 240.0)],
    )
    tip = gripper_axis_tip(jaw)
    assert tip is not None
    assert tip.source == "axis"
    assert tip.confidence == 0.84
    assert tip.v < 50.0
    assert 180.0 < tip.u < 220.0


def test_side_view_does_not_invent_a_tip_from_a_seg_mask():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask, YoloTouch

    class _JawOnly:
        def predict(self, bgr, **_kwargs):
            return YoloTouch(
                masks=[
                    YoloMask(
                        "gripper",
                        0.84,
                        [(420.0, 40.0), (360.0, 260.0), (480.0, 260.0)],
                    )
                ]
            )

    annotator = LiveIbvsAnnotator(detector=_JawOnly())
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    out = annotator.annotate("d435i", frame)
    assert annotator.telemetry()["side_measured"] is False
    assert annotator.telemetry()["left_contact"] is False
    assert int(out[180, 420].max()) == 240


def test_low_confidence_gripper_and_a_tip_off_the_jaw_are_dropped():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask, keep_side_masks, tip_on_mask

    jaw = YoloMask("gripper", 0.89, [(400.0, 20.0), (520.0, 20.0), (510.0, 160.0), (390.0, 150.0)])
    head = YoloMask("gripper", 0.22, [(80.0, 300.0), (180.0, 300.0), (170.0, 420.0), (70.0, 410.0)])
    kept = keep_side_masks([jaw, head])
    assert [mask.conf for mask in kept] == [0.89]
    on_jaw = YoloPoint("gripper_tip", 450.0, 30.0, 0.4)
    on_person = YoloPoint("gripper_tip", 120.0, 360.0, 0.4)
    assert tip_on_mask(on_jaw, kept[0]) is True
    assert tip_on_mask(on_person, kept[0]) is False


def test_low_confidence_keeps_the_scale_button():
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 210.0, 150.0, 0.22)})
    out = apply_pose_buttons(_loc(), touch, conf_min=0.3)
    button = out.button("button_1")
    assert button is not None
    assert button.source == "scale"
    assert button.u == 200.0


def test_point_outside_the_panel_keeps_the_scale_button():
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 12.0, 12.0, 0.95)})
    assert pose_button_ok(touch.point("button_1"), PANEL, 0.3) is False
    out = apply_pose_buttons(_loc(), touch, conf_min=0.3)
    assert out.button("button_1").source == "scale"
    assert out.button("button_1").u == 200.0


def test_confident_point_inside_an_irregular_panel_replaces_the_scale_button():
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 230.0, 160.0, 0.81)})
    out = apply_pose_buttons(_loc(), touch, conf_min=0.3)
    button = out.button("button_1")
    assert button.source == "pose"
    assert button.u == 230.0
    assert button.confidence == 0.81
    assert out.button("button_2").source == "scale"


def test_wide_rectangle_still_keeps_the_scale_button():
    from ibvs_pad_touch.perception.yolo_touch import panel_is_regular_rectangle

    corners = [(40.0, 180.0), (760.0, 190.0), (750.0, 340.0), (50.0, 330.0)]
    assert panel_is_regular_rectangle(corners) is True
    loc = Localization(
        visible=True,
        corners_px=corners,
        buttons={"button_1": CameraPoint(u=640.0, v=210.0, source="scale")},
        message="ok",
    )
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 680.0, 230.0, 0.99)})
    out = apply_pose_buttons(loc, touch, conf_min=0.3)
    assert out.button("button_1").source == "scale"
    assert out.button("button_1").u == 640.0


def test_regular_panel_rectangle_keeps_the_scale_button():
    from ibvs_pad_touch.perception.yolo_touch import panel_is_regular_rectangle

    corners = [(100.0, 80.0), (560.0, 80.0), (560.0, 280.0), (100.0, 280.0)]
    assert panel_is_regular_rectangle(corners) is True
    loc = Localization(
        visible=True,
        corners_px=corners,
        buttons={"button_1": CameraPoint(u=480.0, v=110.0, source="scale")},
        message="ok",
    )
    touch = YoloTouch(points={"button_1": YoloPoint("button_1", 500.0, 130.0, 0.95)})
    out = apply_pose_buttons(loc, touch, conf_min=0.3)
    assert out.button("button_1").source == "scale"
    assert out.button("button_1").u == 480.0


def test_complete_confident_pose_triplet_replaces_regular_panel_scale_points():
    corners = [(100.0, 80.0), (560.0, 80.0), (560.0, 280.0), (100.0, 280.0)]
    loc = Localization(
        visible=True,
        corners_px=corners,
        buttons={
            "button_1": CameraPoint(u=440.0, v=120.0, source="scale"),
            "button_2": CameraPoint(u=480.0, v=120.0, source="scale"),
            "button_3": CameraPoint(u=460.0, v=160.0, source="scale"),
        },
        message="ok",
    )
    touch = YoloTouch(
        points={
            "button_1": YoloPoint("button_1", 480.0, 110.0, 0.88),
            "button_2": YoloPoint("button_2", 520.0, 112.0, 0.81),
            "button_3": YoloPoint("button_3", 500.0, 158.0, 0.94),
        }
    )
    out = apply_pose_buttons(loc, touch, conf_min=0.3)
    assert all(out.button(name).source == "pose" for name in ("button_1", "button_2", "button_3"))
    assert out.button("button_2").u == 520.0


class _FakeDetector:
    def predict(self, bgr):
        return YoloTouch(
            panel_corners=list(PANEL),
            panel_conf=0.92,
            masks=[YoloMask("gripper", 0.86, [(150.0, 180.0), (220.0, 180.0), (220.0, 270.0), (150.0, 270.0)])],
            points={
                "button_1": YoloPoint("button_1", 230.0, 160.0, 0.88),
                "gripper_tip": YoloPoint("gripper_tip", 180.0, 220.0, 0.77),
            },
        )


def test_chest_keeps_a_pose_tip_just_outside_the_jaw_mask():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask

    class _TipBesideJaw:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(80.0, 80.0), (400.0, 80.0), (400.0, 280.0), (80.0, 280.0)],
                panel_conf=0.9,
                masks=[YoloMask("gripper", 0.9, [(520.0, 80.0), (630.0, 100.0), (610.0, 260.0), (500.0, 240.0)])],
                points={"gripper_tip": YoloPoint("gripper_tip", 470.0, 90.0, 0.22)},
            )

    annotator = LiveIbvsAnnotator(detector=_TipBesideJaw())
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["tip_source"] == "pose"
    assert telem["tip_raw_uv"] == [470.0, 90.0]


def test_jaw_on_the_panel_border_still_drives():
    from ibvs_pad_touch.web.live_ibvs import _chest_visual_tip

    corners = [(138.0, 210.0), (506.0, 210.0), (506.0, 388.0), (138.0, 388.0)]
    jaw = _chest_visual_tip(CameraPoint(u=222.0, v=381.0, source="dex1_tip"), None, False, corners)
    assert jaw is not None and jaw.source == "chest_visual"
    deep = _chest_visual_tip(CameraPoint(u=320.0, v=280.0, source="dex1_tip"), None, False, corners)
    assert deep is None


def test_chest_keeps_a_pose_tip_when_the_gripper_mask_is_missing():
    class _PoseTipOnly:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(80.0, 120.0), (520.0, 120.0), (520.0, 320.0), (80.0, 320.0)],
                panel_conf=0.9,
                masks=[],
                points={"gripper_tip": YoloPoint("gripper_tip", 540.0, 180.0, 0.8)},
            )

    annotator = LiveIbvsAnnotator(detector=_PoseTipOnly())
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["tip_source"] == "pose"
    assert telem["tip_raw_uv"] == [540.0, 180.0]


def test_chest_drops_a_pose_tip_beside_the_keys_when_the_jaw_is_elsewhere():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask

    class _PoseBesideKeys:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(80.0, 80.0), (400.0, 80.0), (400.0, 280.0), (80.0, 280.0)],
                panel_conf=0.9,
                masks=[YoloMask("gripper", 0.85, [(550.0, 400.0), (630.0, 410.0), (620.0, 470.0), (540.0, 460.0)])],
                points={"gripper_tip": YoloPoint("gripper_tip", 430.0, 160.0, 0.8)},
            )

    annotator = LiveIbvsAnnotator(detector=_PoseBesideKeys())
    annotator._side = {"measured": False, "contact": False, "gap_px": None, "blocked": True, "message": ""}
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["tip_source"] == "face"
    assert telem["tip_raw_uv"][0] >= 540.0
    assert telem["tip_raw_uv"][1] >= 400.0
    assert telem["phase"] != "BACKOFF"


def test_chest_ignores_a_pose_tip_on_the_buttons_and_uses_the_jaw():
    from ibvs_pad_touch.perception.yolo_touch import YoloMask

    class _JawAtTheEdge:
        def predict(self, bgr):
            return YoloTouch(
                panel_corners=[(80.0, 80.0), (460.0, 80.0), (460.0, 280.0), (80.0, 280.0)],
                panel_conf=0.9,
                masks=[YoloMask("gripper", 0.84, [(560.0, 40.0), (620.0, 80.0), (600.0, 220.0)])],
                points={"gripper_tip": YoloPoint("gripper_tip", 430.0, 120.0, 0.8)},
            )

    annotator = LiveIbvsAnnotator(detector=_JawAtTheEdge())
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    out = annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["tip_source"] == "face"
    assert telem["tip_raw_uv"][0] > 540.0
    assert telem["tip_raw_uv"][1] < 80.0
    assert int(out.sum()) != int(frame.sum())


def test_chest_preview_uses_pose_tip_and_pose_button():
    annotator = LiveIbvsAnnotator(detector=_FakeDetector())
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    annotator.annotate("chest", frame)
    telem = annotator.telemetry()
    assert telem["tip_source"] == "pose"
    assert telem["tip_raw_uv"] == [180.0, 220.0]
    assert telem["target_uv"] == [230.0, 160.0]


def test_revo2_mode_drops_gripper_and_gripper_tip():
    touch = YoloTouch(
        points={
            "gripper_tip": YoloPoint("gripper_tip", 10.0, 20.0, 0.9),
            "button_1": YoloPoint("button_1", 30.0, 40.0, 0.8),
        },
        masks=[
            YoloMask("gripper", 0.95, [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]),
            YoloMask("panel", 0.7, [(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)]),
            YoloMask("screen", 0.6, [(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)]),
        ],
    )
    kept = without_gripper(touch)
    assert kept is not None
    assert "gripper_tip" not in kept.points
    assert "button_1" in kept.points
    assert [mask.name for mask in kept.masks] == ["panel", "screen"]
    finger = YoloTouch(
        points={
            "gripper_tip": YoloPoint("gripper_tip", 10.0, 20.0, 0.9),
            "FingerTip": YoloPoint("FingerTip", 40.0, 50.0, 0.85),
            "button_1": YoloPoint("button_1", 30.0, 40.0, 0.8),
        }
    )
    kept_finger = without_gripper(finger)
    found = pose_finger_tip(kept_finger, 0.3)
    assert found is not None
    assert found.source == "finger"
    assert (found.u, found.v) == (40.0, 50.0)
    assert pose_tip(kept_finger, 0.3) is None


def test_pose_keeps_the_fingertip_from_a_lower_scoring_box():
    class _Tensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __len__(self):
            return len(self.value)

    class _Keypoints:
        def __init__(self, value):
            self.data = _Tensor(value)

    class _Result:
        def __init__(self, value):
            self.keypoints = _Keypoints(value)
            self.boxes = None

    # Higher box has no fingertip. The hand is the second box.
    rows = np.zeros((2, 5, 3), dtype=np.float32)
    rows[0, 0] = (10.0, 10.0, 0.9)
    rows[1, 4] = (200.0, 180.0, 0.4)
    found = _pose_points(_Result(rows))
    assert abs(found["gripper_tip"].conf - 0.9) < 1e-5
    assert (found["FingerTip"].u, found["FingerTip"].v) == (200.0, 180.0)


def test_button3_above_button1_is_swapped_back():
    swapped = normalize_button_layout(
        YoloTouch(
            points={
                "button_1": YoloPoint("button_1", 100.0, 220.0, 0.9),
                "button_3": YoloPoint("button_3", 110.0, 140.0, 0.8),
            }
        )
    )
    assert swapped is not None
    assert (swapped.point("button_1").u, swapped.point("button_1").v) == (110.0, 140.0)
    assert (swapped.point("button_3").u, swapped.point("button_3").v) == (100.0, 220.0)
    kept = normalize_button_layout(
        YoloTouch(
            points={
                "button_1": YoloPoint("button_1", 100.0, 140.0, 0.9),
                "button_3": YoloPoint("button_3", 110.0, 220.0, 0.8),
            }
        )
    )
    assert (kept.point("button_1").v, kept.point("button_3").v) == (140.0, 220.0)


def test_seg_and_pose_run_at_the_same_time():
    class _Empty:
        boxes = None
        masks = None
        keypoints = None

    log = []

    class _Model:
        def __init__(self, name):
            self.name = name

        def predict(self, image, **kwargs):
            assert kwargs["device"] == "cpu"
            log.append(("start", self.name))
            time.sleep(0.12)
            log.append(("end", self.name))
            return [_Empty()]

    detector = YoloTouchDetector(name="chest")
    detector._ready = True
    detector._device = "cpu"
    detector._half = False
    detector._seg = _Model("seg")
    detector._pose = _Model("pose")
    touch = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert touch is not None
    assert [(item[0], item[1]) for item in log] == [
        ("start", "seg"),
        ("end", "seg"),
        ("start", "pose"),
        ("end", "pose"),
    ]
    log.clear()
    pose_only = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8), pose_only=True)
    assert pose_only is not None
    assert pose_only.masks == []
    assert pose_only.panel_corners is None
    assert [(item[0], item[1]) for item in log] == [("start", "pose"), ("end", "pose")]


def test_wrist_pose_uses_its_own_cpu_model():
    class _Empty:
        boxes = None
        masks = None
        keypoints = None

    log = []

    class _Model:
        def predict(self, image, **kwargs):
            log.append(kwargs["device"])
            return [_Empty()]

    detector = YoloTouchDetector(name="wrist", cpu_pose=True)
    detector._ready = True
    detector._device = "cpu"
    detector._half = False
    detector._pose = _Model()
    detector._seg = _Model()
    touch = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert touch is not None
    assert touch.masks == []
    assert log == ["cpu"]
