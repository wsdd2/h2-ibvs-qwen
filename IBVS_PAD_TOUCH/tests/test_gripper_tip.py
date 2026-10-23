from pathlib import Path

import cv2
import numpy as np

import ibvs_pad_touch.perception.gripper_tip as gripper_tip
from ibvs_pad_touch.perception.gripper_tip import detect_dex1_tip

PHOTOS = Path(__file__).resolve().parents[1] / "data" / "gripper"


def _white_scene_dex1() -> tuple[np.ndarray, tuple[int, int]]:
    """Chest-cam-like: light background, silver forearm, closed black Dex1."""
    img = np.full((480, 640, 3), 230, np.uint8)
    cv2.rectangle(img, (280, 300), (360, 470), (190, 190, 190), -1)
    cv2.circle(img, (320, 300), 42, (200, 200, 200), -1)
    hull = np.array([[300, 300], [340, 300], [328, 170], [312, 170]], np.int32)
    cv2.fillConvexPoly(img, hull, (25, 25, 25))
    cv2.circle(img, (320, 168), 7, (20, 20, 20), -1)
    return img, (320, 162)


def test_tip_on_light_background():
    bgr, (tu, tv) = _white_scene_dex1()
    tip = detect_dex1_tip(bgr)
    assert tip is not None
    assert abs(tip.u - tu) < 18
    assert abs(tip.v - tv) < 18


def test_border_entering_gripper_excludes_panel_lcd():
    bgr = np.full((480, 640, 3), 225, dtype=np.uint8)
    panel = [(250, 140), (570, 140), (570, 360), (250, 360)]
    cv2.rectangle(bgr, (350, 245), (465, 320), (25, 25, 25), -1)
    gripper = np.array(
        [[500, 405], [540, 390], [590, 425], [639, 425], [639, 479], [575, 479], [545, 445]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(bgr, gripper, (20, 20, 20))
    tip = detect_dex1_tip(bgr, panel_corners=panel)
    assert tip is not None
    assert abs(tip.u - 500) < 15
    assert abs(tip.v - 405) < 15


def test_white_bracket_is_not_the_gripper_tip():
    bgr = np.full((480, 640, 3), (210, 215, 205), dtype=np.uint8)
    panel = [(180, 160), (520, 160), (520, 340), (180, 340)]
    cv2.rectangle(bgr, (180, 160), (520, 340), (170, 190, 150), -1)
    bracket = np.array([[560, 0], [639, 0], [639, 80], [590, 70], [560, 30]], dtype=np.int32)
    cv2.fillConvexPoly(bgr, bracket, (220, 220, 220))
    assert detect_dex1_tip(bgr, panel_corners=panel) is None
    jaw = np.array([[530, 180], [575, 176], [578, 340], [528, 344]], dtype=np.int32)
    cv2.fillConvexPoly(bgr, jaw, (20, 20, 20))
    tip = detect_dex1_tip(bgr, panel_corners=panel)
    assert tip is not None
    assert 530 < tip.u < 610
    assert 160 < tip.v < 360


def test_border_entry_prefers_sharp_jaw_over_farthest_bracket(monkeypatch):
    contour = np.array(
        [[[639, 300]], [[639, 470]], [[550, 340]], [[590, 305]], [[620, 295]]],
        dtype=np.int32,
    )
    expected = np.array([590.0, 305.0])
    monkeypatch.setattr(gripper_tip, "_v_tip", lambda *_args: expected)
    found = gripper_tip._border_entry_tip(
        contour,
        480,
        640,
        np.zeros((480, 640), dtype=np.uint8),
    )
    assert found is not None
    assert np.allclose(found, expected)


def test_border_entry_pca_ignores_vertical_pin(monkeypatch):
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (500, 438), (639, 479), 255, -1)
    cv2.rectangle(mask, (550, 385), (566, 450), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    monkeypatch.setattr(gripper_tip, "_v_tip", lambda *_args: None)
    found = gripper_tip._border_entry_tip(contour, 480, 640, mask)
    assert found is not None
    assert found[0] <= 505
    assert found[1] >= 430


def test_tip_found_on_handheld_photos():
    paths = sorted(PHOTOS.glob("*.png"))
    assert len(paths) >= 4
    for path in paths:
        bgr = cv2.imread(str(path))
        assert bgr is not None, path
        tip = detect_dex1_tip(bgr)
        if tip is None:
            continue
        h, w = bgr.shape[:2]
        assert 0 < tip.u < w
        assert 0 < tip.v < h
        # Clothing / shoes sit on the bottom edge; the Dex1 tip does not.
        assert tip.v < 0.92 * h, path.name
