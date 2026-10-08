from pathlib import Path

import cv2

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.pipeline import PanelButtonLocalizer
from ibvs_pad_touch.perception.yellow_buttons import assign_poke_buttons
from ibvs_pad_touch.types import Intrinsics

PHOTOS = Path(__file__).resolve().parents[1] / "data" / "photos"


def _photos() -> list[Path]:
    return sorted(PHOTOS.glob("*.png"))


def test_field_photos_exist():
    assert len(_photos()) >= 5


def test_yellow_layout_on_every_field_photo():
    for path in _photos():
        bgr = cv2.imread(str(path))
        assert bgr is not None, path
        found = assign_poke_buttons(bgr)
        assert found is not None, path.name
        b1, b2, b3 = found["button_1"], found["button_2"], found["button_3"]
        assert b1[0] < b2[0], path.name
        assert b3[1] > b1[1], path.name
        assert abs(b3[0] - 0.5 * (b1[0] + b2[0])) < 0.08 * bgr.shape[1], path.name


def test_pipeline_labels_three_poke_targets():
    localizer = PanelButtonLocalizer(load_panel(), refine_color=True)
    for path in _photos():
        bgr = cv2.imread(str(path))
        h, w = bgr.shape[:2]
        K = Intrinsics(fx=0.8 * w, fy=0.8 * w, cx=w / 2, cy=h / 2, width=w, height=h)
        loc = localizer.localize(bgr, None, K)
        assert loc.visible, f"{path.name} {loc.message}"
        assert set(loc.buttons) == {"button_1", "button_2", "button_3"}
        assert loc.buttons["button_1"].u < loc.buttons["button_2"].u
        assert loc.buttons["button_3"].v > loc.buttons["button_1"].v
        assert loc.corners_px is not None, path.name
        xs = [p[0] for p in loc.corners_px]
        ys = [p[1] for p in loc.corners_px]
        assert max(xs) - min(xs) < 0.85 * w
        assert max(ys) - min(ys) < 0.75 * h
        for point in loc.buttons.values():
            assert min(xs) - 25 < point.u < max(xs) + 25
            assert min(ys) - 25 < point.v < max(ys) + 25
