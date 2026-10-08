import cv2
import numpy as np

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.side_contact import estimate_side_gap
from ibvs_pad_touch.servo.dual_view import compose_dual_view_ibvs, servo_depth_m
from ibvs_pad_touch.types import Intrinsics


def _K() -> Intrinsics:
    return Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)


def test_visible_jaw_without_a_panel_backs_off():
    lost = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_blocked=True)
    assert lost.phase == "BACKOFF"
    assert lost.dz < 0.0
    assert lost.contact is False
    blind = compose_dual_view_ibvs(None, None, _K(), side_blocked=True)
    assert blind.dx == blind.dy == blind.dz == 0.0
    assert blind.phase == "SEEK_GRIPPER"


def test_too_close_before_align_backs_off_instead_of_holding():
    jammed = compose_dual_view_ibvs(
        (300.0, 240.0),
        (360.0, 240.0),
        _K(),
        side_measured=True,
        side_contact=True,
        side_gap_px=2.0,
        standoff_px=36.0,
    )
    assert jammed.phase == "ALIGN_XY"
    assert jammed.dz < 0.0
    assert jammed.dx != 0.0
    tight = compose_dual_view_ibvs(
        (300.0, 240.0),
        (360.0, 240.0),
        _K(),
        side_measured=True,
        side_gap_px=12.0,
        standoff_px=36.0,
    )
    assert tight.dz < 0.0


def test_open_side_gap_moves_forward_while_the_chest_is_still_aligning():
    cmd = compose_dual_view_ibvs(
        (300.0, 280.0),
        (360.0, 220.0),
        _K(),
        side_measured=True,
        side_gap_px=63.0,
        standoff_px=36.0,
    )
    assert cmd.phase == "ALIGN_XY"
    assert cmd.dx < 0.0
    assert cmd.dy > 0.0
    assert cmd.dz > 0.0
    assert cmd.contact is False


def test_aligned_side_gap_steps_forward_along_chest_axis():
    cmd = compose_dual_view_ibvs((320.0, 240.0), (322.0, 241.0), _K(), side_measured=True, side_gap_px=30.0)
    assert cmd.aligned
    assert cmd.phase == "APPROACH"
    assert cmd.dx == 0.0 and cmd.dy == 0.0
    assert cmd.dz > 0.0


def test_measured_depth_scales_chest_xy_and_side_z():
    assert servo_depth_m(None, 0.60) == 0.60
    assert servo_depth_m(0.02, 0.60) == 0.08
    assert servo_depth_m(3.0, 0.60) == 1.20
    near = compose_dual_view_ibvs(
        (300.0, 240.0),
        (320.0, 240.0),
        _K(),
        nominal_depth_m=0.30,
        side_depth_m=0.30,
        side_measured=True,
        side_gap_px=24.0,
    )
    far = compose_dual_view_ibvs(
        (300.0, 240.0),
        (320.0, 240.0),
        _K(),
        nominal_depth_m=0.60,
        side_depth_m=0.60,
        side_measured=True,
        side_gap_px=24.0,
    )
    assert abs(near.dx) < abs(far.dx)
    assert near.phase == "ALIGN_XY"
    close = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_measured=True, side_gap_px=24.0, side_depth_m=0.30)
    distant = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_measured=True, side_gap_px=24.0, side_depth_m=0.60)
    assert 0.0 < close.dz < distant.dz


def test_side_pixel_error_scales_the_forward_step():
    near = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_measured=True, side_gap_px=20.0)
    far = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_measured=True, side_gap_px=80.0)
    assert 0.0 < near.dz < far.dz
    assert far.dz <= 0.006 + 1e-9
    tight = compose_dual_view_ibvs(
        (300.0, 240.0),
        (360.0, 240.0),
        _K(),
        side_measured=True,
        side_gap_px=8.0,
        standoff_px=36.0,
    )
    loose = compose_dual_view_ibvs(
        (300.0, 240.0),
        (360.0, 240.0),
        _K(),
        side_measured=True,
        side_gap_px=28.0,
        standoff_px=36.0,
    )
    assert tight.dz < loose.dz < 0.0


def test_side_contact_stops_even_if_fk_would_keep_going():
    cmd = compose_dual_view_ibvs((320.0, 240.0), (321.0, 240.0), _K(), side_measured=True, side_contact=True, side_gap_px=2.0)
    assert cmd.phase == "CONTACT"
    assert cmd.dx == cmd.dy == cmd.dz == 0.0


def test_missing_chest_tip_holds_the_right_arm():
    close = compose_dual_view_ibvs((320.0, 240.0), None, _K(), side_measured=True, side_gap_px=20.0, standoff_px=36.0)
    assert close.phase == "SEEK_GRIPPER"
    assert close.dx == close.dy == close.dz == 0.0
    clear = compose_dual_view_ibvs((320.0, 240.0), None, _K(), side_measured=True, side_gap_px=80.0, standoff_px=36.0)
    assert clear.phase == "SEEK_GRIPPER"
    assert clear.dx == clear.dy == clear.dz == 0.0
    blind = compose_dual_view_ibvs((320.0, 240.0), None, _K())
    assert blind.phase == "SEEK_GRIPPER"
    assert blind.dz == 0.0


def _side_frame(gripper_x0: int) -> np.ndarray:
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (160, 180), (460, 310), (208, 218, 214), -1)
    cv2.rectangle(image, (195, 208), (370, 280), (32, 36, 38), -1)
    cv2.rectangle(image, (gripper_x0, 210), (gripper_x0 + 90, 250), (16, 16, 16), -1)
    return image


def test_side_view_keeps_a_high_panel_when_the_jaw_is_on_it():
    panel = load_panel()
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    high = np.array([(220.0, 30.0), (520.0, 40.0), (510.0, 150.0), (230.0, 140.0)], dtype=np.float32)
    far = estimate_side_gap(image, model=panel, contact_px=14.0, panel_corners=high, tip_uv=(40.0, 400.0))
    assert far.measured is False
    assert far.blocked is True
    near = estimate_side_gap(image, model=panel, contact_px=14.0, panel_corners=high, tip_uv=(200.0, 90.0))
    assert near.measured is True
    assert near.gap_px is not None and near.gap_px < 36.0


def test_side_view_ignores_a_ceiling_rectangle():
    panel = load_panel()
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (220, 20), (520, 150), (210, 214, 208), -1)
    cv2.rectangle(image, (280, 40), (460, 110), (20, 22, 24), -1)
    cv2.rectangle(image, (500, 70), (590, 110), (12, 12, 12), -1)
    overhead = estimate_side_gap(image, model=panel, contact_px=14.0)
    assert overhead.measured is False
    assert overhead.tip_uv is None


def test_side_view_ignores_a_dark_speck_far_from_the_panel():
    panel = load_panel()
    image = _side_frame(20)
    image[:, :110] = 245
    cv2.rectangle(image, (8, 40), (36, 70), (12, 12, 12), -1)
    far = estimate_side_gap(image, model=panel, contact_px=14.0)
    assert far.measured is False
    assert far.tip_uv is None


def test_side_view_reports_gap_then_contact():
    panel = load_panel()
    apart = estimate_side_gap(_side_frame(520), model=panel, contact_px=14.0)
    assert apart.measured
    assert apart.contact is False
    assert apart.gap_px is not None and apart.gap_px > 14.0
    touching = estimate_side_gap(_side_frame(430), model=panel, contact_px=14.0)
    assert touching.measured
    assert touching.contact
