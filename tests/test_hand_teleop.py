"""Webcam fingertip motion mapped onto torso steps."""

from ibvs_pad_touch.teleop.hand_map import TipSample, TipTeleop, torso_from_tips


def test_image_up_raises_the_robot():
    anchor = TipSample(320, 240, 100)
    raised = TipSample(320, 220, 100)
    _x, _y, z = torso_from_tips(anchor, raised, scale=0.1, range_m=0.45)
    assert z > 0
    assert abs(_x) < 1e-9
    assert abs(_y) < 1e-9


def test_image_right_moves_robot_left():
    anchor = TipSample(320, 240, 100)
    right = TipSample(340, 240, 100)
    _x, y, _z = torso_from_tips(anchor, right, scale=0.1, range_m=0.45)
    assert y > 0


def test_d405_depth_closer_moves_toward_the_panel():
    anchor = TipSample(320, 240, 100, 0.30)
    closer = TipSample(320, 240, 100, 0.25)
    x, y, z = torso_from_tips(anchor, closer, scale=0.1, range_m=0.45)
    assert abs(x - 0.005) < 1e-9
    assert abs(y) < 1e-9
    assert abs(z) < 1e-9


def test_a_larger_hand_moves_toward_the_panel():
    anchor = TipSample(320, 240, 100)
    closer = TipSample(320, 240, 120)
    x, y, z = torso_from_tips(anchor, closer, scale=0.1, range_m=0.45)
    assert x > 0
    assert abs(y) < 1e-9
    assert abs(z) < 1e-9


def test_small_motion_is_held_until_it_reaches_two_millimetres():
    teleop = TipTeleop(scale=0.1, min_step_m=0.002, max_step_m=0.008)
    teleop.engage(TipSample(320, 240, 100))
    assert teleop.offer(TipSample(322, 240, 100)) is None
    command = teleop.offer(TipSample(350, 240, 100))
    assert command is not None
    assert command[1] > 0
    assert abs(command[1]) <= 0.008


def test_release_drops_a_pending_motion():
    teleop = TipTeleop()
    teleop.engage(TipSample(320, 240, 100))
    teleop.release()
    assert teleop.offer(TipSample(400, 240, 100)) is None


def test_a_lost_hand_does_not_throw_a_jump():
    teleop = TipTeleop(max_jump_px=50)
    teleop.engage(TipSample(100, 100, 80))
    assert teleop.offer(TipSample(400, 100, 80)) is None
