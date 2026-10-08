from ibvs_pad_touch.servo.reach import ReachGate


def test_reach_requires_stable_frames():
    gate = ReachGate(xy_tol_norm=0.01, z_tol_m=0.008, stable_frames=3, pixel_tol=8)
    assert not gate.update(0.0, 0.0, 0.0, True)
    assert not gate.update(0.0, 0.0, 0.0, True)
    assert gate.update(0.0, 0.0, 0.0, True)


def test_reach_resets_on_outlier():
    gate = ReachGate(xy_tol_norm=0.01, z_tol_m=0.008, stable_frames=2, pixel_tol=8)
    assert not gate.update(0.0, 0.0, 0.0, True)
    assert not gate.update(0.2, 0.0, 40.0, False)
    assert not gate.update(0.0, 0.0, 0.0, True)
    assert gate.update(0.0, 0.0, 0.0, True)
