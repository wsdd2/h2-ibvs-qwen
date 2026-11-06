# -*- coding: utf-8 -*-
"""Screen-progress mode: a changed panel counts as a press, the same panel is repeated."""
from pathlib import Path

import numpy as np

from ibvs_pad_touch.web.arm_motion import SCREEN_PRESS_ATTEMPTS, ArmMover, REST_DEG
from ibvs_pad_touch.web.screen_read import blocks_differ, read_panel


def _half(left: bool) -> np.ndarray:
    image = np.full((180, 320, 3), 210, dtype=np.uint8)
    if left:
        image[:, :140] = 40
    else:
        image[:, 180:] = 40
    return image


def test_identical_crops_match_and_a_moved_panel_does_not():
    same = _half(True)
    assert blocks_differ(same, same.copy(), read_panel(same), read_panel(same)) is False
    other = _half(False)
    assert blocks_differ(same, other, read_panel(same), read_panel(other)) is True


def test_saved_panel_photos_keep_the_solid_block_text():
    root = Path(r"E:\control_panel_screen\screens")
    back = root / "screen_87.jpg"
    info = root / "screen_88.jpg"
    if not back.exists() or not info.exists():
        return
    import cv2

    def load(path: Path) -> np.ndarray:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

    first = load(back)
    second = load(info)
    before = read_panel(first)
    after = read_panel(second)
    assert before["text"] == "返回"
    assert after["text"] == "信息"
    assert blocks_differ(first, second, before, after) is True
    assert blocks_differ(first, first.copy(), before, before) is False


def _sequence_chest(buttons):
    return {
        "wrist_contact": True,
        "panel_visible": True,
        "depth_available": True,
        "using_last_target": False,
        "buttons_torso_m": buttons,
    }


class _Controller:
    def __init__(self, calls):
        self.calls = calls

    def ramp_arm(self, arm, deg, seconds=0, dt=0.02):
        self.calls.append(("ramp", tuple(deg)))

    def measured_arm_rad(self, arm):
        return [0.1] * 7

    def commanded_arm_rad(self, arm):
        return [0.1] * 7

    def move_cartesian(self, arm, goal, rpy, seconds):
        self.calls.append(("clear", tuple(float(value) for value in goal)))
        return True, "ok"

    def release(self, seconds=1.2):
        self.calls.append(("release", seconds))


class _IK:
    def tip_xyz(self, q):
        return (0.50, 0.08, 0.42)


def _mover(calls, frames):
    buttons = {
        "button_1": [0.40, 0.10, 0.20],
        "button_2": [0.42, 0.00, 0.20],
    }
    mover = ArmMover(
        telemetry=lambda: _sequence_chest(buttons),
        confirm=True,
        direct_pbvs_poke=True,
        log=lambda _message: None,
    )
    mover.controller = _Controller(calls)
    mover.ik = _IK()
    mover._drop_arm_sdk = lambda *_args, **_kwargs: calls.append(("drop",))
    mover.screen_settle_s = 0
    mover.read_screen = lambda: frames.pop(0)
    mover.set_screen_progress(True)
    mover.set_sequence_mode("direct_pbvs")
    mover.set_button_sequence(["button_1", "button_2"])

    def direct(telem, release=True):
        calls.append(("direct", telem["target"], release, mover.status()["message"]))
        mover.phase = "direct_pbvs_complete"

    def hybrid(name, goal):
        raise AssertionError(name)

    mover._direct_pbvs_once = direct
    mover._run_hybrid_button = hybrid
    return mover


def test_screen_progress_repeats_a_press_until_the_panel_changes():
    calls = []
    frames = [_half(True), _half(True), _half(False), _half(True)]
    mover = _mover(calls, frames)
    mover._run_button_sequence()
    pressed = [item[:3] for item in calls if item[0] == "direct"]
    assert pressed == [
        ("direct", "button_1", False),
        ("direct", "button_1", False),
        ("direct", "button_2", False),
    ]
    notes = [item[3] for item in calls if item[0] == "direct"]
    assert "【重试】" not in notes[0]
    assert "【重试】第1次" in notes[1]
    assert any("【重试】第1次" in line for line in mover.status()["screen_history"])
    assert any("按到了" in line and "Button1" in line for line in mover.status()["screen_history"])
    assert any("按到了" in line and "Button2" in line for line in mover.status()["screen_history"])
    assert mover.phase == "released"


def test_screen_progress_stops_when_the_panel_never_changes():
    calls = []
    frames = [_half(True)] * (1 + SCREEN_PRESS_ATTEMPTS)
    mover = _mover(calls, frames)
    mover.set_button_sequence(["button_1"])
    mover._run_button_sequence()
    pressed = [item for item in calls if item[0] == "direct"]
    assert len(pressed) == SCREEN_PRESS_ATTEMPTS
    assert ("release", 1.2) not in calls
    assert ("ramp", REST_DEG) not in calls
    assert any("屏幕仍无变化" in line for line in mover.status()["screen_history"])
    assert any("【重试】第2次后仍失败" in line for line in mover.status()["screen_history"])
    assert mover._control_stage == "HOLD"


def test_screen_progress_stays_off_until_requested():
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _message: None)
    assert mover.status()["screen_progress"] is False
    assert "关" in mover.set_screen_progress(False)
    mover._running.set()
    assert "Pause" in mover.set_screen_progress(True)
    assert mover.status()["screen_progress"] is False
