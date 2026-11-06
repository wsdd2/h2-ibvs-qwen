"""Waist yaw hold pulls a drifted yaw back and caps the restoring torque."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Hand_Eye_Calib"))

from handeye_calib.h2_arm_sdk import (  # noqa: E402
    BODY_YAW_MAX_RATE,
    YAW_HOLD_MAX_STEP_RAD,
    YAW_HOLD_TAU_MAX,
    body_yaw_rate,
    waist_yaw_hold,
)


def test_quiet_yaw_stays_on_the_lock_without_a_kick():
    report = waist_yaw_hold(0.10, 0.10, 0.0, 0.02)
    assert report["active"] is False
    assert abs(float(report["q"]) - 0.10) < 1e-9
    assert float(report["tau"]) == 0.0


def test_rightward_drift_is_pulled_back_with_a_bounded_torque():
    lock = 0.0
    measured = math.radians(-4.0)
    integral = 0.0
    last = None
    for _ in range(80):
        last = waist_yaw_hold(lock, measured, integral, 0.02)
        integral = float(last["integral"])
    assert last is not None
    assert last["active"] is True
    assert float(last["error_deg"]) > 0.0
    assert float(last["q"]) > measured
    assert abs(float(last["q"]) - measured) <= YAW_HOLD_MAX_STEP_RAD + 1e-9
    assert abs(float(last["tau"])) <= YAW_HOLD_TAU_MAX + 1e-9
    assert float(last["tau"]) > 0.0


def test_body_heading_rate_turns_back_and_stays_bounded():
    assert body_yaw_rate(0.0, 0.0) == 0.0
    assert body_yaw_rate(0.0, 0.01) == 0.0
    turned_left = body_yaw_rate(0.0, 0.4)
    assert turned_left < 0.0
    assert abs(turned_left) <= BODY_YAW_MAX_RATE + 1e-9


def test_a_large_yaw_error_does_not_command_the_full_jump():
    measured = math.radians(-20.0)
    report = waist_yaw_hold(0.0, measured, 0.0, 0.02)
    assert abs(float(report["q"]) - measured) <= YAW_HOLD_MAX_STEP_RAD + 1e-9
    assert abs(float(report["tau"])) <= YAW_HOLD_TAU_MAX + 1e-9
