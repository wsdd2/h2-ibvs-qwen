"""Waist capture stays fixed while arm commands keep that capture."""
from __future__ import annotations

import math

from ibvs_pad_touch.servo.upper_body_model import UpperBodyModel
from ibvs_pad_touch.servo.waist_lock import (
    arm_delta_for_measured_waist,
    resolve_h2_urdf,
    stamp_waist_command,
    tip_link_name,
    waist_drift,
    waist_lock_from_q,
)
from ibvs_pad_touch.web.arm_motion import RIGHT_ARM_JOINTS, ArmMover


def test_stamp_overwrites_a_requested_waist_move_and_stops_its_velocity():
    command = {12: 1.0, 13: 1.0, 14: 1.0, 22: 0.4}
    dq = {12: 0.2, 22: 0.1}
    lock = waist_lock_from_q({12: 0.11, 13: -0.22, 14: 0.33})
    stamped, velocity = stamp_waist_command(command, dq, lock)
    assert stamped[12] == 0.11
    assert stamped[13] == -0.22
    assert stamped[14] == 0.33
    assert stamped[22] == 0.4
    assert velocity[12] == 0.0 and velocity[13] == 0.0 and velocity[14] == 0.0
    assert velocity[22] == 0.1
    assert lock == {12: 0.11, 13: -0.22, 14: 0.33}


def test_drift_check_does_not_replace_the_capture():
    lock = {12: 0.0, 13: 0.0, 14: 0.0}
    inside = waist_drift(lock, {12: math.radians(0.4), 13: 0.0, 14: 0.0})
    outside = waist_drift(lock, {12: math.radians(1.2), 13: 0.0, 14: 0.0})
    assert inside["ok"] is True
    assert outside["ok"] is False
    assert outside["reason"] == "waist moved; hold approach and relocalize"
    assert lock == {12: 0.0, 13: 0.0, 14: 0.0}


def test_matching_waist_leaves_the_arm_command_unchanged():
    path = resolve_h2_urdf()
    assert path is not None
    model = UpperBodyModel(path)
    tip = tip_link_name(model)
    lock = {12: 0.05, 13: -0.02, 14: 0.01}
    arm = [0.3, -0.4, 0.2, 0.8, 0.1, -0.1, 0.05]
    delta = arm_delta_for_measured_waist(model, lock, arm, lock, tool_offset_m=(0.205, 0.0, 0.0), tip_link=tip)
    assert max(abs(float(value)) for value in delta) < 1e-6


def test_measured_waist_yaw_requests_a_bounded_arm_correction():
    path = resolve_h2_urdf()
    assert path is not None
    model = UpperBodyModel(path)
    tip = tip_link_name(model)
    lock = {12: 0.0, 13: 0.0, 14: 0.0}
    measured = {12: 0.04, 13: 0.0, 14: 0.0}
    arm = [-1.0, -0.7, 0.1, 1.2, 0.0, 0.2, 0.0]
    delta = arm_delta_for_measured_waist(model, lock, arm, measured, tool_offset_m=(0.205, 0.0, 0.0), tip_link=tip)
    assert max(abs(float(value)) for value in delta) > 1e-4
    assert max(abs(float(value)) for value in delta) <= math.radians(2.0) + 1e-12


def test_cartesian_stream_does_not_command_or_stop_for_the_waist():
    written = []

    class _Controller:
        def hold_positions(self):
            return {joint: 0.0 for joint in RIGHT_ARM_JOINTS}

        def measured_arm_rad(self, arm):
            return [0.0] * 7

        def write(self, command, weight=1.0, dq=None):
            written.append((dict(command), dict(dq or {}), weight))

    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _message: None)
    mover.controller = _Controller()
    mover._control_stage = "PBVS_COARSE"
    mover._stream_arm_samples([[0.0] * 7, [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert len(written) == 3
    for command, _dq, weight in written:
        assert weight == 1.0
        assert 12 not in command and 13 not in command and 14 not in command
    assert "腰偏离" not in mover.message
