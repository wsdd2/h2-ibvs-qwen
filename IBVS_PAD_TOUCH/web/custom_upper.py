# -*- coding: utf-8 -*-
"""Send H2 upper-body targets to the custom controller, not to rt/arm_sdk.

The custom stack listens on /upper_body_joint_command
(h2_deploy/msg/UpperBodyCommand) in ROS domain 42. A small bridge process
owns that publisher. This module only keeps the latest 14 joint targets and
hands them to the bridge.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Sequence

from ibvs_pad_touch.web.arm_service import JOINT_FILE, arm_rad_from_joint_file

CUSTOM_BRIDGE_HOST = os.environ.get("H2_CUSTOM_BRIDGE_HOST", "192.168.1.3")
CUSTOM_BRIDGE_PORT = int(os.environ.get("H2_CUSTOM_BRIDGE_PORT", "8093"))
UPPER_TOPIC = "/upper_body_joint_command"
# Left arm motors 15-21, then right arm motors 22-28. Same order as the
# custom controller's 14-length UpperBodyCommand.
LEFT_MOTORS = tuple(range(15, 22))
RIGHT_MOTORS = tuple(range(22, 29))
UPPER_MOTORS = LEFT_MOTORS + RIGHT_MOTORS
UPPER_LIMITS = (
    (-2.618, 1.833),
    (-0.517, 2.494),
    (-2.618, 2.618),
    (-0.986, 3.071),
    (-2.618, 2.618),
    (-0.576, 0.576),
    (-1.22, 1.22),
    (-2.618, 1.833),
    (-2.494, 0.517),
    (-2.618, 2.618),
    (-0.986, 3.071),
    (-2.618, 2.618),
    (-0.576, 0.576),
    (-1.22, 1.22),
)


def _clamp_upper(values: Sequence[float]) -> list[float]:
    clamped: list[float] = []
    for value, (low, high) in zip(values, UPPER_LIMITS):
        number = float(value)
        if not math.isfinite(number):
            number = 0.0
        clamped.append(min(high, max(low, number)))
    if len(clamped) != 14:
        raise ValueError("upper command needs 14 joints")
    return clamped


def upper_positions(left_rad: Sequence[float], right_rad: Sequence[float]) -> list[float]:
    """Pack left then right, seven joints each."""
    if len(left_rad) != 7 or len(right_rad) != 7:
        raise ValueError("each arm needs 7 joints")
    return _clamp_upper([float(value) for value in left_rad] + [float(value) for value in right_rad])


class CustomUpperController:
    """Arm holder that publishes through the custom-controller bridge.

    It has no move_cartesian, so the page solves the fingertip locally and
    streams joint targets. Measured joints still come from the joint sidecar.
    """

    custom_sdk = True

    def __init__(
        self,
        sender=None,
        joint_file: Path | None = None,
        kp: float = 140.0,
        kd: float = 3.0,
    ) -> None:
        self._send = sender if sender is not None else _send_bridge_line
        self._joint_file = Path(joint_file) if joint_file is not None else JOINT_FILE
        self.kp = float(kp)
        self.kd = float(kd)
        self._aborted = False
        self._left = [0.0] * 7
        self._right = [0.0] * 7
        measured_left = self.measured_arm_rad("left")
        measured_right = self.measured_arm_rad("right")
        if len(measured_left) == 7:
            self._left = [float(value) for value in measured_left]
        if len(measured_right) == 7:
            self._right = [float(value) for value in measured_right]
        self._commanded = {"left": list(self._left), "right": list(self._right)}

    def measured_arm_rad(self, arm: str) -> list[float]:
        return arm_rad_from_joint_file(self._joint_file, arm)

    def measured_arm_deg(self, arm: str) -> list[float]:
        return [math.degrees(value) for value in self.measured_arm_rad(arm)]

    def commanded_arm_rad(self, arm: str) -> list[float] | None:
        held = self._commanded.get(arm)
        if held:
            return list(held)
        measured = self.measured_arm_rad(arm)
        return measured or None

    def hold_positions(self) -> dict[int, float]:
        packed = upper_positions(self._left, self._right)
        return {motor: packed[index] for index, motor in enumerate(UPPER_MOTORS)}

    def write(self, joints: dict[int, float], weight: float = 1.0, dq: dict[int, float] | None = None) -> None:
        _ = weight
        if self._aborted:
            return
        for index, motor in enumerate(LEFT_MOTORS):
            if motor in joints:
                self._left[index] = float(joints[motor])
        for index, motor in enumerate(RIGHT_MOTORS):
            if motor in joints:
                self._right[index] = float(joints[motor])
        velocity = [0.0] * 14
        if dq:
            for index, motor in enumerate(UPPER_MOTORS):
                if motor in dq and math.isfinite(float(dq[motor])):
                    velocity[index] = float(dq[motor])
        self._commanded["left"] = list(self._left)
        self._commanded["right"] = list(self._right)
        self._publish(velocity)

    def ramp_arm(self, arm: str, target_deg: Sequence[float], seconds: float, dt: float) -> None:
        _ = seconds, dt
        self.apply_arm_rad(arm, [math.radians(float(value)) for value in target_deg], seconds, dt)

    def apply_arm_rad(self, arm: str, target_rad: Sequence[float], seconds: float, dt: float) -> None:
        _ = seconds, dt
        if arm not in {"left", "right"} or len(target_rad) != 7:
            raise ValueError("apply_arm_rad expects left or right and 7 joints")
        if arm == "left":
            self._left = [float(value) for value in target_rad]
        else:
            self._right = [float(value) for value in target_rad]
        self._commanded[arm] = [float(value) for value in target_rad]
        self._publish([0.0] * 14)

    def abort(self) -> None:
        self._aborted = True
        try:
            self._send({"stop": True})
        except Exception:
            pass

    def release(self, seconds: float = 0.0) -> None:
        _ = seconds
        self.abort()

    def _publish(self, velocity: Sequence[float]) -> None:
        position = upper_positions(self._left, self._right)
        gain_p = [float(self.kp)] * 14
        gain_d = [float(self.kd)] * 14
        self._send(
            {
                "position": position,
                "velocity": [float(value) for value in velocity],
                "kp": gain_p,
                "kd": gain_d,
                "feedforward_torque": [0.0] * 14,
            }
        )


def bridge_is_up(host: str = CUSTOM_BRIDGE_HOST, port: int = CUSTOM_BRIDGE_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _send_bridge_line(payload: dict) -> None:
    line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.create_connection((CUSTOM_BRIDGE_HOST, CUSTOM_BRIDGE_PORT), timeout=0.5) as sock:
        sock.sendall(line)


def ensure_custom_bridge() -> bool:
    """Start the ROS2 bridge when this machine has the custom controller env."""
    if bridge_is_up():
        return True
    env_file = Path.home() / ".lj_ws_env"
    script = Path(__file__).resolve().parents[2] / "scripts" / "custom_sdk_bridge.py"
    if not env_file.is_file() or not script.is_file():
        return False
    subprocess.Popen(
        ["bash", "-lc", f"source {env_file} && exec python3 {script}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if bridge_is_up():
            return True
        time.sleep(0.1)
    return False
