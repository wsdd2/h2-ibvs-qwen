# -*- coding: utf-8 -*-
"""Right Revo2 over the Unitree CDC Modbus port.

This hand's firmware (seen as 1.0.20.U) answers status reads but ignores
``stark_set_finger_positions_and_durations``. Speed commands do move the
motors, so a pose is reached by closing the loop on ``stark_set_finger_speeds``
and then writing zero speed.
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Sequence

RIGHT_PORT = "/dev/serial/by-id/usb-Unitree_Unitree_CDC_COM_26674C9F1974-if00"
RIGHT_SLAVE_ID = 127
BAUD = 460800
PROTOCOL_MODBUS = 1
HARDWARE_REVO2_BASIC = 10
TOLERANCE = 25
MIN_SPEED = 160
MAX_SPEED = 700
SERVO_TIMEOUT_S = 3.5
SERVO_PERIOD_S = 0.05

_LOCK = threading.Lock()
_CANCEL = threading.Event()
_LIB = None


def revo2_test_yaml_candidates() -> list[Path]:
    """Motion team keeps the finger numbers in the source yaml. Save that file."""
    candidates = []
    env = os.environ.get("REVO2_TEST_YAML", "")
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates.append(home / "h2arm_control_ws/src/h2arm_control/config/revo2/revo2_test.yaml")
    candidates.append(
        home / "h2arm_control_ws/install/h2arm_control/share/h2arm_control/config/revo2/revo2_test.yaml"
    )
    return candidates


def parse_revo2_test(text: str) -> dict:
    """Read the fields ControlRevo2Hand uses. Missing keys stay absent."""
    parsed: dict = {}
    port = re.search(r'^\s*port:\s*"([^"]+)"', text, re.M)
    slave = re.search(r"^\s*slave_id:\s*(\d+)", text, re.M)
    baud = re.search(r"^\s*baudrate:\s*(\d+)", text, re.M)
    pose = re.search(r"^\s*test_pose:\s*\[([^\]]+)\]", text, re.M)
    if port:
        parsed["port"] = port.group(1)
    if slave:
        parsed["slave_id"] = int(slave.group(1))
    if baud:
        parsed["baudrate"] = int(baud.group(1))
    if pose:
        numbers = [int(item.strip()) for item in pose.group(1).split(",") if item.strip()]
        if len(numbers) != 6:
            raise ValueError(f"test_pose 需要 6 个数，读到 {numbers}")
        parsed["test_pose"] = numbers
    return parsed


def load_revo2_config() -> dict:
    """Saved source yaml wins. Built-in values are only the fallback."""
    config = {
        "port": RIGHT_PORT,
        "slave_id": RIGHT_SLAVE_ID,
        "baudrate": BAUD,
        "test_pose": [500, 0, 0, 900, 900, 900],
        "path": None,
    }
    path = next((item for item in revo2_test_yaml_candidates() if item.is_file()), None)
    if path is None:
        return config
    parsed = parse_revo2_test(path.read_text(encoding="utf-8"))
    config.update(parsed)
    config["path"] = path
    return config


class _MotorStatus(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.c_uint16 * 6),
        ("speeds", ctypes.c_int16 * 6),
        ("currents", ctypes.c_int16 * 6),
        ("states", ctypes.c_uint8 * 6),
    ]


def device_positions(pose: Sequence[float]) -> list[int]:
    """Accept either normalized 0..1 or the device's 0..1000 integers."""
    values = [float(value) for value in pose]
    if len(values) != 6:
        raise ValueError("右手 Revo2 需要 6 个关节")
    if all(0.0 <= value <= 1.0 for value in values):
        values = [value * 1000.0 for value in values]
    out = []
    for value in values:
        number = int(round(value))
        if number < 0 or number > 1000:
            raise ValueError(f"右手 Revo2 关节 {number} 超出 0 到 1000")
        out.append(number)
    return out


def speeds_toward(target: Sequence[int], current: Sequence[int]) -> list[int]:
    """Positive speed closes a finger, negative speed opens it."""
    speeds = []
    for goal, value in zip(target, current):
        error = int(goal) - int(value)
        if abs(error) <= TOLERANCE:
            speeds.append(0)
            continue
        magnitude = min(MAX_SPEED, max(MIN_SPEED, abs(error)))
        speeds.append(magnitude if error > 0 else -magnitude)
    if len(speeds) != 6:
        raise ValueError("右手 Revo2 需要 6 个关节")
    return speeds


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    candidates = []
    env = os.environ.get("REVO2_STARK_LIB", "")
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates.extend(
        [
            home
            / "h2arm_control_ws/src/h2arm_control/vendor/brainco-hand-sdk/dist/shared/linux/libbc_stark_sdk.so",
            home / "h2arm_control_ws/install/h2arm_control/lib/libbc_stark_sdk.so",
        ]
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError("找不到带 stark_set_finger_speeds 的 libbc_stark_sdk.so")
    lib = ctypes.CDLL(str(path))
    lib.modbus_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.modbus_open.restype = ctypes.c_void_p
    lib.close_device_handler.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
    lib.close_device_handler.restype = None
    lib.stark_set_hardware_type.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_int]
    lib.stark_set_hardware_type.restype = None
    lib.stark_get_motor_status.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
    lib.stark_get_motor_status.restype = ctypes.c_void_p
    lib.free_motor_status_data.argtypes = [ctypes.c_void_p]
    lib.free_motor_status_data.restype = None
    lib.stark_set_finger_speeds.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
    ]
    lib.stark_set_finger_speeds.restype = None
    _LIB = lib
    return lib


def _open(lib, port: str, baud: int, slave: int):
    handle = lib.modbus_open(port.encode("utf-8"), int(baud))
    if not handle:
        raise RuntimeError(f"打不开右手串口 {port}")
    lib.stark_set_hardware_type(handle, int(slave), HARDWARE_REVO2_BASIC)
    return handle


def _read_positions(lib, handle, slave: int) -> list[int] | None:
    pointer = lib.stark_get_motor_status(handle, int(slave))
    if not pointer:
        return None
    status = ctypes.cast(pointer, ctypes.POINTER(_MotorStatus)).contents
    values = [int(value) for value in status.positions]
    lib.free_motor_status_data(pointer)
    return values


def _write_speeds(lib, handle, slave: int, speeds: Sequence[int]) -> None:
    values = (ctypes.c_int16 * 6)(*[int(value) for value in speeds])
    lib.stark_set_finger_speeds(handle, int(slave), values, 6)


def move_right_hand(pose: Sequence[float], duration_ms: int = 1000) -> list[int]:
    """Drive the right hand to a 0..1000 pose, then stop the motors."""
    del duration_ms
    positions = device_positions(pose)
    cfg = load_revo2_config()
    port = str(cfg["port"])
    baud = int(cfg["baudrate"])
    slave = int(cfg["slave_id"])
    _CANCEL.clear()
    with _LOCK:
        lib = _library()
        handle = _open(lib, port, baud, slave)
        measured = None
        try:
            deadline = time.time() + SERVO_TIMEOUT_S
            while time.time() < deadline and not _CANCEL.is_set():
                measured = _read_positions(lib, handle, slave)
                if measured is None:
                    break
                speeds = speeds_toward(positions, measured)
                _write_speeds(lib, handle, slave, speeds)
                if all(speed == 0 for speed in speeds):
                    break
                time.sleep(SERVO_PERIOD_S)
            _write_speeds(lib, handle, slave, [0, 0, 0, 0, 0, 0])
            latest = _read_positions(lib, handle, slave)
            if latest is not None:
                measured = latest
        finally:
            try:
                _write_speeds(lib, handle, slave, [0, 0, 0, 0, 0, 0])
            except Exception:
                pass
            lib.close_device_handler(handle, PROTOCOL_MODBUS)
    if measured is None:
        raise RuntimeError("读不到右手 Revo2 位置")
    return measured


def stop_right_hand() -> None:
    """Cancel an in-progress pose and command zero speed."""
    cfg = load_revo2_config()
    _CANCEL.set()
    with _LOCK:
        lib = _library()
        handle = _open(lib, str(cfg["port"]), int(cfg["baudrate"]), int(cfg["slave_id"]))
        try:
            _write_speeds(lib, handle, int(cfg["slave_id"]), [0, 0, 0, 0, 0, 0])
        finally:
            lib.close_device_handler(handle, PROTOCOL_MODBUS)
