"""Read-only H2 upper-body joints from Unitree LowState. No arm_sdk, no motion."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

JOINT_FILE = Path("/tmp/ibvs_joints.json")

# Same motor indices as Hand_Eye_Calib / official H2_JointIndex.
UPPER_BODY_JOINTS: tuple[tuple[str, int, str, str], ...] = (
    ("waist_yaw_joint", 12, "腰 yaw", "waist"),
    ("waist_roll_joint", 13, "腰 roll", "waist"),
    ("waist_pitch_joint", 14, "腰 pitch", "waist"),
    ("left_shoulder_pitch_joint", 15, "左肩 pitch", "left_arm"),
    ("left_shoulder_roll_joint", 16, "左肩 roll", "left_arm"),
    ("left_shoulder_yaw_joint", 17, "左肩 yaw", "left_arm"),
    ("left_elbow_joint", 18, "左肘", "left_arm"),
    ("left_wrist_roll_joint", 19, "左腕 roll", "left_arm"),
    ("left_wrist_pitch_joint", 20, "左腕 pitch", "left_arm"),
    ("left_wrist_yaw_joint", 21, "左腕 yaw", "left_arm"),
    ("right_shoulder_pitch_joint", 22, "右肩 pitch", "right_arm"),
    ("right_shoulder_roll_joint", 23, "右肩 roll", "right_arm"),
    ("right_shoulder_yaw_joint", 24, "右肩 yaw", "right_arm"),
    ("right_elbow_joint", 25, "右肘", "right_arm"),
    ("right_wrist_roll_joint", 26, "右腕 roll", "right_arm"),
    ("right_wrist_pitch_joint", 27, "右腕 pitch", "right_arm"),
    ("right_wrist_yaw_joint", 28, "右腕 yaw", "right_arm"),
    ("head_pitch_joint", 29, "头 pitch", "head"),
    ("head_yaw_joint", 30, "头 yaw", "head"),
)

_DDS_READY = False
_DDS_LOCK = threading.Lock()


def _motor_field(motor: Any, name: str, default: float = 0.0) -> float:
    value = getattr(motor, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def joints_from_motors(motors: Any) -> list[dict[str, Any]]:
    """Build the upper-body table from a LowState.motor_state sequence."""
    rows: list[dict[str, Any]] = []
    for name, index, label, group in UPPER_BODY_JOINTS:
        q = dq = tau = float("nan")
        try:
            motor = motors[index]
            q = _motor_field(motor, "q")
            dq = _motor_field(motor, "dq")
            tau = _motor_field(motor, "tau_est", _motor_field(motor, "tau"))
        except (IndexError, TypeError, KeyError):
            pass
        deg = None if math.isnan(q) else q * 180.0 / math.pi
        rows.append(
            {
                "name": name,
                "index": index,
                "label": label,
                "group": group,
                "q": None if math.isnan(q) else q,
                "q_deg": None if deg is None else deg,
                "dq": None if math.isnan(dq) else dq,
                "tau": None if math.isnan(tau) else tau,
            }
        )
    return rows


def demo_joints(t: float) -> list[dict[str, Any]]:
    class _Motor:
        def __init__(self, q: float) -> None:
            self.q = q
            self.dq = 0.01 * math.sin(t)
            self.tau_est = 0.0

    motors = [_Motor(0.08 * math.sin(t + 0.15 * i)) for i in range(31)]
    return joints_from_motors(motors)


def ensure_dds(iface: str = "", domain: int = 0) -> None:
    global _DDS_READY
    with _DDS_LOCK:
        if _DDS_READY:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        try:
            if iface:
                ChannelFactoryInitialize(int(domain), str(iface))
            else:
                ChannelFactoryInitialize(int(domain))
        except Exception as exc:
            if "already" not in str(exc).lower():
                raise
        _DDS_READY = True


@dataclass
class JointSnapshot:
    ok: bool
    source: str
    age_s: Optional[float]
    joints: list[dict[str, Any]]
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "age_s": self.age_s,
            "joints": self.joints,
            "message": self.message,
        }


class UpperBodyJointReader:
    """Subscribe rt/lowstate. Safe to construct when unitree_sdk2py is missing."""

    def __init__(
        self,
        *,
        iface: str = "eth0",
        domain: int = 0,
        topic: str = "rt/lowstate",
        demo: bool = False,
    ) -> None:
        self.iface = iface
        self.domain = int(domain)
        self.topic = topic
        self.demo = bool(demo)
        self._lock = threading.Lock()
        self._latest: Any = None
        self._stamp = 0.0
        self._error = ""
        self._source = "disconnected"
        self._subscriber = None
        if self.demo:
            self._source = "demo"
            return
        # Flask is /usr/bin/python3 (cv2). DDS lives in h1_arm sidecar → /tmp/ibvs_joints.json.
        file_snap = self._from_file(max_age_s=3.0, require_ok=False)
        if file_snap is not None:
            self._source = "file"
            return
        try:
            self._connect()
        except Exception as exc:
            self._error = str(exc)
            self._source = "file" if JOINT_FILE.exists() else "unavailable"

    def _connect(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        ensure_dds(self.iface, self.domain)
        sub = ChannelSubscriber(self.topic, LowState_)
        sub.Init(self._on_lowstate, 10)
        self._subscriber = sub
        self._source = "unitree-dds"
        self._error = ""

    def _on_lowstate(self, msg: Any) -> None:
        with self._lock:
            self._latest = msg
            self._stamp = time.time()

    def snapshot(self) -> JointSnapshot:
        if self.demo:
            return JointSnapshot(True, "demo", 0.0, demo_joints(time.time()), "demo joints")
        with self._lock:
            msg = self._latest
            stamp = self._stamp
            error = self._error
            source = self._source
        if msg is None:
            file_snap = self._from_file(max_age_s=3.0, require_ok=False)
            if file_snap is not None:
                return file_snap
            return JointSnapshot(
                False,
                source,
                None,
                joints_from_motors([]),
                error or "waiting for rt/lowstate / joint sidecar",
            )
        motors = getattr(msg, "motor_state", [])
        return JointSnapshot(True, source, time.time() - stamp, joints_from_motors(motors), "")

    def _from_file(self, max_age_s: float = 3.0, require_ok: bool = True) -> JointSnapshot | None:
        try:
            data = json.loads(JOINT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        written = float(data.get("written_at") or 0.0)
        age = None
        if written > 1e6:
            age = time.time() - written
            if age > float(max_age_s):
                return None
        elif data.get("age_s") is not None:
            age = float(data["age_s"])
        joints = data.get("joints") or joints_from_motors([])
        ok = bool(data.get("ok"))
        if require_ok and not ok:
            return None
        message = "" if ok else str(data.get("message") or "joint sidecar not ok")
        return JointSnapshot(ok, str(data.get("source") or "file"), age, joints, message)
