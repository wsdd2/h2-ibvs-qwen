"""Left BrainCo Revo2 only. The right hand is still Dex1 and is not commanded."""

from __future__ import annotations

import threading
import time
from typing import Sequence

# Unitree's BrainCo bridge, same order as the Revo2 table:
# 0 thumb, 1 thumb-aux, 2 index, 3 middle, 4 ring, 5 pinky.
# Published q is normalized: 0 open, 1 closed.
JOINTS = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
INDEX = 2
OPEN = 0.0
CLOSED = 1.0
LEFT_CMD = "rt/brainco/left/cmd"
LEFT_STATE = "rt/brainco/left/state"
HOLD_HZ = 50.0


def index_point_q() -> tuple[float, float, float, float, float, float]:
    """Index straight, every other finger closed."""
    pose = [CLOSED] * len(JOINTS)
    pose[INDEX] = OPEN
    return tuple(pose)  # type: ignore[return-value]


def open_q() -> tuple[float, float, float, float, float, float]:
    return tuple(OPEN for _ in JOINTS)  # type: ignore[return-value]


class LeftRevo2:
    """Publish one left-hand pose. It never opens a right-hand topic."""

    def __init__(self, *, iface: str = "eth0", domain: int = 0) -> None:
        self.iface = str(iface or "eth0")
        self.domain = int(domain)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pose: tuple[float, ...] | None = None
        self._name = ""
        self._state: tuple[float, ...] | None = None
        self._state_at = 0.0
        self._error = ""
        self._pub = None
        self._msg = None
        self._ready = False

    def status(self) -> dict[str, object]:
        with self._lock:
            age = None if self._state_at <= 0 else time.time() - self._state_at
            return {
                "side": "left",
                "ready": self._ready and self._state is not None,
                "holding": self._pose is not None and not self._stop.is_set(),
                "name": self._name,
                "command": None if self._pose is None else [round(float(v), 3) for v in self._pose],
                "state": None if self._state is None else [round(float(v), 3) for v in self._state],
                "age_s": None if age is None else round(age, 3),
                "error": self._error,
                "joints": list(JOINTS),
            }

    def point_index(self) -> str:
        return self.hold(index_point_q(), "左手食指伸出，其余收紧")

    def open_hand(self) -> str:
        return self.hold(open_q(), "左手张开")

    def stop(self) -> str:
        self._stop.set()
        with self._lock:
            self._pose = None
            self._name = ""
        return "已停止发送左手 Revo2"

    def hold(self, pose: Sequence[float], name: str) -> str:
        q = _checked_pose(pose)
        try:
            self._ensure()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            return f"左手 Revo2 没有接上: {exc}"
        if not self._wait_state(2.0):
            return (
                "左手 Revo2 没有 rt/brainco/left/state。"
                "先确认左手手部服务在跑，DDS 网卡是 eth0。"
            )
        with self._lock:
            self._pose = q
            self._name = name
            self._error = ""
        self._stop.clear()
        self._ensure_thread()
        return name

    def _wait_state(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._state is not None:
                    return True
            time.sleep(0.02)
        return False

    def _ensure(self) -> None:
        if self._pub is not None:
            return
        from ibvs_pad_touch.web.dds_bind import ensure_unitree_sdk2py
        from ibvs_pad_touch.web.joints import ensure_dds
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        ensure_unitree_sdk2py()
        ensure_dds(self.iface, self.domain)
        pub = ChannelPublisher(LEFT_CMD, MotorCmds_)
        pub.Init()
        sub = ChannelSubscriber(LEFT_STATE, MotorStates_)
        sub.Init(self._on_state, 10)
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in JOINTS]
        for cmd in msg.cmds:
            cmd.q = OPEN
            cmd.dq = 1.0
        self._pub = pub
        self._msg = msg
        self._sub = sub
        self._ready = True

    def _on_state(self, msg) -> None:
        try:
            states = msg.states
            q = tuple(float(states[i].q) for i in range(len(JOINTS)))
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._state = q
            self._state_at = time.time()

    def _ensure_thread(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="revo2-left", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / HOLD_HZ
        while not self._stop.is_set():
            with self._lock:
                pose = self._pose
                msg = self._msg
                pub = self._pub
            if pose is not None and msg is not None and pub is not None:
                for cmd, value in zip(msg.cmds, pose):
                    cmd.q = float(value)
                    cmd.dq = 1.0
                pub.Write(msg)
            time.sleep(period)


def _checked_pose(pose: Sequence[float]) -> tuple[float, ...]:
    if len(pose) != len(JOINTS):
        raise ValueError(f"Revo2 需要 {len(JOINTS)} 个关节")
    out = []
    for value in pose:
        number = float(value)
        if number < OPEN or number > CLOSED:
            raise ValueError(f"Revo2 关节 {number} 超出 0 到 1")
        out.append(number)
    return tuple(out)
