"""BrainCo Revo2 DDS hand. The default object is the left hand.

The right hand is created only when the web is started with --right-ee revo2.
Dex1 remains the default poke tool.
"""

from __future__ import annotations

import os
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
# Stark speed is 1..1000. dq=1 was only creeping the three fingers to about 0.2 in 4 s.
FINGER_SPEED = 1.0


def index_point_q() -> tuple[float, float, float, float, float, float]:
    """Index straight, every other finger closed."""
    pose = [CLOSED] * len(JOINTS)
    pose[INDEX] = OPEN
    return tuple(pose)  # type: ignore[return-value]


# Device test pose, 0..1000. DDS publishes the same order divided by 1000.
PRESS_Q_RAW = (500, 0, 0, 900, 900, 900)


def press_pose_q() -> tuple[float, float, float, float, float, float]:
    """Right-hand poke pose: thumb 500, thumb-aux and index 0, other fingers 900."""
    return tuple(value / 1000.0 for value in PRESS_Q_RAW)  # type: ignore[return-value]


def fingers_keep_thumb_q(state: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    """Index straight and the three fingers closed. Thumb joints stay as they are."""
    current = _as_normalized(state)
    return (current[0], current[1], 0.0, 0.9, 0.9, 0.9)


def open_q() -> tuple[float, float, float, float, float, float]:
    return tuple(OPEN for _ in JOINTS)  # type: ignore[return-value]


class Revo2Hand:
    """Publish one hand pose on rt/brainco/<side>/{cmd,state}."""

    def __init__(
        self,
        *,
        side: str = "left",
        iface: str = "eth0",
        domain: int = 0,
        bridge: str = "",
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("Revo2 side 只能是 left 或 right")
        self.side = side
        self.label = "左手" if side == "left" else "右手"
        self.cmd_topic = f"rt/brainco/{side}/cmd"
        self.state_topic = f"rt/brainco/{side}/state"
        self.iface = str(iface or "eth0")
        self.domain = int(domain)
        self.bridge = str(bridge or os.environ.get("REVO2_BRIDGE", "")).strip()
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
                "side": self.side,
                "ready": self._ready and self._state is not None,
                "holding": self._pose is not None and not self._stop.is_set(),
                "name": self._name,
                "command": None if self._pose is None else [round(float(v), 3) for v in self._pose],
                "state": None if self._state is None else [round(float(v), 3) for v in self._state],
                "age_s": None if age is None else round(age, 3),
                "error": self._error,
                "joints": list(JOINTS),
                "bridge": self.bridge,
            }

    def point_index(self) -> str:
        return self.hold(index_point_q(), f"{self.label}食指伸出，其余收紧")

    def prepare_outward_then_thumb(self) -> str:
        """Other fingers first. The thumb moves only after those fingers arrive."""
        try:
            self._ensure()
        except Exception as exc:
            return f"{self.label} Revo2 没有接上: {exc}"
        if not self._wait_state(2.0):
            return (
                f"{self.label} Revo2 没有 {self.state_topic}。"
                "先确认 brainco_hand_service 在跑，DDS 网卡是 eth0。"
            )
        with self._lock:
            state = None if self._state is None else tuple(self._state)
        if state is None:
            return f"{self.label}还没有手指状态，先不摆手指"
        message = self.hold(fingers_keep_thumb_q(state), f"{self.label}四指预备，大拇指保持")
        if "四指" not in message:
            return message
        if not self.wait_near(10.0, atol=0.08):
            return f"{self.label}四指还没到位，当前 [{self.state_text()}]，先不举臂"
        return self.hold_press()

    def _press_raw(self) -> tuple[int, ...]:
        """Right hand reads the saved revo2_test.yaml. Left hand keeps the built-in pose."""
        if self.side != "right":
            return PRESS_Q_RAW
        from ibvs_pad_touch.web.revo2_serial import load_revo2_config

        pose = load_revo2_config().get("test_pose")
        if not pose or len(pose) != 6:
            return PRESS_Q_RAW
        return tuple(int(value) for value in pose)

    def hold_press_now(self) -> str:
        """Publish the press pose and return. Does not wait or move the arm."""
        raw = self._press_raw()
        shown = ",".join(str(value) for value in raw)
        return self.hold(tuple(value / 1000.0 for value in raw), f"{self.label}食指指腹预备 {shown}")

    def hold_press(self) -> str:
        raw = self._press_raw()
        shown = ",".join(str(value) for value in raw)
        message = self.hold(tuple(value / 1000.0 for value in raw), f"{self.label}食指指腹预备 {shown}")
        if "食指" not in message:
            return message
        if not self.wait_near(10.0, atol=0.08):
            return f"{self.label}手指还没到预备姿势，当前 [{self.state_text()}]，先不举臂"
        return message

    def state_text(self) -> str:
        with self._lock:
            state = None if self._state is None else tuple(self._state)
        if state is None:
            return "无状态"
        return ", ".join(f"{float(value):.3f}" for value in state)

    def wait_near(self, timeout_s: float = 4.0, atol: float = 0.08) -> bool:
        """Block until the reported finger pose matches the command."""
        with self._lock:
            pose = None if self._pose is None else tuple(self._pose)
        if pose is None:
            return False
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                state = None if self._state is None else tuple(self._state)
            if state is not None and _near_pose(state, pose, atol):
                return True
            time.sleep(0.05)
        return False

    def open_hand(self) -> str:
        return self.hold(open_q(), f"{self.label}张开")

    def stop(self) -> str:
        self._stop.set()
        if self.side == "right":
            try:
                if self.bridge:
                    from ibvs_pad_touch.web.revo2_bridge import stop_over_bridge

                    stop_over_bridge(self.bridge)
                else:
                    from ibvs_pad_touch.web.revo2_serial import stop_right_hand

                    stop_right_hand()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                return f"{self.label}停止失败: {exc}"
        with self._lock:
            self._pose = None
            self._name = ""
        return f"已停止发送{self.label} Revo2"

    def hold(self, pose: Sequence[float], name: str) -> str:
        if self.side == "right":
            return self._hold_serial(pose, name)
        q = _checked_pose(pose)
        try:
            self._ensure()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            return f"{self.label} Revo2 没有接上: {exc}"
        if not self._wait_state(2.0):
            return (
                f"{self.label} Revo2 没有 {self.state_topic}。"
                "先确认 brainco_hand_service 在跑，DDS 网卡是 eth0。"
            )
        with self._lock:
            self._pose = q
            self._name = name
            self._error = ""
        self._stop.clear()
        self._ensure_thread()
        return name

    def _hold_serial(self, pose: Sequence[float], name: str) -> str:
        from ibvs_pad_touch.web.revo2_serial import device_positions

        try:
            if self.bridge:
                from ibvs_pad_touch.web.revo2_bridge import move_over_bridge

                sent = move_over_bridge(self.bridge, pose)
            else:
                from ibvs_pad_touch.web.revo2_serial import move_right_hand

                sent = move_right_hand(pose)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            where = "桥" if self.bridge else "串口"
            return f"{self.label}{where}没有写上: {exc}"
        requested = device_positions(pose)
        shown = tuple(value / 1000.0 for value in sent)
        with self._lock:
            self._pose = tuple(value / 1000.0 for value in requested)
            self._state = shown
            self._state_at = time.time()
            self._name = name
            self._error = ""
            self._ready = True
        gap = max(abs(int(a) - int(b)) for a, b in zip(sent, requested))
        if gap > 80:
            return f"{name}，测到 {list(sent)}，还没到目标 {requested}"
        return f"{name}，测到 {list(sent)}"

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
        pub = ChannelPublisher(self.cmd_topic, MotorCmds_)
        pub.Init()
        sub = ChannelSubscriber(self.state_topic, MotorStates_)
        sub.Init(self._on_state, 10)
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in JOINTS]
        for cmd in msg.cmds:
            cmd.q = OPEN
            cmd.dq = FINGER_SPEED
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
        self._thread = threading.Thread(target=self._loop, name=f"revo2-{self.side}", daemon=True)
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
                    cmd.dq = FINGER_SPEED
                pub.Write(msg)
            time.sleep(period)


LeftRevo2 = Revo2Hand


def _as_normalized(state: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    values = [float(value) for value in list(state)[: len(JOINTS)]]
    if len(values) < len(JOINTS):
        raise ValueError("Revo2 状态不足 6 个关节")
    if any(abs(value) > 1.5 for value in values):
        values = [value / 1000.0 for value in values]
    return tuple(values)  # type: ignore[return-value]


def _near_pose(state: Sequence[float], pose: Sequence[float], atol: float) -> bool:
    try:
        values = _as_normalized(state)
    except ValueError:
        return False
    return all(abs(actual - float(target)) <= atol for actual, target in zip(values, pose))


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
