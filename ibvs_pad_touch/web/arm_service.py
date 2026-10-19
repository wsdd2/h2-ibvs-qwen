# -*- coding: utf-8 -*-
"""Move the H2 arms through h2_arm_service_node. This process does not publish arm_sdk.

The shipped services are std_srvs/Trigger and read a yaml on each call. A live
target is written into that yaml, then the matching service is called. The
service node is the only arm_sdk publisher.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from ibvs_pad_touch.web.joints import JOINT_FILE

SERVICE_CARTESIAN = "/h2_arm/singlearmcartesian"
SERVICE_DELTA = "/h2_arm/singlearmdeltacartesian"
SERVICE_JOINT = "/h2_arm/singlearmjoint"
# Plan() rejects the whole move when any Cartesian sample exceeds these.
# The node default is 5 mm / 0.05 rad. A hang-to-button line failed at
# 8.4 mm on step 25, then again at 30.0 mm on step 79, before the hand moved.
SERVICE_IK_POS_ALLOW_M = 0.040
SERVICE_IK_ROT_ALLOW_RAD = math.radians(25.0)
PLANNER_LOG = Path("/tmp/ibvs_h2_arm_service.log")


def resolve_arm_config_dir() -> Path:
    """Install-space config the running h2_arm_service_node actually reads."""
    override = os.environ.get("H2_ARM_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    home = Path.home() / "h2arm_control_ws/install/h2arm_control/share/h2arm_control/config"
    if home.is_dir():
        return home
    return home


def _fmt(values: Sequence[float]) -> str:
    return ", ".join(f"{float(value):.6f}" for value in values)


def write_cartesian_yaml(
    path: Path,
    *,
    arm: str,
    xyz_m: Sequence[float],
    rpy_deg: Sequence[float],
    seconds: float,
) -> None:
    pose = [float(value) for value in xyz_m] + [float(value) for value in rpy_deg]
    text = (
        f'control_mode: "{arm}"\n'
        "feedrate: 0.3\n"
        f"move_duration: {max(0.4, float(seconds)):.3f}\n"
        "control_dt: 0.02\n"
        "arm_kp: 140.0\n"
        "arm_kd: 3.0\n"
        "joint_kp: [160.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0]\n"
        "joint_kd: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]\n"
        f"allow_ik_pos_error: {SERVICE_IK_POS_ALLOW_M:.3f}\n"
        f"allow_ik_rot_error: {SERVICE_IK_ROT_ALLOW_RAD:.6f}\n"
        f"target_pose: [{_fmt(pose)}]\n"
    )
    _atomic_write(path, text)


def write_delta_yaml(
    path: Path,
    *,
    arm: str,
    delta_m: Sequence[float],
    seconds: float,
) -> None:
    delta = [float(value) for value in delta_m] + [0.0, 0.0, 0.0]
    text = (
        f'control_mode: "{arm}"\n'
        "feedrate: 0.3\n"
        f"move_duration: {max(0.3, float(seconds)):.3f}\n"
        "control_dt: 0.02\n"
        "arm_kp: 140.0\n"
        "arm_kd: 3.0\n"
        f"allow_ik_pos_error: {SERVICE_IK_POS_ALLOW_M:.3f}\n"
        f"allow_ik_rot_error: {SERVICE_IK_ROT_ALLOW_RAD:.6f}\n"
        f"delta_pose: [{_fmt(delta)}]\n"
    )
    _atomic_write(path, text)


def write_joint_yaml(
    path: Path,
    *,
    arm: str,
    left_deg: Sequence[float],
    right_deg: Sequence[float],
    seconds: float,
) -> None:
    text = (
        f'control_mode: "{arm}"\n'
        f"move_duration: {max(0.4, float(seconds)):.3f}\n"
        "control_dt: 0.02\n"
        "arm_kp: 140.0\n"
        "arm_kd: 3.0\n"
        f"left_target: [{_fmt(left_deg)}]\n"
        f"right_target: [{_fmt(right_deg)}]\n"
    )
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def ros_shell_prefix() -> str:
    """Match the domain of h2_arm_service_node. setup.bash does not set it."""
    domain = os.environ.get("H2_ARM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "")).strip()
    lines = []
    if domain:
        lines.append(f"export ROS_DOMAIN_ID={domain}")
    localhost = os.environ.get("ROS_LOCALHOST_ONLY", "").strip()
    if localhost:
        lines.append(f"export ROS_LOCALHOST_ONLY={localhost}")
    lines.append("source /opt/ros/humble/setup.bash")
    lines.append(
        'if [ -f "$HOME/h2arm_control_ws/install/setup.bash" ]; then '
        'source "$HOME/h2arm_control_ws/install/setup.bash"; fi'
    )
    return " && ".join(lines)


# Linux comm is 15 characters. h2_arm_service_node shows up as h2_arm_service_.
_SERVICE_NODE_PGREP = (
    ["pgrep", "-x", "h2_arm_service_"],
    ["pgrep", "-f", "[h]2_arm_service_node"],
)
_SERVICE_NODE_PKILL = (
    ["pkill", "-x", "h2_arm_service_"],
    ["pkill", "-f", "[h]2_arm_service_node"],
    ["pkill", "-9", "-x", "h2_arm_service_"],
    ["pkill", "-9", "-f", "[h]2_arm_service_node"],
)


def _service_process_alive() -> bool:
    for args in _SERVICE_NODE_PGREP:
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                timeout=1.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return True
    return False


def service_node_running() -> bool:
    """True when h2_arm_service_node is alive. That process already publishes rt/arm_sdk."""
    if _service_process_alive():
        return True
    pid_path = Path("/tmp/ibvs_h2_arm_service.pid")
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return pid > 0 and Path(f"/proc/{pid}").is_dir()


def _signal_pid(pid: int, sig: int) -> None:
    if pid <= 0:
        return
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(pid, sig)
            return
        except OSError:
            pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def stop_service_node(immediate: bool = False) -> bool:
    """Stop h2_arm_service_node so this page is the only rt/arm_sdk publisher."""
    pid_path = Path("/tmp/ibvs_h2_arm_service.pid")
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = 0
    if immediate:
        _signal_pid(pid, signal.SIGKILL)
        for args in _SERVICE_NODE_PKILL:
            try:
                subprocess.run(args, capture_output=True, timeout=1.0, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            pid_path.unlink()
        except OSError:
            pass
        return not service_node_running()
    _signal_pid(pid, signal.SIGTERM)
    time.sleep(0.3)
    if service_node_running():
        _signal_pid(pid, signal.SIGKILL)
        for args in _SERVICE_NODE_PKILL:
            if not service_node_running():
                break
            try:
                subprocess.run(args, capture_output=True, timeout=2.0, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            time.sleep(0.2)
    try:
        pid_path.unlink()
    except OSError:
        pass
    return not service_node_running()


def drop_arm_sdk_now(iface: str = "eth0") -> None:
    """Publish weight 0 and turn arm_sdk off. Used when this page has no SDK holder."""
    domain_text = os.environ.get("ROS_DOMAIN_ID", "").strip()
    domain_id = int(domain_text) if domain_text.isdigit() else 0
    from handeye_calib.h2_arm_sdk import H2ArmSdkController

    sdk = H2ArmSdkController(iface=iface or "eth0", domain_id=domain_id, gravity=False)
    sdk.abort()


def start_service_node() -> bool:
    """Start h2_arm_service_node in its own session when the Service link is selected."""
    if service_node_running():
        return True
    humble = Path("/opt/ros/humble/setup.bash")
    install = Path.home() / "h2arm_control_ws/install/setup.bash"
    if not humble.is_file() or not install.is_file():
        return False
    script = (
        "echo $$ > /tmp/ibvs_h2_arm_service.pid\n"
        "set +u\n"
        "source /opt/ros/humble/setup.bash\n"
        'source "$HOME/h2arm_control_ws/install/setup.bash"\n'
        'if [ -n "${ROS_DOMAIN_ID:-}" ]; then export ROS_DOMAIN_ID; fi\n'
        'if [ -n "${ROS_LOCALHOST_ONLY:-}" ]; then export ROS_LOCALHOST_ONLY; fi\n'
        'if [ -n "${RMW_IMPLEMENTATION:-}" ]; then export RMW_IMPLEMENTATION; fi\n'
        "exec ros2 run h2arm_control h2_arm_service_node\n"
    )
    log = open("/tmp/ibvs_h2_arm_service.log", "ab", buffering=0)
    try:
        subprocess.Popen(
            ["setsid", "bash", "-c", script],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
    except OSError:
        log.close()
        return False
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if service_node_running():
            return True
        time.sleep(0.4)
    return service_node_running()


def ros_domain_label() -> str:
    return os.environ.get("H2_ARM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "")).strip() or "0"


def service_missing_message(service: str) -> str:
    return (
        f"找不到 {service}。请先启动 h2_arm_service_node，"
        f"并与网页使用同一个 ROS_DOMAIN_ID（当前 {ros_domain_label()}）。"
    )


def latest_planner_error(log_path: Path | None = None) -> str:
    path = PLANNER_LOG if log_path is None else Path(log_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    hits = [line.strip() for line in lines if "IK Error" in line]
    return hits[-1] if hits else ""


def explain_service_failure(detail: str, log_path: Path | None = None) -> str:
    """The Trigger only says the Cartesian run failed. The step error is in the node log."""
    ik = latest_planner_error(log_path)
    if ik and ("执行失败" in detail or "Planning failed" in detail or not detail.strip()):
        return f"运控直线插值 IK 未通过：{ik}"
    return detail


def call_timeout_message(service: str, timeout_s: float) -> str:
    return (
        f"{service} 在 {timeout_s:.0f}s 内没有返回。"
        "若手臂已在动，运控节点仍在执行，不要立刻再点 Start。"
    )


def _run_ros(script: str, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=max(3.0, float(timeout_s)),
        check=False,
    )


def service_is_listed(service: str, timeout_s: float = 8.0) -> tuple[bool, str]:
    """ros2 service call waits forever when the node is down or on another domain."""
    try:
        proc = _run_ros(f"{ros_shell_prefix()} && ros2 service list", timeout_s)
    except subprocess.TimeoutExpired:
        return False, service_missing_message(service)
    except OSError as exc:
        return False, f"无法执行 ros2：{exc}"
    output = f"{proc.stdout}\n{proc.stderr}"
    if service not in output:
        return False, service_missing_message(service)
    return True, ""


def call_trigger(service: str, timeout_s: float) -> tuple[bool, str]:
    """Block until h2_arm_service_node finishes the Trigger."""
    found, why = service_is_listed(service)
    if not found:
        return False, why
    wait_s = max(90.0, float(timeout_s))
    script = f"{ros_shell_prefix()} && ros2 service call {service} std_srvs/srv/Trigger"
    try:
        proc = _run_ros(script, wait_s)
    except subprocess.TimeoutExpired:
        return False, call_timeout_message(service, wait_s)
    except OSError as exc:
        return False, f"无法执行 ros2：{exc}"
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    compact = output.replace(" ", "")
    ok = proc.returncode == 0 and ("success=True" in compact or "success:true" in compact.lower())
    if not ok and "success: true" in output.lower():
        ok = True
    detail = output.splitlines()[-1] if output else f"exit {proc.returncode}"
    return ok, detail


def arm_rad_from_joint_file(path: Path, arm: str, max_age_s: float = 2.0) -> list[float]:
    """Measured arm joints from the read-only sidecar. No arm_sdk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    written = float(data.get("written_at") or 0.0)
    if written > 1e6 and time.time() - written > max_age_s:
        return []
    group = "left_arm" if arm == "left" else "right_arm"
    rows = [row for row in (data.get("joints") or []) if row.get("group") == group and row.get("q") is not None]
    if len(rows) != 7:
        return []
    return [float(row["q"]) for row in rows]


class H2ArmServiceController:
    """Joint and Cartesian moves via the existing Trigger services."""

    def __init__(
        self,
        config_dir: Path | None = None,
        caller: Callable[[str, float], tuple[bool, str]] | None = None,
        joint_file: Path | None = None,
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else resolve_arm_config_dir()
        self._caller = caller or call_trigger
        self._joint_file = Path(joint_file) if joint_file is not None else JOINT_FILE
        self._commanded: dict[str, list[float]] = {}

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

    def abort(self) -> None:
        """Kill the service node now. The page then drops arm_sdk."""
        stop_service_node(immediate=True)

    def release(self, seconds: float = 0.0) -> None:
        """This process never took arm_sdk. The service node keeps it."""
        _ = seconds

    def move_cartesian(
        self,
        arm: str,
        xyz_m: Sequence[float],
        rpy_deg: Sequence[float],
        seconds: float = 2.0,
    ) -> tuple[bool, str]:
        path = self.config_dir / "single" / "simple_cartesian.yaml"
        write_cartesian_yaml(path, arm=arm, xyz_m=xyz_m, rpy_deg=rpy_deg, seconds=seconds)
        ok, detail = self._caller(SERVICE_CARTESIAN, float(seconds) + 90.0)
        if not ok:
            detail = explain_service_failure(detail)
        return ok, detail

    def move_delta(
        self,
        arm: str,
        delta_m: Sequence[float],
        seconds: float = 0.45,
    ) -> tuple[bool, str]:
        path = self.config_dir / "single" / "delta_cartesian.yaml"
        write_delta_yaml(path, arm=arm, delta_m=delta_m, seconds=seconds)
        return self._caller(SERVICE_DELTA, float(seconds) + 60.0)

    def ramp_arm(
        self,
        arm: str,
        target_deg: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        _ = dt
        if arm not in {"left", "right"} or len(target_deg) != 7:
            raise ValueError("ramp_arm expects left or right and 7 joint degrees")
        left = self.measured_arm_deg("left") or [0.0] * 7
        right = self.measured_arm_deg("right") or [0.0] * 7
        if arm == "left":
            left = [float(value) for value in target_deg]
        else:
            right = [float(value) for value in target_deg]
        path = self.config_dir / "single" / "joint_point.yaml"
        write_joint_yaml(path, arm=arm, left_deg=left, right_deg=right, seconds=seconds)
        ok, detail = self._caller(SERVICE_JOINT, float(seconds) + 60.0)
        if not ok:
            raise RuntimeError(detail)
        self._commanded[arm] = [math.radians(float(value)) for value in target_deg]

    def apply_arm_rad(
        self,
        arm: str,
        target_rad: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        self.ramp_arm(arm, [math.degrees(float(value)) for value in target_rad], seconds, dt)
