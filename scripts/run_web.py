#!/usr/bin/env python3
"""Serve chest D435i + H2 upper-body joints over Flask."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.web.dashboard import RealRobotDashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--records-port",
        type=int,
        default=8082,
        help="客户记录页端口，默认 8082。0 表示不打开。",
    )
    parser.add_argument("--serial", default="", help="D435i serial; default is configs/camera_d435i.yaml")
    parser.add_argument("--iface", default="eth0", help="Unitree DDS NIC, e.g. eth0")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--demo", action="store_true", help="No camera / no DDS; local browser smoke")
    parser.add_argument("--no-autostart-camera", action="store_true")
    parser.add_argument(
        "--confirm-arm-motion",
        action="store_true",
        help="点 Raise 时左臂平举侧视、右臂去点触，并发 rt/arm_sdk。必须先 AI 站稳。",
    )
    parser.add_argument(
        "--direct-pbvs-poke",
        action="store_true",
        help="实验模式：不需要先 Raise。按运控姿态直接解到按键坐标。",
    )
    parser.add_argument(
        "--direct-pbvs-max-m",
        type=float,
        default=1.20,
        help="Direct PBVS 允许的一次规划最大距离。垂臂到按钮大约 0.95m，默认 1.20m。",
    )
    parser.add_argument(
        "--direct-pbvs-max-joint-deg",
        type=float,
        default=150.0,
        help="Direct PBVS 单关节一次最大变化。垂臂抬到运控姿态大约 130deg，默认 150deg。",
    )
    parser.add_argument(
        "--eye-to-hand-json",
        default="",
        help="胸口相机 eye-to-hand 标定 JSON；留空自动选最新 eye_to_hand 输出。",
    )
    parser.add_argument(
        "--eye-to-hand-x-bias-m",
        type=float,
        default=0.03,
        help="现场修正 cam2reference 的 Torso X 平移；默认 +0.03m。",
    )
    parser.add_argument(
        "--eye-to-hand-y-bias-m",
        type=float,
        default=0.0,
        help="现场修正 cam2reference 的 Torso Y 平移；正方向向左，默认 0。",
    )
    parser.add_argument(
        "--eye-to-hand-z-bias-m",
        type=float,
        default=0.025,
        help="现场修正 cam2reference 的 Torso Z 平移；默认 +0.025m。",
    )
    parser.add_argument(
        "--pbvs-first",
        action="store_true",
        default=True,
        help="Start 先用胸口 PBVS 对准按钮，再用右腕图像 IBVS 进近。",
    )
    parser.add_argument(
        "--direct-ee-poke",
        action="store_true",
        help="实验模式：不需要 Raise；Start 直接把 right_ee_link 解到按钮点和指定姿态。",
    )
    parser.add_argument("--direct-ee-poke-depth-m", type=float, default=0.005)
    parser.add_argument(
        "--direct-ee-max-m",
        type=float,
        default=0.0,
        help="Direct EE 距离门槛；0 表示关闭。分段 Y/Z→X 模式默认关闭该门槛。",
    )
    parser.add_argument("--direct-ee-max-joint-deg", type=float, default=130.0)
    parser.add_argument(
        "--pbvs-step-m",
        type=float,
        default=0.020,
        help="已不切片。第一阶段 PBVS 一次解到按钮坐标。",
    )
    parser.add_argument(
        "--pbvs-step-seconds",
        type=float,
        default=0.45,
        help="已不使用。一次到位时长按距离计算，约 0.10m/s，1.2–3.5s。",
    )
    parser.add_argument(
        "--pbvs-max-joint-deg",
        type=float,
        default=20.0,
        help="已不使用。一次到位按运控 right_ee 位姿求解，关节变化上限 120deg。",
    )
    parser.add_argument(
        "--right-ee",
        choices=("dex1", "revo2"),
        default="dex1",
        help="右手点按末端。dex1 保持原夹爪；revo2 用食指指尖并拉起右手 Revo2。",
    )
    parser.add_argument(
        "--revo2-bridge",
        default="",
        help="右手串口在机器人 PC2 上时填写，例如 192.168.123.162:8091。Orin 只发送目标姿势。",
    )
    args = parser.parse_args()
    dash = RealRobotDashboard(
        host=args.host,
        port=args.port,
        serial=args.serial,
        jpeg_quality=args.jpeg_quality,
        iface=args.iface,
        demo=args.demo,
        autostart_camera=not args.no_autostart_camera,
        confirm_arm_motion=args.confirm_arm_motion,
        direct_pbvs_poke=args.direct_pbvs_poke,
        direct_pbvs_max_m=args.direct_pbvs_max_m,
        direct_pbvs_max_joint_deg=args.direct_pbvs_max_joint_deg,
        eye_to_hand_json=args.eye_to_hand_json,
        eye_to_hand_x_bias_m=args.eye_to_hand_x_bias_m,
        eye_to_hand_y_bias_m=args.eye_to_hand_y_bias_m,
        eye_to_hand_z_bias_m=args.eye_to_hand_z_bias_m,
        pbvs_first=args.pbvs_first,
        direct_ee_poke=args.direct_ee_poke,
        direct_ee_poke_depth_m=args.direct_ee_poke_depth_m,
        direct_ee_max_m=args.direct_ee_max_m,
        direct_ee_max_joint_deg=args.direct_ee_max_joint_deg,
        pbvs_step_m=args.pbvs_step_m,
        pbvs_step_seconds=args.pbvs_step_seconds,
        pbvs_max_joint_deg=args.pbvs_max_joint_deg,
        right_ee=args.right_ee,
        records_port=args.records_port,
        revo2_bridge=args.revo2_bridge,
    )

    interrupting = False

    def _interrupt(signum, frame) -> None:
        nonlocal interrupting
        if interrupting:
            print(f"[ibvs-web] ignore repeated signal {signum}; arm is lowering", flush=True)
            return
        interrupting = True
        print(f"[ibvs-web] signal {signum}, lowering the right arm then exit", flush=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt)
    try:
        dash.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dash.shutdown()


if __name__ == "__main__":
    main()
