#!/usr/bin/env python3
"""Repeat the latest upper-body command on the custom controller's ROS2 topic.

Run this on the H2, in the custom controller environment:

    source ~/.lj_ws_env
    python3 ~/IBVS_PAD_TOUCH/scripts/custom_sdk_bridge.py

The web sends one JSON line per joint update to 127.0.0.1:8093. This process
republishes that line at 20 Hz so the controller's 0.5 s upper-body timeout
does not drop the arms between samples. It does not start h2_controller and
does not publish rt/arm_sdk.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading

HOST = "0.0.0.0"
PORT = 8093
TOPIC = "/upper_body_joint_command"


def _drop_conda_cyclonedds() -> None:
    """ROS Humble rclpy must load Ubuntu's CycloneDDS.

    ~/.lj_ws_env puts a conda build of libddsc first on LD_LIBRARY_PATH.
    rclpy.init() still returns, then Node() segfaults inside that library.
    Re-exec once with the conda build removed. The running controller is a
    different process and is left alone.
    """
    if os.environ.get("IBVS_CUSTOM_BRIDGE_REEXEC") == "1":
        return
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    home = os.environ.get("CYCLONEDDS_HOME", "")
    if "cyclonedds/install-conda" not in library_path and "install-conda" not in home:
        return
    kept = [
        item
        for item in library_path.split(":")
        if item and "cyclonedds/install-conda" not in item
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(kept)
    os.environ.pop("CYCLONEDDS_HOME", None)
    os.environ["IBVS_CUSTOM_BRIDGE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> None:
    _drop_conda_cyclonedds()
    import rclpy
    from h2_deploy.msg import UpperBodyCommand
    from rclpy.node import Node

    rclpy.init()
    node = Node("ibvs_custom_sdk_bridge")
    publisher = node.create_publisher(UpperBodyCommand, TOPIC, 10)
    latest: dict = {}
    lock = threading.Lock()
    running = True

    def repeat() -> None:
        with lock:
            payload = dict(latest)
        if not payload or payload.get("stop"):
            return
        message = UpperBodyCommand()
        message.position = [float(value) for value in payload["position"]]
        message.velocity = [float(value) for value in payload["velocity"]]
        message.kp = [float(value) for value in payload["kp"]]
        message.kd = [float(value) for value in payload["kd"]]
        message.feedforward_torque = [float(value) for value in payload["feedforward_torque"]]
        publisher.publish(message)

    node.create_timer(0.05, repeat)

    def serve() -> None:
        nonlocal running
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(0.5)
        print(f"[custom-sdk] listening {HOST}:{PORT} -> {TOPIC}", flush=True)
        while running and rclpy.ok():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            with conn:
                buffer = b""
                while running and rclpy.ok():
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        if not raw.strip():
                            continue
                        try:
                            payload = json.loads(raw.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
                        with lock:
                            latest.clear()
                            latest.update(payload)
        server.close()

    threading.Thread(target=serve, name="custom-sdk-listen", daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
