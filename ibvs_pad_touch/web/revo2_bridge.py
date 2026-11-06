# -*- coding: utf-8 -*-
"""Right Revo2 pose bridge.

The hand USB stays on the robot PC2. That machine runs this server, opens the
serial port, and closes the loop with stark_set_finger_speeds. The Orin sends
one 6-joint target and waits for the measured pose.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Callable, Sequence

DEFAULT_PORT = 8091
MOVE_TIMEOUT_S = 8.0


def parse_bridge(text: str) -> tuple[str, int]:
    """Accept host or host:port. The default port is 8091."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Revo2 桥地址是空的")
    if ":" not in raw:
        return raw, DEFAULT_PORT
    host, port_text = raw.rsplit(":", 1)
    if not host or not port_text.isdigit():
        raise ValueError(f"Revo2 桥地址无法解析: {text}")
    port = int(port_text)
    if port <= 0 or port > 65535:
        raise ValueError(f"Revo2 桥端口超出范围: {text}")
    return host, port


def move_over_bridge(address: str, pose: Sequence[float], timeout_s: float = MOVE_TIMEOUT_S) -> list[int]:
    """Send a target pose and return the 6 measured device positions."""
    host, port = parse_bridge(address)
    reply = _exchange(host, port, {"cmd": "move", "pose": [float(value) for value in pose]}, timeout_s)
    measured = reply.get("measured")
    if not isinstance(measured, list) or len(measured) != 6:
        raise RuntimeError("Revo2 桥没有返回 6 个关节")
    return [int(value) for value in measured]


def stop_over_bridge(address: str, timeout_s: float = MOVE_TIMEOUT_S) -> None:
    """Ask the bridge to cancel the pose and write zero speed."""
    host, port = parse_bridge(address)
    _exchange(host, port, {"cmd": "stop"}, timeout_s)


def serve(host: str = "0.0.0.0", port: int = DEFAULT_PORT, move=None, stop=None) -> None:
    """Listen until the process is interrupted. Each request uses the local serial."""
    move_fn, stop_fn = _handlers(move, stop)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    bound = server.getsockname()
    print(f"[revo2-bridge] {bound[0]}:{bound[1]} 右手串口在本机", flush=True)
    try:
        serve_until(server, move_fn, stop_fn, threading.Event())
    finally:
        server.close()


def serve_until(server: socket.socket, move, stop, done: threading.Event) -> None:
    """Accept connections until done is set. Used by the process and by tests."""
    server.settimeout(0.2)
    server.listen(4)
    while not done.is_set():
        try:
            conn, _addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            if done.is_set():
                return
            raise
        threading.Thread(
            target=_handle,
            args=(conn, move, stop),
            name="revo2-bridge",
            daemon=True,
        ).start()


def _handlers(move, stop) -> tuple[Callable, Callable]:
    if move is not None and stop is not None:
        return move, stop
    from ibvs_pad_touch.web.revo2_serial import move_right_hand, stop_right_hand

    return move or move_right_hand, stop or stop_right_hand


def _exchange(host: str, port: int, payload: dict, timeout_s: float) -> dict:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        conn = socket.create_connection((host, int(port)), timeout=min(2.0, timeout_s))
    except OSError as exc:
        raise RuntimeError(f"连不上 Revo2 桥 {host}:{port}: {exc}") from exc
    try:
        conn.settimeout(timeout_s)
        conn.sendall(line.encode("utf-8"))
        reply = _read_line(conn)
    except socket.timeout as exc:
        raise RuntimeError(f"Revo2 桥 {host}:{port} 没有在 {timeout_s:.0f}s 内回复") from exc
    finally:
        conn.close()
    if not isinstance(reply, dict):
        raise RuntimeError("Revo2 桥回复不是一条命令")
    if not reply.get("ok"):
        raise RuntimeError(str(reply.get("error") or "Revo2 桥拒绝"))
    return reply


def _read_line(conn: socket.socket) -> dict:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 65536:
            raise RuntimeError("Revo2 桥回复过长")
    if b"\n" not in buf:
        raise RuntimeError("Revo2 桥回复中断")
    text = buf.split(b"\n", 1)[0].decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Revo2 桥回复无法解析") from exc


def _handle(conn: socket.socket, move, stop) -> None:
    try:
        conn.settimeout(2.0)
        request = _read_line(conn)
        if not isinstance(request, dict):
            raise RuntimeError("Revo2 桥命令无法解析")
        conn.settimeout(None)
        command = str(request.get("cmd") or "")
        if command == "move":
            measured = move(request.get("pose"))
            reply = {"ok": True, "measured": [int(value) for value in measured]}
        elif command == "stop":
            stop()
            reply = {"ok": True}
        else:
            reply = {"ok": False, "error": f"未知命令 {command}"}
    except Exception as exc:
        reply = {"ok": False, "error": str(exc)}
    try:
        conn.sendall((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
    except OSError:
        pass
    finally:
        conn.close()
