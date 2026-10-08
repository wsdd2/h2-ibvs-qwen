#!/usr/bin/env python3
"""Copy IBVS_PAD_TOUCH to H2 and start the Flask monitor."""

from __future__ import annotations

import argparse
import os
import posixpath
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = {".git", ".pytest_cache", "__pycache__", "outputs", ".venv", "venv"}


def _paramiko():
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("pip install paramiko   # only needed on the Windows deploy PC") from exc
    return paramiko


def connect(host: str, password: str, port: int = 22):
    paramiko = _paramiko()
    if "@" in host:
        user, addr = host.split("@", 1)
    else:
        user, addr = "unitree", host
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(addr, port=port, username=user, password=password, timeout=12, allow_agent=False, look_for_keys=True)
    return client, user, addr


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_NAMES or part.endswith(".pyc") for part in path.parts)


def upload(client, local: Path, remote_dir: str) -> None:
    sftp = client.open_sftp()
    try:
        _mkdirs(sftp, remote_dir)
        count = 0
        for path in local.rglob("*"):
            if not path.is_file() or _should_skip(path.relative_to(local)):
                continue
            rel = path.relative_to(local).as_posix()
            dest = posixpath.join(remote_dir, rel)
            _mkdirs(sftp, posixpath.dirname(dest))
            if path.suffix == ".sh":
                text = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                with sftp.file(dest, "wb") as handle:
                    handle.write(text)
            else:
                sftp.put(str(path), dest)
            count += 1
        print(f"[ssh] uploaded {count} files to {remote_dir}")
    finally:
        sftp.close()


def _mkdirs(sftp, remote: str) -> None:
    if not remote or remote == "/":
        return
    parts = []
    head = remote
    while head not in {"", "/", "."}:
        parts.append(head)
        head = posixpath.dirname(head)
    for folder in reversed(parts):
        try:
            sftp.stat(folder)
        except OSError:
            sftp.mkdir(folder)


def run(client, command: str) -> tuple[int, str, str]:
    print(f"[ssh] $ {command}")
    _stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return code, out, err


def expand_home(client, remote_dir: str) -> str:
    if remote_dir.startswith("~/"):
        code, out, _err = run(client, "printf %s \"$HOME\"")
        if code == 0 and out.strip():
            return posixpath.join(out.strip(), remote_dir[2:])
    return remote_dir


def _relay(src: socket.socket, chan) -> None:
    try:
        while True:
            readable, _w, _x = __import__("select").select([src, chan], [], [])
            if src in readable:
                data = src.recv(32768)
                if not data:
                    break
                chan.sendall(data)
            if chan in readable:
                data = chan.recv(32768)
                if not data:
                    break
                src.sendall(data)
    finally:
        try:
            src.close()
        except OSError:
            pass
        try:
            chan.close()
        except Exception:
            pass


def _forward_local(client, local_port: int, remote_host: str, remote_port: int) -> None:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", local_port))
    sock.listen(8)
    transport = client.get_transport()
    try:
        while True:
            src, _addr = sock.accept()
            chan = transport.open_channel("direct-tcpip", (remote_host, remote_port), src.getpeername())
            threading.Thread(target=_relay, args=(src, chan), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def deploy(args: argparse.Namespace) -> None:
    client, _user, addr = connect(args.host, args.password)
    try:
        remote = expand_home(client, args.remote_dir)
        if not args.skip_upload:
            upload(client, ROOT, remote)
        extra = " --demo" if args.demo else ""
        start = (
            f"chmod +x {remote}/scripts/*.sh && "
            f"{remote}/scripts/start_web_remote.sh --detach {int(args.port)}{extra}"
        )
        code, _out, err = run(client, start)
        if code != 0:
            raise SystemExit(f"remote start failed: {err}")
        print(f"[ssh] open http://{addr}:{int(args.port)}/")
        if args.tunnel:
            print(f"[ssh] tunnel http://127.0.0.1:{int(args.port)}/  (Ctrl+C to close)")
            _forward_local(client, int(args.port), "127.0.0.1", int(args.port))
    finally:
        if not args.tunnel:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "probe", "check", "camera", "logs"))
    parser.add_argument("--host", default="unitree@192.168.1.3")
    parser.add_argument("--password", default=os.environ.get("IBVS_SSH_PASSWORD", ""))
    parser.add_argument("--remote-dir", default="~/IBVS_PAD_TOUCH")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--tunnel", action="store_true")
    args = parser.parse_args()
    if args.action == "probe":
        client, _user, addr = connect(args.host, args.password)
        try:
            run(client, "hostname; python3 -V; ls /dev/video* 2>/dev/null | head")
            print(f"[ssh] reachable {addr}")
        finally:
            client.close()
        return
    if args.action == "logs":
        client, _user, _addr = connect(args.host, args.password)
        try:
            run(client, "echo '--- web ---'; tail -n 40 /tmp/ibvs_web.log; echo '--- joints ---'; tail -n 20 /tmp/ibvs_joints.log")
        finally:
            client.close()
        return
    if args.action == "camera":
        client, _user, _addr = connect(args.host, args.password)
        try:
            remote = expand_home(client, args.remote_dir)
            upload(client, ROOT, remote)
            run(
                client,
                f"if [ -f /tmp/ibvs_web.pid ]; then kill $(cat /tmp/ibvs_web.pid) || true; sleep 1; fi; "
                f"/usr/bin/python3 {remote}/scripts/camera_probe.py",
            )
        finally:
            client.close()
        return
    if args.action == "check":
        client, _user, _addr = connect(args.host, args.password)
        try:
            remote = expand_home(client, args.remote_dir)
            upload(client, ROOT, remote)
            run(client, f"chmod +x {remote}/scripts/*.sh && {remote}/scripts/remote_env_check.sh")
        finally:
            client.close()
        return
    deploy(args)


if __name__ == "__main__":
    main()
