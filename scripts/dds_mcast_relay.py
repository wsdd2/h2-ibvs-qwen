#!/usr/bin/env python3
"""Repeat DDS discovery multicast between br123 and the direct cable."""

from __future__ import annotations

import fcntl
import socket
import struct
import sys
import threading

MCAST = "239.255.0.1"
PORT = 7400
SIDES = ("br123", "eth1")


def if_addr(name: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        info = fcntl.ioctl(sock, 0x8915, struct.pack("16s16s", name.encode(), b""))
    finally:
        sock.close()
    return socket.inet_ntoa(info[20:24])


def open_socket(addr: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", PORT))
    membership = struct.pack("4s4s", socket.inet_aton(MCAST), socket.inet_aton(addr))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(addr))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    return sock


def pump(src: socket.socket, dst: socket.socket, mine: set[str]) -> None:
    while True:
        data, addr = src.recvfrom(65535)
        if addr[0] in mine:
            continue
        dst.sendto(data, (MCAST, PORT))


def main() -> None:
    addrs = {name: if_addr(name) for name in SIDES}
    sockets = {name: open_socket(addr) for name, addr in addrs.items()}
    mine = set(addrs.values())
    print(
        f"[dds-relay] br123 {addrs['br123']} <-> eth1 {addrs['eth1']}",
        flush=True,
    )
    threads = [
        threading.Thread(
            target=pump,
            args=(sockets["br123"], sockets["eth1"], mine),
            daemon=True,
        ),
        threading.Thread(
            target=pump,
            args=(sockets["eth1"], sockets["br123"], mine),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
