import socket
import threading

from ibvs_pad_touch.web.revo2_bridge import (
    move_over_bridge,
    parse_bridge,
    serve_until,
    stop_over_bridge,
)
from ibvs_pad_touch.web.revo2_hand import Revo2Hand
from ibvs_pad_touch.web.revo2_serial import device_positions


def _listen(move, stop):
    done = threading.Event()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    thread = threading.Thread(target=serve_until, args=(server, move, stop, done), daemon=True)
    thread.start()
    return server, port, done, thread


def test_parse_bridge_accepts_host_and_port():
    assert parse_bridge("192.168.123.162") == ("192.168.123.162", 8091)
    assert parse_bridge("192.168.123.162:8091") == ("192.168.123.162", 8091)


def test_bridge_returns_the_measured_pose_and_stop():
    seen = {}

    def move(pose):
        seen["pose"] = list(pose)
        return device_positions(pose)

    def stop():
        seen["stopped"] = True

    server, port, done, _thread = _listen(move, stop)
    try:
        address = f"127.0.0.1:{port}"
        measured = move_over_bridge(address, (0.5, 0, 0, 0.9, 0.9, 0.9))
        assert measured == [500, 0, 0, 900, 900, 900]
        assert seen["pose"] == [0.5, 0.0, 0.0, 0.9, 0.9, 0.9]
        stop_over_bridge(address)
        assert seen["stopped"] is True
    finally:
        done.set()
        server.close()


def test_right_hand_sends_the_press_pose_over_the_bridge():
    seen = {}

    def move(pose):
        seen["pose"] = device_positions(pose)
        return seen["pose"]

    server, port, done, _thread = _listen(move, lambda: None)
    try:
        hand = Revo2Hand(side="right", bridge=f"127.0.0.1:{port}")
        message = hand.hold_press_now()
        assert seen["pose"] == [500, 0, 0, 900, 900, 900]
        assert "500,0,0,900,900,900" in message
        assert hand.status()["bridge"].endswith(str(port))
    finally:
        done.set()
        server.close()


def test_missing_bridge_is_reported_without_opening_a_serial_port():
    hand = Revo2Hand(side="right", bridge="127.0.0.1:1")
    message = hand.hold_press_now()
    assert message.startswith("右手桥没有写上")
    assert "连不上 Revo2 桥" in message
