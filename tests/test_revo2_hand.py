from ibvs_pad_touch.web.revo2_hand import (
    INDEX,
    JOINTS,
    PRESS_Q_RAW,
    _near_pose,
    fingers_keep_thumb_q,
    index_point_q,
    open_q,
    press_pose_q,
)
from ibvs_pad_touch.web.revo2_serial import device_positions, parse_revo2_test, speeds_toward


def test_index_point_opens_only_the_index_finger():
    pose = index_point_q()
    assert len(pose) == len(JOINTS)
    assert pose[INDEX] == 0.0
    assert all(value == 1.0 for i, value in enumerate(pose) if i != INDEX)


def test_open_pose_is_all_zeros():
    assert open_q() == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_press_pose_uses_the_device_values():
    pose = press_pose_q()
    assert PRESS_Q_RAW == (500, 0, 0, 900, 900, 900)
    assert pose == tuple(value / 1000.0 for value in PRESS_Q_RAW)
    assert len(pose) == len(JOINTS)
    assert _near_pose((520, 20, 10, 880, 910, 890), pose, 0.08)
    assert not _near_pose((0, 0, 0, 0, 0, 0), pose, 0.08)
    kept = fingers_keep_thumb_q((0.2, 0.4, 0.7, 0.1, 0.1, 0.1))
    assert kept == (0.2, 0.4, 0.0, 0.9, 0.9, 0.9)


def test_right_hand_serial_pose_is_zero_to_one_thousand():
    assert device_positions(press_pose_q()) == [500, 0, 0, 900, 900, 900]
    assert device_positions((500, 0, 0, 900, 900, 900)) == [500, 0, 0, 900, 900, 900]
    assert speeds_toward((500, 0, 0, 900, 900, 900), (500, 0, 0, 111, 132, 221))[:3] == [0, 0, 0]
    assert all(speed > 0 for speed in speeds_toward((900, 900, 900, 900, 900, 900), (0, 0, 0, 0, 0, 0)))
    assert speeds_toward((0, 0, 0, 0, 0, 0), (200, 0, 0, 0, 0, 0))[0] < 0


def test_revo2_yaml_pose_is_the_saved_test_pose():
    text = """
revo2_hand:
  port: "/dev/serial/by-id/usb-Unitree_Unitree_CDC_COM_26674C9F1974-if00"
  slave_id: 127
  baudrate: 460800
  test_pose: [500, 0, 0, 900, 900, 900]                # saved
"""
    parsed = parse_revo2_test(text)
    assert parsed["test_pose"] == [500, 0, 0, 900, 900, 900]
    assert parsed["slave_id"] == 127
    assert parsed["baudrate"] == 460800
    assert parsed["port"].endswith("26674C9F1974-if00")
