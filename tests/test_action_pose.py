# -*- coding: utf-8 -*-
"""Action-end pose lines stay on the operator page."""
from ibvs_pad_touch.web.arm_motion import ArmMover


class _Arm:
    def measured_arm_rad(self, _arm):
        return [0.1] * 7


class _IK:
    def tip_xyz(self, _q):
        return (0.610, -0.110, -0.420)


def _mover() -> ArmMover:
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _message: None)
    mover.controller = _Arm()
    mover.ik = _IK()
    mover._thread = None
    return mover


def test_unfinished_action_keeps_tip_and_goal():
    mover = _mover()
    mover._begin_action_pose("Button1", (0.600, -0.115, -0.424))
    mover._finish_direct_pbvs(False, "Direct PBVS 实测未到位")
    line = mover.status()["action_pose_history"][-1]
    assert line.startswith("Button1 未到位")
    assert "到位 X 610 Y -110 Z -420 mm" in line
    assert "目标 X 600 Y -115 Z -424 mm" in line
    assert mover._action_pose_pending is False


def test_release_mid_action_records_the_same_pair():
    mover = _mover()
    mover.controller = None
    mover._begin_action_pose("Button2", (0.500, 0.0, 0.200))
    mover.release_for_operator()
    line = mover.status()["action_pose_history"][-1]
    assert "Button2 已释放" in line
    assert "到位 未读到" in line
    assert "目标 X 500 Y 0 Z 200 mm" in line
    again = len(mover.status()["action_pose_history"])
    mover.release_for_operator()
    assert len(mover.status()["action_pose_history"]) == again


def test_direct_reject_without_a_finish_still_records():
    mover = _mover()
    mover._direct_pbvs_once({"target": "button_1"})
    line = mover.status()["action_pose_history"][-1]
    assert line.startswith("Button1 未到位")
    assert "目标 未读到" in line


if __name__ == "__main__":
    test_unfinished_action_keeps_tip_and_goal()
    test_release_mid_action_records_the_same_pair()
    test_direct_reject_without_a_finish_still_records()
    print("ok")
