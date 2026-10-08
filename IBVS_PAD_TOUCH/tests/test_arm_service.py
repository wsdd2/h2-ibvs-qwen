import json
import time

from ibvs_pad_touch.web.arm_motion import DIRECT_EE_RPY_DEG, ArmMover
from ibvs_pad_touch.web.arm_service import (
    SERVICE_CARTESIAN,
    SERVICE_DELTA,
    SERVICE_IK_POS_ALLOW_M,
    SERVICE_IK_ROT_ALLOW_RAD,
    SERVICE_JOINT,
    H2ArmServiceController,
    call_timeout_message,
    explain_service_failure,
    service_missing_message,
    write_cartesian_yaml,
)


def test_missing_service_names_the_domain_instead_of_a_raw_timeout():
    missing = service_missing_message("/h2_arm/singlearmcartesian")
    assert "h2_arm_service_node" in missing
    assert "ROS_DOMAIN_ID" in missing
    assert "timed out" not in missing
    assert "bash" not in call_timeout_message("/h2_arm/singlearmcartesian", 90)


def test_cartesian_yaml_records_the_button_pose(tmp_path):
    path = tmp_path / "simple_cartesian.yaml"
    write_cartesian_yaml(
        path,
        arm="right",
        xyz_m=(0.56, -0.02, 0.44),
        rpy_deg=(-88.8, -4.0, 3.2),
        seconds=2.0,
    )
    text = path.read_text(encoding="utf-8")
    assert 'control_mode: "right"' in text
    assert "0.560000, -0.020000, 0.440000, -88.800000, -4.000000, 3.200000" in text
    assert f"allow_ik_pos_error: {SERVICE_IK_POS_ALLOW_M:.3f}" in text
    assert f"allow_ik_rot_error: {SERVICE_IK_ROT_ALLOW_RAD:.6f}" in text
    assert SERVICE_IK_POS_ALLOW_M >= 0.030
    assert SERVICE_IK_ROT_ALLOW_RAD > 0.031


def test_service_controller_writes_yaml_and_calls_the_trigger(tmp_path):
    calls = []

    def _caller(service, timeout_s):
        calls.append((service, timeout_s))
        return True, "ok"

    joint_file = tmp_path / "joints.json"
    joint_file.write_text(
        json.dumps(
            {
                "ok": True,
                "written_at": time.time(),
                "joints": [
                    {"group": "right_arm", "q": 0.1 * i} for i in range(7)
                ],
            }
        ),
        encoding="utf-8",
    )
    controller = H2ArmServiceController(config_dir=tmp_path, caller=_caller, joint_file=joint_file)
    ok, _detail = controller.move_cartesian("right", (0.5, 0.0, 0.4), (-88.0, -4.0, 3.0), seconds=2.0)
    assert ok
    assert calls[0][0] == SERVICE_CARTESIAN
    assert "target_pose:" in (tmp_path / "single" / "simple_cartesian.yaml").read_text(encoding="utf-8")
    ok, _detail = controller.move_delta("right", (0.006, 0.0, 0.0), seconds=0.45)
    assert ok
    assert calls[1][0] == SERVICE_DELTA
    controller.ramp_arm("left", [1, 2, 3, 4, 5, 6, 7], seconds=1.0, dt=0.02)
    assert calls[2][0] == SERVICE_JOINT
    assert 'control_mode: "left"' in (tmp_path / "single" / "joint_point.yaml").read_text(encoding="utf-8")
    assert controller.measured_arm_rad("right")[1] == 0.1


def test_cartesian_failure_quotes_the_planner_step(tmp_path):
    log = tmp_path / "h2_arm_service.log"
    log.write_text(
        "[IK Error] Failed to converge at step 25 | pos_err: 0.00840934m, rot_err: 0.0307329rad\n",
        encoding="utf-8",
    )
    detail = explain_service_failure("末端笛卡尔轨迹执行失败！", log)
    assert "step 25" in detail
    assert "0.00840934" in detail
    assert "bash" not in detail


def test_direct_pbvs_uses_the_service_instead_of_local_ik():
    sent = []

    class _Controller:
        def measured_arm_rad(self, arm):
            return []

        def commanded_arm_rad(self, arm):
            return None

        def move_cartesian(self, arm, xyz, rpy, seconds=2.0):
            sent.append((arm, tuple(xyz), tuple(rpy), seconds))
            return True, "ok"

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.55, -0.01, 0.42]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = None
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DIRECT_PBVS"
    mover._servo_once()
    assert len(sent) == 2
    assert sent[0][0] == "right"
    assert sent[0][1] == (0.55, -0.01, 0.42)
    assert sent[1][1] == (0.55, -0.01, 0.42)
    assert sent[0][2] == tuple(DIRECT_EE_RPY_DEG)
    assert sent[0][3] == 2.0
    assert "两段" in mover.message


def test_direct_pbvs_service_slows_a_long_reach_then_corrects():
    sent = []
    moved = {"n": 0}

    class _Controller:
        def measured_arm_rad(self, arm):
            return [0.0] * 7

        def commanded_arm_rad(self, arm):
            return None

        def move_cartesian(self, arm, xyz, rpy, seconds=2.0):
            sent.append(seconds)
            moved["n"] += 1
            return True, "ok"

    class _IK:
        def tip_xyz(self, q_rad):
            if moved["n"] >= 1:
                return [0.470, -0.030, 0.440]
            return [0.016, -0.361, -0.309]

    mover = ArmMover(
        telemetry=lambda: {"target_torso_m": [0.477, -0.0237, 0.4472]},
        confirm=True,
        log=lambda _m: None,
    )
    mover.controller = _Controller()
    mover.ik = _IK()
    mover._ibvs_enabled = True
    mover._height_locked = True
    mover._seek_done = True
    mover._control_stage = "DIRECT_PBVS"
    mover.direct_pbvs_max_m = 1.20
    mover._servo_once()
    assert len(sent) == 2
    assert sent[0] > 2.0
    assert sent[0] <= 6.0
    assert sent[1] == 2.0
    assert "两段" in mover.message
