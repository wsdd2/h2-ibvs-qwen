import json
import tempfile
import time
from pathlib import Path

import numpy as np

from ibvs_pad_touch.camera.realsense_bind import _is_usable, usable_realsense
from ibvs_pad_touch.camera.image_io import read_depth, write_depth
import ibvs_pad_touch.web.dashboard as dashboard_module
from ibvs_pad_touch.web.dashboard import COMMANDS, RealRobotDashboard
from ibvs_pad_touch.web.joints import UPPER_BODY_JOINTS, UpperBodyJointReader, demo_joints, joints_from_motors


class _Motor:
    def __init__(self, q: float) -> None:
        self.q = q
        self.dq = 0.1
        self.tau_est = 0.2


def test_upper_body_includes_waist_arms_and_head():
    names = [item[0] for item in UPPER_BODY_JOINTS]
    assert "right_shoulder_pitch_joint" in names
    assert "left_wrist_yaw_joint" in names
    assert "head_yaw_joint" in names
    motors = [_Motor(0.01 * i) for i in range(31)]
    rows = joints_from_motors(motors)
    right = next(item for item in rows if item["name"] == "right_elbow_joint")
    assert right["index"] == 25
    assert abs(right["q"] - 0.25) < 1e-9


def test_usable_realsense_is_none_or_has_pipeline():
    class _Stub:
        pass

    assert not _is_usable(_Stub())
    rs = usable_realsense()
    if rs is not None:
        assert hasattr(rs, "pipeline")


def test_joint_sidecar_file_is_enough_without_sdk(tmp_path, monkeypatch):
    path = tmp_path / "ibvs_joints.json"
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "source": "unitree-dds",
                "written_at": time.time(),
                "joints": demo_joints(0.0),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ibvs_pad_touch.web.joints.JOINT_FILE", path)
    reader = UpperBodyJointReader(demo=False)
    snap = reader.snapshot()
    assert snap.ok
    assert len(snap.joints) == len(UPPER_BODY_JOINTS)


def test_demo_joints_have_all_rows():
    rows = demo_joints(0.0)
    assert len(rows) == len(UPPER_BODY_JOINTS)
    assert all(item["q"] is not None for item in rows)


def test_camera_worker_reads_sidecar_bgr(tmp_path):
    import threading

    import numpy as np

    from ibvs_pad_touch.camera.image_io import write_bgr
    from ibvs_pad_touch.web.dashboard import ChestCameraWorker

    dest = tmp_path / "ibvs_cam_chest.jpg"
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (20, 80, 180)
    write_bgr(dest, frame)
    worker = ChestCameraWorker("1", 64, 48, 15, 70, False, name="chest", sidecar_path=str(dest))
    thread = threading.Thread(target=worker._run_sidecar_file, daemon=True)
    thread.start()
    deadline = time.time() + 3
    while time.time() < deadline and worker.status()["state"] != "running":
        time.sleep(0.05)
    worker._stop.set()
    thread.join(timeout=2)
    assert worker.status()["state"] == "running"


def test_camera_worker_reads_sidecar_ppm(tmp_path):
    import threading

    import numpy as np

    from ibvs_pad_touch.camera.image_io import write_ppm
    from ibvs_pad_touch.web.dashboard import ChestCameraWorker

    jpeg = tmp_path / "ibvs_cam_chest.jpg"
    stale = jpeg
    stale.write_bytes(b"not-a-real-jpeg-but-long-enough-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (20, 80, 180)
    write_ppm(jpeg, frame)
    worker = ChestCameraWorker("1", 64, 48, 15, 70, False, name="chest", sidecar_path=str(jpeg))
    thread = threading.Thread(target=worker._run_sidecar_file, daemon=True)
    thread.start()
    deadline = time.time() + 3
    while time.time() < deadline and worker.status()["state"] != "running":
        time.sleep(0.05)
    worker._stop.set()
    thread.join(timeout=2)
    assert worker.status()["state"] == "running"


def test_depth_sidecar_round_trip(tmp_path):
    import numpy as np

    raw = np.array([[0, 1000], [500, 250]], dtype=np.uint16)
    path = tmp_path / "depth.bin"
    write_depth(path, raw, 0.001)
    depth = read_depth(path)
    assert depth is not None
    assert depth.shape == raw.shape
    assert abs(float(depth[0, 1]) - 1.0) < 1e-6


def test_chest_depth_preview_and_hover_value(tmp_path, monkeypatch):
    path = tmp_path / "chest_depth.bin"
    raw = np.full((48, 64), 800, dtype=np.uint16)
    raw[20, 30] = 1234
    write_depth(path, raw, 0.001)
    monkeypatch.setitem(dashboard_module.DEPTH_PATHS, "chest", path)
    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    client = dash.app.test_client()
    preview = client.get("/depth/chest.png")
    assert preview.status_code == 200
    assert preview.mimetype == "image/png"
    value = client.get("/depth/chest/value?u=30&v=20").get_json()
    assert value["ok"] is True
    assert abs(value["depth_m"] - 1.234) < 1e-4
    assert value["size"] == [64, 48]


def test_start_pause_and_button_switch_control_motion_take(tmp_path):
    class _Arm:
        def __init__(self):
            self.enabled = False

        def start(self):
            self.enabled = True
            return "started"

        def pause(self):
            self.enabled = False
            return "paused"

        def release_for_operator(self):
            self.enabled = False
            return "released"

        def stop(self):
            self.enabled = False
            return "stopped"

        def status(self):
            return {
                "ibvs_enabled": self.enabled,
                "phase": "manual" if not self.enabled else "PBVS",
                "control_stage": "MANUAL" if not self.enabled else "PBVS_COARSE",
            }

        def update_joint_readback(self, q):
            return None

        def shutdown(self):
            return None

    dash = RealRobotDashboard(
        host="127.0.0.1",
        port=0,
        demo=True,
        autostart_camera=False,
    )
    dash.arm = _Arm()
    dash.motion_trajectories = dashboard_module.MotionTrajectoryRecorder(tmp_path, hz=50)
    dash._apply_command("start")
    assert dash.motion_trajectories.status()["enabled"] is False
    assert dash.motion_trajectories.status()["recording"] is False
    dash.motion_trajectories.set_enabled(True)
    dash._apply_command("start")
    deadline = time.time() + 1.0
    while time.time() < deadline and dash.motion_trajectories.status()["samples"] < 2:
        time.sleep(0.02)
    assert dash.motion_trajectories.status()["recording"] is True
    dash._apply_command("pause")
    assert dash.motion_trajectories.status()["recording"] is True
    dash._apply_command("button_2")
    assert dash.motion_trajectories.status()["recording"] is False
    rows = dash.motion_trajectories.list()
    assert rows and rows[0]["reason"] == "button_switch"


def test_camera_worker_reads_sidecar_jpeg(tmp_path):
    import threading

    import cv2
    import numpy as np

    from ibvs_pad_touch.web.dashboard import ChestCameraWorker

    jpeg = tmp_path / "ibvs_cam_chest.jpg"
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (20, 80, 180)
    ok, blob = cv2.imencode(".jpg", frame)
    assert ok
    jpeg.write_bytes(blob.tobytes())
    worker = ChestCameraWorker("1", 64, 48, 15, 70, False, name="chest", sidecar_path=str(jpeg))
    thread = threading.Thread(target=worker._run_sidecar_file, daemon=True)
    thread.start()
    deadline = time.time() + 3
    while time.time() < deadline and worker.status()["state"] != "running":
        time.sleep(0.05)
    worker._stop.set()
    thread.join(timeout=2)
    assert worker.status()["state"] == "running"


def test_flask_page_has_reserved_buttons_and_state():
    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=True)
    client = dash.app.test_client()
    html = client.get("/").get_data(as_text=True)
    for label in ("Button 1", "Button 2", "Button 3", "Cycle", "Raise", "Right level", "Start", "组合启动", "急停", "Pause / 细调", "释放双臂", "下垂", "Reset"):
        assert label in html
    assert "/stream/chest.mjpg" in html
    assert "/stream/d435i.mjpg" in html
    assert "/snapshot/chest.jpg" in html
    assert 'id="chestDepth"' in html
    assert "/depth/chest.png" in html
    assert "depthHover(event)" in html
    assert 'id="trajectorySelect"' in html
    assert 'id="trajectoryToggle"' in html
    assert 'id="trajectoryName"' in html
    assert 'id="sequenceList"' in html
    assert 'id="screenShot"' in html
    assert 'id="shadow-actual-x"' in html
    assert 'id="shadow-vit-x"' in html
    assert "ViT shadow 对照" in html
    assert "/shadow/reference" in html
    assert "/trajectories/play" in html
    assert "/trajectories/delete" in html
    assert "删除所选轨迹" in html
    assert "/trajectories/delete_all" in html
    assert "删除全部轨迹" in html
    assert "Save Screenshot" in html
    assert "/dataset/save/" in html
    with tempfile.TemporaryDirectory() as tmp:
        dash.dataset_root = Path(tmp)
        saved = client.post("/dataset/save/chest")
        assert saved.status_code == 200
        assert saved.mimetype == "image/jpeg"
        assert "chest_" in (saved.headers.get("Content-Disposition") or "")
        assert (dash.dataset_root / "labels.txt").read_text(encoding="utf-8").splitlines()[0] == "panel"
        assert list((dash.dataset_root / "chest").glob("chest_*.jpg"))
    assert "chestStill" in html
    assert "stills" in html
    assert "height:calc(100vh - 46px)" in html
    assert "object-position:center" in html
    assert 'class="stage"' in html
    assert "dx dy dz" in html
    assert "Direct PBVS" in html
    assert "Hybrid VS" in html
    assert "LLM Hybrid" in html
    assert "backend_arm_sdk" in html
    assert "left_index" in html
    assert "左手食指" in html
    assert "左手张开" in html
    assert "左手停止" in html
    assert "backend_service" in html
    assert "only after Release" in html
    assert 'data-cmd="hang"' in html
    assert "h2_arm_service_node" in html
    assert "左手侧视" in html
    assert "mode_direct_pbvs" in html
    assert "mode_hybrid" in html
    assert "mode_llm_hybrid" in html
    assert "sequence_llm_hybrid" in html
    assert 'id="recordToggle"' in html
    assert "录制关" in html
    assert "No Raise required" in html
    assert "Ctrl+C" in html
    assert "droll dpitch dyaw" in html
    assert "/manual/jog" in html
    assert "右肘" in html
    assert "/manual/elbow" in html
    assert "肩 pitch" in html
    assert "肩 yaw" in html
    assert "腕 roll" in html
    assert "腕 yaw" in html
    assert "/manual/joint" in html
    assert 'id="ee-xyz"' in html
    assert 'id="ee-rpy"' in html
    assert 'id="world-target-xyz"' in html
    assert 'id="world-ee-xyz"' in html
    assert 'id="world-error-xyz"' in html
    assert "right_ee_link" in html
    assert "user-select:text" in html
    assert "selectingPanelText" in html
    assert "sidecar-bgr" in html
    assert "/tip_seed" in html
    assert "/wrist_yellow_target" in html
    assert "pickWristYellow" in html
    assert 'id="chestView"' in html
    assert 'onclick="setTip(event)"' in html
    snap = client.get("/snapshot/chest.jpg")
    assert snap.status_code == 200
    assert snap.headers["Content-Type"].startswith("image/jpeg")
    seeded = client.post("/tip_seed", json={"u": 320, "v": 240}).get_json()
    assert seeded["ok"]
    cleared = client.post("/tip_seed", json={"clear": True}).get_json()
    assert cleared["ok"] and cleared["cleared"]
    manual = client.post("/manual/jog", json={"dxyz_m": [0.002, 0, 0], "drpy_deg": [0, 0, 0]})
    assert manual.status_code == 409
    assert manual.get_json()["ok"] is False
    trajectories = client.get("/trajectories").get_json()
    assert trajectories["ok"] is True
    bad_playback = client.post("/trajectories/play", json={"id": "missing", "speed": 2})
    assert bad_playback.status_code == 400
    bad_delete = client.post("/trajectories/delete", json={"id": "missing"})
    assert bad_delete.status_code == 400
    assert bad_delete.get_json()["ok"] is False
    idle = client.get("/state").get_json()
    assert idle["revo2_left"]["side"] == "left"
    assert idle["revo2_left"]["holding"] is False
    dash.revo2.point_index = lambda: "左手食指伸出，其余收紧"
    dash.revo2.open_hand = lambda: "左手张开"
    dash.revo2.stop = lambda: "已停止发送左手 Revo2"
    left = client.post("/command", json={"command": "left_index"}).get_json()
    assert left["ok"]
    assert "食指" in (left["state"]["message"] or "")
    opened = client.post("/command", json={"command": "left_open"}).get_json()
    assert "张开" in (opened["state"]["message"] or "")
    stopped = client.post("/command", json={"command": "left_stop"}).get_json()
    assert "停止" in (stopped["state"]["message"] or "")
    assert idle["record"]["enabled"] is False
    assert idle["record"]["recording"] is False
    armed = client.post("/record", json={"enabled": True}).get_json()
    assert armed["ok"]
    assert armed["record"]["enabled"] is True
    assert client.get("/state").get_json()["record"]["enabled"] is True
    client.post("/record", json={"enabled": False})
    health = client.get("/health").get_json()
    assert health["ok"]
    assert health["serial"] == "347622074029"
    posted = client.post("/command", json={"command": "button_3"}).get_json()
    assert posted["ok"]
    assert posted["state"]["target"] == "button_3"
    client.post("/command", json={"command": "cycle"})
    client.post("/command", json={"command": "start"})
    state = client.get("/state").get_json()
    assert state["cycle"] is True
    assert state["running"] is False
    assert state["target"] == "button_3"
    assert len(state["upper_body"]["joints"]) == len(UPPER_BODY_JOINTS)
    assert set(state["cameras"]) == {"chest", "d435i"}
    assert state["cameras"]["d435i"]["serial"] == "233722073797"
    assert "ibvs" in state
    assert {"dx", "dy", "dz", "droll", "dpitch", "dyaw"} <= set(state["ibvs"])
    reset = client.post("/command", json={"command": "reset"}).get_json()
    assert reset["state"]["target"] == "button_1"
    assert reset["state"]["running"] is False
    handed = client.post("/command", json={"command": "release"}).get_json()
    assert handed["ok"]
    assert handed["state"]["running"] is False
    assert "释放" in (handed["state"]["message"] or "")
    assert set(COMMANDS) >= {
        "button_1",
        "button_2",
        "button_3",
        "cycle",
        "scan",
        "start",
        "stop",
        "release",
        "reset",
        "mode_direct_pbvs",
        "mode_hybrid",
        "mode_llm_hybrid",
        "sequence_llm_hybrid",
    }
    selected = client.post("/command", json={"command": "mode_direct_pbvs"}).get_json()
    assert selected["ok"]
    assert "Direct PBVS" in (selected["state"]["message"] or "")
    hybrid = client.post("/command", json={"command": "mode_hybrid"}).get_json()
    assert hybrid["ok"]
    assert "Hybrid VS" in (hybrid["state"]["message"] or "")
    llm = client.post("/command", json={"command": "mode_llm_hybrid"}).get_json()
    assert llm["ok"]
    assert "LLM Hybrid" in (llm["state"]["message"] or "")
    seq_llm = client.post("/command", json={"command": "sequence_llm_hybrid"}).get_json()
    assert seq_llm["ok"]
    assert "LLM Hybrid" in (seq_llm["state"]["message"] or "")
    for worker in dash.cameras.values():
        worker.stop()


def test_wrist_yellow_click_is_only_accepted_during_the_approach():
    import cv2

    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    client = dash.app.test_client()
    refused = client.post("/wrist_yellow_target", json={"u": 50, "v": 48})
    assert refused.status_code == 409
    assert "进近" in refused.get_json()["error"]
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30, 40), (78, 58), (0, 220, 255), -1)
    dash.arm._control_stage = "WRIST_APPROACH"
    dash.ibvs._last_wrist_bgr = frame
    dash.ibvs.set_target("button_1")
    picked = client.post("/wrist_yellow_target", json={"u": 50, "v": 48}).get_json()
    assert picked["ok"] is True
    assert dash.ibvs.wrist_pbvs_view()["yellow_lock"] is True
    cleared = client.post("/wrist_yellow_target", json={"clear": True}).get_json()
    assert cleared["ok"] and cleared["cleared"]
    assert dash.ibvs.wrist_pbvs_view()["yellow_lock"] is False


def test_sequence_route_and_trajectory_switch():
    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    client = dash.app.test_client()
    state = client.get("/state").get_json()
    assert state["trajectory"]["enabled"] is False
    assert state["arm"]["motion_backend"] == "arm_sdk"
    assert state["arm"]["screen_progress"] is False
    assert state["screen"]["ready"] is False
    assert state["screen"]["progress"] is False
    blocked = client.post("/command", json={"command": "start"}).get_json()
    assert "confirm-arm-motion" in (blocked["state"]["message"] or "")
    enabled = client.post("/trajectories/enabled", json={"enabled": True}).get_json()
    assert enabled["trajectory"]["enabled"] is True
    posted = client.post(
        "/sequence",
        json={"steps": ["button_1", "button_2", "button_1", "button_3", "button_9"]},
    ).get_json()
    assert posted["steps"] == ["button_1", "button_2", "button_1", "button_3"]
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Next Step" in html
    assert "向下、向右" in html
    assert "sequence_direct_pbvs" in html
    assert "sequence_hybrid" in html
    assert 'id="screenProgressToggle"' in html
    assert "读取屏幕判断进度" in html
    opened = client.post("/screen_progress", json={"enabled": True}).get_json()
    assert opened["ok"] is True
    assert opened["arm"]["screen_progress"] is True
    assert "开" in (opened["message"] or "")
    closed = client.post("/screen_progress", json={"enabled": False}).get_json()
    assert closed["arm"]["screen_progress"] is False
    chosen = client.post("/command", json={"command": "sequence_direct_pbvs"}).get_json()
    assert chosen["ok"] is True
    assert "Direct PBVS" in (chosen["state"]["message"] or "")
    assert client.get("/state").get_json()["arm"]["sequence_mode"] == "direct_pbvs"
    back = client.post("/command", json={"command": "sequence_hybrid"}).get_json()
    assert "Hybrid" in (back["state"]["message"] or "")
    assert client.get("/state").get_json()["arm"]["sequence_mode"] == "hybrid"
    waiting = client.post("/command", json={"command": "sequence_next"}).get_json()
    assert waiting["ok"] is True
    assert "组合启动" in (waiting["state"]["message"] or "")
    assert client.get("/screen/latest.jpg").status_code == 404
    client.post("/sequence", json={"steps": []})
    empty = client.post("/command", json={"command": "start_sequence"}).get_json()
    assert empty["ok"] is True
    assert "为空" in (empty["state"]["message"] or "")
    assert empty["state"]["running"] is False
    for worker in dash.cameras.values():
        worker.stop()


def test_screen_mask_and_eye_in_hand_file(tmp_path):
    import cv2

    from ibvs_pad_touch.web.live_ibvs import load_eye_in_hand_transform, screen_mask_jpeg

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 180)
    image[6:16, 8:20] = (0, 220, 0)
    blob = screen_mask_jpeg(image, [(8, 6), (19, 6), (19, 15), (8, 15)])
    frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None and frame.shape[0] < 24
    assert int(frame[frame.shape[0] // 2, frame.shape[1] // 2, 1]) > 180
    path = tmp_path / "eye_in_hand_test.json"
    transform = [
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1],
    ]
    path.write_text(
        json.dumps({"calibration_method": "eye_in_hand", "cam2hand": {"transform": transform}}),
        encoding="utf-8",
    )
    matrix, info = load_eye_in_hand_transform(path)
    assert matrix is not None
    assert info["method"] == "eye_in_hand"
    assert abs(float(matrix[0, 3]) - 0.1) < 1e-9
    rejected, empty = load_eye_in_hand_transform(tmp_path / "missing.json")
    assert rejected is None and empty == {}


def test_eye_bias_form_defaults_to_startup_arguments_and_submit_replaces_them():
    dash = RealRobotDashboard(
        host="127.0.0.1",
        port=0,
        demo=True,
        autostart_camera=False,
        eye_to_hand_x_bias_m=0.03,
        eye_to_hand_y_bias_m=-0.01,
        eye_to_hand_z_bias_m=0.03,
    )
    client = dash.app.test_client()
    page = client.get("/").get_data(as_text=True)
    assert 'id="bias-x"' in page and 'id="bias-z"' in page
    state = client.get("/state").get_json()
    assert state["eye_to_hand_bias_m"] == {"x": 0.03, "y": -0.01, "z": 0.03}

    class _Bias:
        def __init__(self) -> None:
            self.eye_to_hand = {
                "torso_x_bias_m": 0.03,
                "torso_y_bias_m": -0.01,
                "torso_z_bias_m": 0.03,
            }

        def set_eye_to_hand_bias(self, x_m: float, y_m: float, z_m: float):
            self.eye_to_hand = {
                "torso_x_bias_m": float(x_m),
                "torso_y_bias_m": float(y_m),
                "torso_z_bias_m": float(z_m),
            }
            return True, ""

        def telemetry(self):
            return {}

    dash.ibvs = _Bias()
    posted = client.post(
        "/eye_to_hand_bias",
        json={"x_m": 0.01, "y_m": 0.0, "z_m": 0.04},
    )
    assert posted.status_code == 200
    assert posted.get_json()["eye_to_hand_bias_m"]["z"] == 0.04
    updated = client.get("/state").get_json()
    assert updated["eye_to_hand_bias_m"] == {"x": 0.01, "y": 0.0, "z": 0.04}
    rejected = client.post("/eye_to_hand_bias", json={"x_m": "nope", "y_m": 0, "z_m": 0})
    assert rejected.status_code == 400


def test_shadow_panel_compares_without_changing_the_command():
    class _Shift:
        name = "fake"

        def describe(self, bgr):
            base = np.zeros((4, 5, 20), dtype=np.float64)
            for index in range(20):
                row, col = divmod(index, 5)
                base[row, col, index] = 1.0
            if float(bgr.mean()) < 1.0:
                return base
            moved = np.zeros_like(base)
            moved[:, 1:] = base[:, :-1]
            moved[:, 0, -1] = 1.0
            return moved

    dash = RealRobotDashboard(host="127.0.0.1", port=0, demo=True, autostart_camera=False)
    dash.shadow._extractor = _Shift()
    client = dash.app.test_client()
    idle = client.get("/state").get_json()
    assert idle["shadow"]["applied"] is False
    assert idle["arm"]["wrist_command_step_m"] is None
    worker = dash.cameras["d435i"]
    worker._publish(np.zeros((48, 64, 3), dtype=np.uint8), "running")
    saved = client.post("/shadow/reference").get_json()
    assert saved["ok"] is True
    assert saved["shadow"]["reference"] is True
    assert saved["shadow"]["applied"] is False
    worker._publish(np.full((48, 64, 3), 255, dtype=np.uint8), "running")
    dash.arm._wrist_ibvs_step = (0.001, 0.0, 0.0)
    compared = dash.shadow.refresh()
    assert compared["applied"] is False
    state = client.get("/state").get_json()
    assert state["shadow"]["actual_step_mm"][0] == 1.0
    assert state["shadow"]["shadow_step_mm"][0] > 0.0
    assert state["arm"]["wrist_command_step_m"] == [0.001, 0.0, 0.0]
    dash.shadow.close()
