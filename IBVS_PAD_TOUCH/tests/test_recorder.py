import json
import time

import cv2
import numpy as np

from ibvs_pad_touch.web.recorder import MotionTrajectoryRecorder, SessionRecorder, slim_trajectory


class _Cam:
    def __init__(self) -> None:
        frame = np.full((48, 64, 3), 40, dtype=np.uint8)
        cv2.putText(frame, "rec", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ok, blob = cv2.imencode(".jpg", frame)
        assert ok
        self._jpeg = blob.tobytes()

    def jpeg(self) -> bytes:
        return self._jpeg


def test_recording_stays_off_until_enabled():
    recorder = SessionRecorder("outputs/recordings-test-off", video_hz=20, keyframe_s=0.05)
    status = recorder.begin({"chest": _Cam()}, lambda: {"target": "button_1"})
    assert status["enabled"] is False
    assert status["recording"] is False
    assert status["dir"] is None


def test_start_records_video_keyframes_and_trajectory(tmp_path):
    recorder = SessionRecorder(tmp_path, video_hz=20, keyframe_s=0.05)
    recorder.set_enabled(True)
    state = {
        "target": "button_1",
        "running": True,
        "message": "servo",
        "arm": {"phase": "ALIGN_XY", "message": "move", "control_stage": "DUAL_IBVS", "ibvs_enabled": True},
        "ibvs": {"phase": "ALIGN_XY", "dx": 0.001, "dy": 0.0, "dz": 0.0, "target_uv": [1, 2], "tip_source": "chest_visual"},
        "upper_body": {"joints": [{"name": "right_elbow_joint", "group": "right_arm", "q_deg": 12.0}]},
    }
    status = recorder.begin({"chest": _Cam(), "d435i": _Cam()}, lambda: state)
    assert status["enabled"] is True
    deadline = time.time() + 2.0
    while time.time() < deadline and recorder.status()["keyframes"] < 2:
        time.sleep(0.05)
    ended = recorder.end()
    assert ended["recording"] is False
    session = tmp_path / next(path.name for path in tmp_path.iterdir())
    lines = (session / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    sample = json.loads(lines[0])
    assert sample["arm"]["control_stage"] == "DUAL_IBVS"
    assert sample["ibvs"]["tip_source"] == "chest_visual"
    assert sample["joints"][0]["q_deg"] == 12.0
    assert (session / "keyframes" / "chest" / "0001.jpg").is_file()
    assert (session / "keyframes" / "d435i" / "0001.jpg").is_file()
    assert (session / "chest.avi").is_file()
    assert (session / "d435i.avi").is_file()
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    assert "ended_at" in meta


def test_slim_trajectory_keeps_servo_fields():
    slim = slim_trajectory(
        {
            "target": "button_3",
            "ibvs": {"dz": 0.004, "left_contact": False},
            "upper_body": {"joints": [{"name": "head_yaw_joint", "group": "head", "q_deg": 1.0}]},
        },
        now=10.0,
    )
    assert slim["t"] == 10.0
    assert slim["ibvs"]["dz"] == 0.004
    assert slim["joints"] == []


def test_motion_trajectory_records_pause_samples_and_can_reload(tmp_path):
    recorder = MotionTrajectoryRecorder(tmp_path, hz=50)
    state = {
        "target": "button_2",
        "running": False,
        "arm": {"phase": "manual", "control_stage": "MANUAL"},
        "upper_body": {
            "joints": [
                {
                    "name": "right_elbow_joint",
                    "group": "right_arm",
                    "q": 0.25,
                    "q_deg": 14.32,
                }
            ]
        },
    }
    recorder.begin(lambda: state, target="button_2")
    deadline = time.time() + 1.0
    while time.time() < deadline and recorder.status()["samples"] < 3:
        time.sleep(0.02)
    ended = recorder.end("button_switch")
    assert ended["recording"] is False
    rows = recorder.list()
    assert len(rows) == 1
    assert rows[0]["target"] == "button_2"
    assert rows[0]["reason"] == "button_switch"
    samples = recorder.load(rows[0]["id"])
    assert samples[0]["arm"]["control_stage"] == "MANUAL"
    assert samples[0]["joints"][0]["q"] == 0.25
    deleted = recorder.delete(rows[0]["id"])
    assert deleted == {"ok": True, "id": rows[0]["id"]}
    assert recorder.list() == []
    assert recorder.status()["id"] is None
    assert not (tmp_path / rows[0]["id"]).exists()


def test_motion_trajectory_delete_rejects_bad_ids_and_active_takes(tmp_path):
    import pytest

    recorder = MotionTrajectoryRecorder(tmp_path, hz=50)
    for bad in ("", "..", "../secret", "nested/id"):
        with pytest.raises(ValueError):
            recorder.delete(bad)
    assert list(tmp_path.iterdir()) == []
    state = {
        "target": "button_1",
        "upper_body": {"joints": [{"name": "right_elbow_joint", "group": "right_arm", "q": 0.1}]},
    }
    recorder.begin(lambda: state, target="button_1")
    active = recorder.status()["id"]
    try:
        with pytest.raises(ValueError, match="recording"):
            recorder.delete(active)
        assert (tmp_path / active).is_dir()
    finally:
        recorder.end("release")


def _plant_trajectory(root, name: str) -> None:
    folder = root / name
    folder.mkdir()
    (folder / "meta.json").write_text(
        json.dumps({"id": name, "started_at": 1.0, "target": "button_1"}),
        encoding="utf-8",
    )
    (folder / "trajectory.jsonl").write_text(
        '{"joints":[{"q":0.1}]}\n{"joints":[{"q":0.2}]}\n',
        encoding="utf-8",
    )


def test_motion_trajectory_delete_all_keeps_the_active_take(tmp_path):
    recorder = MotionTrajectoryRecorder(tmp_path, hz=50)
    _plant_trajectory(tmp_path, "20260928_100000_001")
    _plant_trajectory(tmp_path, "20260928_100100_002")
    cleared = recorder.delete_all()
    assert cleared["deleted"] == 2
    assert cleared["kept"] is None
    assert recorder.list() == []

    _plant_trajectory(tmp_path, "20260928_100200_003")
    state = {
        "target": "button_1",
        "upper_body": {"joints": [{"name": "right_elbow_joint", "group": "right_arm", "q": 0.1}]},
    }
    recorder.begin(lambda: state, target="button_1")
    active = recorder.status()["id"]
    deadline = time.time() + 1.0
    while time.time() < deadline and not (tmp_path / active / "trajectory.jsonl").is_file():
        time.sleep(0.02)
    try:
        result = recorder.delete_all()
        assert result["kept"] == active
        assert "20260928_100200_003" in result["ids"]
        assert (tmp_path / active).is_dir()
        assert not (tmp_path / "20260928_100200_003").exists()
    finally:
        recorder.end("release")
