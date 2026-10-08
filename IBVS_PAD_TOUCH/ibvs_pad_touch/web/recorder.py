# -*- coding: utf-8 -*-
"""Optional local recording of the forwarded camera streams and the arm trajectory.

Off by default. The web switch only arms it. A take starts on Start and ends
on Stop, release, reset, or when the switch is turned off.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


VIDEO_HZ = 10.0
KEYFRAME_S = 1.0


def slim_trajectory(state: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    ibvs = state.get("ibvs") or {}
    arm = state.get("arm") or {}
    joints = []
    for joint in (state.get("upper_body") or {}).get("joints") or []:
        group = str(joint.get("group") or "")
        if group not in {"left_arm", "right_arm", "waist"}:
            continue
        joints.append(
            {
                "name": joint.get("name"),
                "group": group,
                "q": joint.get("q"),
                "q_deg": joint.get("q_deg"),
            }
        )
    return {
        "t": float(time.time() if now is None else now),
        "target": state.get("target"),
        "running": bool(state.get("running")),
        "message": state.get("message"),
        "arm": {
            "phase": arm.get("phase"),
            "message": arm.get("message"),
            "control_stage": arm.get("control_stage"),
            "ibvs_enabled": arm.get("ibvs_enabled"),
        },
        "ibvs": {
            "phase": ibvs.get("phase"),
            "message": ibvs.get("message"),
            "dx": ibvs.get("dx"),
            "dy": ibvs.get("dy"),
            "dz": ibvs.get("dz"),
            "target_uv": ibvs.get("target_uv"),
            "tip_uv": ibvs.get("tip_uv"),
            "tip_source": ibvs.get("tip_source"),
            "side_measured": ibvs.get("side_measured"),
            "side_gap_px": ibvs.get("side_gap_px"),
            "left_contact": ibvs.get("left_contact"),
        },
        "joints": joints,
    }


class SessionRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        video_hz: float = VIDEO_HZ,
        keyframe_s: float = KEYFRAME_S,
    ) -> None:
        self.root = Path(root)
        self.video_hz = float(video_hz)
        self.keyframe_s = float(keyframe_s)
        self._lock = threading.RLock()
        self._enabled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: Path | None = None
        self._frames = 0
        self._keyframes = 0

    def status(self) -> dict[str, Any]:
        with self._lock:
            recording = self._thread is not None and self._thread.is_alive()
            return {
                "enabled": self._enabled,
                "recording": recording,
                "dir": None if self._session is None else str(self._session),
                "frames": self._frames,
                "keyframes": self._keyframes,
            }

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._enabled = bool(enabled)
        if not enabled:
            self.end()
        return self.status()

    def begin(
        self,
        cameras: dict[str, Any],
        state_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            if not self._enabled:
                return self.status()
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            session = self.root / stamp
            session.mkdir(parents=True, exist_ok=True)
            (session / "keyframes").mkdir(exist_ok=True)
            self._session = session
            self._frames = 0
            self._keyframes = 0
            self._stop.clear()
            meta = {
                "started_at": time.time(),
                "cameras": list(cameras),
                "video_hz": self.video_hz,
                "keyframe_s": self.keyframe_s,
            }
            (session / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._thread = threading.Thread(
                target=self._loop,
                args=(dict(cameras), state_fn, session),
                name="ibvs-record",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def end(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            session = self._session
            self._thread = None
        if session is not None:
            meta_path = session / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            meta["ended_at"] = time.time()
            meta["frames"] = self._frames
            meta["keyframes"] = self._keyframes
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.status()

    def _loop(
        self,
        cameras: dict[str, Any],
        state_fn: Callable[[], dict[str, Any]],
        session: Path,
    ) -> None:
        writers: dict[str, cv2.VideoWriter] = {}
        last_key = 0.0
        key_index: dict[str, int] = {}
        period = 1.0 / max(self.video_hz, 1.0)
        trajectory = session / "trajectory.jsonl"
        try:
            with trajectory.open("a", encoding="utf-8") as handle:
                while not self._stop.is_set():
                    started = time.time()
                    try:
                        sample = slim_trajectory(state_fn() or {}, started)
                    except Exception as exc:
                        sample = {"t": started, "error": str(exc)}
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    handle.flush()
                    save_key = started - last_key >= self.keyframe_s
                    if save_key:
                        last_key = started
                    for name, worker in cameras.items():
                        frame = _decode_jpeg(worker.jpeg())
                        if frame is None:
                            continue
                        writer = _writer(writers, session, name, frame, self.video_hz)
                        if writer is not None:
                            writer.write(frame)
                            with self._lock:
                                self._frames += 1
                        if save_key:
                            folder = session / "keyframes" / name
                            folder.mkdir(parents=True, exist_ok=True)
                            key_index[name] = key_index.get(name, 0) + 1
                            cv2.imwrite(str(folder / f"{key_index[name]:04d}.jpg"), frame)
                            with self._lock:
                                self._keyframes += 1
                    remaining = period - (time.time() - started)
                    if remaining > 0:
                        self._stop.wait(remaining)
        finally:
            for writer in writers.values():
                writer.release()


def _decode_jpeg(blob: bytes) -> np.ndarray | None:
    if not blob:
        return None
    frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def _writer(
    writers: dict[str, cv2.VideoWriter],
    session: Path,
    name: str,
    frame: np.ndarray,
    fps: float,
) -> cv2.VideoWriter | None:
    current = writers.get(name)
    height, width = frame.shape[:2]
    if current is not None and current.isOpened():
        return current
    path = session / f"{name}.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        return None
    writers[name] = writer
    return writer


class MotionTrajectoryRecorder:
    """Motion-only take from Start. Recording stays off until the page enables it."""

    def __init__(self, root: str | Path, hz: float = 25.0) -> None:
        self.root = Path(root)
        self.hz = max(5.0, float(hz))
        self._lock = threading.RLock()
        self._enabled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: Path | None = None
        self._started_at = 0.0
        self._samples = 0

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._enabled = bool(enabled)
        if not enabled:
            self.end("disabled")
        return self.status()

    def begin(
        self,
        state_fn: Callable[[], dict[str, Any]],
        *,
        target: str,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._enabled:
                return self.status()
        self.end("new_start")
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        session = self.root / stamp
        session.mkdir(parents=True, exist_ok=True)
        started = time.time()
        meta = {
            "id": stamp,
            "target": str(target),
            "started_at": started,
            "hz": self.hz,
            "complete": False,
        }
        (session / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._lock:
            self._session = session
            self._started_at = started
            self._samples = 0
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                args=(state_fn, session),
                name="motion-trajectory-record",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def end(self, reason: str = "ended") -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            session = self._session
            started = self._started_at
            samples = self._samples
            self._thread = None
        if session is not None:
            meta_path = session / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {"id": session.name}
            ended = time.time()
            meta.update(
                {
                    "ended_at": ended,
                    "duration_s": max(0.0, ended - started),
                    "samples": samples,
                    "reason": str(reason),
                    "complete": samples >= 2,
                }
            )
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            recording = self._thread is not None and self._thread.is_alive()
            return {
                "enabled": self._enabled,
                "recording": recording,
                "id": None if self._session is None else self._session.name,
                "dir": None if self._session is None else str(self._session),
                "samples": self._samples,
                "hz": self.hz,
            }

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for meta_path in self.root.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not (meta_path.parent / "trajectory.jsonl").is_file():
                continue
            rows.append(meta)
        return sorted(rows, key=lambda row: float(row.get("started_at") or 0.0), reverse=True)

    def delete(self, trajectory_id: str) -> dict[str, Any]:
        folder = self._trajectory_folder(trajectory_id)
        with self._lock:
            active_name = None if self._session is None else self._session.name
            if self._thread is not None and active_name == folder.name:
                raise ValueError("trajectory is still recording")
            if not folder.is_dir() or not (folder / "meta.json").is_file():
                raise FileNotFoundError(folder.name)
            shutil.rmtree(folder)
            if active_name == folder.name:
                self._session = None
                self._samples = 0
                self._started_at = 0.0
        return {"ok": True, "id": folder.name}

    def delete_all(self) -> dict[str, Any]:
        deleted: list[str] = []
        kept = None
        for row in self.list():
            trajectory_id = str(row.get("id") or "")
            if not trajectory_id:
                continue
            try:
                self.delete(trajectory_id)
            except ValueError as exc:
                if "recording" not in str(exc):
                    raise
                kept = trajectory_id
                continue
            except FileNotFoundError:
                continue
            deleted.append(trajectory_id)
        return {"ok": True, "deleted": len(deleted), "ids": deleted, "kept": kept}

    def load(self, trajectory_id: str) -> list[dict[str, Any]]:
        folder = self._trajectory_folder(trajectory_id)
        path = folder / "trajectory.jsonl"
        if not path.is_file():
            raise FileNotFoundError(folder.name)
        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("joints"):
                samples.append(sample)
        if len(samples) < 2:
            raise ValueError("trajectory has fewer than two samples")
        return samples

    def _loop(
        self,
        state_fn: Callable[[], dict[str, Any]],
        session: Path,
    ) -> None:
        period = 1.0 / self.hz
        path = session / "trajectory.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                started = time.time()
                try:
                    sample = slim_trajectory(state_fn() or {}, started)
                except Exception as exc:
                    sample = {"t": started, "error": str(exc)}
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.flush()
                with self._lock:
                    self._samples += 1
                remaining = period - (time.time() - started)
                if remaining > 0:
                    self._stop.wait(remaining)

    def _trajectory_folder(self, trajectory_id: str) -> Path:
        text = str(trajectory_id).strip()
        safe = Path(text).name
        if not text or safe != text or safe in {".", ".."}:
            raise ValueError("invalid trajectory id")
        folder = (self.root / safe).resolve()
        if folder.parent != self.root.resolve():
            raise ValueError("invalid trajectory id")
        return folder
