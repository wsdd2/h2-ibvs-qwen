"""Flask MJPEG + upper-body joints. Same pattern as Hand_Eye_Calib debug_stream."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Optional
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import cv2
import numpy as np
from flask import Flask, Response, jsonify, make_response, request

from ibvs_pad_touch.camera.devices import read_seen
from ibvs_pad_touch.camera.image_io import pick_sidecar, read_depth, read_image
from ibvs_pad_touch.camera.realsense_bind import usable_realsense as _usable_realsense
from ibvs_pad_touch.config import camera_views, load_camera_config
from ibvs_pad_touch.web.joints import UpperBodyJointReader
from ibvs_pad_touch.web.recorder import MotionTrajectoryRecorder, SessionRecorder


COMMANDS = {
    "button_1",
    "button_2",
    "button_3",
    "cycle",
    "scan",
    "start",
    "start_sequence",
    "sequence_next",
    "sequence_direct_pbvs",
    "sequence_hybrid",
    "pause",
    "stop",
    "release",
    "reset",
    "mode_direct_pbvs",
    "mode_hybrid",
    "backend_arm_sdk",
    "backend_service",
    "hang",
    "level_right",
    "estop",
}
COMMAND_ALIASES = {
    "button1": "button_1",
    "b1": "button_1",
    "button2": "button_2",
    "b2": "button_2",
    "button3": "button_3",
    "b3": "button_3",
    "ok": "button_3",
    "释放": "release",
    "handoff": "release",
}
TARGET_LABELS = {
    "button_1": "Button 1  <",
    "button_2": "Button 2  >",
    "button_3": "Button 3  OK",
}
DEPTH_PATHS = {
    "chest": Path("/tmp/ibvs_depth_chest.bin"),
    "d435i": Path("/tmp/ibvs_depth_d435i.bin"),
}


def _placeholder(width: int, height: int, text: str) -> np.ndarray:
    frame = np.full((max(120, height), max(240, width), 3), 28, dtype=np.uint8)
    y = max(40, height // 2 - 20)
    for i, start in enumerate(range(0, len(text), 42)):
        cv2.putText(
            frame,
            text[start : start + 42],
            (16, y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return frame


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def serve_wsgi(app, host: str, port: int) -> None:
    """Avoid Flask/Werkzeug's built-in server. H2 --user Flask often has broken metadata."""
    print(f"[ibvs-web] wsgiref http://{host}:{port}/", flush=True)
    httpd = make_server(host, int(port), app, server_class=_ThreadingWSGIServer, handler_class=_QuietHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[ibvs-web] interrupt, leaving serve loop", flush=True)
    finally:
        httpd.server_close()


def _encode_jpeg(frame: np.ndarray, quality: int) -> Optional[bytes]:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return encoded.tobytes() if ok else None


def _depth_preview_png(depth_m: np.ndarray) -> bytes | None:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05)
    clipped = np.clip(np.where(valid, depth, 0.20), 0.20, 1.50)
    gray = np.round(255.0 * (clipped - 0.20) / 1.30).astype(np.uint8)
    color = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.putText(
        color,
        "0.20m",
        (8, color.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        color,
        "1.50m",
        (color.shape[1] - 58, color.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".png", color)
    return encoded.tobytes() if ok else None


def _start_rs_pipeline(rs, serial: str, width: int, height: int, fps: int, timeout_s: float = 8.0):
    box: dict[str, Any] = {"pipe": None, "err": None}

    def work() -> None:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            if serial:
                cfg.enable_device(str(serial))
            cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
            pipe.start(cfg)
            box["pipe"] = pipe
        except Exception as exc:
            box["err"] = exc

    thread = threading.Thread(target=work, name="rs-start", daemon=True)
    thread.start()
    thread.join(float(timeout_s))
    if thread.is_alive():
        return None, TimeoutError(f"pipeline.start {width}x{height}@{fps} hung {timeout_s:.0f}s")
    if box["pipe"] is None:
        return None, box["err"] or RuntimeError("pipeline.start returned nothing")
    return box["pipe"], None


class ChestCameraWorker:
    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        demo: bool,
        name: str = "chest",
        label: str = "",
        model: str = "Intel RealSense",
        reset_hardware: bool = False,
        sidecar_path: str = "",
    ) -> None:
        self.name = str(name)
        self.label = str(label or name)
        self.model = str(model)
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.demo = bool(demo)
        self.annotate = None
        self.reset_hardware = bool(reset_hardware)
        self.sidecar_path = Path(sidecar_path or f"/tmp/ibvs_cam_{self.name}.jpg")
        self._sidecar_paths = [
            self.sidecar_path.with_suffix(".bgr"),
            self.sidecar_path.with_suffix(".ppm"),
            self.sidecar_path,
            self.sidecar_path.with_suffix(".jpg"),
        ]
        self._claim_path = Path(f"/tmp/ibvs_cam_{self.name}.serial")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jpeg = _encode_jpeg(_placeholder(self.width, self.height, "camera idle"), self.jpeg_quality)
        self._raw_jpeg = self._jpeg
        self._state = "idle"
        self._error = ""
        self._frames = 0
        self._stamp = 0.0

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"starting", "running"}:
                return self.status()
            self._state = "starting"
            self._error = ""
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._publish(_placeholder(self.width, self.height, f"{self.label} starting {self.serial}"), "starting")
        self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self._state = "stopped"
            self._thread = None
            self._jpeg = _encode_jpeg(_placeholder(self.width, self.height, "camera stopped"), self.jpeg_quality)
            self._raw_jpeg = self._jpeg
        return self.status()

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg or b""

    def raw_jpeg(self) -> bytes:
        """Camera frame before the IBVS overlay, for dataset screenshots."""
        with self._lock:
            return self._raw_jpeg or self._jpeg or b""

    def _live_serial(self) -> str:
        try:
            text = self._claim_path.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        return text or self.serial

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = None if self._stamp <= 0 else time.time() - self._stamp
            return {
                "name": self.name,
                "label": self.label,
                "serial": self._live_serial(),
                "model": self.model,
                "state": self._state,
                "frames": self._frames,
                "age_s": age,
                "error": self._error,
                "size": [self.width, self.height],
                "fps": self.fps,
            }

    def _publish(self, frame: np.ndarray, state: str) -> None:
        raw_blob = _encode_jpeg(frame, self.jpeg_quality)
        shown = frame
        if self.annotate is not None and state == "running":
            try:
                shown = self.annotate(self.name, frame)
            except Exception as exc:
                cv2.putText(shown, f"ibvs overlay: {exc}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        blob = _encode_jpeg(shown, self.jpeg_quality)
        if blob is None:
            return
        with self._lock:
            self._raw_jpeg = raw_blob or blob
            self._jpeg = blob
            self._state = state
            self._frames += 1
            self._stamp = time.time()

    def _run(self) -> None:
        if self.demo:
            self._run_demo()
            return
        try:
            self._publish(_placeholder(self.width, self.height, f"{self.label} wait sidecar"), "starting")
            if self._run_sidecar_file():
                return
            raise RuntimeError(f"{self.label}: sidecar stopped")
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
            self._publish(_placeholder(self.width, self.height, f"camera error: {exc}"), "error")
            print(f"[ibvs-cam] {exc}", flush=True)

    def _run_realsense(self) -> bool:
        rs = _usable_realsense()
        if rs is None:
            print("[ibvs-cam] no usable pyrealsense2.pipeline", flush=True)
            return False
        print(f"[ibvs-cam] {self.name} bind={getattr(rs, '__file__', rs)}", flush=True)
        if self.reset_hardware:
            self._hardware_reset(rs)
        profiles = (
            (self.width, self.height, 15),
            (self.width, self.height, int(self.fps)),
            (640, 480, 15),
            (640, 480, 6),
            (848, 480, 15),
        )
        seen: set[tuple[int, int, int]] = set()
        for width, height, fps in profiles:
            key = (int(width), int(height), int(fps))
            if key in seen:
                continue
            seen.add(key)
            if self._stop.is_set():
                return False
            self._publish(
                _placeholder(self.width, self.height, f"pipeline {key[0]}x{key[1]}@{key[2]}"),
                "starting",
            )
            pipeline, err = _start_rs_pipeline(rs, self.serial, key[0], key[1], key[2], timeout_s=8.0)
            if pipeline is None:
                print(f"[ibvs-cam] start fail {key}: {err}", flush=True)
                continue
            try:
                if self._pump_rs(pipeline, first_wait_s=8.0):
                    return True
            finally:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        return False

    def _run_sidecar_file(self) -> bool:
        path = None
        last_hint = 0.0
        while not self._stop.is_set():
            path = pick_sidecar(self._sidecar_paths, max_age_s=4.0)
            if path is not None:
                break
            now = time.time()
            if now - last_hint > 2.0:
                seen = read_seen() or "no RealSense yet"
                self._publish(
                    _placeholder(self.width, self.height, f"{self.label} wait live sidecar; usb {seen}"),
                    "starting",
                )
                last_hint = now
            time.sleep(0.2)
        if path is None:
            print(f"[ibvs-cam] {self.name} no sidecar file {self.sidecar_path}", flush=True)
            return False
        print(f"[ibvs-cam] {self.name} reading {path}", flush=True)
        last = 0.0
        while not self._stop.is_set():
            try:
                live = pick_sidecar(self._sidecar_paths, max_age_s=6.0) or path
                stamp = live.stat().st_mtime
                if stamp == last:
                    time.sleep(0.03)
                    continue
                frame = read_image(live)
                if frame is None:
                    time.sleep(0.03)
                    continue
                path = live
                last = stamp
                self._publish(frame, "running")
            except OSError:
                time.sleep(0.05)
        return True

    def _hardware_reset(self, rs) -> None:
        if not self.serial or not hasattr(rs, "context"):
            return
        try:
            for dev in rs.context().query_devices():
                serial = dev.get_info(rs.camera_info.serial_number)
                if serial != self.serial:
                    continue
                print(f"[ibvs-cam] hardware_reset {serial}", flush=True)
                dev.hardware_reset()
                time.sleep(2.0)
                return
        except Exception as exc:
            print(f"[ibvs-cam] hardware_reset skipped: {exc}", flush=True)

    def _pump_rs(self, pipeline, first_wait_s: float) -> bool:
        got = False
        deadline = time.time() + float(first_wait_s)
        while not self._stop.is_set() and time.time() < deadline:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1500)
            except Exception:
                continue
            color = frames.get_color_frame()
            if color is None:
                continue
            got = True
            self._publish(np.asanyarray(color.get_data()), "running")
            break
        if not got:
            return False
        while not self._stop.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                continue
            color = frames.get_color_frame()
            if color is None:
                continue
            self._publish(np.asanyarray(color.get_data()), "running")
        return True

    def _run_demo(self) -> None:
        while not self._stop.is_set():
            frame = np.full((self.height, self.width, 3), 36, dtype=np.uint8)
            t = time.time()
            cv2.putText(frame, f"DEMO  {self.label}", (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2)
            cv2.putText(frame, self.serial, (24, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            cv2.putText(frame, time.strftime("%H:%M:%S"), (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.circle(frame, (int(320 + 80 * np.sin(t)), 240), 18, (0, 200, 255), -1)
            self._publish(frame, "running")
            time.sleep(1.0 / max(1, self.fps))


class RealRobotDashboard:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        serial: str = "",
        jpeg_quality: int = 80,
        iface: str = "eth0",
        demo: bool = False,
        autostart_camera: bool = True,
        confirm_arm_motion: bool = False,
        direct_pbvs_poke: bool = False,
        direct_pbvs_max_m: float = 1.20,
        direct_pbvs_max_joint_deg: float = 150.0,
        eye_to_hand_json: str = "",
        eye_to_hand_x_bias_m: float = 0.03,
        eye_to_hand_y_bias_m: float = 0.0,
        eye_to_hand_z_bias_m: float = 0.025,
        pbvs_first: bool = True,
        direct_ee_poke: bool = False,
        direct_ee_poke_depth_m: float = 0.005,
        direct_ee_max_m: float = 0.0,
        direct_ee_max_joint_deg: float = 130.0,
        pbvs_step_m: float = 0.020,
        pbvs_step_seconds: float = 0.45,
        pbvs_max_joint_deg: float = 20.0,
    ) -> None:
        cam = load_camera_config()["camera"]
        self.host = host
        self.port = int(port)
        views = camera_views({"camera": cam})
        if serial:
            views = [dict(views[0], serial_number=str(serial))] if views else [{"name": "chest", "serial_number": str(serial)}]
        size = cam.get("color_size") or [640, 480]
        self.cameras = {
            str(view["name"]): ChestCameraWorker(
                str(view.get("serial_number") or ""),
                int(size[0]),
                int(size[1]),
                min(15, int(cam.get("fps", 15))),
                jpeg_quality,
                demo,
                name=str(view["name"]),
                label=str(view.get("label") or view["name"]),
                model=str(view.get("model") or cam.get("model") or "RealSense"),
            )
            for view in views
        }
        self.camera = self.cameras.get("chest") or next(iter(self.cameras.values()))
        self.serial = self.camera.serial
        self._startup_eye_bias_m = (
            float(eye_to_hand_x_bias_m),
            float(eye_to_hand_y_bias_m),
            float(eye_to_hand_z_bias_m),
        )
        self.joints = UpperBodyJointReader(iface=iface, demo=demo)
        try:
            from ibvs_pad_touch.web.live_ibvs import LiveIbvsAnnotator

            self.ibvs = LiveIbvsAnnotator(
                use_yolo=not demo,
                eye_to_hand_json=(eye_to_hand_json or None),
                eye_to_hand_x_bias_m=eye_to_hand_x_bias_m,
                eye_to_hand_y_bias_m=eye_to_hand_y_bias_m,
                eye_to_hand_z_bias_m=eye_to_hand_z_bias_m,
            )
        except Exception as exc:
            self.ibvs = None
            print(f"[ibvs-web] live overlay off: {exc}", flush=True)
        if self.ibvs is not None:
            for worker in self.cameras.values():
                worker.annotate = self.ibvs.annotate
        from ibvs_pad_touch.web.arm_motion import ArmMover

        self.recorder = SessionRecorder(Path(__file__).resolve().parents[2] / "outputs" / "recordings")
        self.motion_trajectories = MotionTrajectoryRecorder(
            Path(__file__).resolve().parents[2] / "outputs" / "trajectories"
        )
        self.dataset_root = Path(__file__).resolve().parents[2] / "outputs" / "dataset"
        self.arm = ArmMover(
            iface=iface,
            telemetry=(lambda: {} if self.ibvs is None else self.ibvs.telemetry()),
            confirm=bool(confirm_arm_motion) and not demo,
            on_ee_pose=(None if self.ibvs is None else self.ibvs.set_ee_pose),
            direct_pbvs_poke=direct_pbvs_poke,
            direct_pbvs_max_m=direct_pbvs_max_m,
            direct_pbvs_max_joint_deg=direct_pbvs_max_joint_deg,
            pbvs_first=bool(pbvs_first and self.ibvs is not None and self.ibvs.eye_to_hand_valid),
            camera_R_torso=(None if self.ibvs is None else self.ibvs.camera_R_torso),
            direct_ee_poke=direct_ee_poke,
            direct_ee_poke_depth_m=direct_ee_poke_depth_m,
            direct_ee_max_m=direct_ee_max_m,
            direct_ee_max_joint_deg=direct_ee_max_joint_deg,
            pbvs_step_m=pbvs_step_m,
            pbvs_step_seconds=pbvs_step_seconds,
            pbvs_max_joint_deg=pbvs_max_joint_deg,
        )
        if self.ibvs is not None:
            self.ibvs.set_side_ready(lambda: bool(self.arm.status().get("left_side_ready")))
            self.ibvs.set_control_stage(lambda: str(self.arm.status().get("control_stage") or ""))
            self.arm.wrist_watch = self.ibvs.set_wrist_pbvs_watch
            self.arm.wrist_hold = self.ibvs.hold_wrist_pbvs
            self.arm.wrist_view = self.ibvs.wrist_pbvs_view
            self.arm.cam2hand_matrix = self.ibvs.cam2hand_matrix
            self.arm.on_screen_capture = self._capture_screen
            self.arm.on_select_target = self.ibvs.set_target
            self.arm.on_reidentify = self.ibvs.forget_button_lock
        self._screen_jpeg: bytes | None = None
        self._screen_at = 0.0
        self._app = Flask(__name__)
        self._lock = threading.Lock()
        self._started = False
        self._target = "button_1"
        self._cycle = False
        self._running = False
        self._commands: list[dict[str, Any]] = []
        self._last_command: dict[str, Any] | None = None
        self._message = "按钮已预留，尚未接入实机 IBVS 伺服"
        self._configure_routes()
        if autostart_camera:
            for worker in self.cameras.values():
                worker.start()

    @property
    def app(self) -> Flask:
        return self._app

    def start(self) -> None:
        if self._started:
            return
        threading.Thread(
            target=lambda: serve_wsgi(self._app, self.host, self.port),
            daemon=True,
        ).start()
        self._started = True
        print(f"[ibvs-web] http://{self.host}:{self.port}/")
        print(f"[ibvs-web] 本机浏览器: http://127.0.0.1:{self.port}/   局域网: http://<h2-ip>:{self.port}/")

    def run_forever(self) -> None:
        print(
            f"[ibvs-web] cams="
            + ",".join(f"{name}:{worker.serial}" for name, worker in self.cameras.items())
            + f" demo={self.camera.demo}"
        )
        print("[ibvs-web] camera frames from sidecars; Start uses cyclonedds._clayer in this process")
        print("[ibvs-web] Ctrl+C / SIGTERM 会先放下右臂再退出；网页 Stop 仍保持姿态")
        try:
            serve_wsgi(self._app, self.host, self.port)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        print("[ibvs-web] shutting down cameras + arm", flush=True)
        for worker in self.cameras.values():
            try:
                worker.stop()
            except Exception:
                pass
        self.recorder.end()
        self.motion_trajectories.end("shutdown")
        self.arm.shutdown()

    def pop_command(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self._commands:
                return None
            return self._commands.pop(0)

    def _session(self) -> dict[str, Any]:
        with self._lock:
            return {
                "target": self._target,
                "target_label": TARGET_LABELS[self._target],
                "cycle": self._cycle,
                "running": self._running,
                "last_command": self._last_command,
                "message": self._message,
                "queued": len(self._commands),
            }

    def _apply_command(self, command: str) -> dict[str, Any]:
        begin_record = False
        end_record = False
        begin_trajectory = False
        end_trajectory_reason = ""
        with self._lock:
            if command in TARGET_LABELS:
                if command != self._target:
                    end_trajectory_reason = "button_switch"
                self._target = command
                if self.ibvs is not None:
                    self.ibvs.set_target(command)
                self._message = f"目标 {TARGET_LABELS[command]}（画面伺服，不发 arm_sdk）"
            elif command == "cycle":
                self._cycle = not self._cycle
                self._message = f"Cycle={'on' if self._cycle else 'off'}（未伺服）"
            elif command == "scan":
                self._running = False
                self._message = self.arm.scan()
            elif command == "start":
                self._message = self.arm.start()
                self._running = bool(self.arm.status()["ibvs_enabled"])
                begin_record = True
                begin_trajectory = self._running
            elif command == "start_sequence":
                self._message = self.arm.start_sequence()
                self._running = bool(self.arm.status()["ibvs_enabled"])
                begin_record = self._running
                begin_trajectory = self._running
            elif command == "sequence_next":
                self._message = self.arm.advance_sequence()
                status = self.arm.status()
                self._running = bool(status["ibvs_enabled"] or status["running"])
            elif command == "sequence_direct_pbvs":
                self._message = self.arm.set_sequence_mode("direct_pbvs")
            elif command == "sequence_hybrid":
                self._message = self.arm.set_sequence_mode("hybrid")
            elif command in {"pause", "stop"}:
                self._running = False
                self._message = self.arm.pause()
                end_record = True
            elif command == "estop":
                self._running = False
                self._message = self.arm.emergency_stop()
                end_record = True
                end_trajectory_reason = "estop"
            elif command == "release":
                self._running = False
                self._message = self.arm.release_for_operator()
                end_record = True
                end_trajectory_reason = "release"
            elif command == "mode_direct_pbvs":
                self._message = self.arm.set_servo_mode("direct_pbvs")
            elif command == "mode_hybrid":
                self._message = self.arm.set_servo_mode("hybrid")
            elif command == "backend_arm_sdk":
                self._message = self.arm.set_motion_backend("arm_sdk")
            elif command == "backend_service":
                self._message = self.arm.set_motion_backend("service")
            elif command == "hang":
                self._running = False
                self._message = self.arm.hang_arms()
                end_record = True
                end_trajectory_reason = "hang"
            elif command == "level_right":
                self._running = False
                self._message = self.arm.level_right_arm()
                end_record = True
                end_trajectory_reason = "level_right"
            elif command == "reset":
                self._target = "button_1"
                self._cycle = False
                self._running = False
                self._commands.clear()
                if self.ibvs is not None:
                    self.ibvs.set_target("button_1")
                self._message = self.arm.return_to_ready()
                end_record = True
                end_trajectory_reason = "reset"
            payload = {
                "command": command,
                "target": self._target,
                "cycle": self._cycle,
                "running": self._running,
                "at": time.time(),
            }
            self._last_command = payload
            self._commands.append(payload)
        if end_record:
            self.recorder.end()
        if end_trajectory_reason:
            self.motion_trajectories.end(end_trajectory_reason)
        if begin_record:
            self.recorder.begin(self.cameras, self.state)
        if begin_trajectory:
            self.motion_trajectories.begin(self.state, target=self._target)
        return payload

    def state(self) -> dict[str, Any]:
        payload = self._session()
        payload["camera"] = self.camera.status()
        payload["cameras"] = {name: worker.status() for name, worker in self.cameras.items()}
        upper = self.joints.snapshot()
        right_rows = sorted(
            (
                row
                for row in upper.joints
                if row.get("group") == "right_arm" and row.get("q") is not None
            ),
            key=lambda row: int(row.get("index") or 0),
        )
        if len(right_rows) == 7:
            self.arm.update_joint_readback([float(row["q"]) for row in right_rows])
        payload["upper_body"] = upper.as_dict()
        payload["ibvs"] = None if self.ibvs is None else self.ibvs.telemetry()
        payload["arm"] = self.arm.status()
        payload["record"] = self.recorder.status()
        payload["trajectory"] = self.motion_trajectories.status()
        payload["screen"] = {"ready": self._screen_jpeg is not None, "at": self._screen_at}
        payload["eye_to_hand_bias_m"] = self._eye_bias_payload()
        payload["updated_at"] = time.time()
        return payload

    def _eye_bias_payload(self) -> dict[str, float]:
        x_m, y_m, z_m = self._startup_eye_bias_m
        eye = getattr(self.ibvs, "eye_to_hand", None) if self.ibvs is not None else None
        if isinstance(eye, dict) and "torso_z_bias_m" in eye:
            x_m = float(eye["torso_x_bias_m"])
            y_m = float(eye["torso_y_bias_m"])
            z_m = float(eye["torso_z_bias_m"])
        return {"x": round(x_m, 6), "y": round(y_m, 6), "z": round(z_m, 6)}

    def _capture_screen(self) -> None:
        jpeg = None if self.ibvs is None else self.ibvs.screen_jpeg()
        with self._lock:
            self._screen_jpeg = jpeg
            self._screen_at = time.time() if jpeg else 0.0

    def _configure_routes(self) -> None:
        @self._app.get("/")
        def index() -> Response:
            response = make_response(PAGE_HTML)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return response

        @self._app.get("/health")
        def health() -> Response:
            cam = self.camera.status()
            return jsonify({"ok": True, "has_frame": cam["frames"] > 0, "camera": cam["state"], "serial": self.serial})

        @self._app.get("/state")
        def state() -> Response:
            return jsonify(self.state())

        @self._app.get("/stream.mjpg")
        def stream() -> Response:
            resp = Response(self._mjpeg(self.camera), mimetype="multipart/x-mixed-replace; boundary=frame")
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return resp

        @self._app.get("/stream/<name>.mjpg")
        def stream_named(name: str) -> Response:
            worker = self.cameras.get(name) or self.camera
            resp = Response(self._mjpeg(worker), mimetype="multipart/x-mixed-replace; boundary=frame")
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            return resp

        @self._app.get("/snapshot/<name>.jpg")
        def snapshot_named(name: str) -> Response:
            worker = self.cameras.get(name) or self.camera
            jpeg = worker.jpeg()
            if not jpeg:
                return make_response("no frame", 503)
            resp = make_response(jpeg)
            resp.headers["Content-Type"] = "image/jpeg"
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return resp

        @self._app.get("/depth/<name>.png")
        def depth_preview(name: str) -> Response:
            path = DEPTH_PATHS.get(name)
            depth = None if path is None else read_depth(path)
            if depth is None:
                return make_response("depth unavailable", 503)
            blob = _depth_preview_png(depth)
            if blob is None:
                return make_response("depth encode failed", 500)
            resp = make_response(blob)
            resp.headers["Content-Type"] = "image/png"
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return resp

        @self._app.get("/depth/<name>/value")
        def depth_value(name: str) -> Response:
            path = DEPTH_PATHS.get(name)
            depth = None if path is None else read_depth(path)
            if depth is None:
                return jsonify({"ok": False, "error": "depth unavailable"}), 503
            try:
                u = int(round(float(request.args["u"])))
                v = int(round(float(request.args["v"])))
            except (KeyError, TypeError, ValueError):
                return jsonify({"ok": False, "error": "u/v required"}), 400
            height, width = depth.shape[:2]
            if not (0 <= u < width and 0 <= v < height):
                return jsonify({"ok": False, "error": "u/v out of range"}), 400
            value = float(depth[v, u])
            radius = 2
            patch = depth[
                max(0, v - radius) : min(height, v + radius + 1),
                max(0, u - radius) : min(width, u + radius + 1),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0.05)]
            median = None if valid.size == 0 else float(np.median(valid))
            return jsonify(
                {
                    "ok": True,
                    "u": u,
                    "v": v,
                    "depth_m": value if np.isfinite(value) and value > 0.05 else None,
                    "median_5x5_m": median,
                    "size": [width, height],
                }
            )

        @self._app.post("/dataset/save/<name>")
        def save_dataset_shot(name: str) -> Response:
            worker = self.cameras.get(name)
            if worker is None:
                return jsonify({"ok": False, "error": f"unknown camera: {name}"}), 404
            blob = worker.raw_jpeg()
            if not blob:
                return jsonify({"ok": False, "error": "no frame"}), 503
            folder_name = "chest" if name == "chest" else "side"
            folder = self.dataset_root / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            labels = (
                "panel\n"
                "gripper\n"
                "screen\n"
                "gripper_tip\n"
                "button_1\n"
                "button_2\n"
                "button_3\n"
            )
            for labels_path in (self.dataset_root / "labels.txt", folder / "labels.txt"):
                if not labels_path.exists():
                    labels_path.write_text(labels, encoding="utf-8")
            filename = f"{folder_name}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            path = folder / filename
            path.write_bytes(blob)
            resp = make_response(blob)
            resp.headers["Content-Type"] = "image/jpeg"
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            resp.headers["X-Dataset-Path"] = str(path)
            resp.headers["Cache-Control"] = "no-store"
            return resp

        @self._app.post("/record")
        def record() -> Response:
            payload = request.get_json(silent=True) or {}
            status = self.recorder.set_enabled(bool(payload.get("enabled")))
            return jsonify({"ok": True, "record": status})

        @self._app.post("/trajectories/enabled")
        def trajectories_enabled() -> Response:
            payload = request.get_json(silent=True) or {}
            status = self.motion_trajectories.set_enabled(bool(payload.get("enabled")))
            return jsonify({"ok": True, "trajectory": status})

        @self._app.post("/sequence")
        def sequence() -> Response:
            payload = request.get_json(silent=True) or {}
            steps = payload.get("steps") or []
            if not isinstance(steps, list):
                return jsonify({"ok": False, "error": "steps must be a list"}), 400
            cleaned = self.arm.set_button_sequence(steps)
            return jsonify({"ok": True, "steps": cleaned})

        @self._app.get("/screen/latest.jpg")
        def screen_latest() -> Response:
            blob = self._screen_jpeg
            if not blob:
                return make_response("no screen", 404)
            resp = make_response(blob)
            resp.headers["Content-Type"] = "image/jpeg"
            resp.headers["Cache-Control"] = "no-store"
            if request.args.get("download"):
                resp.headers["Content-Disposition"] = 'attachment; filename="screen.jpg"'
            return resp

        @self._app.get("/trajectories")
        def trajectories() -> Response:
            return jsonify(
                {
                    "ok": True,
                    "trajectories": self.motion_trajectories.list(),
                    "status": self.motion_trajectories.status(),
                }
            )

        @self._app.post("/trajectories/play")
        def play_trajectory() -> Response:
            payload = request.get_json(silent=True) or {}
            trajectory_id = str(payload.get("id") or "")
            try:
                speed = float(payload.get("speed") or 2.0)
                samples = self.motion_trajectories.load(trajectory_id)
            except (TypeError, ValueError, FileNotFoundError) as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            self.motion_trajectories.end("playback")
            result = self.arm.start_playback(trajectory_id, samples, speed)
            return jsonify(result), (200 if result.get("ok") else 409)

        @self._app.post("/trajectories/delete")
        def delete_trajectory() -> Response:
            payload = request.get_json(silent=True) or {}
            trajectory_id = str(payload.get("id") or "")
            try:
                result = self.motion_trajectories.delete(trajectory_id)
            except (TypeError, ValueError, FileNotFoundError) as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            return jsonify(result)

        @self._app.post("/trajectories/delete_all")
        def delete_all_trajectories() -> Response:
            return jsonify(self.motion_trajectories.delete_all())

        @self._app.post("/eye_to_hand_bias")
        def eye_to_hand_bias() -> Response:
            payload = request.get_json(silent=True) or {}
            try:
                x_m = float(payload.get("x_m"))
                y_m = float(payload.get("y_m"))
                z_m = float(payload.get("z_m"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "x_m y_m z_m 需要是数字，单位 m"}), 400
            if self.ibvs is None or not hasattr(self.ibvs, "set_eye_to_hand_bias"):
                return jsonify({"ok": False, "error": "胸口手眼未加载"}), 409
            ok, detail = self.ibvs.set_eye_to_hand_bias(x_m, y_m, z_m)
            if not ok:
                return jsonify({"ok": False, "error": detail}), 409
            return jsonify({"ok": True, "eye_to_hand_bias_m": self._eye_bias_payload()})

        @self._app.post("/command")
        def command() -> Response:
            payload = request.get_json(silent=True) or {}
            raw = str(payload.get("command", "")).strip().lower()
            cmd = COMMAND_ALIASES.get(raw, raw)
            if cmd not in COMMANDS:
                return jsonify({"ok": False, "error": f"unsupported: {raw}"}), 400
            applied = self._apply_command(cmd)
            return jsonify({"ok": True, "command": applied, "state": self._session()})

        @self._app.post("/manual/jog")
        def manual_jog() -> Response:
            payload = request.get_json(silent=True) or {}
            try:
                dxyz = tuple(float(value) for value in payload.get("dxyz_m", (0.0, 0.0, 0.0)))
                drpy = tuple(float(value) for value in payload.get("drpy_deg", (0.0, 0.0, 0.0)))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "invalid jog values"}), 400
            if len(dxyz) != 3 or len(drpy) != 3:
                return jsonify({"ok": False, "error": "dxyz_m/drpy_deg need 3 values"}), 400
            result = self.arm.manual_jog(dxyz, drpy)
            return jsonify(result), (200 if result.get("ok") else 409)

        @self._app.post("/manual/elbow")
        def manual_elbow() -> Response:
            payload = request.get_json(silent=True) or {}
            try:
                delta_deg = float(payload.get("delta_deg"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "delta_deg required"}), 400
            result = self.arm.manual_elbow(delta_deg)
            return jsonify(result), (200 if result.get("ok") else 409)

        @self._app.post("/manual/joint")
        def manual_joint() -> Response:
            payload = request.get_json(silent=True) or {}
            try:
                index = int(payload.get("index"))
                delta_deg = float(payload.get("delta_deg"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "index and delta_deg required"}), 400
            result = self.arm.manual_joint(index, delta_deg)
            return jsonify(result), (200 if result.get("ok") else 409)

        @self._app.post("/tip_seed")
        def tip_seed() -> Response:
            if self.ibvs is None:
                return jsonify({"ok": False, "error": "live IBVS unavailable"}), 503
            payload = request.get_json(silent=True) or {}
            if bool(payload.get("clear")):
                self.ibvs.clear_tip_seed()
                return jsonify({"ok": True, "cleared": True})
            try:
                u = float(payload["u"])
                v = float(payload["v"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"ok": False, "error": "u/v required"}), 400
            if not (0.0 <= u < 4096.0 and 0.0 <= v < 4096.0):
                return jsonify({"ok": False, "error": "u/v out of range"}), 400
            self.ibvs.set_tip_seed(u, v)
            return jsonify({"ok": True, "tip_uv": [u, v]})

        @self._app.post("/wrist_yellow_target")
        def wrist_yellow_target() -> Response:
            if self.ibvs is None or not hasattr(self.ibvs, "pick_wrist_yellow"):
                return jsonify({"ok": False, "error": "右腕画面未加载"}), 503
            payload = request.get_json(silent=True) or {}
            if bool(payload.get("clear")):
                self.ibvs.clear_wrist_yellow("已取消黄色点选")
                return jsonify({"ok": True, "cleared": True})
            try:
                u = float(payload["u"])
                v = float(payload["v"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"ok": False, "error": "u/v required"}), 400
            ok, detail = self.ibvs.pick_wrist_yellow(u, v)
            if not ok:
                return jsonify({"ok": False, "error": detail}), 409
            return jsonify({"ok": True, "message": detail, "u": u, "v": v})

    def _mjpeg(self, worker: ChestCameraWorker | None = None):
        cam = worker or self.camera
        boundary = b"--frame"
        while True:
            jpeg = cam.jpeg()
            if jpeg:
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(0.04)


PAGE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>H2 IBVS 实机监视</title>
  <meta http-equiv="Cache-Control" content="no-store"/>
  <style>
    html,body{height:100%;overflow:hidden}
    body{margin:0;background:#111;color:#eee;font-family:Arial,"Microsoft YaHei",sans-serif}
    header{padding:10px 14px;background:#1b1b1b;border-bottom:1px solid #333;display:flex;gap:12px;align-items:center;flex:0 0 auto}
    h1{margin:0;font-size:18px;font-weight:600}
    .meta{color:#9fd0ff;font:12px Consolas,monospace}
    .layout{display:flex;height:calc(100vh - 46px);min-height:0}
    .videos-col{flex:1;min-width:0;min-height:0;overflow:auto;background:#000;display:flex;flex-direction:column}
    .videos{display:grid;grid-template-columns:1fr 1fr;background:#000;min-height:min(48vh,520px);flex:0 0 auto}
    .video{display:flex;flex-direction:column;min-width:0;min-height:280px;border-right:1px solid #333}
    .video h2,.still-box h2{margin:0;padding:8px 10px;font-size:13px;font-weight:600;background:#141414;color:#9fd0ff;flex:0 0 auto}
    .stage{flex:1;min-height:220px;display:flex;align-items:center;justify-content:center;background:#000}
    .stage img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;object-position:center center;background:#000}
    #sideView{cursor:crosshair}
    .shotbar{display:flex;align-items:center;gap:8px;padding:6px 10px;background:#141414;border-top:1px solid #333}
    .shotbar button{padding:6px 10px;font-size:12px}
    .shotbar span{color:#9fd0ff;font:11px Consolas,monospace}
    .depth-box{border-top:1px solid #365568;background:#050505}
    .depth-head{display:flex;justify-content:space-between;padding:6px 10px;background:#101820;color:#9fd0ff;font:12px Consolas,monospace}
    .depth-wrap{position:relative;display:flex;justify-content:center;min-height:180px;background:#000;cursor:crosshair}
    .depth-wrap img{max-width:100%;height:auto;object-fit:contain}
    .depth-tip{position:absolute;display:none;padding:4px 6px;border-radius:4px;background:rgba(0,0,0,.82);color:#fff;font:12px Consolas,monospace;pointer-events:none;white-space:nowrap}
    .stills{display:none;grid-template-columns:1fr 1fr;border-top:2px solid #4a6;flex:0 0 auto}
    .stills.show{display:grid}
    .still-box{display:flex;flex-direction:column;min-height:300px;border-right:1px solid #333;background:#0c0c0c}
    .still-stage{flex:1;min-height:260px;display:flex;align-items:center;justify-content:center;background:#111;padding:8px}
    .still-wrap{position:relative;display:inline-block;line-height:0;max-width:100%}
    .still-wrap img{max-width:100%;height:auto;cursor:crosshair;background:#000}
    .cross{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border:2px solid #ff4dff;border-radius:50%;box-shadow:0 0 0 1px #000;pointer-events:none}
    .panel{width:460px;flex:0 0 460px;padding:14px;background:#1b1b1b;border-left:1px solid #333;overflow:auto;height:100%;box-sizing:border-box}
    .panel,.panel table,.panel td,.panel pre,.panel .pose-value{-webkit-user-select:text;user-select:text}
    .panel button,.panel select,.panel input{-webkit-user-select:none;user-select:none}
    .controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
    button{padding:8px 12px;border:1px solid rgba(255,255,255,.12);border-radius:6px;color:#fff;cursor:pointer}
    button:hover{filter:brightness(1.08)}
    button.active{outline:2px solid #fff}
    .b1{background:#315a9b} .b2{background:#2f7a7a} .b3{background:#8a5a1d}
    .cycle{background:#5a3d7a} .scan{background:#24627a} .snap{background:#6a4a1d} .start{background:#1f7a3a} .stop{background:#9b2c2c} .estop{background:#ff1a1a;color:#fff;font-weight:700} .release{background:#8a3d12} .reset{background:#8a6d1d} .mode{background:#1d4e8a}
    .record-toggle{display:inline-flex;align-items:center;gap:8px;color:#ddd;font-size:13px;margin-left:4px}
    .record-toggle input{position:absolute;opacity:0;width:0;height:0}
    .record-toggle .track{width:42px;height:24px;background:#444;border-radius:12px;position:relative;cursor:pointer;transition:background .15s}
    .record-toggle .track::after{content:"";position:absolute;top:3px;left:3px;width:18px;height:18px;background:#eee;border-radius:50%;transition:transform .15s}
    .record-toggle input:checked + .track{background:#9b2c2c}
    .record-toggle input:checked + .track::after{transform:translateX(18px)}
    .hint{color:#888;font-size:12px;line-height:1.45}
    .manual{margin:10px 0;padding:10px;border:1px solid #3b4b58;border-radius:7px;background:#121a20}
    .manual-title{display:flex;justify-content:space-between;align-items:center;color:#9fd0ff;font-size:13px;margin-bottom:8px}
    .manual-grid{display:grid;grid-template-columns:74px 1fr 1fr;gap:6px;align-items:center}
    .manual-grid span{font:12px Consolas,monospace;color:#bbb}
    .manual-grid button{padding:6px 8px;background:#29445a;font:12px Consolas,monospace}
    .manual-steps{display:flex;gap:12px;align-items:center;margin-top:8px;color:#aaa;font-size:12px}
    .manual select{background:#222;color:#eee;border:1px solid #555;border-radius:4px;padding:3px}
    .trajectory-box{margin:10px 0;padding:10px;border:1px solid #66513b;border-radius:7px;background:#1b1712}
    .trajectory-row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
    .trajectory-row select{min-width:190px;background:#222;color:#eee;border:1px solid #555;border-radius:4px;padding:5px}
    .trajectory-row button{background:#725126;padding:6px 9px}
    .trajectory-row button.danger{background:#6d2a2a}
    .sequence-box{margin:10px 0;padding:10px;border:1px solid #3d5a48;border-radius:7px;background:#121a16}
    .sequence-list{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-direction:column;gap:6px}
    .sequence-list li{display:flex;align-items:center;gap:8px;padding:6px 8px;background:#243028;border:1px solid #3d5a48;border-radius:5px;cursor:grab}
    .sequence-list li.drag{opacity:.45}
    .sequence-list button{padding:2px 8px;background:#5a3030}
    .screen-box{border-top:1px solid #365568;background:#050505;padding:8px 10px}
    .screen-box h2{margin:0 0 6px;font-size:13px;color:#9fd0ff}
    .screen-box img{max-width:100%;background:#000;display:block}
    .screen-box a{color:#9fd0ff;font-size:12px}
    .pose-value{font:12px Consolas,monospace;color:#d8f1ff}
    .coord-panel{border:1px solid #41677d;border-radius:7px;background:#101820;padding:8px;margin-top:10px}
    .coord-panel h3{margin-top:0;color:#9fd0ff}
    .bias-form{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:flex-start}
    .bias-form label{display:flex;align-items:center;gap:4px;color:#9fd0ff}
    .bias-form input{width:6.2em;background:#0c1218;color:#d7e7f3;border:1px solid #41677d;border-radius:4px;padding:3px 4px}
    .bias-form button{background:#1d4c38;color:#e8fff4;border:0;border-radius:4px;padding:4px 10px;cursor:pointer}
    #bias-status{color:#c8ffb0;font-size:12px}
    .bias-hint{display:block;margin-top:4px;color:#8aa4b8;font:12px sans-serif;text-align:left}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#24351f;color:#c8ffb0;margin:2px 4px 2px 0;font-size:12px}
    table{width:100%;border-collapse:collapse;font:12px Consolas,monospace}
    th,td{padding:4px 6px;border-bottom:1px solid #2a2a2a;text-align:right}
    th:first-child,td:first-child{text-align:left}
    h3{margin:16px 0 6px;font-size:14px;color:#bbb}
    pre{white-space:pre-wrap;font-size:12px;line-height:1.35}
  </style>
</head>
<body>
<header>
  <h1>H2 IBVS 实机监视</h1>
  <span class="meta" id="camMeta">sidecar-bgr …</span>
</header>
<div class="layout">
  <div class="videos-col">
    <div class="videos">
      <div class="video">
        <h2 id="title-chest">D435 胸口 · 直播</h2>
        <div class="stage"><img id="chestView" src="/stream/chest.mjpg?v=bgr" alt="chest D435"/></div>
        <div class="shotbar">
          <button class="snap" onclick="saveShot('chest')">Save Screenshot</button>
          <span id="shot-chest"></span>
        </div>
        <div class="depth-box">
          <div class="depth-head"><span>胸口对齐深度图</span><span id="depth-readout">悬停读取深度</span></div>
          <div class="depth-wrap" id="depthWrap">
            <img id="chestDepth" src="/depth/chest.png" alt="chest aligned depth" onmousemove="depthHover(event)" onmouseleave="depthLeave()"/>
            <div class="depth-tip" id="depthTip"></div>
          </div>
        </div>
      </div>
      <div class="video">
        <h2 id="title-d435i">D435I 左手侧视 · 直播</h2>
        <div class="stage"><img id="sideView" src="/stream/d435i.mjpg?v=bgr" alt="right wrist D435I" onclick="pickWristYellow(event)" oncontextmenu="clearWristYellow(event)"/></div>
        <div class="shotbar">
          <button class="snap" onclick="saveShot('d435i')">Save Screenshot</button>
          <span id="shot-side">进近中左键点黄色按键，右键取消</span>
        </div>
        <div class="screen-box">
          <h2>Screen</h2>
          <img id="screenShot" alt="screen" hidden/>
          <div class="shotbar">
            <a id="screenDownload" href="/screen/latest.jpg?download=1">Download</a>
            <span id="screenStatus">no screen yet</span>
          </div>
        </div>
      </div>
    </div>
    <div class="stills" id="stills">
      <div class="still-box">
        <h2>胸口静帧 · 在这里点真实尖端</h2>
        <div class="still-stage">
          <div class="still-wrap">
            <img id="chestStill" alt="chest still" onclick="setTip(event)" oncontextmenu="clearTip(event)"/>
            <div id="chestMark" class="cross" hidden></div>
          </div>
        </div>
      </div>
      <div class="still-box">
        <h2>左手侧视静帧 · 只对照接触，不点选</h2>
        <div class="still-stage">
          <div class="still-wrap">
            <img id="wristStill" alt="wrist still"/>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="panel">
    <div class="hint">Hybrid VS：胸口 PBVS 粗定位，右腕图像 IBVS 把按钮对准夹爪尖后进近，并由右腕视觉确认接触。选择按钮后 Start；No Raise required。arm_sdk 和 h2_arm_service_node 均保留；运动中不要切换控制链路，only after Release 才切换正在占用的链路。Reset 和 Ctrl+C 会放下双臂。</div>
    <div class="controls">
      <button class="b1" data-cmd="button_1" onclick="send('button_1',this)">Button 1</button>
      <button class="b2" data-cmd="button_2" onclick="send('button_2',this)">Button 2</button>
      <button class="b3" data-cmd="button_3" onclick="send('button_3',this)">Button 3</button>
      <button class="cycle" data-cmd="cycle" onclick="send('cycle',this)">Cycle</button>
      <button class="mode" data-cmd="mode_direct_pbvs" onclick="send('mode_direct_pbvs',this)">Direct PBVS</button>
      <button class="mode" data-cmd="mode_hybrid" onclick="send('mode_hybrid',this)">Hybrid VS</button>
      <button class="mode" data-cmd="backend_arm_sdk" onclick="send('backend_arm_sdk',this)">arm_sdk</button>
      <button class="mode" data-cmd="backend_service" onclick="send('backend_service',this)">Service</button>
      <button class="scan" data-cmd="scan" onclick="send('scan',this)">Raise</button>
      <button class="scan" data-cmd="level_right" onclick="send('level_right',this)" title="已关闭，不会动右臂">Right level off</button>
      <button class="snap" onclick="captureStill()">截图</button>
      <button class="start" data-cmd="start" onclick="send('start',this)">Start</button>
      <button class="start" data-cmd="start_sequence" onclick="send('start_sequence',this)">组合启动</button>
      <button class="estop" data-cmd="estop" onclick="send('estop',this)">急停</button>
      <button class="stop" data-cmd="pause" onclick="send('pause',this)">Pause / 细调</button>
      <button class="release" data-cmd="release" onclick="send('release',this)">释放双臂</button>
      <button class="reset" data-cmd="hang" onclick="send('hang',this)">下垂</button>
      <button class="reset" data-cmd="reset" onclick="send('reset',this)">Reset</button>
      <label class="record-toggle" title="默认关闭。打开后，下一次 Start 才开始录制">
        <input id="recordToggle" type="checkbox" onchange="setRecord(this.checked)"/>
        <span class="track"></span>
        <span id="recordLabel">录制关</span>
      </label>
    </div>
    <div class="manual">
      <div class="manual-title"><span>右臂细调（先 Pause）</span><span id="manual-state">未启用</span></div>
      <div class="manual-grid">
        <span>dx 前后</span><button onclick="jog('x',-1)">X− 后</button><button onclick="jog('x',1)">X+ 前</button>
        <span>dy 左右</span><button onclick="jog('y',-1)">Y− 右</button><button onclick="jog('y',1)">Y+ 左</button>
        <span>dz 上下</span><button onclick="jog('z',-1)">Z− 下</button><button onclick="jog('z',1)">Z+ 上</button>
        <span>droll</span><button onclick="jog('roll',-1)">R−</button><button onclick="jog('roll',1)">R+</button>
        <span>dpitch</span><button onclick="jog('pitch',-1)">P−</button><button onclick="jog('pitch',1)">P+</button>
        <span>dyaw</span><button onclick="jog('yaw',-1)">Y−</button><button onclick="jog('yaw',1)">Y+</button>
        <span>肩 pitch</span><button type="button" data-joint="0" data-sign="-1">−</button><button type="button" data-joint="0" data-sign="1">+</button>
        <span>肩 roll</span><button type="button" data-joint="1" data-sign="-1">−</button><button type="button" data-joint="1" data-sign="1">+</button>
        <span>肩 yaw</span><button type="button" data-joint="2" data-sign="-1">−</button><button type="button" data-joint="2" data-sign="1">+</button>
        <span>右肘</span><button type="button" data-joint="3" data-sign="-1">− 伸</button><button type="button" data-joint="3" data-sign="1">+ 弯</button>
        <span>腕 roll</span><button type="button" data-joint="4" data-sign="-1">−</button><button type="button" data-joint="4" data-sign="1">+</button>
        <span>腕 pitch</span><button type="button" data-joint="5" data-sign="-1">−</button><button type="button" data-joint="5" data-sign="1">+</button>
        <span>腕 yaw</span><button type="button" data-joint="6" data-sign="-1">−</button><button type="button" data-joint="6" data-sign="1">+</button>
      </div>
      <div class="manual-steps">
        <label>平移 <select id="xyzStep"><option value="0.001">1 mm</option><option value="0.002" selected>2 mm</option><option value="0.005">5 mm</option></select></label>
        <label>旋转 <select id="rpyStep"><option value="0.5">0.5°</option><option value="1" selected>1°</option><option value="2">2°</option></select></label>
      </div>
      <div class="hint" id="jog-message">坐标系：躯干 X 向前、Y 向左、Z 向上。下面七个关节沿用运控限位，每一下只转当前这一个轴。右肘往弯的方向加，戳击更平。</div>
    </div>
    <div class="trajectory-box">
      <div class="manual-title"><span>运动轨迹录制 / 加速回放</span><span id="trajectory-status">未录制</span></div>
      <div class="trajectory-row">
        <label class="record-toggle" title="默认关闭。打开后，下一次 Start 才录制运动轨迹">
          <input id="trajectoryToggle" type="checkbox" onchange="setTrajectoryRecord(this.checked)"/>
          <span class="track"></span>
          <span id="trajectoryRecordLabel">轨迹关</span>
        </label>
      </div>
      <div class="trajectory-row">
        <select id="trajectorySelect"><option value="">暂无轨迹</option></select>
        <label>速度
          <select id="trajectorySpeed">
            <option value="1">1×</option>
            <option value="1.5">1.5×</option>
            <option value="2" selected>2×</option>
            <option value="3">3×</option>
            <option value="4">4×</option>
          </select>
        </label>
        <button onclick="loadTrajectories()">刷新</button>
        <button onclick="playTrajectory()">回放</button>
      </div>
      <div class="trajectory-row">
        <button class="danger" onclick="deleteTrajectory()">删除所选轨迹</button>
        <button class="danger" onclick="deleteAllTrajectories()">删除全部轨迹</button>
      </div>
      <div class="hint">默认关。打开后 Start 才录制；Pause 微调继续记录；切换 Button 或释放双臂时保存。</div>
    </div>
    <div class="sequence-box">
      <div class="manual-title"><span>动作组合</span><span id="sequence-status">空</span></div>
      <div class="trajectory-row">
        <button type="button" class="mode" data-cmd="sequence_direct_pbvs" onclick="send('sequence_direct_pbvs', this)">Direct PBVS</button>
        <button type="button" class="mode" data-cmd="sequence_hybrid" onclick="send('sequence_hybrid', this)">Hybrid</button>
        <button type="button" onclick="addSequence('button_1')">+ Button 1</button>
        <button type="button" onclick="addSequence('button_2')">+ Button 2</button>
        <button type="button" onclick="addSequence('button_3')">+ Button 3</button>
        <button type="button" class="danger" onclick="clearSequence()">清空</button>
        <button type="button" onclick="send('sequence_next', this)">Next Step</button>
      </div>
      <ol class="sequence-list" id="sequenceList"></ol>
      <div class="hint">拖动排序，可以重复。先选 Direct PBVS 或 Hybrid，再点上面的「组合启动」。Direct PBVS 每颗一次到位；Hybrid 是胸口 PBVS 再右腕进近。组合为空会报错。Start 仍只按当前选中的一颗，并使用上面的模式按钮。做完一个动作后，右臂向下、向右移出面板，胸口重新识别，再做下一颗。已经按到但没检测到时，点 Next Step 会立刻移开手臂并进入下一颗；已经是最后一颗则放下右臂并释放控制。</div>
    </div>
    <div id="pills"></div>
    <div class="hint" id="message">loading…</div>
    <div class="coord-panel">
      <h3>目标 / right_ee_link（Torso_Link 坐标）</h3>
      <table>
        <tbody>
          <tr><td>目标 XYZ m</td><td class="pose-value" id="world-target-xyz">—</td></tr>
          <tr><td>目标来源</td><td class="pose-value" id="world-target-source">—</td></tr>
          <tr><td>末端 XYZ m</td><td class="pose-value" id="world-ee-xyz">—</td></tr>
          <tr><td>目标−末端 m</td><td class="pose-value" id="world-error-xyz">—</td></tr>
          <tr><td>末端 RPY deg</td><td class="pose-value" id="world-ee-rpy">—</td></tr>
          <tr>
            <td>胸口偏移 m</td>
            <td>
              <form id="bias-form" class="bias-form" onsubmit="submitEyeBias(event)">
                <label>X<input id="bias-x" type="number" step="0.001" required oninput="clearBiasStatus()"></label>
                <label>Y<input id="bias-y" type="number" step="0.001" required oninput="clearBiasStatus()"></label>
                <label>Z<input id="bias-z" type="number" step="0.001" required oninput="clearBiasStatus()"></label>
                <button type="submit">提交</button>
                <span id="bias-status"></span>
              </form>
              <span class="bias-hint">躯干坐标，单位 m。默认是本次启动的 --eye-to-hand-x/y/z-bias-m。提交后立刻生效，重启网页回到启动参数。</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <h3>IBVS / 目标</h3>
    <table>
      <tbody>
        <tr><td>phase</td><td id="ibvs-phase">—</td></tr>
        <tr><td>target uv</td><td id="ibvs-uv">—</td></tr>
        <tr><td>control tip uv</td><td id="ibvs-tip-uv">—</td></tr>
        <tr><td>tip source</td><td id="ibvs-tip-source">—</td></tr>
        <tr><td>target xyz m</td><td id="ibvs-xyz">—</td></tr>
        <tr><td>FK / control tip xyz m</td><td id="ibvs-tip-xyz">—</td></tr>
        <tr><td>PBVS error cam m</td><td id="ibvs-pbvs-error">—</td></tr>
        <tr><td>PBVS error torso m</td><td id="ibvs-pbvs-error-torso">—</td></tr>
        <tr><td>PBVS tip source</td><td id="ibvs-pbvs-source">—</td></tr>
        <tr><td>dx dy dz m</td><td id="ibvs-dxyz">—</td></tr>
        <tr><td>droll dpitch dyaw deg</td><td id="ibvs-drpy">—</td></tr>
        <tr><td>handeye</td><td id="ibvs-handeye">—</td></tr>
      </tbody>
    </table>
    <h3>右腕关节原点实时位姿（Torso_Link）</h3>
    <table>
      <tbody>
        <tr><td>xyz m</td><td class="pose-value" id="ee-xyz">—</td></tr>
        <tr><td>roll pitch yaw deg</td><td class="pose-value" id="ee-rpy">—</td></tr>
      </tbody>
    </table>
    <h3>上肢关节</h3>
    <table>
      <thead><tr><th>关节</th><th>q rad</th><th>q deg</th><th>dq</th></tr></thead>
      <tbody id="joints"></tbody>
    </table>
    <h3>State</h3>
    <pre id="state">waiting…</pre>
  </div>
</div>
<script>
function imageUv(event){
  const img=event.currentTarget, r=img.getBoundingClientRect();
  const nw=img.naturalWidth||640, nh=img.naturalHeight||480;
  const scale=Math.min(r.width/nw,r.height/nh);
  const ox=(r.width-nw*scale)/2, oy=(r.height-nh*scale)/2;
  const u=(event.clientX-r.left-ox)/scale, v=(event.clientY-r.top-oy)/scale;
  return {img,u,v,nw,nh,scale,ox,oy};
}
let lastDepthQuery=0;
let depthQuerySerial=0;
async function depthHover(event){
  const p=imageUv(event);
  if(p.u<0||p.v<0||p.u>=p.nw||p.v>=p.nh) return;
  const wrap=document.getElementById('depthWrap');
  const tip=document.getElementById('depthTip');
  const readout=document.getElementById('depth-readout');
  if(wrap&&tip){
    const r=wrap.getBoundingClientRect();
    tip.style.left=(event.clientX-r.left+12)+'px';
    tip.style.top=(event.clientY-r.top+12)+'px';
    tip.style.display='block';
  }
  const now=performance.now();
  if(now-lastDepthQuery<80) return;
  lastDepthQuery=now;
  const serial=++depthQuerySerial;
  try{
    const data=await fetch(`/depth/chest/value?u=${p.u.toFixed(2)}&v=${p.v.toFixed(2)}`,{cache:'no-store'}).then(r=>r.json());
    if(serial!==depthQuerySerial) return;
    const exact=data.depth_m==null?'invalid':`${Number(data.depth_m).toFixed(3)} m`;
    const median=data.median_5x5_m==null?'invalid':`${Number(data.median_5x5_m).toFixed(3)} m`;
    const text=`u=${data.u} v=${data.v}  z=${exact}  median=${median}`;
    if(tip) tip.textContent=text;
    if(readout) readout.textContent=text;
  }catch(e){
    if(tip) tip.textContent='depth unavailable';
    if(readout) readout.textContent='depth unavailable';
  }
}
function depthLeave(){
  const tip=document.getElementById('depthTip');
  if(tip) tip.style.display='none';
}
function refreshDepth(){
  const img=document.getElementById('chestDepth');
  if(img) img.src='/depth/chest.png?t='+Date.now();
}
function markStill(img,u,v){
  const mark=document.getElementById('chestMark');
  if(!mark||!img) return;
  const nw=img.naturalWidth||640, nh=img.naturalHeight||480;
  const scale=img.clientWidth/nw;
  mark.style.left=(u*scale)+'px';
  mark.style.top=(v*scale)+'px';
  mark.hidden=false;
}
async function saveShot(name){
  const res = await fetch('/dataset/save/'+name,{method:'POST'});
  if(!res.ok){
    const slot=document.getElementById(name==='chest'?'shot-chest':'shot-side');
    if(slot) slot.textContent='save failed';
    return;
  }
  const blob=await res.blob();
  const header=res.headers.get('Content-Disposition')||'';
  const match=/filename="([^"]+)"/.exec(header);
  const filename=match?match[1]:(name+'.jpg');
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');
  link.href=url;
  link.download=filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  const saved=res.headers.get('X-Dataset-Path')||filename;
  const slot=document.getElementById(name==='chest'?'shot-chest':'shot-side');
  if(slot) slot.textContent=saved;
}
async function captureStill(){
  const t=Date.now();
  const chest=document.getElementById('chestStill');
  const wrist=document.getElementById('wristStill');
  const mark=document.getElementById('chestMark');
  if(mark) mark.hidden=true;
  chest.src='/snapshot/chest.jpg?t='+t;
  wrist.src='/snapshot/d435i.jpg?t='+t;
  document.getElementById('stills').classList.add('show');
  document.getElementById('stills').scrollIntoView({block:'nearest'});
}
async function pickWristYellow(event){
  const p=imageUv(event);
  if(p.u<0||p.v<0||p.u>=p.nw||p.v>=p.nh) return;
  const res=await fetch('/wrist_yellow_target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({u:p.u,v:p.v})});
  const data=await res.json().catch(()=>({}));
  const slot=document.getElementById('shot-side');
  if(slot) slot.textContent=data.ok?(data.message||'已点选'):(data.error||'点选失败');
  refresh();
}
async function clearWristYellow(event){
  event.preventDefault();
  const res=await fetch('/wrist_yellow_target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clear:true})});
  const data=await res.json().catch(()=>({}));
  const slot=document.getElementById('shot-side');
  if(slot) slot.textContent=data.ok?'已取消黄色点选':(data.error||'取消失败');
  refresh();
}
async function setTip(event){
  const p=imageUv(event);
  if(p.u<0||p.v<0||p.u>=p.nw||p.v>=p.nh) return;
  markStill(p.img,p.u,p.v);
  await fetch('/tip_seed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({u:p.u,v:p.v})});
  refresh();
}
async function clearTip(event){
  event.preventDefault();
  const mark=document.getElementById('chestMark');
  if(mark) mark.hidden=true;
  await fetch('/tip_seed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clear:true})});
  refresh();
}
let sequenceSteps=[];
let sequenceMode='hybrid';
let sequenceTouched=false;
let dragIndex=-1;
const SEQUENCE_LABELS={button_1:'Button 1',button_2:'Button 2',button_3:'Button 3'};
function renderSequence(){
  const list=document.getElementById('sequenceList');
  const status=document.getElementById('sequence-status');
  const modeLabel=sequenceMode==='direct_pbvs'?'Direct PBVS':'Hybrid';
  const names=sequenceSteps.length?sequenceSteps.map(name=>SEQUENCE_LABELS[name]||name).join(' → '):'空';
  if(status) status.textContent=modeLabel+' · '+names;
  if(!list) return;
  list.innerHTML=sequenceSteps.map((name,index)=>`
    <li draggable="true" data-index="${index}">
      <span>${index+1}. ${SEQUENCE_LABELS[name]||name}</span>
      <button type="button" onclick="removeSequence(${index})">×</button>
    </li>`).join('');
  list.querySelectorAll('li').forEach(item=>{
    item.addEventListener('dragstart',()=>{ dragIndex=index; item.classList.add('drag'); });
    item.addEventListener('dragend',()=>{ item.classList.remove('drag'); });
    item.addEventListener('dragover',event=>event.preventDefault());
    item.addEventListener('drop',event=>{
      event.preventDefault();
      const from=dragIndex;
      const to=Number(item.dataset.index);
      if(Number.isNaN(from)||Number.isNaN(to)||from===to) return;
      const next=sequenceSteps.slice();
      const [moved]=next.splice(from,1);
      next.splice(to,0,moved);
      sequenceSteps=next;
      sequenceTouched=true;
      renderSequence();
      saveSequence();
    });
  });
}
async function saveSequence(){
  await fetch('/sequence',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({steps:sequenceSteps})});
}
function addSequence(name){
  sequenceSteps.push(name);
  sequenceTouched=true;
  renderSequence();
  saveSequence();
}
function removeSequence(index){
  sequenceSteps.splice(index,1);
  sequenceTouched=true;
  renderSequence();
  saveSequence();
}
function clearSequence(){
  sequenceSteps=[];
  sequenceTouched=true;
  renderSequence();
  saveSequence();
}
async function setTrajectoryRecord(enabled){
  const label=document.getElementById('trajectoryRecordLabel');
  if(label) label.textContent=enabled?'轨迹开':'轨迹关';
  await fetch('/trajectories/enabled',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})});
  refresh();
}
async function setRecord(enabled){
  const label=document.getElementById('recordLabel');
  if(label) label.textContent = enabled ? '录制开' : '录制关';
  await fetch('/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})});
  refresh();
}
let biasFilled=false;
let biasStatusTimer=0;
function clearBiasStatus(){
  if(biasStatusTimer){ clearTimeout(biasStatusTimer); biasStatusTimer=0; }
  const status=document.getElementById('bias-status');
  if(status) status.textContent='';
}
function fillEyeBias(state){
  const bias=state.eye_to_hand_bias_m;
  if(!bias || biasFilled) return;
  const ids=['bias-x','bias-y','bias-z'];
  if(ids.some(id=>document.activeElement && document.activeElement.id===id)) return;
  document.getElementById('bias-x').value=bias.x;
  document.getElementById('bias-y').value=bias.y;
  document.getElementById('bias-z').value=bias.z;
  biasFilled=true;
}
async function submitEyeBias(ev){
  ev.preventDefault();
  const status=document.getElementById('bias-status');
  const body={
    x_m:Number(document.getElementById('bias-x').value),
    y_m:Number(document.getElementById('bias-y').value),
    z_m:Number(document.getElementById('bias-z').value),
  };
  const res=await fetch('/eye_to_hand_bias',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  });
  const data=await res.json();
  if(!status) return;
  clearBiasStatus();
  if(data.ok){
    status.textContent='已提交';
    biasStatusTimer=setTimeout(clearBiasStatus, 1600);
  }else{
    status.textContent='提交失败：'+(data.error||'');
  }
}
async function send(command, btn){
  if(btn){ btn.classList.add('active'); setTimeout(()=>btn.classList.remove('active'),180); }
  await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command})});
  refresh();
}
async function jog(axis, sign){
  const dxyz=[0,0,0], drpy=[0,0,0];
  const xyzStep=Number(document.getElementById('xyzStep').value||0.002);
  const rpyStep=Number(document.getElementById('rpyStep').value||1);
  const xyzIndex={x:0,y:1,z:2}[axis];
  const rpyIndex={roll:0,pitch:1,yaw:2}[axis];
  if(xyzIndex!==undefined) dxyz[xyzIndex]=sign*xyzStep;
  if(rpyIndex!==undefined) drpy[rpyIndex]=sign*rpyStep;
  const slot=document.getElementById('jog-message');
  const res=await fetch('/manual/jog',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({dxyz_m:dxyz,drpy_deg:drpy})
  });
  const data=await res.json();
  if(slot) slot.textContent=data.message||data.error||(data.ok?'完成':'失败');
  refresh();
}
async function joint(index, sign){
  const step=Number(document.getElementById('rpyStep').value||1);
  const slot=document.getElementById('jog-message');
  const res=await fetch('/manual/joint',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index, delta_deg:sign*step})
  });
  const data=await res.json();
  if(slot) slot.textContent=data.message||data.error||(data.ok?'完成':'失败');
  refresh();
}
let jointHoldDelay=null, jointHoldRepeat=null;
function stopJointHold(){
  if(jointHoldDelay!==null) clearTimeout(jointHoldDelay);
  if(jointHoldRepeat!==null) clearInterval(jointHoldRepeat);
  jointHoldDelay=null;
  jointHoldRepeat=null;
}
function startJointHold(event, btn){
  event.preventDefault();
  stopJointHold();
  const index=Number(btn.dataset.joint);
  const sign=Number(btn.dataset.sign);
  joint(index, sign);
  jointHoldDelay=setTimeout(()=>{
    jointHoldRepeat=setInterval(()=>joint(index, sign), 450);
  }, 420);
}
document.querySelectorAll('[data-joint]').forEach((btn)=>{
  btn.addEventListener('pointerdown', (event)=>startJointHold(event, btn));
  btn.addEventListener('pointerup', stopJointHold);
  btn.addEventListener('pointercancel', stopJointHold);
  btn.addEventListener('pointerleave', stopJointHold);
  btn.addEventListener('contextmenu', (event)=>event.preventDefault());
});
async function elbow(sign){
  const step=Number(document.getElementById('rpyStep').value||1);
  const slot=document.getElementById('jog-message');
  const res=await fetch('/manual/elbow',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({delta_deg:sign*step})
  });
  const data=await res.json();
  if(slot) slot.textContent=data.message||data.error||(data.ok?'完成':'失败');
  refresh();
}
async function loadTrajectories(){
  const select=document.getElementById('trajectorySelect');
  const previous=select?.value||'';
  try{
    const data=await fetch('/trajectories',{cache:'no-store'}).then(r=>r.json());
    const rows=data.trajectories||[];
    if(select){
      select.innerHTML=rows.length ? rows.map(row=>{
        const duration=Number(row.duration_s||0).toFixed(1);
        return `<option value="${row.id}">${row.id} · ${row.target||'-'} · ${duration}s · ${row.samples||0}帧</option>`;
      }).join('') : '<option value="">暂无轨迹</option>';
      if(rows.some(row=>row.id===previous)) select.value=previous;
    }
  }catch(e){
    if(select) select.innerHTML='<option value="">轨迹读取失败</option>';
  }
}
async function playTrajectory(){
  const id=document.getElementById('trajectorySelect')?.value||'';
  const speed=Number(document.getElementById('trajectorySpeed')?.value||2);
  const slot=document.getElementById('trajectory-status');
  if(!id){ if(slot) slot.textContent='请选择轨迹'; return; }
  const res=await fetch('/trajectories/play',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,speed})
  });
  const data=await res.json();
  if(slot) slot.textContent=data.ok?`回放 ${id} ×${speed}`:(data.error||'回放失败');
  refresh();
}
async function deleteTrajectory(){
  const id=document.getElementById('trajectorySelect')?.value||'';
  const slot=document.getElementById('trajectory-status');
  if(!id){ if(slot) slot.textContent='请选择轨迹'; return; }
  if(!confirm('删除轨迹 '+id+'？')) return;
  const res=await fetch('/trajectories/delete',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id})
  });
  const data=await res.json();
  if(slot) slot.textContent=data.ok?`已删除 ${id}`:(data.error||'删除失败');
  loadTrajectories();
  refresh();
}
async function deleteAllTrajectories(){
  const slot=document.getElementById('trajectory-status');
  if(!confirm('删除全部已保存轨迹？正在录制的那条会保留。')) return;
  const res=await fetch('/trajectories/delete_all',{method:'POST'});
  const data=await res.json();
  if(slot){
    slot.textContent=data.ok
      ? (data.kept ? `已删除 ${data.deleted} 条，录制中的已保留` : `已删除 ${data.deleted} 条轨迹`)
      : (data.error||'删除失败');
  }
  loadTrajectories();
  refresh();
}
function num(v, d=3){ return (v===null||v===undefined||Number.isNaN(v)) ? '—' : Number(v).toFixed(d); }
async function refresh(){
  try{
    const s = await fetch('/state',{cache:'no-store'}).then(r=>r.json());
    const selection=window.getSelection ? window.getSelection() : null;
    const panel=document.querySelector('.panel');
    const selectingPanelText=!!(
      selection && !selection.isCollapsed && panel &&
      ((selection.anchorNode&&panel.contains(selection.anchorNode)) ||
       (selection.focusNode&&panel.contains(selection.focusNode)))
    );
    if(selectingPanelText) return;
    const cams = s.cameras || {chest:s.camera};
    document.getElementById('camMeta').textContent = Object.values(cams).map(c=>
      (c.label||c.name||'')+' '+(c.serial||'')+' '+(c.state||'')
    ).join('  |  ');
    const chest = cams.chest || {};
    const d435i = cams.d435i || {};
    const t1 = document.getElementById('title-chest');
    const t2 = document.getElementById('title-d435i');
    if(t1) t1.textContent = (chest.label||'D435 胸口')+' '+(chest.serial||'');
    if(t2) t2.textContent = (d435i.label||'D435I 左手侧视')+' '+(d435i.serial||'');
    const ibvs = s.ibvs || {};
    document.getElementById('pills').innerHTML = [
      `target:${s.target||'-'}`,
      `phase:${ibvs.phase||'-'}`,
      `panel:${ibvs.panel_visible?'ok':(ibvs.panel_partial?'partial':'lost')}`,
      `wristView:eye-in-hand`,
      `wristContact:${ibvs.wrist_contact?'yes':'no'}`,
      `tip:${ibvs.tip_source||(ibvs.tip_visible?'manual':'fk/lost')}`,
      `fkTip:${ibvs.virtual_tip_xyz_m?'ok':'wait'}`,
      `tipTrack:${ibvs.tip_tracker?.mode||'auto'}:${ibvs.tip_tracker?.reason||'-'}`,
      `depth:${ibvs.depth_available?'ok':'lost'}`,
      `cycle:${s.cycle?'on':'off'}`,
      `session:${s.running?'START':'STOP'}`,
      `arm:${(s.arm&&s.arm.phase)||'idle'}`,
      `ibvs:${s.arm&&s.arm.ibvs_enabled?'on':'off'}`,
      `stage:${(s.arm&&s.arm.control_stage)||'IDLE'}`,
      `mode:${s.arm?.servo_mode==='direct_pbvs'?'PBVS-DIRECT':(s.arm?.servo_mode==='hybrid'?'HYBRID-VS':(s.arm?.servo_mode==='direct_ee'?'DIRECT-EE':'DUAL-IBVS'))}`,
      `link:${s.arm?.motion_backend==='service'?'service':'arm_sdk'}`,
      `gravity:${s.arm&&s.arm.gravity?'on':'off'}`,
      `sdk:${s.arm&&s.arm.sdk_active?'hold':'released'}`,
      `rec:${s.record?.recording?'REC':(s.record?.enabled?'armed':'off')}`,
      `traj:${s.trajectory?.recording?'REC':(s.trajectory?.enabled?'armed':'off')}`,
      `seq:${(s.arm?.button_sequence||[]).length}`,
      `eih:${s.arm?.pbvs_source||'-'}`,
      `joints:${s.upper_body?.ok?'ok':'wait'}`,
    ].map(t=>`<span class="pill">${t}</span>`).join('');
    const sequenceStage=s.arm?.sequence_stage||'';
    const detail=(s.arm?.message||ibvs.message||s.message||'');
    document.getElementById('message').textContent=(sequenceStage && detail && !String(detail).startsWith(sequenceStage))
      ? (sequenceStage+' · '+detail)
      : (detail||sequenceStage);
    document.getElementById('ibvs-phase').textContent = ibvs.phase||'—';
    document.getElementById('ibvs-uv').textContent = ibvs.target_uv ? ibvs.target_uv.join(', ') : '—';
    document.getElementById('ibvs-tip-uv').textContent = ibvs.tip_uv ? ibvs.tip_uv.join(', ') : '—';
    document.getElementById('ibvs-tip-source').textContent = ibvs.tip_source || '—';
    document.getElementById('ibvs-xyz').textContent = ibvs.target_xyz_m ? ibvs.target_xyz_m.join(', ') : '—';
    document.getElementById('ibvs-tip-xyz').textContent = (ibvs.virtual_tip_xyz_m||ibvs.tip_xyz_m) ? (ibvs.tip_xyz_m||ibvs.virtual_tip_xyz_m).join(', ') : '—';
    document.getElementById('ibvs-pbvs-error').textContent = ibvs.pbvs_error_cam_m ? ibvs.pbvs_error_cam_m.join(', ') : '—';
    document.getElementById('ibvs-pbvs-error-torso').textContent = ibvs.pbvs_error_torso_m ? ibvs.pbvs_error_torso_m.join(', ') : '—';
    document.getElementById('ibvs-pbvs-source').textContent = ibvs.pbvs_tip_source || '—';
    document.getElementById('ibvs-dxyz').textContent = [ibvs.dx,ibvs.dy,ibvs.dz].map(v=>num(v,4)).join('  ');
    document.getElementById('ibvs-drpy').textContent = [ibvs.droll,ibvs.dpitch,ibvs.dyaw].map(v=>num(v,2)).join('  ');
    document.getElementById('ibvs-handeye').textContent = (ibvs.handeye&&ibvs.handeye.file) ? ibvs.handeye.file : 'missing';
    const trajToggle=document.getElementById('trajectoryToggle');
    if(trajToggle && document.activeElement!==trajToggle){
      trajToggle.checked=!!(s.trajectory&&s.trajectory.enabled);
      const trajLabel=document.getElementById('trajectoryRecordLabel');
      if(trajLabel) trajLabel.textContent=trajToggle.checked?'轨迹开':'轨迹关';
    }
    sequenceMode=s.arm?.sequence_mode||'hybrid';
    if(!sequenceTouched && Array.isArray(s.arm?.button_sequence)){
      sequenceSteps=s.arm.button_sequence.slice();
      renderSequence();
    }else{
      const sequenceStatus=document.getElementById('sequence-status');
      const modeLabel=sequenceMode==='direct_pbvs'?'Direct PBVS':'Hybrid';
      const names=sequenceSteps.length?sequenceSteps.map(name=>SEQUENCE_LABELS[name]||name).join(' → '):'空';
      if(sequenceStatus) sequenceStatus.textContent=modeLabel+' · '+names;
    }
    if(sequenceStage){
      const sequenceStatus=document.getElementById('sequence-status');
      if(sequenceStatus) sequenceStatus.textContent=sequenceStage;
    }
    const screen=s.screen||{};
    const screenImg=document.getElementById('screenShot');
    const screenStatus=document.getElementById('screenStatus');
    if(screen.ready && screenImg){
      const next='/screen/latest.jpg?t='+screen.at;
      if(screenImg.getAttribute('data-at')!==String(screen.at)){
        screenImg.src=next;
        screenImg.hidden=false;
        screenImg.setAttribute('data-at',String(screen.at));
      }
      if(screenStatus) screenStatus.textContent='ready';
    }else if(screenStatus && !screen.ready){
      screenStatus.textContent='no screen yet';
    }
    const arm=s.arm||{};
    document.getElementById('manual-state').textContent=arm.manual_enabled?'细调已启用':'先点 Pause';
    document.getElementById('ee-xyz').textContent=arm.ee_xyz_m ? arm.ee_xyz_m.map(v=>num(v,4)).join('  ') : '—';
    document.getElementById('ee-rpy').textContent=arm.ee_rpy_deg ? arm.ee_rpy_deg.map(v=>num(v,2)).join('  ') : '—';
    const targetTorso=ibvs.target_torso_m||null;
    const rightEe=arm.right_ee_xyz_m||null;
    const worldError=(targetTorso&&rightEe) ? targetTorso.map((v,i)=>Number(v)-Number(rightEe[i])) : null;
    document.getElementById('world-target-xyz').textContent=targetTorso ? targetTorso.map(v=>num(v,4)).join('  ') : '—';
    document.getElementById('world-target-source').textContent=ibvs.target_source||'—';
    document.getElementById('world-ee-xyz').textContent=rightEe ? rightEe.map(v=>num(v,4)).join('  ') : '—';
    document.getElementById('world-error-xyz').textContent=worldError ? worldError.map(v=>num(v,4)).join('  ') : '—';
    document.getElementById('world-ee-rpy').textContent=arm.right_ee_rpy_deg ? arm.right_ee_rpy_deg.map(v=>num(v,2)).join('  ') : '—';
    fillEyeBias(s);
    const trajectory=s.trajectory||{};
    document.getElementById('trajectory-status').textContent=arm.playback
      ? `回放 ${arm.playback_id||''} ×${num(arm.playback_speed,1)}`
      : (trajectory.recording ? `录制中 ${trajectory.samples||0}帧` : (trajectory.id?`已保存 ${trajectory.id}`:'未录制'));
    const rows = (s.upper_body?.joints||[]).map(j=>
      `<tr><td>${j.label} <span style="color:#666">${j.index}</span></td><td>${num(j.q,4)}</td><td>${num(j.q_deg,1)}</td><td>${num(j.dq,3)}</td></tr>`
    ).join('');
    document.getElementById('joints').innerHTML = rows || '<tr><td colspan="4">无关节数据</td></tr>';
    const slim = Object.assign({}, s);
    delete slim.upper_body;
    document.getElementById('state').textContent = JSON.stringify({
      target:s.target, cycle:s.cycle, running:s.running, ibvs:s.ibvs, cameras:s.cameras,
      joints_source:s.upper_body?.source, joints_age_s:s.upper_body?.age_s,
      joints_message:s.upper_body?.message, last_command:s.last_command
    }, null, 2);
    document.querySelectorAll('button[data-cmd]').forEach(b=>b.classList.remove('active'));
    const tbtn = document.querySelector(`button[data-cmd="${s.target}"]`);
    if(tbtn) tbtn.classList.add('active');
    if(s.cycle) document.querySelector('button[data-cmd="cycle"]')?.classList.add('active');
    if(s.arm?.scanning) document.querySelector('button[data-cmd="scan"]')?.classList.add('active');
    if(s.arm?.leveling_right || s.arm?.phase==='leveling_right') document.querySelector('button[data-cmd="level_right"]')?.classList.add('active');
    if(s.running && s.arm?.sequence_run) document.querySelector('button[data-cmd="start_sequence"]')?.classList.add('active');
    else if(s.running) document.querySelector('button[data-cmd="start"]')?.classList.add('active');
    if(s.arm?.manual_enabled) document.querySelector('button[data-cmd="pause"]')?.classList.add('active');
    if(s.arm?.phase==='estop') document.querySelector('button[data-cmd="estop"]')?.classList.add('active');
    if(s.arm?.servo_mode==='direct_pbvs') document.querySelector('button[data-cmd="mode_direct_pbvs"]')?.classList.add('active');
    if(s.arm?.servo_mode==='hybrid') document.querySelector('button[data-cmd="mode_hybrid"]')?.classList.add('active');
    if(s.arm?.sequence_mode==='direct_pbvs') document.querySelector('button[data-cmd="sequence_direct_pbvs"]')?.classList.add('active');
    else document.querySelector('button[data-cmd="sequence_hybrid"]')?.classList.add('active');
    if(s.arm?.motion_backend==='service') document.querySelector('button[data-cmd="backend_service"]')?.classList.add('active');
    else document.querySelector('button[data-cmd="backend_arm_sdk"]')?.classList.add('active');
    const rec=document.getElementById('recordToggle');
    const recLabel=document.getElementById('recordLabel');
    if(rec && document.activeElement!==rec) rec.checked=!!(s.record&&s.record.enabled);
    if(recLabel) recLabel.textContent=(s.record&&s.record.recording)?'录制中':((s.record&&s.record.enabled)?'录制开':'录制关');
  }catch(e){ document.getElementById('state').textContent = String(e); }
}
setInterval(refresh, 400); refresh();
setInterval(refreshDepth, 800);
setInterval(loadTrajectories, 3000); loadTrajectories();
</script>
</body>
</html>
"""
