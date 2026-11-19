#!/usr/bin/env python3
"""Drive the H2 fingertip from the operator's index finger.

The desktop D405 faces the operator. MediaPipe tracks the hand, and the index
tip — not the shoulder or elbow — moves the robot fingertip at one tenth of
the motion. Forward comes from the D405 depth at that fingertip: a smaller
range means the hand moved toward the camera, which is robot +X toward the
panel. Space engages and releases. The robot web page must already be Paused,
because /manual/jog only runs in that mode.

Run this with the Windows Python. The D405 is on the Windows USB bus.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.teleop.hand_map import TipSample, TipTeleop

WRIST = 0
INDEX_MCP = 5
INDEX_TIP = 8
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def model_path() -> Path:
    path = Path.home() / ".cache" / "ibvs" / "hand_landmarker.task"
    if path.is_file() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载手部模型到 {path}", flush=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """MediaPipe hand landmarker. Returns the index tip in pixels."""

    def __init__(self, task_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._stamp_ms = 0

    def read(self, frame_bgr) -> tuple[TipSample | None, list[tuple[int, int]]]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._stamp_ms += 33
        result = self._landmarker.detect_for_video(image, self._stamp_ms)
        hands = result.hand_landmarks or []
        if not hands:
            return None, []
        height, width = frame_bgr.shape[:2]
        points = [
            (int(point.x * width), int(point.y * height))
            for point in hands[0]
        ]
        if len(points) <= INDEX_TIP:
            return None, points
        wrist = points[WRIST]
        knuckle = points[INDEX_MCP]
        tip = points[INDEX_TIP]
        span = ((knuckle[0] - wrist[0]) ** 2 + (knuckle[1] - wrist[1]) ** 2) ** 0.5
        return TipSample(float(tip[0]), float(tip[1]), float(span)), points


def draw_hand(frame, points: list[tuple[int, int]], engaged: bool) -> None:
    import cv2

    color = (40, 220, 40) if engaged else (180, 180, 180)
    for start, end in HAND_EDGES:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, 2)
    for point in points:
        cv2.circle(frame, point, 3, color, -1)
    if len(points) > INDEX_TIP:
        cv2.circle(frame, points[INDEX_TIP], 8, (0, 180, 255), 2)


class LegacyHandTracker:
    """Older MediaPipe solutions API. Same tip sample, no downloaded model."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def read(self, frame_bgr) -> tuple[TipSample | None, list[tuple[int, int]]]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        landmarks = result.multi_hand_landmarks or []
        if not landmarks:
            return None, []
        height, width = frame_bgr.shape[:2]
        points = [
            (int(point.x * width), int(point.y * height))
            for point in landmarks[0].landmark
        ]
        if len(points) <= INDEX_TIP:
            return None, points
        wrist = points[WRIST]
        knuckle = points[INDEX_MCP]
        tip = points[INDEX_TIP]
        span = ((knuckle[0] - wrist[0]) ** 2 + (knuckle[1] - wrist[1]) ** 2) ** 0.5
        return TipSample(float(tip[0]), float(tip[1]), float(span)), points


def open_tracker() -> HandTracker | LegacyHandTracker:
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "缺少 mediapipe。摄像头在 Windows 上，用 Windows Python 安装后再跑：\n"
            "/mnt/e/miniconda3/python.exe -m pip install mediapipe"
        ) from exc
    try:
        return HandTracker(model_path())
    except Exception:
        return LegacyHandTracker()


class JogClient:
    def __init__(self, url: str, dry_run: bool, log_path: Path) -> None:
        self.url = url.rstrip("/") + "/manual/jog"
        self.dry_run = dry_run
        self.log_path = log_path
        self._lock = threading.Lock()
        self.busy = False
        self.last_error = ""

    def submit(self, delta: tuple[float, float, float]) -> bool:
        with self._lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._send, args=(delta,), daemon=True).start()
        return True

    def _send(self, delta: tuple[float, float, float]) -> None:
        record = {"t": time.time(), "dxyz_m": [round(value, 5) for value in delta]}
        try:
            if self.dry_run:
                self.last_error = ""
                print(f"干跑 { _fmt(delta) }", flush=True)
            else:
                body = json.dumps({"dxyz_m": list(delta), "drpy_deg": [0, 0, 0]}).encode()
                request = urllib.request.Request(
                    self.url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=1.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not payload.get("ok"):
                    self.last_error = str(payload.get("error") or payload.get("message") or "细调被拒绝")
                else:
                    self.last_error = ""
            record["ok"] = self.last_error == ""
            record["error"] = self.last_error
            self._log(record)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"HTTP {exc.code} {detail[:120]}"
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            with self._lock:
                self.busy = False


    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fmt(delta: tuple[float, float, float]) -> str:
    return " ".join(f"{axis} {value * 1000:+.1f}mm" for axis, value in zip("XYZ", delta))


def depth_at(depth_m, u: float, v: float, radius: int = 4) -> float | None:
    """Median D405 range around the fingertip. Holes and far background are ignored."""
    if depth_m is None:
        return None
    import numpy as np

    height, width = depth_m.shape[:2]
    x0 = max(0, int(u) - radius)
    x1 = min(width, int(u) + radius + 1)
    y0 = max(0, int(v) - radius)
    y1 = min(height, int(v) + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[(patch > 0.07) & (patch < 0.50)]
    if valid.size < 6:
        return None
    return float(np.median(valid))


class D405:
    """Color plus depth from the local RealSense D405."""

    def __init__(self, serial: str = "") -> None:
        import numpy as np
        import pyrealsense2 as rs

        self._np = np
        self._rs = rs
        chosen, name = _find_d405(rs, serial)
        self.serial = chosen
        self.name = name
        self.pipe = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        last_error: Exception | None = None
        started = False
        for width, height, fps in ((640, 480, 30), (848, 480, 30), (1280, 720, 15)):
            config = rs.config()
            config.enable_device(chosen)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = self.pipe.start(config)
                started = True
                self.size = (width, height, fps)
                sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = float(sensor.get_depth_scale())
                break
            except Exception as exc:
                last_error = exc
        if not started:
            raise SystemExit(f"D405 {chosen} 开不起来：{last_error}")
        print(f"D405 {name} {chosen} {self.size[0]}x{self.size[1]} @{self.size[2]}", flush=True)

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames(timeout_ms=1000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color:
            return None, None
        image = self._np.asanyarray(color.get_data())
        depth_m = None
        if depth:
            depth_m = self._np.asanyarray(depth.get_data()).astype(self._np.float32) * self.depth_scale
        return image, depth_m

    def close(self) -> None:
        try:
            self.pipe.stop()
        except Exception:
            pass


def _find_d405(rs, serial: str) -> tuple[str, str]:
    devices = []
    for device in rs.context().query_devices():
        number = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else ""
        name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else ""
        devices.append((number, name))
    if serial:
        match = next((item for item in devices if item[0] == serial), None)
        if match is None:
            raise SystemExit(f"没有序列号 {serial} 的 RealSense。看到的是：{devices or '无'}")
        return match
    d405 = next((item for item in devices if "405" in item[1]), None)
    if d405 is not None:
        return d405
    if len(devices) == 1:
        return devices[0]
    raise SystemExit(f"没找到 D405。插上相机后再跑。当前设备：{devices or '无'}")


def pause_robot(base: str) -> None:
    body = json.dumps({"command": "pause"}).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/command",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="用本地摄像头的食指遥操 H2 指尖")
    parser.add_argument("--url", default="http://192.168.1.40:8080")
    parser.add_argument("--serial", default="", help="多台 RealSense 时指定 D405 序列号")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--flip-y", action="store_true", help="左右感觉反了时打开")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pause-first", action="store_true")
    parser.add_argument("--log", type=Path, default=ROOT / "outputs" / "teleop" / "hand_teleop.jsonl")
    args = parser.parse_args()

    import cv2

    if args.pause_first and not args.dry_run:
        pause_robot(args.url)
        print("已请求 Pause。", flush=True)
    tracker = open_tracker()
    teleop = TipTeleop(scale=args.scale, flip_y=args.flip_y)
    client = JogClient(args.url, args.dry_run, args.log)
    camera = None
    try:
        camera = D405(args.serial)
    except ImportError as exc:
        raise SystemExit(
            "缺少 pyrealsense2。D405 在这台 Windows 台式机上，用 Windows Python 安装：\n"
            "/mnt/e/miniconda3/python.exe -m pip install pyrealsense2"
        ) from exc
    print("空格：开始/停止跟随。Q：退出。面对 D405，抬手机器人向上，靠近相机机器人朝面板向前。", flush=True)
    print("网页上要先 Pause，细调才会接受。", flush=True)
    window = "hand teleop"
    try:
        while True:
            frame, depth = camera.read()
            if frame is None:
                continue
            sample, points = tracker.read(frame)
            if sample is not None:
                sample = TipSample(sample.u, sample.v, sample.span_px, depth_at(depth, sample.u, sample.v))
            draw_hand(frame, points, teleop.engaged)
            if teleop.engaged and sample is not None and not client.busy:
                command = teleop.offer(sample)
                if command is not None and not client.submit(command):
                    teleop.undo()
            status = "跟随" if teleop.engaged else "空格后跟随"
            if sample is None:
                status += "  没有看到手"
            if client.last_error:
                status += "  " + client.last_error
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if teleop.engaged:
                    teleop.release()
                elif sample is not None:
                    teleop.engage(sample)
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
