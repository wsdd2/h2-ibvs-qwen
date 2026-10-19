from __future__ import annotations

from typing import Optional

import numpy as np

from ibvs_pad_touch.config import load_camera_config
from ibvs_pad_touch.types import Intrinsics


def intrinsics_from_config(cfg: dict | None = None) -> Intrinsics:
    cam = (cfg or load_camera_config())["camera"]
    k = cam["fallback_intrinsics"]
    return Intrinsics(
        fx=float(k["fx"]),
        fy=float(k["fy"]),
        cx=float(k["cx"]),
        cy=float(k["cy"]),
        width=int(k["width"]),
        height=int(k["height"]),
    )


class D435iCamera:
    """Color + aligned depth. Optional; synthetic / still-image paths do not need this."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or load_camera_config())["camera"]
        self._pipeline = None
        self._align = None
        self.intrinsics = intrinsics_from_config({"camera": self.cfg})

    def start(self) -> Intrinsics:
        from ibvs_pad_touch.camera.realsense_bind import usable_realsense

        rs = usable_realsense()
        if rs is None:
            from ibvs_pad_touch.platform import environment_name

            hint = "need librealsense python (rs.pipeline). pip pyrealsense2 stubs do not work."
            if environment_name() == "wsl":
                hint += " WSL USB passthrough is unreliable; use H2 native Ubuntu."
            raise ImportError(f"D435iCamera: {hint}")

        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        serial = self.cfg.get("serial_number")
        if serial:
            rs_cfg.enable_device(str(serial))
        w, h = self.cfg["color_size"]
        fps = int(self.cfg.get("fps", 30))
        rs_cfg.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, fps)
        dw, dh = self.cfg["depth_size"]
        rs_cfg.enable_stream(rs.stream.depth, int(dw), int(dh), rs.format.z16, fps)
        profile = pipeline.start(rs_cfg)
        align = rs.align(rs.stream.color) if self.cfg.get("align_depth_to_color", True) else None
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        k = color_stream.get_intrinsics()
        self.intrinsics = Intrinsics(fx=k.fx, fy=k.fy, cx=k.ppx, cy=k.ppy, width=k.width, height=k.height)
        self._pipeline = pipeline
        self._align = align
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        return self.intrinsics

    def read(self) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
        if self._pipeline is None:
            raise RuntimeError("D435iCamera.start() was not called")
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self._depth_scale
        return color, depth, self.intrinsics

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
