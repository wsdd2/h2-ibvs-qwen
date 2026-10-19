#!/usr/bin/env python3
"""Grab one RealSense color stream and write /tmp/ibvs_cam_<name>.jpg for Flask."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.camera.sidecar_env import apply_conda_path_guard

apply_conda_path_guard()

import numpy as np

from ibvs_pad_touch.camera.devices import (
    claimed_serials,
    format_device_line,
    list_realsense,
    pick_serial,
    write_claim,
    write_seen,
)
from ibvs_pad_touch.camera.image_io import write_bgr, write_depth
from ibvs_pad_touch.camera.realsense_bind import usable_realsense


def _open_profiles(fps: int, want_depth: bool) -> list[tuple[int, bool]]:
    """Keep depth on. Distance is read from the aligned depth frame.

    640x480@15 with depth can miss the first frameset on the second camera.
    Retry at 6 fps with depth. Do not drop the depth stream.
    """
    ordered: list[tuple[int, bool]] = []
    for item in ((int(fps), bool(want_depth)), (6, bool(want_depth))):
        if item[0] <= 0 or item in ordered:
            continue
        ordered.append(item)
    return ordered


def _start(rs, serial: str, width: int, height: int, fps: int, enable_depth: bool = False):
    pipe = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    if enable_depth:
        cfg.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
    profile = pipe.start(cfg)
    return pipe, profile


def _color_sensor(rs, profile):
    device = profile.get_device()
    fallback = None
    for sensor in device.query_sensors():
        try:
            if not sensor.supports(rs.option.exposure):
                continue
            if fallback is None:
                fallback = sensor
            name = ""
            if sensor.supports(rs.camera_info.name):
                name = str(sensor.get_info(rs.camera_info.name)).lower()
            if "rgb" in name or "color" in name:
                return sensor
        except Exception:
            continue
    return fallback


def _prepare_auto_exposure(rs, sensor) -> None:
    if sensor is None:
        return
    if sensor.supports(rs.option.auto_exposure_priority):
        sensor.set_option(rs.option.auto_exposure_priority, 0)
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)


def _lock_current_exposure(rs, sensor) -> tuple[float | None, float | None]:
    """Freeze the settled RGB exposure so a nearby bright arm cannot darken it."""
    if sensor is None or not sensor.supports(rs.option.exposure):
        return None, None
    exposure = float(sensor.get_option(rs.option.exposure))
    gain = (
        float(sensor.get_option(rs.option.gain))
        if sensor.supports(rs.option.gain)
        else None
    )
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, exposure)
    if gain is not None:
        sensor.set_option(rs.option.gain, gain)
    return exposure, gain


def _set_manual_exposure(rs, sensor, exposure: float, gain: float) -> tuple[float, float | None]:
    if sensor is None or not sensor.supports(rs.option.exposure):
        raise RuntimeError("RGB sensor has no exposure option")
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, float(exposure))
    actual_gain = None
    if gain >= 0.0 and sensor.supports(rs.option.gain):
        sensor.set_option(rs.option.gain, float(gain))
        actual_gain = float(sensor.get_option(rs.option.gain))
    return float(sensor.get_option(rs.option.exposure)), actual_gain


def _write_frame(bgr: np.ndarray, out: Path) -> Path:
    return write_bgr(out, bgr)


def _read_frames(
    pipe,
    align,
    depth_out: Path | None,
    depth_scale: float | None,
    out: Path,
    args,
    serial: str,
    rs,
    sensor,
    manual_exposure: bool,
) -> bool:
    """Read until the stream dies. Return False when the first color frame never arrives."""
    frames = 0
    empty = 0
    exposure_ready = False
    exposure_locked = False
    deadline = time.monotonic() + 8.0
    while True:
        try:
            pack = pipe.wait_for_frames(timeout_ms=2000)
        except Exception as exc:
            empty += 1
            if frames == 0 and (empty == 1 or empty % 3 == 0):
                print(f"[ibvs-cam] {args.name} no frame yet: {exc}", flush=True)
            if frames == 0 and time.monotonic() > deadline:
                return False
            continue
        if align is not None:
            try:
                pack = align.process(pack)
            except Exception as exc:
                if frames == 0 and time.monotonic() > deadline:
                    print(f"[ibvs-cam] {args.name} align failed: {exc}", flush=True)
                    return False
                continue
        color = pack.get_color_frame()
        if color is None:
            if frames == 0 and time.monotonic() > deadline:
                print(f"[ibvs-cam] {args.name} color frame missing", flush=True)
                return False
            continue
        if depth_out is not None:
            depth_frame = None if depth_scale is None else pack.get_depth_frame()
            if depth_frame is None:
                if frames == 0 and time.monotonic() > deadline:
                    print(f"[ibvs-cam] {args.name} depth frame missing", flush=True)
                    return False
                continue
            write_depth(
                depth_out,
                np.asanyarray(depth_frame.get_data()),
                depth_scale,
            )
        written = _write_frame(np.asanyarray(color.get_data()), out)
        frames += 1
        if frames == 1:
            print(f"[ibvs-cam] {args.name} first frame {written} serial={serial}", flush=True)
            if manual_exposure:
                try:
                    exposure, gain = _set_manual_exposure(
                        rs,
                        sensor,
                        float(args.manual_exposure),
                        float(args.manual_gain),
                    )
                    print(
                        f"[ibvs-cam] {args.name} manual exposure={exposure} gain={gain}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[ibvs-cam] {args.name} manual exposure failed: {exc}", flush=True)
            else:
                try:
                    _prepare_auto_exposure(rs, sensor)
                    exposure_ready = True
                except Exception as exc:
                    print(f"[ibvs-cam] {args.name} auto exposure setup: {exc}", flush=True)
        if (
            exposure_ready
            and not exposure_locked
            and not manual_exposure
            and int(args.lock_exposure_after) > 0
            and frames >= int(args.lock_exposure_after)
        ):
            try:
                exposure, gain = _lock_current_exposure(rs, sensor)
                exposure_locked = True
                print(
                    f"[ibvs-cam] {args.name} exposure locked exposure={exposure} gain={gain}",
                    flush=True,
                )
            except Exception as exc:
                exposure_locked = True
                print(f"[ibvs-cam] {args.name} exposure lock failed: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--serial", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--depth-out", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--lock-exposure-after",
        type=int,
        default=45,
        help="let RGB auto exposure settle N frames, then freeze it; 0 keeps auto",
    )
    parser.add_argument("--manual-exposure", type=float, default=0.0)
    parser.add_argument("--manual-gain", type=float, default=-1.0)
    args = parser.parse_args()
    out = Path(args.out or f"/tmp/ibvs_cam_{args.name}.jpg")
    depth_out = Path(args.depth_out) if args.depth_out else None
    rs = usable_realsense()
    if rs is None:
        print(f"[ibvs-cam] {args.name} no rs.pipeline in {sys.executable}", flush=True)
        sys.exit(2)
    print(f"[ibvs-cam] {args.name} prefer={args.serial or '-'} py={sys.executable} bind={getattr(rs, '__file__', '')}", flush=True)
    while True:
        devices = list_realsense(rs)
        write_seen(devices)
        line = format_device_line(devices)
        serial = pick_serial(args.serial, devices, claimed_serials(except_name=args.name))
        if not serial:
            wanted = f" for {args.serial}" if args.serial else " unused"
            print(f"[ibvs-cam] {args.name} waiting{wanted}; usb={line}", flush=True)
            time.sleep(3.0)
            continue
        print(f"[ibvs-cam] {args.name} usb={line}", flush=True)
        if args.serial and serial != args.serial:
            print(f"[ibvs-cam] {args.name} prefer {args.serial} missing; using {serial}", flush=True)
        opened = False
        for fps, enable_depth in _open_profiles(int(args.fps), depth_out is not None):
            try:
                pipe, profile = _start(
                    rs,
                    serial,
                    args.width,
                    args.height,
                    fps,
                    enable_depth=enable_depth,
                )
            except Exception as exc:
                print(
                    f"[ibvs-cam] {args.name} start {serial} @{fps} depth={int(enable_depth)} fail: {exc}",
                    flush=True,
                )
                continue
            write_claim(args.name, serial)
            sensor = _color_sensor(rs, profile)
            align = rs.align(rs.stream.color) if enable_depth else None
            depth_scale = None
            if enable_depth:
                try:
                    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
                    print(
                        f"[ibvs-cam] {args.name} aligned depth scale={depth_scale}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[ibvs-cam] {args.name} depth setup failed: {exc}", flush=True)
            manual_exposure = float(args.manual_exposure) > 0.0
            print(
                f"[ibvs-cam] {args.name} streaming {args.width}x{args.height}@{fps} depth={int(enable_depth)}",
                flush=True,
            )
            try:
                opened = _read_frames(
                    pipe,
                    align,
                    depth_out if enable_depth else None,
                    depth_scale,
                    out,
                    args,
                    serial,
                    rs,
                    sensor,
                    manual_exposure,
                )
            except Exception as exc:
                print(f"[ibvs-cam] {args.name} stream died: {exc}", flush=True)
                opened = False
            finally:
                try:
                    pipe.stop()
                except Exception:
                    pass
            if opened:
                break
            print(
                f"[ibvs-cam] {args.name} no depth frame @{fps}; retrying with depth still on",
                flush=True,
            )
            time.sleep(0.5)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
