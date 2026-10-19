#!/usr/bin/env python3
import time

import pyrealsense2 as rs

ctx = rs.context()
devs = list(ctx.query_devices())
print("devices", len(devs))
for dev in devs:
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(" ", name, serial)
    for sensor in dev.query_sensors():
        print("  sensor", sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else sensor)
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.is_video_stream_profile():
                video = profile.as_video_stream_profile()
                print(f"    color {video.width()}x{video.height()} {video.fps()} {profile.format()}")

serial = "347622074029"
for w, h, fps in ((640, 480, 30), (848, 480, 30), (1280, 720, 15)):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    print("try", w, h, fps)
    try:
        pipeline.start(cfg)
        ok = 0
        for _ in range(8):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                color = frames.get_color_frame()
                if color:
                    ok += 1
            except Exception as exc:
                print("  wait", exc)
        print("  ok_frames", ok)
        pipeline.stop()
    except Exception as exc:
        print("  start FAIL", exc)
    time.sleep(0.5)
