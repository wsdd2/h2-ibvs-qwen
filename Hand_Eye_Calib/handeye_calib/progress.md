# Hand-Eye Calibration Progress

## Environment

- Local development environment: WSL2 Ubuntu, project mounted at `/mnt/e/MscapeTech`.
- Remote robot compute: Unitree Orin Ubuntu 20.04 / Jetson L4T 35.3.1 / arm64.
- Remote project path: `~/MscapeTech`.
- Python environment on remote: `~/MscapeTech/.venv-calib-py38`.
- Main working directory on remote: `~/MscapeTech/Hand_Eye_Calib`.
- RealSense / OpenCV / Flask are used for camera preview and calibration.

Common remote setup:

```bash
cd ~/MscapeTech/Hand_Eye_Calib
source ~/MscapeTech/.venv-calib-py38/bin/activate
```

## Camera Devices

`pyrealsense2` currently reports two RealSense devices:

```text
index 0:
  name: Intel RealSense D435
  serial: 349622074791
  physical_port: /sys/devices/platform/3610000.xhci/usb2/2-2/2-2.3/2-2.3:1.0/video4linux/video6
  firmware: 5.15.1.55

index 1:
  name: Intel RealSense D435I
  serial: 254322072703
  physical_port: /sys/devices/platform/3610000.xhci/usb2/2-3/2-3:1.0/video4linux/video0
  firmware: 5.15.1.55
```

`v4l2-ctl --list-devices` mapping:

```text
Intel RealSense D435 on usb-3610000.xhci-2.3:
  /dev/video6
  /dev/video7
  /dev/video8
  /dev/video9
  /dev/video10
  /dev/video11

Intel RealSense D435I on usb-3610000.xhci-3:
  /dev/video0
  /dev/video1
  /dev/video2
  /dev/video3
  /dev/video4
  /dev/video5
```

Observed behavior:

- `serial=254322072703` opens a color stream, but it is the head camera.
- `serial=349622074791` is the target arm/chest-side RealSense candidate.
- `/dev/video4` belongs to the head camera group and was held by `videohub_pc4`.
- `/dev/video10` belongs to the arm/chest camera group and was held by `videohub_pc4_chest`.

## Unitree Video Hub Conflict

Problem:

```text
RuntimeError: xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy
```

Confirmed with:

```bash
sudo lsof /dev/video10
sudo lsof /dev/video4
```

Relevant processes:

```text
/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4
/unitree/module/video_hub_pc4/videohub_pc4_chest /dev/video10
```

This matches Unitree upstream issue:

```text
https://github.com/unitreerobotics/xr_teleoperate/issues/299
```

The service is not systemd-managed. It is controlled by Unitree `master_service`.

Known service files:

```text
/unitree/etc/master_service/service/video_hub_pc4
/unitree/etc/master_service/service/video_hub_pc4_chest.disabled
```

The `video_hub_pc4_chest` file was already renamed to `.disabled` during debugging, but `videohub_pc4_chest` still respawned through Unitree tooling/watchdog. If direct RealSense access is needed, the most reliable temporary workaround is to isolate the binary, then restore it later:

```bash
sudo mv /unitree/module/video_hub_pc4/videohub_pc4_chest \
  /unitree/module/video_hub_pc4/videohub_pc4_chest.disabled

sudo kill -9 <current_videohub_pc4_chest_pid>
sleep 2
sudo lsof /dev/video10
```

Restore after calibration:

```bash
sudo mv /unitree/module/video_hub_pc4/videohub_pc4_chest.disabled \
  /unitree/module/video_hub_pc4/videohub_pc4_chest

sudo mv /unitree/etc/master_service/service/video_hub_pc4_chest.disabled \
  /unitree/etc/master_service/service/video_hub_pc4_chest

sudo /unitree/sbin/mscli startservice video_hub_pc4_chest
```

Do not permanently delete Unitree files. This service likely supports Unitree video / XR / teleoperation.

## Code Changes Made

### `handeye_calib/debug_stream.py`

Added a Flask MJPEG debug server.

Features:

- `/stream`: latest OpenCV debug frame.
- `/state`: JSON state panel.
- `/command`: browser buttons send `save`, `solve`, or `quit`.
- Web UI includes `Save / SPACE`, `Solve / S`, and `Quit / Q` buttons.

### `handeye_calib/chessboard.py`

Added:

- `--stream-debug`
- `--stream-host`
- `--stream-port`
- `--stream-jpeg-quality`
- `--headless`
- `--color-only`
- `--camera-backend {realsense,opencv}`
- `--video-device`
- `--video-index`

Behavior:

- Can run headless over SSH.
- Browser preview shows chessboard overlay.
- Browser `Save / SPACE` saves calibration images.
- Browser `Quit / Q` exits.
- `--camera-backend realsense --cam-serial ...` can open by RealSense serial.
- `--camera-backend opencv --cam-serial ... --video-index ...` can resolve `/dev/v4l/by-id` symlinks.

### `handeye_calib/camera.py`

Added:

- `RealSenseD435i(color_only=True)` support. It opens only color stream and avoids depth stream.
- `OpenCVVideoCamera` for V4L2/OpenCV capture.

### `capture_handeye.py`

Added:

- `--stream-debug`
- `--stream-fk`
- `--stream-port`
- `--fk-network-interface`
- `--fk-target`
- `--color-only`
- `--headless`

It can stream the OpenCV hand-eye capture view and FK status panel. Browser buttons can now trigger `save`, `solve`, and `quit`; when `--stream-fk` is enabled, browser `Save / SPACE` uses the current FK transform for `--hand-frame` instead of requiring keyboard input on the robot.

The FK status panel now reports only the seven right-arm joints under `right_arm_joints`; right leg joints such as `right_hip_*`, `right_knee_*`, and `right_ankle_*` are intentionally excluded.

### `auto_capture_handeye.py`

New automatic eye-in-hand capture script.

Behavior:

- Uses Unitree `rt/arm_sdk` to move the right arm through small relative joint offsets from the current posture.
- Requires `--confirm-robot-motion` before publishing robot commands.
- Starts in browser-controlled idle mode; robot motion only begins after pressing `Start Calib`.
- Idle state now continuously reports right-arm joint values, waist joint values, MotionSwitcher mode, and FK for `--hand-frame`.
- Browser `Switch SDK Mode` calls `MotionSwitcher.ReleaseMode()` without moving the robot.
- Browser `Test Move` runs one visible right-wrist test motion and reports the observed lowstate joint delta.
- Browser `Stop` interrupts the current automatic run, releases `arm_sdk`, and returns to idle without exiting the program.
- During automatic capture, the script now checks lowstate after commanded motion; if the expected joint delta is not observed, it stops the run and does not keep saving samples.
- Waits for each pose to settle, detects the chessboard, and saves the image plus FK-derived `T_base_hand` only when the board is visible.
- Defaults to conservative right-arm offsets; custom offset sequences can be supplied with `--auto-poses-json`.
- Uses the same camera intrinsics and save format as `capture_handeye.py`, so results can be solved later with `solve_offline.py`.

### `calibrate_camera.py`

New camera intrinsics calibration script.

Reads chessboard images, runs `cv2.calibrateCamera`, and writes:

```text
outputs/camera_intrinsics_<timestamp>.json
outputs/camera_intrinsics_<timestamp>_npy/camera_matrix.npy
outputs/camera_intrinsics_<timestamp>_npy/dist_coeffs.npy
```

## Camera Intrinsics Result

Collected images at:

```text
~/MscapeTech/Hand_Eye_Calib/camera_calib_image/
```

Calibration command used:

```bash
python calibrate_camera.py \
  --image-dir camera_calib_image \
  --cols 11 \
  --rows 8 \
  --square-mm 30 \
  --output-dir outputs
```

Result:

```text
[DONE] rms=0.0936px used=10 rejected=0
```

Generated files:

```text
outputs/camera_intrinsics_20260624_141347.json
outputs/camera_intrinsics_20260624_141347_npy/camera_matrix.npy
outputs/camera_intrinsics_20260624_141347_npy/dist_coeffs.npy
```

Print intrinsics:

```bash
cd ~/MscapeTech/Hand_Eye_Calib
source ~/MscapeTech/.venv-calib-py38/bin/activate

python - <<'PY'
import json
import numpy as np
from pathlib import Path

base = Path("outputs/camera_intrinsics_20260624_141347_npy")
camera_matrix = np.load(base / "camera_matrix.npy")
dist_coeffs = np.load(base / "dist_coeffs.npy")

print("camera_matrix:")
print(camera_matrix)
print("\ndist_coeffs:")
print(dist_coeffs.reshape(-1))

report = json.loads(Path("outputs/camera_intrinsics_20260624_141347.json").read_text())
print("\nrms_reprojection_error_px:")
print(report["rms_reprojection_error_px"])
PY
```

## Recommended Next Step

Use the generated intrinsics for hand-eye capture:

```bash
cd ~/MscapeTech/Hand_Eye_Calib
source ~/MscapeTech/.venv-calib-py38/bin/activate

export CYCLONEDDS_HOME=$HOME/cyclonedds/install
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CYCLONEDDS_HOME:$CMAKE_PREFIX_PATH

python capture_handeye.py \
  --mode eye-in-hand \
  --cam-serial 349622074791 \
  --cols 11 \
  --rows 8 \
  --square-mm 30 \
  --hand-frame right_wrist_yaw_link \
  --camera-name right_hand_d435 \
  --camera-mount wrist \
  --camera-matrix-npy outputs/camera_intrinsics_20260624_141347_npy/camera_matrix.npy \
  --dist-coeffs-npy outputs/camera_intrinsics_20260624_141347_npy/dist_coeffs.npy \
  --stream-debug \
  --stream-fk \
  --fk-network-interface eth0 \
  --fk-target right_wrist_yaw_link \
  --headless \
  --color-only
```

If using the arm/chest D435 directly, make sure `videohub_pc4_chest` is not holding `/dev/video10` or the matching RealSense serial.

Open the debug page from another machine and use the browser buttons:

- `Save / SPACE`: save the current sample using the latest FK pose for `right_wrist_yaw_link`.
- `Solve / S`: solve after enough samples have been collected.
- `Quit / Q`: exit the capture script.

Optional automatic robot-motion capture:

```bash
python auto_capture_handeye.py \
  --confirm-robot-motion \
  --network-interface eth0 \
  --mode eye-in-hand \
  --cam-serial 349622074791 \
  --cols 11 \
  --rows 8 \
  --square-mm 30 \
  --hand-frame right_wrist_yaw_link \
  --camera-name right_hand_d435 \
  --camera-mount wrist \
  --camera-matrix-npy outputs/camera_intrinsics_20260624_141347_npy/camera_matrix.npy \
  --dist-coeffs-npy outputs/camera_intrinsics_20260624_141347_npy/dist_coeffs.npy \
  --color-only \
  --return-to-start
```

Open the debug page and first confirm the idle state is showing live right-arm joints and FK. Press `Switch SDK Mode`, then press `Test Move`. Confirm the robot visibly moves and the state panel reports a nonzero observed joint delta. Then confirm the chessboard is visible and press `Start Calib`. The automatic script only saves a sample after the target pose is reached and the chessboard is detected. It skips poses where the board leaves the camera view. Press `Stop` to stop the current automatic run and keep the program alive for another start attempt.

## Useful Commands

List RealSense devices:

```bash
python - <<'PY'
import pyrealsense2 as rs

ctx = rs.context()
for i, dev in enumerate(ctx.query_devices()):
    print("index:", i)
    for info in [
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.physical_port,
        rs.camera_info.product_line,
        rs.camera_info.firmware_version,
    ]:
        if dev.supports(info):
            print(" ", str(info), "=", dev.get_info(info))
    print()
PY
```

List V4L2 devices:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
sudo lsof /dev/video10
sudo lsof /dev/video4
```

Run chessboard capture at 1280x720@30:

```bash
python handeye_calib/chessboard.py \
  --camera-backend realsense \
  --cam-serial 349622074791 \
  --task camera_calib \
  --cols 11 \
  --rows 8 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --color-only \
  --stream-debug \
  --stream-port 8080 \
  --headless
```

