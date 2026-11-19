# -*- coding: utf-8 -*-
"""Live panel / IBVS overlay for the Flask dashboard. No arm_sdk."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ibvs_pad_touch.camera.image_io import read_depth
from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.features import exposed_panel_depth, sample_depth
from ibvs_pad_touch.perception.pipeline import align_buttons_to_exposed_panel
from ibvs_pad_touch.perception.gripper_tip import point_inside_panel
from ibvs_pad_touch.perception.overlay import draw_localization, draw_pose_buttons, draw_seg_masks
from ibvs_pad_touch.perception.side_contact import _outside_distance, estimate_side_gap
from ibvs_pad_touch.perception.yellow_buttons import (
    YELLOW_TRACK_MISS_FRAMES,
    detect_red_bar,
    match_yellow_blob,
    nearest_long_yellow_bar,
    yellow_blob_at,
    yellow_blobs,
)
from ibvs_pad_touch.perception.yolo_touch import (
    YoloTouchDetector,
    apply_pose_buttons,
    gripper_toward_panel_tip,
    keep_side_masks,
    normalize_button_layout,
    pose_finger_tip,
    pose_tip,
    tip_on_mask,
    without_gripper,
)
from ibvs_pad_touch.perception.tip_tracker import StableTipTracker
from ibvs_pad_touch.servo.controller import IBVSController
from ibvs_pad_touch.servo.dual_view import compose_dual_view_ibvs, servo_depth_m
from ibvs_pad_touch.servo.virtual_tip import (
    choose_control_tip,
    virtual_tip_from_ee,
    virtual_tip_in_front,
    visual_bias_cam_m,
)
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization, Observation


HANDEYE_CANDIDATES = (
    Path("/home/unitree/MscapeTech/Hand_Eye_Calib/artifacts/eye_in_hand/outputs"),
    Path(__file__).resolve().parents[3] / "Hand_Eye_Calib" / "artifacts" / "eye_in_hand" / "outputs",
)
EYE_IN_HAND_PREFERRED = "eye_in_hand_20261029_160039.json"
EYE_TO_HAND_CANDIDATES = (
    Path("/home/unitree/MscapeTech/Hand_Eye_Calib/artifacts/eye_to_hand/outputs"),
    Path(__file__).resolve().parents[3] / "Hand_Eye_Calib" / "artifacts" / "eye_to_hand" / "outputs",
)
EYE_TO_HAND_PREFERRED = "eye_to_hand_20261012_104628.json"
WRIST_SERIAL = "233722073797"
# Kept in telemetry for compatibility with older dashboards. The camera is
# physically mounted on the right wrist; it is not a fixed left-side camera.
LEFT_SERIAL = WRIST_SERIAL
CHEST_SERIAL = "347622074029"
CHEST_DEPTH_PATH = Path("/tmp/ibvs_depth_chest.bin")
SIDE_DEPTH_PATH = Path("/tmp/ibvs_depth_d435i.bin")
CHEST_CAMERA_HEIGHT_M = 0.40
CHEST_CAMERA_FORWARD_M = 0.025


def camera_xyz_to_torso(
    xyz: list[float] | tuple[float, float, float],
    camera_height_m: float = CHEST_CAMERA_HEIGHT_M,
    camera_forward_m: float = CHEST_CAMERA_FORWARD_M,
    torso_T_camera: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Chest camera X right/Y down/Z forward -> torso X forward/Y left/Z up.

    Camera origin is 2.5 cm in front of torso_link and 40 cm above it.
    Optical axis is parallel to the ground, so camera Z maps only onto torso X.
    """
    if torso_T_camera is not None:
        point = np.asarray([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
        mapped = np.asarray(torso_T_camera, dtype=np.float64).reshape(4, 4) @ point
        return float(mapped[0]), float(mapped[1]), float(mapped[2])
    x, y, z = (float(value) for value in xyz)
    return z + float(camera_forward_m), -x, float(camera_height_m) - y


def torso_xyz_to_camera(
    xyz: list[float] | tuple[float, float, float],
    torso_T_camera: np.ndarray,
) -> tuple[float, float, float]:
    point = np.asarray([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    mapped = np.linalg.inv(np.asarray(torso_T_camera, dtype=np.float64).reshape(4, 4)) @ point
    return float(mapped[0]), float(mapped[1]), float(mapped[2])


def discover_eye_to_hand_json() -> Optional[Path]:
    found: list[Path] = []
    for root in EYE_TO_HAND_CANDIDATES:
        preferred = root / EYE_TO_HAND_PREFERRED
        if preferred.is_file():
            return preferred
        if root.is_dir():
            found.extend(root.glob("eye_to_hand_*.json"))
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def load_eye_to_hand_transform(path: Path | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    if path is None or not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = data.get("cam2reference") or data.get("cam2base") or {}
        transform = np.asarray(record.get("transform"), dtype=np.float64).reshape(4, 4)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, {}
    if not np.all(np.isfinite(transform)):
        return None, {}
    reference = str(data.get("reference_frame") or data.get("context", {}).get("base_frame") or "")
    method = str(data.get("calibration_method") or "")
    serial = str(data.get("context", {}).get("camera_info", {}).get("serial") or "")
    if method != "eye_to_hand" or reference != "torso_link" or (serial and serial != CHEST_SERIAL):
        return None, {}
    quality = data.get("quality") or {}
    return transform, {
        "file": str(path),
        "method": method,
        "reference_frame": reference,
        "serial": serial,
        "translation_residual_mean_mm": round(
            1000.0 * float(quality.get("translation_residual_norm_mean_m") or 0.0),
            2,
        ),
        "rotation_error_mean_deg": round(float(quality.get("rotation_error_deg_mean") or 0.0), 3),
        "pass": bool(quality.get("pass")),
    }


def discover_eye_in_hand_json() -> Optional[Path]:
    found: list[Path] = []
    for root in HANDEYE_CANDIDATES:
        preferred = root / EYE_IN_HAND_PREFERRED
        if preferred.is_file():
            return preferred
        if root.is_dir():
            found.extend(path for path in root.glob("eye_in_hand_*.json") if path.is_file())
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def eye_in_hand_npy_dir(path: Path, data: dict[str, Any] | None = None) -> Path | None:
    """Directory saved beside the wrist json, containing T_cam2hand.npy."""
    recorded = ""
    if isinstance(data, dict):
        recorded = str(data.get("npy_dir") or "")
    candidates = []
    if recorded:
        candidates.append(Path(recorded))
    candidates.append(path.with_name(path.stem + "_npy"))
    for candidate in candidates:
        if (candidate / "T_cam2hand.npy").is_file():
            return candidate
    return None


def load_eye_in_hand_transform(path: Path | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """cam2hand: a camera point maps into the right wrist frame. Runtime pose supplies the torso."""
    if path is None or not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = data.get("cam2hand") or {}
        transform = np.asarray(record.get("transform"), dtype=np.float64).reshape(4, 4)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, {}
    method = str(data.get("calibration_method") or "")
    npy_dir = eye_in_hand_npy_dir(path, data if isinstance(data, dict) else None)
    npy_file = None if npy_dir is None else npy_dir / "T_cam2hand.npy"
    if npy_file is not None:
        try:
            stored = np.load(npy_file)
            if stored.shape == (4, 4) and np.all(np.isfinite(stored)):
                transform = np.asarray(stored, dtype=np.float64)
        except (OSError, ValueError):
            npy_file = None
    if (method and method != "eye_in_hand") or not np.all(np.isfinite(transform)):
        return None, {}
    quality = data.get("quality") or {}
    serial = str(data.get("context", {}).get("camera_info", {}).get("serial") or "")
    hand = str(data.get("context", {}).get("hand_frame") or data.get("hand_frame") or "")
    if (serial and serial != WRIST_SERIAL) or (hand and hand != "right_wrist_yaw_link"):
        return None, {}
    return transform, {
        "file": str(path),
        "npy_dir": "" if npy_dir is None else str(npy_dir),
        "npy_file": "" if npy_file is None else str(npy_file),
        "method": method or "eye_in_hand",
        "serial": serial,
        "hand_frame": hand,
        "translation_residual_mean_mm": round(
            1000.0 * float(quality.get("translation_residual_norm_mean_m") or 0.0),
            2,
        ),
        "rotation_error_mean_deg": round(float(quality.get("rotation_error_deg_mean") or 0.0), 3),
        "pass": bool(quality.get("pass")),
    }


def screen_crop(bgr: np.ndarray | None, polygon) -> np.ndarray | None:
    """Keep the YOLO screen mask and black out everything around it."""
    if bgr is None or polygon is None:
        return None
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3 or bgr.size == 0:
        return None
    height, width = bgr.shape[:2]
    clipped = np.column_stack(
        (
            np.clip(np.rint(poly[:, 0]), 0, width - 1),
            np.clip(np.rint(poly[:, 1]), 0, height - 1),
        )
    ).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [clipped.reshape(-1, 1, 2)], 255)
    x, y, w, h = cv2.boundingRect(clipped.reshape(-1, 1, 2))
    if w < 2 or h < 2:
        return None
    crop = bgr[y : y + h, x : x + w].copy()
    crop[mask[y : y + h, x : x + w] == 0] = 0
    return crop


def screen_mask_jpeg(bgr: np.ndarray | None, polygon) -> bytes | None:
    crop = screen_crop(bgr, polygon)
    if crop is None:
        return None
    ok, blob = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return blob.tobytes()


def _screen_polygon(touch) -> list[tuple[float, float]] | None:
    for mask in keep_side_masks([] if touch is None else getattr(touch, "masks", None), gripper_min=0.0):
        if mask.name == "screen" and len(mask.polygon) >= 3:
            return [(float(point[0]), float(point[1])) for point in mask.polygon]
    return None


LAST_TARGET_HOLD_S = 3.0
POSE_BUTTON_NAMES = ("button_1", "button_2", "button_3")
# A 14 px image gap is not a press. The jaw sits beside the panel outline
# while the real gap is still many centimetres along the camera axis.
WRIST_CONTACT_DEPTH_M = 0.02
WRIST_BUTTON_CONF_MIN = 0.3


def _wrist_depth_gap_m(depth_m, tip_uv, corners, touch) -> float | None:
    """Range difference between the jaw and the visible panel face."""
    if depth_m is None or tip_uv is None or corners is None:
        return None
    tip_z = sample_depth(depth_m, float(tip_uv[0]), float(tip_uv[1]), radius=6, percentile=20.0)
    panel_z = exposed_panel_depth(depth_m, corners, _cover_polygons(touch))
    if tip_z is None or panel_z is None:
        return None
    return abs(float(tip_z) - float(panel_z))


def wrist_depth_confirms_contact(pixel_contact: bool, depth_gap_m: float | None) -> bool:
    """Pixel contact is only a press when the jaw and the panel are at the same range."""
    if not pixel_contact or depth_gap_m is None:
        return False
    return float(depth_gap_m) <= WRIST_CONTACT_DEPTH_M


def select_wrist_button(touch, conf_min: float, target_name: str, tip_uv):
    """The wrist only needs a button. The button is already on the panel."""
    if touch is None:
        return None
    points = []
    for name in POSE_BUTTON_NAMES:
        point = touch.point(name) if hasattr(touch, "point") else None
        if point is None or float(point.conf) < float(conf_min):
            continue
        points.append((name, float(point.u), float(point.v)))
    if not points:
        return None
    for item in points:
        if item[0] == target_name:
            return item
    if tip_uv is None:
        return points[0]
    return min(
        points,
        key=lambda item: math.hypot(item[1] - float(tip_uv[0]), item[2] - float(tip_uv[1])),
    )


def _point_depth_gap_m(depth_m, tip_uv, button_uv) -> float | None:
    if depth_m is None or tip_uv is None or button_uv is None:
        return None
    tip_z = sample_depth(depth_m, float(tip_uv[0]), float(tip_uv[1]), radius=6, percentile=20.0)
    button_z = sample_depth(depth_m, float(button_uv[0]), float(button_uv[1]), radius=6, percentile=20.0)
    if tip_z is None or button_z is None:
        return None
    return abs(float(tip_z) - float(button_z))
# After coarse PBVS, keep the chest pose fix while the moving right-wrist
# camera supplies the fresh close-range target.
POST_PBVS_IBVS_STAGES = frozenset({"WRIST_APPROACH", "IBVS", "POKE"})
# The chest camera is fixed to the torso. Button positions come only from pose
# keypoints. A scale or PnP guess never replaces the last pose frame.
GRIPPER_PANEL_COVER_FRACTION = 0.05


def remember_pose_buttons(
    memory: dict[str, dict[str, Any]],
    buttons: dict[str, Any],
    *,
    freeze: bool,
) -> dict[str, dict[str, Any]]:
    """Keep pose fixes for Button 1/2/3. A frozen IBVS stage does not take new ones."""
    if freeze:
        return memory
    out = dict(memory)
    for name in POSE_BUTTON_NAMES:
        point = buttons.get(name)
        if point is None or str(getattr(point, "source", "")) != "pose":
            continue
        old = out.get(name)
        if old is not None:
            jump_px = math.hypot(float(point.u) - float(old["u"]), float(point.v) - float(old["v"]))
            if jump_px > 40.0:
                continue
        xyz = getattr(point, "xyz_m", None)
        out[name] = {
            "u": float(point.u),
            "v": float(point.v),
            "z_m": None if point.z_m is None else float(point.z_m),
            "xyz_m": None if xyz is None else [float(value) for value in xyz],
        }
    return out


def _polygon_area(points) -> float:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if contour.shape[0] < 3:
        return 0.0
    return abs(float(cv2.contourArea(contour)))


def polygon_overlap_fraction(cover, panel) -> float:
    """Intersection area divided by the panel area. Zero when either polygon is unusable."""
    panel_area = _polygon_area(panel)
    if panel_area < 1.0 or _polygon_area(cover) < 1.0:
        return 0.0
    both = np.vstack(
        (
            np.asarray(cover, dtype=np.float32).reshape(-1, 2),
            np.asarray(panel, dtype=np.float32).reshape(-1, 2),
        )
    )
    min_xy = np.floor(both.min(axis=0)).astype(np.int32) - 1
    max_xy = np.ceil(both.max(axis=0)).astype(np.int32) + 2
    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])
    if width < 2 or height < 2 or width * height > 4_000_000:
        return 0.0

    def paint(poly) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted = np.round(np.asarray(poly, dtype=np.float32).reshape(-1, 2) - min_xy).astype(np.int32)
        cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
        return mask

    overlap = int(np.count_nonzero(paint(cover) & paint(panel)))
    return float(overlap) / panel_area


def _point_in_polygon(point, polygon) -> bool:
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if poly.shape[0] < 3:
        return False
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def gripper_occludes_panel(gripper_polygon, panel_corners, button_points=()) -> bool:
    """True once the jaw covers the faceplate or lies on a button."""
    if gripper_polygon is None or len(gripper_polygon) < 3:
        return False
    if panel_corners and polygon_overlap_fraction(gripper_polygon, panel_corners) >= GRIPPER_PANEL_COVER_FRACTION:
        return True
    return any(_point_in_polygon(point, gripper_polygon) for point in button_points)


def _is_pose_button(point) -> bool:
    return str(getattr(point, "source", "")).startswith("pose")


def hold_last_pose(
    loc: Localization,
    held: dict[str, CameraPoint],
    held_corners,
) -> tuple[Localization, dict[str, CameraPoint], bool]:
    """Accept pose keypoints only. Missing pose keeps the previous pose frame."""
    fresh = {name: point for name, point in loc.buttons.items() if _is_pose_button(point)}
    if fresh:
        merged = {name: point for name, point in held.items() if _is_pose_button(point)}
        merged.update(fresh)
        corners = list(loc.corners_px) if loc.corners_px else held_corners
        return (
            Localization(
                visible=True,
                corners_px=corners,
                buttons=merged,
                message=loc.message,
            ),
            merged,
            False,
        )
    if not held:
        return (
            Localization(
                visible=False,
                corners_px=loc.corners_px,
                buttons={},
                message="no_pose",
            ),
            held,
            False,
        )
    return (
        Localization(
            visible=True,
            corners_px=list(held_corners) if held_corners else loc.corners_px,
            buttons={name: point for name, point in held.items() if _is_pose_button(point)},
            message="pose_held",
        ),
        held,
        True,
    )


def hold_buttons_under_occlusion(
    loc: Localization,
    held: dict[str, CameraPoint],
    *,
    covered: bool,
    held_corners=None,
) -> tuple[Localization, dict[str, CameraPoint], bool]:
    """Keep the last clear-view buttons while the gripper hides the panel."""
    if covered:
        if not held:
            return (
                Localization(
                    visible=False,
                    corners_px=loc.corners_px,
                    buttons={},
                    message="gripper_occludes_panel",
                ),
                held,
                True,
            )
        corners = list(held_corners) if held_corners else loc.corners_px
        return (
            Localization(
                visible=True,
                corners_px=corners,
                buttons={name: point for name, point in held.items()},
                message="gripper_occludes_panel",
            ),
            held,
            True,
        )
    if loc.buttons:
        held = {name: point for name, point in loc.buttons.items()}
    return loc, held, False


def chest_control_under_occlusion(loc, touch, held, held_corners):
    """Chest only. A jaw on the faceplate must not replace the clear-view buttons."""
    gripper = next(
        (
            mask
            for mask in keep_side_masks([] if touch is None else touch.masks, gripper_min=0.3)
            if mask.name == "gripper"
        ),
        None,
    )
    points = []
    if touch is not None:
        for name in POSE_BUTTON_NAMES:
            point = touch.point(name)
            if point is not None:
                points.append((float(point.u), float(point.v)))
    covered = gripper_occludes_panel(
        None if gripper is None else gripper.polygon,
        loc.corners_px,
        points,
    )
    return hold_buttons_under_occlusion(
        loc,
        held,
        covered=covered,
        held_corners=held_corners,
    )


def fresh_chest_pose(
    touch,
    depth_m,
    K,
    torso_T_camera,
    *,
    target_name: str = "",
    conf_min: float = 0.3,
) -> dict[str, Any] | None:
    """Live chest pose button. A held or locked button does not count.

    The pixel is returned even when that pixel has no depth, so the approach
    can still judge whether the gripper is visibly above or below the button.
    """
    if touch is None:
        return None
    # Only the selected key. Borrowing Button 3's depth while the name stays
    # Button 1 sends the fingertip onto the OK key.
    names = [str(target_name)] if target_name in POSE_BUTTON_NAMES else list(POSE_BUTTON_NAMES)
    for name in names:
        point = touch.point(name)
        if point is None or float(point.conf) < float(conf_min):
            continue
        image_uv = [round(float(point.u), 1), round(float(point.v), 1)]
        torso = None
        if depth_m is not None and K is not None:
            z_m = sample_depth(depth_m, float(point.u), float(point.v), radius=6, percentile=20.0)
            if z_m is not None:
                xyz = K.backproject(float(point.u), float(point.v), float(z_m))
                mapped = camera_xyz_to_torso(xyz, torso_T_camera=torso_T_camera)
                torso = [round(float(value), 4) for value in mapped]
        return {"uv": image_uv, "torso_m": torso}
    return None


def fresh_chest_pose_torso(
    touch,
    depth_m,
    K,
    torso_T_camera,
    *,
    target_name: str = "",
    conf_min: float = 0.3,
) -> list[float] | None:
    """Torso XYZ of a button the chest pose model sees in this frame."""
    found = fresh_chest_pose(
        touch,
        depth_m,
        K,
        torso_T_camera,
        target_name=target_name,
        conf_min=conf_min,
    )
    if found is None:
        return None
    torso = found.get("torso_m")
    return None if torso is None else list(torso)


def pose_button_from_memory(memory: dict[str, dict[str, Any]], name: str) -> CameraPoint | None:
    record = memory.get(str(name))
    if not record:
        return None
    xyz = record.get("xyz_m")
    return CameraPoint(
        u=float(record["u"]),
        v=float(record["v"]),
        z_m=None if record.get("z_m") is None else float(record["z_m"]),
        xyz_m=None if xyz is None else (float(xyz[0]), float(xyz[1]), float(xyz[2])),
        source="pose_locked",
    )


def snapshot_good_target(payload: dict[str, Any], now: float) -> dict[str, Any] | None:
    if not (payload.get("panel_visible") and payload.get("target_uv")):
        return None
    return {
        "t": float(now),
        "target_uv": payload.get("target_uv"),
        "target_xyz_m": payload.get("target_xyz_m"),
        "target_torso_m": payload.get("target_torso_m"),
        "pbvs_error_cam_m": payload.get("pbvs_error_cam_m"),
        "pbvs_error_torso_m": payload.get("pbvs_error_torso_m"),
        "dx": payload.get("dx"),
        "dy": payload.get("dy"),
        "dz": payload.get("dz"),
        "tip_xyz_m": payload.get("tip_xyz_m"),
        "virtual_tip_xyz_m": payload.get("virtual_tip_xyz_m"),
    }


def hold_last_target(
    payload: dict[str, Any],
    last: dict[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    """Keep servoing for a few seconds after the close-up view loses the panel."""
    out = dict(payload)
    if snapshot_good_target(out, now) is not None:
        out["using_last_target"] = False
        return out
    if last is None or now - float(last.get("t") or 0.0) > LAST_TARGET_HOLD_S:
        out["using_last_target"] = False
        return out
    out["target_uv"] = last.get("target_uv") or out.get("target_uv")
    out["target_xyz_m"] = last.get("target_xyz_m") or out.get("target_xyz_m")
    out["target_torso_m"] = last.get("target_torso_m") or out.get("target_torso_m")
    out["pbvs_error_cam_m"] = last.get("pbvs_error_cam_m") or out.get("pbvs_error_cam_m")
    out["pbvs_error_torso_m"] = last.get("pbvs_error_torso_m") or out.get("pbvs_error_torso_m")
    if max(abs(float(out.get("dx") or 0.0)), abs(float(out.get("dy") or 0.0)), abs(float(out.get("dz") or 0.0))) < 1e-6:
        out["dx"] = last.get("dx") or 0.0
        out["dy"] = last.get("dy") or 0.0
        out["dz"] = last.get("dz") or 0.0
    out["using_last_target"] = True
    out["message"] = "panel_lost_hold_last_target"
    return out


def discover_handeye_json(day: str = "20260920") -> Optional[Path]:
    found: list[Path] = []
    for root in HANDEYE_CANDIDATES:
        if not root.is_dir():
            continue
        found.extend(
            path
            for path in root.glob(f"eye_in_hand_{day}*.json")
            if "strict" not in path.name
        )
    if not found:
        return None
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(rotation[2, 0]) < 0.999999:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return roll, pitch, yaw


def panel_rpy_deg(corners_px: list, K: Intrinsics, width_m: float, height_m: float) -> tuple[float, float, float] | None:
    if not corners_px or len(corners_px) != 4:
        return None
    obj = np.array(
        [[0.0, 0.0, 0.0], [width_m, 0.0, 0.0], [width_m, height_m, 0.0], [0.0, height_m, 0.0]],
        dtype=np.float64,
    )
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    camera = np.array([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(obj, img, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = _rpy_from_matrix(rotation)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


class LiveIbvsAnnotator:
    def __init__(
        self,
        handeye_json: str | Path | None = None,
        eye_to_hand_json: str | Path | None = None,
        eye_to_hand_x_bias_m: float = 0.0,
        eye_to_hand_y_bias_m: float = 0.0,
        eye_to_hand_z_bias_m: float = 0.0,
        day: str = "20260920",
        use_yolo: bool = False,
        detector=None,
        ignore_gripper: bool = False,
        async_preview: bool = False,
        use_onnx: bool = False,
        use_rknn: bool = False,
    ) -> None:
        self.ignore_gripper = bool(ignore_gripper)
        if self.ignore_gripper:
            print("[ibvs-yolo] revo2: FingerTip approach, ignore gripper and gripper_tip", flush=True)
        self.ctrl = IBVSController.from_configs()
        self.panel = load_panel()
        cam = __import__("ibvs_pad_touch.config", fromlist=["load_camera_config"]).load_camera_config()["camera"]
        fallback = cam.get("fallback_intrinsics") or {}
        self.K = Intrinsics(
            fx=float(fallback.get("fx", 615.0)),
            fy=float(fallback.get("fy", 615.0)),
            cx=float(fallback.get("cx", 320.0)),
            cy=float(fallback.get("cy", 240.0)),
            width=int(fallback.get("width", 640)),
            height=int(fallback.get("height", 480)),
        )
        # The D435I is rigidly mounted on right_wrist_yaw_link.
        path = Path(handeye_json) if handeye_json else None
        self.handeye_path = str(path) if path and path.is_file() else ""
        chest_path = Path(eye_to_hand_json) if eye_to_hand_json else discover_eye_to_hand_json()
        self.torso_T_camera, self.eye_to_hand = load_eye_to_hand_transform(chest_path)
        self._eye_bias_m = (
            float(eye_to_hand_x_bias_m),
            float(eye_to_hand_y_bias_m),
            float(eye_to_hand_z_bias_m),
        )
        self._torso_T_camera_base = None
        if not isinstance(self.eye_to_hand, dict):
            self.eye_to_hand = {}
        if self.torso_T_camera is not None:
            self.torso_T_camera = np.asarray(self.torso_T_camera, dtype=np.float64).copy()
            self._torso_T_camera_base = self.torso_T_camera.copy()
            self._apply_eye_bias(self._eye_bias_m)
        else:
            self.eye_to_hand["torso_x_bias_m"] = self._eye_bias_m[0]
            self.eye_to_hand["torso_y_bias_m"] = self._eye_bias_m[1]
            self.eye_to_hand["torso_z_bias_m"] = self._eye_bias_m[2]
        self.eye_to_hand_path = "" if chest_path is None else str(chest_path)
        wrist_path = discover_eye_in_hand_json()
        self.cam2hand, self.eye_in_hand = load_eye_in_hand_transform(wrist_path)
        self.handeye = dict(self.eye_to_hand)
        if not self.handeye and self.handeye_path:
            self.handeye = self._load_handeye(path)
        self._lock = threading.Lock()
        self._telemetry: dict[str, Any] = self._empty_telemetry()
        self._last_chest_ts = 0.0
        self._last_chest_vis: Optional[np.ndarray] = None
        self._last_wrist_ts = 0.0
        self._last_wrist_vis: Optional[np.ndarray] = None
        self.tip_tracker = StableTipTracker()
        self._ee_xyz: list[float] | None = None
        self._ee_rpy: list[float] | None = None
        self._tip_bias_cam: list[float] | None = None
        self._last_good_target: dict[str, Any] | None = None
        self._pose_buttons: dict[str, dict[str, Any]] = {}
        self._clear_view_buttons: dict[str, CameraPoint] = {}
        self._clear_view_corners = None
        self._chest_occlusion_buttons: dict[str, CameraPoint] = {}
        self._chest_occlusion_corners = None
        self._wrist_watch = False
        self._wrist_seen = False
        self._wrist_buttons_cam: dict[str, list[float]] = {}
        self._wrist_target_uv: list[float] | None = None
        self._wrist_target_z: float | None = None
        self._wrist_tip_uv: list[float] | None = None
        self._wrist_tip_z: float | None = None
        self._wrist_occluded = False
        self._wrist_updated_at = 0.0
        self._wrist_buttons_fresh = False
        self._wrist_red_bar = False
        self._wrist_close_range = False
        self._wrist_target_source = ""
        self._wrist_yellow: dict[str, Any] | None = None
        self._wrist_yellow_id = 0
        self._wrist_yellow_message = ""
        self._last_wrist_bgr: np.ndarray | None = None
        self._last_chest_bgr: np.ndarray | None = None
        self._last_screen_polygon = None
        self._last_side_bgr: np.ndarray | None = None
        self._last_side_screen = None
        self._side_ready_fn = lambda: False
        self._sdk_active = False
        self._finger_lock: CameraPoint | None = None
        self._finger_recapture = False
        self._finger_lock_enabled = False
        self._stage_fn = lambda: ""
        self._async_preview = bool(async_preview)
        self._latest_frame: dict[str, np.ndarray] = {}
        self._preview: dict[str, tuple] = {}
        self._infer_guard = threading.Lock()
        self._infer_event = threading.Event()
        self._infer_started = False
        if detector is not None:
            self.detector = detector
            self.wrist_detector = detector
        elif use_yolo:
            self.detector = YoloTouchDetector(name="chest", use_onnx=use_onnx, use_rknn=use_rknn)
            self.wrist_detector = YoloTouchDetector(
                name="wrist",
                cpu_pose=True,
                use_onnx=use_onnx,
                use_rknn=use_rknn,
            )
            self.detector._ensure()
            self.wrist_detector._ensure()
        else:
            self.detector = None
            self.wrist_detector = None
        self._touch_cache: dict[str, tuple[float, Any]] = {}
        self._yolo_warned: set[str] = set()
        self._side: dict[str, Any] = {
            "measured": False,
            "contact": False,
            "gap_px": None,
            "blocked": False,
            "depth_m": None,
            "message": "",
        }
        self._side_contact_since: float | None = None
        self._wrist_contact_since: float | None = None

    def _apply_eye_bias(self, bias: tuple[float, float, float]) -> None:
        base = self._torso_T_camera_base
        if base is None:
            return
        updated = np.array(base, dtype=np.float64, copy=True)
        updated[0, 3] += float(bias[0])
        updated[1, 3] += float(bias[1])
        updated[2, 3] += float(bias[2])
        self.torso_T_camera = updated
        self.eye_to_hand["torso_x_bias_m"] = float(bias[0])
        self.eye_to_hand["torso_y_bias_m"] = float(bias[1])
        self.eye_to_hand["torso_z_bias_m"] = float(bias[2])
        self._eye_bias_m = (float(bias[0]), float(bias[1]), float(bias[2]))

    def set_eye_to_hand_bias(self, x_m: float, y_m: float, z_m: float) -> tuple[bool, str]:
        """Replace the field XYZ bias. Values are metres in the torso frame, not increments."""
        try:
            bias = (float(x_m), float(y_m), float(z_m))
        except (TypeError, ValueError):
            return False, "x_m y_m z_m 需要是数字，单位 m"
        if not all(math.isfinite(value) and abs(value) <= 0.5 for value in bias):
            return False, "偏移需要是 ±0.5 m 以内的数"
        if self._torso_T_camera_base is None:
            return False, "没有胸口手眼标定，不能改偏移"
        with self._lock:
            self._apply_eye_bias(bias)
            self.handeye = dict(self.eye_to_hand)
            if isinstance(self._telemetry, dict):
                self._telemetry["eye_to_hand"] = dict(self.eye_to_hand)
                self._telemetry["handeye"] = dict(self.eye_to_hand)
        return True, ""

    def _load_handeye(self, path: Path | None) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        record = data.get("cam2hand") or {}
        tvec = record.get("tvec_m") or [None, None, None]
        rotation = np.asarray(record.get("rotation_matrix") or np.eye(3), dtype=np.float64)
        rpy = _rpy_from_matrix(rotation)
        return {
            "file": path.name,
            "serial": WRIST_SERIAL,
            "xyz_m": [None if v is None else float(v) for v in tvec],
            "rpy_deg": [math.degrees(v) for v in rpy],
        }

    def _empty_telemetry(self) -> dict[str, Any]:
        return {
            "phase": "SEEK_PANEL",
            "target": self.ctrl.target_name,
            "target_uv": None,
            "target_source": "",
            "tip_uv": None,
            "tip_visible": False,
            "tip_raw_uv": None,
            "tip_tracker": {"mode": "auto_stable", "confirmed": False, "reason": "waiting"},
            "tip_source": "",
            "virtual_tip_uv": None,
            "virtual_tip_xyz_m": None,
            "target_xyz_m": None,
            "tip_xyz_m": None,
            "target_torso_m": None,
            "tip_torso_m": None,
            "pbvs_error_cam_m": None,
            "pbvs_error_torso_m": None,
            "depth_available": False,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "droll": 0.0,
            "dpitch": 0.0,
            "dyaw": 0.0,
            "panel_visible": False,
            "panel_partial": False,
            "wrist_panel_visible": False,
            "wrist_panel_partial": False,
            "wrist_target_uv": None,
            "wrist_tip_uv": None,
            "wrist_updated_at": 0.0,
            "left_contact": False,
            "side_measured": False,
            "side_gap_px": None,
            "using_last_target": False,
            "message": "waiting_chest_frame",
            "handeye": {},
            "eye_to_hand": dict(self.eye_to_hand),
            "eye_in_hand": dict(self.eye_in_hand),
            "buttons_torso_m": {},
            "wrist_serial": WRIST_SERIAL,
            "left_serial": LEFT_SERIAL,
            "chest_serial": CHEST_SERIAL,
            "updated_at": 0.0,
        }

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._telemetry)
        payload.update(self.finger_tip_status())
        return payload

    def set_sdk_active(self, active: bool) -> None:
        """Arm the fingertip latch only after arm_sdk is holding the body."""
        with self._lock:
            self._sdk_active = bool(active)

    def recapture_finger_tip(self) -> str:
        """Drop in the next wrist FingerTip. Task changes do not do this."""
        if not self.ignore_gripper:
            return "当前右手不是 Revo2，指尖不锁定"
        with self._lock:
            if not self._finger_lock_enabled:
                return "指尖锁定是关的，正在用每帧检测"
            self._finger_recapture = True
        return "等待下一帧右腕 FingerTip，用来替换已锁定的指尖"

    def toggle_finger_lock(self) -> str:
        """Turn the fixed fingertip off, or arm it again for the next frame."""
        if not self.ignore_gripper:
            return "当前右手不是 Revo2，指尖不锁定"
        with self._lock:
            self._finger_lock_enabled = not self._finger_lock_enabled
            enabled = self._finger_lock_enabled
            if not enabled:
                self._finger_lock = None
                self._finger_recapture = False
                return "指尖锁定已关闭，改用实时检测"
            self._finger_recapture = True
            return "指尖锁定已打开，等待下一帧"

    def finger_tip_status(self) -> dict[str, Any]:
        with self._lock:
            locked = self._finger_lock
            recapture = self._finger_recapture
            enabled = self._finger_lock_enabled
        if not self.ignore_gripper:
            return {
                "finger_tip_lock": None,
                "finger_tip_recapture": False,
                "finger_tip_lock_enabled": False,
                "finger_tip_text": "",
            }
        if not enabled:
            text = "指尖实时检测"
        elif recapture:
            text = "等待重新检测指尖"
        elif locked is None:
            text = "等待第一次指尖"
        else:
            depth = "" if locked.z_m is None else f" z {locked.z_m * 1000:.0f}mm"
            text = f"指尖锁定 {locked.u:.0f},{locked.v:.0f}{depth}"
        point = None
        if locked is not None:
            point = {
                "u": round(float(locked.u), 1),
                "v": round(float(locked.v), 1),
                "z_m": None if locked.z_m is None else round(float(locked.z_m), 4),
            }
        return {
            "finger_tip_lock": point,
            "finger_tip_recapture": recapture,
            "finger_tip_lock_enabled": enabled,
            "finger_tip_text": text,
        }

    def _take_wrist_finger(self, live: CameraPoint | None) -> tuple[CameraPoint | None, CameraPoint | None]:
        """Control tip, plus the live detection when it should still be drawn."""
        if not self.ignore_gripper:
            return live, None
        with self._lock:
            if not self._finger_lock_enabled:
                return live, None
            locked = self._finger_lock
            recapture = self._finger_recapture
        control, stored, pending = latch_wrist_finger_tip(
            locked,
            live,
            recapture=recapture,
        )
        if stored is not None and stored.z_m is None:
            filled = self._finger_depth_once(stored)
            if filled is not None:
                stored = filled
                control = filled
        with self._lock:
            self._finger_lock = stored
            self._finger_recapture = pending
        return control, live

    def _finger_depth_once(self, point: CameraPoint) -> CameraPoint | None:
        """Freeze the first depth at the locked pixel. Later frames do not move it."""
        depth = read_depth(SIDE_DEPTH_PATH)
        if depth is None:
            return None
        z_m = sample_depth(depth, float(point.u), float(point.v), radius=6, percentile=20.0)
        if z_m is None:
            return None
        frozen = point.with_depth(self.K, float(z_m))
        return CameraPoint(
            u=frozen.u,
            v=frozen.v,
            z_m=frozen.z_m,
            xyz_m=frozen.xyz_m,
            source="finger_lock",
            confidence=frozen.confidence,
        )

    def set_target(self, name: str) -> None:
        self.ctrl.set_target(name)

    def set_tip_seed(self, u: float, v: float) -> None:
        self.tip_tracker.seed(u, v)

    def clear_tip_seed(self) -> None:
        self.tip_tracker.clear()
        with self._lock:
            self._tip_bias_cam = None

    def forget_button_lock(self) -> None:
        """Drop the pose held from the last press so the next chest frame is a new fix."""
        with self._lock:
            self._pose_buttons = {}
            self._clear_view_buttons = {}
            self._clear_view_corners = None
            self._chest_occlusion_buttons = {}
            self._chest_occlusion_corners = None
            self._last_good_target = None
            self._wrist_yellow = None
            self._wrist_yellow_message = ""

    def set_side_ready(self, ready_fn) -> None:
        """Compatibility hook for the retired fixed left-side approach."""
        self._side_ready_fn = ready_fn

    def set_control_stage(self, stage_fn) -> None:
        """IBVS after PBVS reads this so a covered button cannot replace the pose fix."""
        self._stage_fn = stage_fn

    def set_wrist_pbvs_watch(self, enabled: bool) -> None:
        """Latch right-wrist button observations for PBVS and final approach."""
        with self._lock:
            self._wrist_watch = bool(enabled)
            if enabled:
                self._wrist_seen = False
                self._wrist_buttons_cam = {}
                self._wrist_target_uv = None
                self._wrist_target_z = None
                self._wrist_tip_uv = None
                self._wrist_tip_z = None
                self._wrist_occluded = False
                self._wrist_updated_at = 0.0
                self._wrist_buttons_fresh = False
                self._wrist_red_bar = False
                self._wrist_close_range = False
                self._wrist_target_source = ""
                self._wrist_yellow = None
                self._wrist_yellow_message = ""

    def hold_wrist_pbvs(self) -> None:
        """Drop samples taken while the forearm was still moving. The settle wait refills them."""
        with self._lock:
            self._wrist_seen = False
            self._wrist_buttons_cam = {}
            self._wrist_target_uv = None
            self._wrist_target_z = None
            self._wrist_tip_uv = None
            self._wrist_tip_z = None
            self._wrist_occluded = False
            self._wrist_updated_at = 0.0
            self._wrist_buttons_fresh = False
            self._wrist_close_range = False
            self._wrist_target_source = ""

    def wrist_pbvs_view(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seen": bool(self._wrist_seen),
                "buttons_cam_m": {name: list(value) for name, value in self._wrist_buttons_cam.items()},
                "target_uv": None if self._wrist_target_uv is None else list(self._wrist_target_uv),
                "target_z_m": self._wrist_target_z,
                "tip_uv": None if self._wrist_tip_uv is None else list(self._wrist_tip_uv),
                "tip_z_m": self._wrist_tip_z,
                "occluded": bool(self._wrist_occluded),
                "intrinsics": {
                    "fx": float(self.K.fx),
                    "fy": float(self.K.fy),
                    "cx": float(self.K.cx),
                    "cy": float(self.K.cy),
                },
                "updated_at": float(self._wrist_updated_at),
                "contact": bool(self._side.get("wrist_contact")),
                "buttons_fresh": bool(self._wrist_buttons_fresh),
                "red_bar": bool(self._wrist_red_bar),
                "close_range": bool(self._wrist_close_range),
                "target_source": str(self._wrist_target_source or ""),
                "yellow_lock": self._wrist_yellow is not None,
                "yellow_id": 0 if self._wrist_yellow is None else int(self._wrist_yellow["id"]),
                "yellow_uv": (
                    None
                    if self._wrist_yellow is None
                    else [float(self._wrist_yellow["u"]), float(self._wrist_yellow["v"])]
                ),
            }

    def clear_wrist_yellow(self, reason: str = "") -> None:
        with self._lock:
            self._wrist_yellow = None
            if reason:
                self._wrist_yellow_message = reason

    def pick_wrist_yellow(self, u: float, v: float) -> tuple[bool, str]:
        """Lock the yellow region under a click. Only during the wrist approach."""
        try:
            stage = str(self._stage_fn() or "")
        except Exception:
            stage = ""
        if stage != "WRIST_APPROACH":
            return False, "只在右腕进近时可以点选黄色区域"
        try:
            cu = float(u)
            cv = float(v)
        except (TypeError, ValueError):
            return False, "u/v 无效"
        if not math.isfinite(cu) or not math.isfinite(cv):
            return False, "u/v 无效"
        with self._lock:
            frame = None if self._last_wrist_bgr is None else self._last_wrist_bgr
        if frame is None:
            return False, "还没有右腕画面"
        height, width = frame.shape[:2]
        if not (0.0 <= cu < float(width) and 0.0 <= cv < float(height)):
            return False, "点击超出画面"
        blob = yellow_blob_at(frame, cu, cv)
        if blob is None:
            return False, "这里没有黄色区域"
        with self._lock:
            self._wrist_yellow_id += 1
            self._wrist_yellow = {
                "id": self._wrist_yellow_id,
                "u": float(blob.cx),
                "v": float(blob.cy),
                "target": str(self.ctrl.target_name),
                "misses": 0,
            }
            self._wrist_yellow_message = f"已点选黄色区域 {blob.cx:.0f},{blob.cy:.0f}"
            return True, self._wrist_yellow_message

    def _refresh_wrist_yellow(self, bgr: np.ndarray) -> None:
        """Follow the clicked region through a small shift, then drop it."""
        with self._lock:
            lock = None if self._wrist_yellow is None else dict(self._wrist_yellow)
            target_name = str(self.ctrl.target_name)
        if lock is None:
            return
        try:
            stage = str(self._stage_fn() or "")
        except Exception:
            stage = ""
        if stage != "WRIST_APPROACH" or str(lock.get("target") or "") != target_name:
            self.clear_wrist_yellow("状态已切换，取消黄色点选")
            return
        blobs = yellow_blobs(bgr)
        if not blobs:
            self.clear_wrist_yellow("画面里没有黄色按键，取消点选")
            return
        matched = match_yellow_blob(blobs, float(lock["u"]), float(lock["v"]))
        with self._lock:
            current = self._wrist_yellow
            if current is None or int(current.get("id") or 0) != int(lock.get("id") or 0):
                return
            if matched is None:
                misses = int(current.get("misses") or 0) + 1
                current["misses"] = misses
                if misses >= YELLOW_TRACK_MISS_FRAMES:
                    self._wrist_yellow = None
                    self._wrist_yellow_message = "点选的黄色区域跟丢了"
                return
            current["u"] = float(matched.cx)
            current["v"] = float(matched.cy)
            current["misses"] = 0

    def cam2hand_matrix(self):
        if self.cam2hand is None:
            return None
        return np.asarray(self.cam2hand, dtype=np.float64).copy()

    def _screen_view(self):
        with self._lock:
            bgr = None if self._last_chest_bgr is None else self._last_chest_bgr.copy()
            polygon = None if self._last_screen_polygon is None else list(self._last_screen_polygon)
            if bgr is None or polygon is None:
                bgr = None if self._last_side_bgr is None else self._last_side_bgr.copy()
                polygon = None if self._last_side_screen is None else list(self._last_side_screen)
        return bgr, polygon

    def screen_bgr(self) -> np.ndarray | None:
        bgr, polygon = self._screen_view()
        return screen_crop(bgr, polygon)

    def screen_jpeg(self) -> bytes | None:
        crop = self.screen_bgr()
        if crop is None:
            return None
        ok, blob = cv2.imencode(".jpg", crop)
        if not ok:
            return None
        return blob.tobytes()

    def _buttons_torso(self, loc) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name in POSE_BUTTON_NAMES:
            point = loc.button(name) if hasattr(loc, "button") else None
            xyz = None if point is None else getattr(point, "xyz_m", None)
            if xyz is None or len(xyz) != 3:
                continue
            torso = camera_xyz_to_torso(xyz, torso_T_camera=self.torso_T_camera)
            out[name] = [round(float(value), 4) for value in torso]
        return out

    def _note_wrist_pbvs(self, bgr: np.ndarray, touch, tip=None) -> None:
        with self._lock:
            watching = self._wrist_watch
        screen = _screen_polygon(touch)
        depth = read_depth(SIDE_DEPTH_PATH)
        if depth is not None and depth.shape[:2] != bgr.shape[:2]:
            depth = None
        buttons: dict[str, list[float]] = {}
        if touch is not None and depth is not None:
            for name in POSE_BUTTON_NAMES:
                point = touch.point(name)
                if point is None or float(point.conf) < self._wrist_button_conf():
                    continue
                z_m = sample_depth(depth, float(point.u), float(point.v), radius=6, percentile=20.0)
                if z_m is None:
                    continue
                xyz = self.K.backproject(float(point.u), float(point.v), float(z_m))
                buttons[name] = [float(value) for value in xyz]
        tip_z = None
        if tip is not None and tip.source == "finger_lock" and tip.z_m is not None:
            tip_z = float(tip.z_m)
        elif depth is not None and tip is not None:
            sampled_tip = sample_depth(depth, float(tip.u), float(tip.v), radius=6, percentile=20.0)
            if sampled_tip is not None:
                tip_z = float(sampled_tip)
        red_bar = detect_red_bar(bgr)
        self._refresh_wrist_yellow(bgr)
        image_target = None if touch is None else touch.point(self.ctrl.target_name)
        image_ok = (
            image_target is not None
            and float(image_target.conf) >= self._wrist_button_conf()
        )
        yellow_uv = None
        if not image_ok and tip is not None:
            yellow_uv = nearest_long_yellow_bar(bgr, (float(tip.u), float(tip.v)))
        with self._lock:
            if screen is not None:
                self._last_side_bgr = bgr.copy()
                self._last_side_screen = screen
            self._wrist_red_bar = bool(red_bar)
            if watching:
                # The wrist camera only ever sees part of the panel, and the jaw
                # is in frame. That is not a reason to drop this sample.
                self._wrist_occluded = False
            target_name = self.ctrl.target_name
            target_has_depth = bool(image_ok and target_name in buttons)
            self._wrist_target_source = ""
            # A confident button with no depth is already inside the camera's
            # minimum range, about 5 cm. Keep the pixels and do not invent Z.
            close_range = bool(watching and image_ok and tip is not None and not target_has_depth)
            if watching and buttons:
                merged = dict(self._wrist_buttons_cam)
                merged.update(buttons)
                self._wrist_buttons_cam = merged
                self._wrist_seen = True
            if watching and target_has_depth:
                self._wrist_buttons_fresh = True
                self._wrist_close_range = False
                self._wrist_target_source = "button"
                self._wrist_target_uv = [float(image_target.u), float(image_target.v)]
                self._wrist_target_z = float(buttons[target_name][2])
                self._wrist_tip_uv = None if tip is None else [float(tip.u), float(tip.v)]
                self._wrist_tip_z = None if tip is None else tip_z
                self._wrist_updated_at = time.time()
            elif close_range:
                self._wrist_buttons_fresh = False
                self._wrist_close_range = True
                self._wrist_seen = True
                self._wrist_target_source = "button"
                self._wrist_buttons_cam.pop(target_name, None)
                self._wrist_target_uv = [float(image_target.u), float(image_target.v)]
                self._wrist_target_z = None
                self._wrist_tip_uv = [float(tip.u), float(tip.v)]
                self._wrist_tip_z = None
                self._wrist_updated_at = time.time()
            elif watching and yellow_uv is not None and tip is not None:
                yu, yv = yellow_uv
                yellow_z = None
                if depth is not None:
                    sampled = sample_depth(depth, yu, yv, radius=6, percentile=20.0)
                    if sampled is not None:
                        yellow_z = float(sampled)
                self._wrist_target_uv = [yu, yv]
                self._wrist_tip_uv = [float(tip.u), float(tip.v)]
                self._wrist_tip_z = tip_z
                self._wrist_seen = True
                self._wrist_target_source = "yellow_bar"
                self._wrist_updated_at = time.time()
                if yellow_z is not None:
                    self._wrist_target_z = yellow_z
                    self._wrist_buttons_fresh = True
                    self._wrist_close_range = False
                else:
                    self._wrist_target_z = None
                    self._wrist_buttons_fresh = False
                    self._wrist_close_range = True
            elif watching and buttons:
                self._wrist_buttons_fresh = True
                self._wrist_close_range = False
                self._wrist_tip_uv = None if tip is None else [float(tip.u), float(tip.v)]
                self._wrist_tip_z = None if tip is None else tip_z
                self._wrist_updated_at = time.time()
            elif watching:
                self._wrist_buttons_fresh = False
                self._wrist_close_range = False
                if self._wrist_buttons_cam:
                    self._wrist_seen = True
            yellow = self._wrist_yellow
            if yellow is not None:
                yu = float(yellow["u"])
                yv = float(yellow["v"])
                yellow_z = None
                if depth is not None:
                    sampled = sample_depth(depth, yu, yv, radius=6, percentile=20.0)
                    if sampled is not None:
                        yellow_z = float(sampled)
                self._wrist_target_uv = [yu, yv]
                self._wrist_tip_uv = None if tip is None else [float(tip.u), float(tip.v)]
                self._wrist_tip_z = None if tip is None else tip_z
                self._wrist_seen = True
                self._wrist_updated_at = time.time()
                if yellow_z is not None:
                    self._wrist_target_z = yellow_z
                    self._wrist_buttons_fresh = True
                    self._wrist_close_range = False
                    self._wrist_target_source = "yellow_lock"
                elif tip is not None:
                    self._wrist_target_z = None
                    self._wrist_buttons_fresh = False
                    self._wrist_close_range = True
                    self._wrist_target_source = "yellow_lock"
                else:
                    self._wrist_target_z = None
                    self._wrist_buttons_fresh = False
                    self._wrist_close_range = False

    def _pose_lock_frozen(self) -> bool:
        try:
            stage = str(self._stage_fn() or "")
        except Exception:
            return False
        return stage in POST_PBVS_IBVS_STAGES

    def set_ee_pose(self, xyz_m: list[float], rpy_rad: list[float]) -> None:
        with self._lock:
            self._ee_xyz = [float(v) for v in xyz_m]
            self._ee_rpy = [float(v) for v in rpy_rad]

    @property
    def eye_to_hand_valid(self) -> bool:
        return self.torso_T_camera is not None

    @property
    def camera_R_torso(self) -> np.ndarray | None:
        if self.torso_T_camera is None:
            return None
        return np.asarray(self.torso_T_camera[:3, :3], dtype=np.float64).copy()

    def _ee_pose(self) -> tuple[list[float], list[float]] | None:
        with self._lock:
            if self._ee_xyz is None or self._ee_rpy is None:
                return None
            return list(self._ee_xyz), list(self._ee_rpy)

    def _virtual_tip(self, apply_bias: bool = True):
        pose = self._ee_pose()
        if pose is None:
            return None
        xyz, rpy = pose
        bias = None
        if apply_bias:
            with self._lock:
                bias = None if self._tip_bias_cam is None else list(self._tip_bias_cam)
        return virtual_tip_from_ee(
            xyz,
            rpy,
            self.K,
            bias_cam_m=bias,
            torso_T_camera=self.torso_T_camera,
        )

    def _paint_side(self, bgr: np.ndarray, touch):
        """Draw the fingertip and the pose buttons. The wrist does not use segmentation."""
        vis = bgr
        seg_corners = None
        live_finger = None
        if self.ignore_gripper:
            live_finger = pose_finger_tip(touch, self._point_conf())
            if live_finger is None:
                live_finger = pose_finger_tip(touch, 0.15)
            tip, live_finger = self._take_wrist_finger(live_finger)
        else:
            tip = pose_tip(touch, self._point_conf())
        if (
            live_finger is not None
            and tip is not None
            and math.hypot(float(live_finger.u) - float(tip.u), float(live_finger.v) - float(tip.v)) > 8.0
        ):
            if vis is bgr:
                vis = bgr.copy()
            seen = (int(round(live_finger.u)), int(round(live_finger.v)))
            cv2.circle(vis, seen, 6, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(
                vis,
                "NEW",
                (seen[0] + 8, max(16, seen[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
        if tip is not None:
            if vis is bgr:
                vis = bgr.copy()
            px = (int(round(tip.u)), int(round(tip.v)))
            if tip.source == "finger_lock":
                label = "LOCK"
            elif tip.source == "finger":
                label = "FINGER"
            elif tip.source == "axis":
                label = "AXIS"
            else:
                label = "TIP"
            cv2.drawMarker(vis, px, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(
                vis,
                label,
                (px[0] + 8, max(16, px[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        vis = draw_pose_buttons(
            vis,
            {} if touch is None else touch.points,
            conf_min=self._wrist_button_conf(),
        )
        return vis, seg_corners, tip

    def _point_conf(self) -> float:
        return float(self.ctrl.cfg.get("yolo_point_conf", 0.3))

    def _wrist_button_conf(self) -> float:
        """Right-wrist buttons below this are noise. They are not drawn and do not move the arm."""
        return max(WRIST_BUTTON_CONF_MIN, self._point_conf())

    def _touch(self, key: str, bgr: np.ndarray, detector=None):
        detector = self.detector if detector is None else detector
        if detector is None:
            return None
        now = time.monotonic()
        cached = self._touch_cache.get(key)
        # The wrist pose is one network. Do not keep a stale fingertip while
        # the chest segmentation is still using the GPU.
        limit = 0.05 if key == "wrist" else 0.20
        if cached is not None and now - float(cached[0]) < limit:
            return cached[1]
        try:
            if key == "wrist":
                result = normalize_button_layout(detector.predict(bgr, pose_only=True))
            else:
                result = normalize_button_layout(detector.predict(bgr))
            if self.ignore_gripper:
                result = without_gripper(result)
        except Exception as exc:
            if key not in self._yolo_warned:
                print(f"[ibvs-yolo] {key} {exc}", flush=True)
                self._yolo_warned.add(key)
            result = None
        self._touch_cache[key] = (now, result)
        return result

    def annotate(self, name: str, frame: np.ndarray) -> np.ndarray:
        if self._async_preview and name in {"chest", "d435i"}:
            with self._infer_guard:
                self._latest_frame[name] = frame.copy()
                self._infer_event.set()
                if not self._infer_started:
                    self._infer_started = True
                    threading.Thread(
                        target=self._infer_loop, args=("chest",), name="ibvs-chest", daemon=True
                    ).start()
                    threading.Thread(
                        target=self._infer_loop, args=("d435i",), name="ibvs-wrist", daemon=True
                    ).start()
            return self._preview_frame(name, frame)
        if name == "chest":
            return self._annotate_chest(frame)
        if name == "d435i":
            return self._annotate_right_wrist(frame)
        return frame

    def _infer_loop(self, camera: str) -> None:
        """One thread per camera. The wrist pose must not wait for chest segmentation."""
        while True:
            self._infer_event.wait(0.02)
            with self._infer_guard:
                frame = self._latest_frame.pop(camera, None)
                if not self._latest_frame:
                    self._infer_event.clear()
            if frame is None:
                continue
            try:
                if camera == "chest":
                    self._annotate_chest(frame)
                elif camera == "d435i":
                    self._annotate_right_wrist(frame)
            except Exception as exc:
                print(f"[ibvs-yolo] {camera} {exc}", flush=True)
                self._stash_view(camera, ("error", f"{camera} pose failed: {exc}"))

    def _stash_view(self, name: str, payload) -> None:
        with self._infer_guard:
            self._preview[name] = payload

    def _preview_frame(self, name: str, bgr: np.ndarray) -> np.ndarray:
        with self._infer_guard:
            pack = self._preview.get(name)
        if not pack:
            return bgr
        if pack[0] == "error":
            vis = bgr.copy()
            cv2.putText(
                vis,
                str(pack[1])[:90],
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            return vis
        if pack[0] == "chest":
            return self._paint_chest_view(bgr, pack)
        if pack[0] == "wrist":
            return self._paint_wrist_view(bgr, pack)
        return bgr

    def _paint_chest_view(self, bgr: np.ndarray, pack) -> np.ndarray:
        """Same panel fill, screen contour, and colored button labels as the full overlay."""
        _kind, loc, touch, lines, model_lines = pack
        vis = bgr
        if loc.corners_px or loc.buttons:
            vis = draw_localization(bgr, loc)
        vis = draw_pose_buttons(
            vis,
            {} if touch is None else touch.points,
            skip={name: (point.u, point.v) for name, point in loc.buttons.items()},
        )
        regions = [
            mask
            for mask in keep_side_masks([] if touch is None else touch.masks, gripper_min=0.3)
            if mask.name != "panel"
        ]
        if regions:
            vis = draw_seg_masks(vis, regions)
        if lines:
            vis = _hud(vis, list(lines), max_width=int(0.52 * vis.shape[1]))
        if model_lines:
            vis = _hud(
                vis,
                list(model_lines),
                origin=(int(0.56 * vis.shape[1]), 8),
                max_width=int(0.42 * vis.shape[1]),
            )
        return vis

    def _paint_wrist_view(self, bgr: np.ndarray, pack) -> np.ndarray:
        _kind, touch, tip, yellow, lines, model_lines, red_bar = pack
        vis = bgr
        vis = draw_pose_buttons(
            vis,
            {} if touch is None else touch.points,
            conf_min=self._wrist_button_conf(),
        )
        if tip is not None:
            if vis is bgr:
                vis = bgr.copy()
            px = (int(round(tip[0])), int(round(tip[1])))
            cv2.drawMarker(vis, px, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(
                vis,
                tip[2],
                (px[0] + 8, max(16, px[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if yellow is not None:
            if vis is bgr:
                vis = bgr.copy()
            mark = (int(round(yellow[0])), int(round(yellow[1])))
            cv2.circle(vis, mark, 16, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, mark, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.putText(
                vis,
                "YELLOW",
                (mark[0] + 12, max(16, mark[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if red_bar:
            if vis is bgr:
                vis = bgr.copy()
            cv2.putText(
                vis,
                "RED",
                (8, max(18, vis.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if lines:
            vis = _hud(vis, list(lines), max_width=int(0.52 * vis.shape[1]))
        if model_lines:
            vis = _hud(
                vis,
                list(model_lines),
                origin=(int(0.56 * vis.shape[1]), 8),
                max_width=int(0.42 * vis.shape[1]),
            )
        return vis

    def _annotate_chest(self, bgr: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        self._last_chest_ts = now
        depth_m = read_depth(CHEST_DEPTH_PATH)
        if depth_m is not None and depth_m.shape[:2] != bgr.shape[:2]:
            depth_m = None
        touch = self._touch("chest", bgr)
        seg_corners = _seg_panel_corners(touch, bgr.shape, self._point_conf())
        loc = self.ctrl.localize(bgr, depth_m, self.K, corners_px=seg_corners)
        loc = apply_pose_buttons(
            loc,
            touch,
            conf_min=self._point_conf(),
            depth_m=depth_m,
            K=self.K,
        )
        loc = align_buttons_to_exposed_panel(
            loc,
            depth_m,
            self.K,
            _cover_polygons(touch),
        )
        loc, self._chest_occlusion_buttons, occlusion_held = chest_control_under_occlusion(
            loc,
            touch,
            self._chest_occlusion_buttons,
            self._chest_occlusion_corners,
        )
        if not occlusion_held:
            self._chest_occlusion_corners = list(loc.corners_px) if loc.corners_px else None
        loc, self._clear_view_buttons, buttons_frozen = hold_last_pose(
            loc,
            self._clear_view_buttons,
            self._clear_view_corners,
        )
        if not buttons_frozen and loc.corners_px:
            self._clear_view_corners = list(loc.corners_px)
        pose_frozen = self._pose_lock_frozen()
        self._pose_buttons = remember_pose_buttons(
            self._pose_buttons,
            loc.buttons,
            freeze=pose_frozen,
        )
        locked_button = (
            pose_button_from_memory(self._pose_buttons, self.ctrl.target_name)
            if pose_frozen
            else None
        )
        raw_tip = _chest_raw_tip(
            touch,
            loc.corners_px,
            self._point_conf(),
            finger=self.ignore_gripper,
        )
        tracked = self.tip_tracker.update(bgr, raw_tip)
        tracker = self.tip_tracker.status()
        virtual = self._virtual_tip(apply_bias=True)
        raw_virtual = self._virtual_tip(apply_bias=False)
        if tracked is not None and tracker.get("mode") == "manual_flow" and depth_m is not None:
            z_tip = sample_depth(depth_m, tracked.u, tracked.v, radius=10, percentile=5.0)
            if z_tip is not None:
                tracked = tracked.with_depth(self.K, z_tip)
        control_tip = choose_control_tip(
            virtual=virtual,
            manual=tracked,
            K=self.K,
            tracker_manual=tracker.get("mode") == "manual_flow",
            panel_corners=loc.corners_px,
        )
        if tracker.get("mode") == "manual_flow" and control_tip is not None and raw_virtual is not None:
            bias = visual_bias_cam_m(control_tip, raw_virtual)
            if bias is not None:
                with self._lock:
                    self._tip_bias_cam = list(bias)
        cmd = self.ctrl.step(
            bgr,
            depth_m,
            self.K,
            observation=Observation(localization=loc, tip=control_tip),
            detect_tip=False,
        )
        visual = _chest_visual_tip(raw_tip, tracked, tracker.get("mode") == "manual_flow", loc.corners_px)
        button = locked_button if locked_button is not None else (
            loc.button(self.ctrl.target_name) if loc.visible else None
        )
        with self._lock:
            side = dict(self._side)
        cfg = self.ctrl.cfg
        nominal_depth_m = float(cfg.get("image_servo_nominal_depth_m", 0.60))
        depth_low = float(cfg.get("min_depth_m", 0.08))
        depth_high = float(cfg.get("max_depth_m", 1.20))
        face_z = button.z_m if button is not None else None
        if face_z is None:
            face_z = exposed_panel_depth(depth_m, loc.corners_px, _cover_polygons(touch))
        chest_z = servo_depth_m(face_z, nominal_depth_m, depth_low, depth_high)
        side_z = servo_depth_m(side.get("depth_m"), nominal_depth_m, depth_low, depth_high)
        standoff_px = float(cfg.get("side_standoff_px", 36.0))
        jaw_clear = (
            visual is not None
            and bool(loc.corners_px)
            and _outside_distance(np.asarray([visual.u, visual.v], dtype=np.float64), loc.corners_px) > standoff_px
        )
        dual = compose_dual_view_ibvs(
            None if button is None else (button.u, button.v),
            None if visual is None else (visual.u, visual.v),
            self.K,
            side_measured=bool(side.get("measured")),
            side_contact=bool(side.get("contact")),
            side_gap_px=side.get("gap_px"),
            side_blocked=bool(side.get("blocked")) and not jaw_clear,
            pixel_tol=float(cfg.get("pixel_tol", 8.0)),
            nominal_depth_m=chest_z,
            side_depth_m=side_z,
            gain=float(cfg.get("lambda_xyz", 0.35)),
            max_step_m=float(cfg.get("max_delta_m", 0.006)),
            approach_step_m=float(cfg.get("side_approach_step_m", 0.004)),
            standoff_px=standoff_px,
            contact_px=float(cfg.get("side_contact_px", 14.0)),
        )
        screen_poly = _screen_polygon(touch)
        if screen_poly is not None:
            with self._lock:
                self._last_chest_bgr = bgr.copy()
                self._last_screen_polygon = screen_poly
        vis = bgr
        if loc.corners_px or loc.buttons:
            vis = draw_localization(bgr, loc)
        vis = draw_pose_buttons(
            vis,
            {} if touch is None else touch.points,
            skip={name: (point.u, point.v) for name, point in loc.buttons.items()},
        )
        jaw = [mask for mask in keep_side_masks([] if touch is None else touch.masks, gripper_min=0.3) if mask.name != "panel"]
        if jaw:
            vis = draw_seg_masks(vis, jaw)
        if raw_tip is not None:
            auto_px = (int(round(raw_tip.u)), int(round(raw_tip.v)))
            tip_label = {"pose": "POSE TIP", "axis": "AXIS TIP", "face": "JAW TIP"}.get(raw_tip.source, "VIS TIP")
            cv2.drawMarker(vis, auto_px, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.putText(vis, tip_label, (auto_px[0] + 8, auto_px[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if virtual_tip_in_front(virtual):
            fk_px = (int(round(virtual.u)), int(round(virtual.v)))
            cv2.drawMarker(vis, fk_px, (255, 220, 0), cv2.MARKER_CROSS, 26, 2)
            fk_label = "FK TIP"
            if virtual.xyz_m is not None and button is not None and button.xyz_m is not None:
                gap_m = math.sqrt(
                    sum((float(button.xyz_m[i]) - float(virtual.xyz_m[i])) ** 2 for i in range(3))
                )
                fk_label = f"FK TIP {gap_m * 1000:.0f}mm"
            cv2.putText(vis, fk_label, (fk_px[0] + 10, fk_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2)
        if tracker.get("mode") == "manual_flow" and tracked is not None:
            man_px = (int(round(tracked.u)), int(round(tracked.v)))
            cv2.drawMarker(vis, man_px, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(vis, "CLICK TIP", (man_px[0] + 10, man_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if locked_button is not None:
            lock_px = (int(round(locked_button.u)), int(round(locked_button.v)))
            cv2.circle(vis, lock_px, 10, (0, 180, 255), 2)
            cv2.putText(
                vis,
                "POSE LOCK",
                (lock_px[0] + 12, lock_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),
                1,
            )
        if visual is not None and button is not None:
            tip_px = (int(round(visual.u)), int(round(visual.v)))
            target_px = (int(round(button.u)), int(round(button.v)))
            cv2.line(vis, tip_px, target_px, (255, 180, 0), 2, cv2.LINE_AA)
        rpy = panel_rpy_deg(loc.corners_px or [], self.K, self.panel.width_m, self.panel.height_m)
        droll, dpitch, dyaw = (0.0, 0.0, 0.0) if rpy is None else rpy
        target_uv = None if button is None else [round(button.u, 1), round(button.v, 1)]
        tip_uv = None if visual is None else [round(visual.u, 1), round(visual.v, 1)]
        if locked_button is not None and locked_button.xyz_m is not None:
            target_xyz = [float(value) for value in locked_button.xyz_m]
        else:
            target_xyz = None if cmd.button_xyz_m is None else [float(v) for v in cmd.button_xyz_m]
            if target_xyz is None and button is not None and button.xyz_m is not None:
                target_xyz = [float(v) for v in button.xyz_m]
        tip_xyz = None if cmd.tip_xyz_m is None else [float(v) for v in cmd.tip_xyz_m]
        virtual_xyz = None if virtual is None or virtual.xyz_m is None else [float(v) for v in virtual.xyz_m]
        virtual_uv = (
            [round(virtual.u, 1), round(virtual.v, 1)] if virtual_tip_in_front(virtual) else None
        )
        pbvs_tip_xyz, pbvs_tip_source = _pbvs_tip_observation(visual, depth_m, self.K, virtual_xyz)
        target_torso = (
            None
            if target_xyz is None
            else list(camera_xyz_to_torso(target_xyz, torso_T_camera=self.torso_T_camera))
        )
        chest_pose = fresh_chest_pose(
            touch,
            depth_m,
            self.K,
            self.torso_T_camera,
            target_name=str(self.ctrl.target_name),
            conf_min=self._point_conf(),
        )
        chest_pose_torso = None if chest_pose is None else chest_pose.get("torso_m")
        chest_pose_uv = None if chest_pose is None else chest_pose.get("uv")
        tip_torso = (
            None
            if pbvs_tip_xyz is None
            else list(camera_xyz_to_torso(pbvs_tip_xyz, torso_T_camera=self.torso_T_camera))
        )
        pbvs_error = (
            None
            if target_xyz is None or pbvs_tip_xyz is None
            else [target_xyz[i] - pbvs_tip_xyz[i] for i in range(3)]
        )
        pbvs_error_torso = (
            None
            if target_torso is None or tip_torso is None
            else [target_torso[i] - tip_torso[i] for i in range(3)]
        )
        payload = {
            "phase": dual.phase,
            "target": cmd.target,
            "target_uv": target_uv,
            "target_source": "" if button is None else str(button.source),
            "tip_uv": tip_uv,
            "tip_visible": tracker.get("mode") == "manual_flow" and tracked is not None,
            "tip_raw_uv": None if raw_tip is None else [round(raw_tip.u, 1), round(raw_tip.v, 1)],
            "tip_tracker": tracker,
            "tip_source": "" if visual is None else visual.source,
            "virtual_tip_uv": virtual_uv,
            "virtual_tip_xyz_m": None if virtual_xyz is None else [round(v, 4) for v in virtual_xyz],
            "target_xyz_m": None if target_xyz is None else [round(v, 4) for v in target_xyz],
            "tip_xyz_m": None if tip_xyz is None else [round(v, 4) for v in tip_xyz],
            "target_torso_m": None if target_torso is None else [round(v, 4) for v in target_torso],
            "chest_pose_button_torso_m": chest_pose_torso,
            "chest_pose_button_uv": chest_pose_uv,
            "chest_gripper_uv": _chest_gripper_uv(touch),
            "tip_torso_m": None if tip_torso is None else [round(v, 4) for v in tip_torso],
            "pbvs_error_cam_m": None if pbvs_error is None else [round(v, 4) for v in pbvs_error],
            "pbvs_error_torso_m": (
                None if pbvs_error_torso is None else [round(v, 4) for v in pbvs_error_torso]
            ),
            "pbvs_tip_source": pbvs_tip_source,
            "depth_available": depth_m is not None,
            "chest_servo_depth_m": round(chest_z, 3),
            "side_servo_depth_m": round(side_z, 3),
            "dx": float(dual.dx),
            "dy": float(dual.dy),
            "dz": float(dual.dz),
            "droll": float(droll),
            "dpitch": float(dpitch),
            "dyaw": float(dyaw),
            "panel_visible": bool(loc.visible),
            "panel_partial": bool(loc.corners_px) and not loc.visible,
            "using_last_target": False,
            "message": dual.message,
            "side_measured": bool(side.get("measured")),
            "side_gap_px": side.get("gap_px"),
            "side_message": str(side.get("message") or ""),
            "left_contact": bool(side.get("contact")),
            "handeye": dict(self.eye_to_hand),
            "eye_to_hand": dict(self.eye_to_hand),
            "eye_in_hand": dict(self.eye_in_hand),
            "buttons_torso_m": self._buttons_torso(loc),
            "wrist_serial": WRIST_SERIAL,
            "left_serial": LEFT_SERIAL,
            "chest_serial": CHEST_SERIAL,
            "updated_at": time.time(),
        }
        if locked_button is not None:
            payload["using_last_target"] = True
            payload["target_source"] = "pose_locked"
        else:
            good = snapshot_good_target(payload, now)
            if good is not None:
                self._last_good_target = good
            payload = hold_last_target(payload, self._last_good_target, now)
            if not loc.visible and dual.phase != "BACKOFF":
                payload["dx"] = 0.0
                payload["dy"] = 0.0
                payload["dz"] = 0.0
                payload["using_last_target"] = False
                payload["phase"] = dual.phase
                payload["message"] = dual.message
        with self._lock:
            payload["wrist_panel_visible"] = bool(self._wrist_seen)
            payload["wrist_panel_partial"] = False
            payload["wrist_target_uv"] = (
                None if self._wrist_target_uv is None else list(self._wrist_target_uv)
            )
            payload["wrist_tip_uv"] = (
                None if self._wrist_tip_uv is None else list(self._wrist_tip_uv)
            )
            payload["wrist_updated_at"] = float(self._wrist_updated_at)
            payload["left_contact"] = bool(self._side.get("contact"))
            payload["wrist_contact"] = bool(self._side.get("wrist_contact"))
            payload["wrist_gap_px"] = self._side.get("wrist_gap_px")
            payload["wrist_message"] = str(self._side.get("wrist_message") or "")
            payload["side_measured"] = bool(self._side.get("measured"))
            payload["side_gap_px"] = self._side.get("gap_px")
            self._telemetry = payload
        lines = [
            f"{payload['phase']}  {cmd.target}",
            _preview_line(str(payload.get("message") or "")),
            f"depth chest {chest_z:.2f}m side {side_z:.2f}m",
        ]
        if loc.corners_px and not loc.visible:
            lines.append("panel partial (incomplete buttons still drawn)")
        if locked_button is not None:
            lines.append("pose lock")
        if buttons_frozen:
            lines.append("保持上一帧姿态点")
        vis = _hud(vis, lines, max_width=int(0.52 * vis.shape[1]))
        vis = _hud(
            vis,
            _model_status_lines(touch),
            origin=(int(0.56 * vis.shape[1]), 8),
            max_width=int(0.42 * vis.shape[1]),
        )
        self._stash_view(
            "chest",
            ("chest", loc, touch, tuple(lines), tuple(_model_status_lines(touch))),
        )
        self._last_chest_vis = vis
        return vis

    def _note_wrist_contact(self, bgr: np.ndarray, touch, tip, seg_corners, now: float) -> str:
        """Right-wrist image: the jaw against a button. A button is already on the panel."""
        tip_uv = None if tip is None else (float(tip.u), float(tip.v))
        contact_px = float(self.ctrl.cfg.get("side_contact_px", 14.0))
        depth = read_depth(SIDE_DEPTH_PATH)
        if depth is not None and depth.shape[:2] != bgr.shape[:2]:
            depth = None
        button = select_wrist_button(touch, self._wrist_button_conf(), self.ctrl.target_name, tip_uv)
        if button is not None:
            _name, button_u, button_v = button
            button_uv = (button_u, button_v)
            if tip_uv is None:
                gap_px = None
                pixel_contact = False
                measured = False
            else:
                gap_px = math.hypot(tip_uv[0] - button_u, tip_uv[1] - button_v)
                pixel_contact = gap_px <= contact_px
                measured = True
            depth_gap_m = _point_depth_gap_m(depth, tip_uv, button_uv)
            pressed = wrist_depth_confirms_contact(pixel_contact, depth_gap_m)
            confirmed, self._wrist_contact_since = _debounce_side_contact(
                measured,
                pressed,
                self._wrist_contact_since,
                now,
            )
            tip_name = "指尖" if self.ignore_gripper else "夹爪尖"
            if tip_uv is None:
                message = f"右腕已看到按键，还没看到{tip_name}"
            elif pressed and confirmed:
                message = f"右腕相机：{tip_name}已碰到面板"
            elif pressed:
                message = "右腕接触确认中"
            elif pixel_contact and depth_gap_m is not None:
                message = (
                    f"右腕像素间隙 {float(gap_px or 0):.0f}px，"
                    f"深度还差 {depth_gap_m * 1000:.0f}mm，尚未接触"
                )
            elif pixel_contact:
                message = f"右腕已看到按键，像素间隙 {float(gap_px or 0):.0f}px，没有深度，不能判定接触"
            else:
                message = f"右腕已看到按键，尖端还差 {float(gap_px or 0):.0f}px"
            with self._lock:
                self._side["wrist_contact"] = confirmed
                self._side["wrist_gap_px"] = gap_px
                self._side["wrist_message"] = message
            return message
        gap = estimate_side_gap(
            bgr,
            model=self.panel,
            contact_px=contact_px,
            panel_corners=seg_corners,
            tip_uv=tip_uv,
        )
        depth_gap_m = _wrist_depth_gap_m(depth, tip_uv, seg_corners, touch)
        pressed = wrist_depth_confirms_contact(bool(gap.contact), depth_gap_m)
        confirmed, self._wrist_contact_since = _debounce_side_contact(
            gap.measured,
            pressed,
            self._wrist_contact_since,
            now,
        )
        if pressed and confirmed:
            message = "右腕相机：夹爪已碰到面板"
        elif pressed:
            message = "右腕接触确认中"
        elif gap.contact and depth_gap_m is not None:
            message = (
                f"右腕像素间隙 {float(gap.gap_px or 0):.0f}px，"
                f"深度还差 {depth_gap_m * 1000:.0f}mm，尚未接触"
            )
        elif gap.contact:
            message = f"右腕像素间隙 {float(gap.gap_px or 0):.0f}px，没有深度，不能判定接触"
        elif gap.measured and gap.gap_px is not None:
            message = f"右腕间隙 {float(gap.gap_px):.0f}px"
        else:
            message = "右腕相机还没看到按键"
        with self._lock:
            self._side["wrist_contact"] = confirmed
            self._side["wrist_gap_px"] = gap.gap_px
            self._side["wrist_message"] = message
        return message

    def _annotate_right_wrist(self, bgr: np.ndarray) -> np.ndarray:
        """Right-wrist Eye-in-Hand view used for close PBVS and contact."""
        now = time.monotonic()
        self._last_wrist_ts = now
        self._last_wrist_bgr = bgr.copy()
        try:
            armed = bool(self._side_ready_fn())
        except Exception:
            armed = False
        touch = self._touch("wrist", bgr, self.wrist_detector)
        vis, seg_corners, tip = self._paint_side(bgr, touch)
        self._note_wrist_pbvs(bgr, touch, tip)
        with self._lock:
            yellow = None if self._wrist_yellow is None else dict(self._wrist_yellow)
            yellow_message = self._wrist_yellow_message
        if yellow is not None:
            if vis is bgr:
                vis = bgr.copy()
            mark = (int(round(float(yellow["u"]))), int(round(float(yellow["v"]))))
            cv2.circle(vis, mark, 16, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, mark, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.putText(
                vis,
                "YELLOW",
                (mark[0] + 12, max(16, mark[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        wrist_buttons = select_wrist_button(
            touch,
            self._wrist_button_conf(),
            self.ctrl.target_name,
            None if tip is None else (float(tip.u), float(tip.v)),
        )
        if self._wrist_red_bar:
            if vis is bgr:
                vis = bgr.copy()
            cv2.putText(
                vis,
                "RED",
                (8, max(18, vis.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        wrist_message = self._note_wrist_contact(bgr, touch, tip, seg_corners, now)
        if not armed:
            self._side_contact_since = None
            with self._lock:
                wrist_contact = bool(self._side.get("wrist_contact"))
                wrist_gap = self._side.get("wrist_gap_px")
                self._side = {
                    "measured": False,
                    "contact": False,
                    "gap_px": None,
                    "blocked": False,
                    "depth_m": None,
                    "message": "侧视未抬起，只显示检测",
                    "wrist_contact": wrist_contact,
                    "wrist_gap_px": wrist_gap,
                    "wrist_message": wrist_message,
                }
                self._telemetry = {
                    **self._telemetry,
                    "wrist_panel_visible": seg_corners is not None or wrist_buttons is not None,
                    "wrist_panel_partial": False,
                    "wrist_target_uv": (
                        None if self._wrist_target_uv is None else list(self._wrist_target_uv)
                    ),
                    "wrist_tip_uv": (
                        None if self._wrist_tip_uv is None else list(self._wrist_tip_uv)
                    ),
                    "wrist_updated_at": float(self._wrist_updated_at),
                    "left_contact": False,
                    "wrist_contact": wrist_contact,
                    "wrist_gap_px": wrist_gap,
                    "wrist_message": wrist_message,
                    "side_measured": False,
                    "side_gap_px": None,
                    "left_updated_at": time.time(),
                }
            unarmed_lines = [wrist_message]
            finger_text = self.finger_tip_status().get("finger_tip_text") or ""
            if finger_text:
                unarmed_lines.append(finger_text)
            if yellow_message:
                unarmed_lines.append(yellow_message)
            vis = _hud(vis, unarmed_lines, max_width=int(0.52 * vis.shape[1]))
            vis = _hud(
                vis,
                _model_status_lines(touch, seg=False),
                origin=(int(0.56 * vis.shape[1]), 8),
                max_width=int(0.42 * vis.shape[1]),
            )
            self._stash_wrist_preview(seg_corners, tip, wrist_buttons, yellow, unarmed_lines, touch)
            self._last_wrist_vis = vis
            return vis
        side_depth = read_depth(SIDE_DEPTH_PATH)
        if side_depth is not None and side_depth.shape[:2] != bgr.shape[:2]:
            side_depth = None
        gap = estimate_side_gap(
            bgr,
            model=self.panel,
            contact_px=float(self.ctrl.cfg.get("side_contact_px", 14.0)),
            panel_corners=_seg_panel_corners(touch, bgr.shape, self._point_conf()),
            tip_uv=None if tip is None else (float(tip.u), float(tip.v)),
        )
        contact_confirmed, self._side_contact_since = _debounce_side_contact(
            gap.measured,
            gap.contact,
            self._side_contact_since,
            now,
        )
        side_message = (
            gap.message
            if not gap.contact or contact_confirmed
            else "侧视接触确认中"
        )
        with self._lock:
            self._side = {
                "measured": gap.measured,
                "contact": contact_confirmed,
                "gap_px": gap.gap_px,
                "blocked": gap.blocked,
                "target_uv": gap.target_uv,
                "depth_m": _side_range_m(side_depth, gap.tip_uv, gap.target_uv),
                "message": side_message,
                "wrist_contact": bool(self._side.get("wrist_contact")),
                "wrist_gap_px": self._side.get("wrist_gap_px"),
                "wrist_message": wrist_message,
            }
        if gap.target_uv is not None and gap.tip_uv is not None:
            tip_px = (int(round(gap.tip_uv[0])), int(round(gap.tip_uv[1])))
            goal_px = (int(round(gap.target_uv[0])), int(round(gap.target_uv[1])))
            cv2.line(vis, tip_px, goal_px, (255, 180, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, goal_px, (255, 180, 0), cv2.MARKER_CROSS, 16, 2)
        with self._lock:
            self._telemetry = {
                **self._telemetry,
                "wrist_panel_visible": seg_corners is not None or wrist_buttons is not None,
                "wrist_panel_partial": False,
                "wrist_target_uv": (
                    None if self._wrist_target_uv is None else list(self._wrist_target_uv)
                ),
                "wrist_tip_uv": (
                    None if self._wrist_tip_uv is None else list(self._wrist_tip_uv)
                ),
                "wrist_updated_at": float(self._wrist_updated_at),
                "left_contact": contact_confirmed,
                "wrist_contact": bool(self._side.get("wrist_contact")),
                "wrist_gap_px": self._side.get("wrist_gap_px"),
                "wrist_message": wrist_message,
                "side_measured": gap.measured,
                "side_gap_px": gap.gap_px,
                "side_message": side_message,
                "left_updated_at": time.time(),
            }
        lines = [
            _preview_line(side_message),
            "side z " + ("lost" if self._side.get("depth_m") is None else f"{float(self._side['depth_m']):.2f}m"),
        ]
        if yellow_message:
            lines.append(yellow_message)
        finger_text = self.finger_tip_status().get("finger_tip_text") or ""
        if finger_text:
            lines.append(finger_text)
        vis = _hud(vis, lines, max_width=int(0.52 * vis.shape[1]))
        vis = _hud(
            vis,
            _model_status_lines(touch, seg=False),
            origin=(int(0.56 * vis.shape[1]), 8),
            max_width=int(0.42 * vis.shape[1]),
        )
        self._stash_wrist_preview(seg_corners, tip, wrist_buttons, yellow, lines, touch)
        self._last_wrist_vis = vis
        return vis

    def _stash_wrist_preview(self, seg_corners, tip, wrist_buttons, yellow, lines, touch) -> None:
        del seg_corners, wrist_buttons
        if tip is None:
            tip_mark = None
        elif getattr(tip, "source", "") == "finger_lock":
            tip_mark = (float(tip.u), float(tip.v), "LOCK")
        elif getattr(tip, "source", "") == "finger":
            tip_mark = (float(tip.u), float(tip.v), "FINGER")
        elif getattr(tip, "source", "") == "axis":
            tip_mark = (float(tip.u), float(tip.v), "AXIS")
        else:
            tip_mark = (float(tip.u), float(tip.v), "TIP")
        yellow_mark = None if yellow is None else (float(yellow["u"]), float(yellow["v"]))
        self._stash_view(
            "d435i",
            (
                "wrist",
                touch,
                tip_mark,
                yellow_mark,
                tuple(str(line) for line in lines if line),
                tuple(_model_status_lines(touch, seg=False)),
                bool(self._wrist_red_bar),
            ),
        )


def latch_wrist_finger_tip(
    locked: CameraPoint | None,
    live: CameraPoint | None,
    *,
    recapture: bool,
) -> tuple[CameraPoint | None, CameraPoint | None, bool]:
    """Keep one wrist FingerTip only while the operator has turned locking on.

    The camera and the hand move relative to each other, so the default is
    live detection. Returns (tip used for approach, stored lock, recapture
    still waiting).
    """
    if recapture:
        if live is None:
            return locked, locked, True
        stored = _copy_finger_tip(live)
        return stored, stored, False
    if locked is None and live is not None:
        stored = _copy_finger_tip(live)
        return stored, stored, False
    if locked is not None:
        return locked, locked, False
    return live, None, False


def _copy_finger_tip(point: CameraPoint) -> CameraPoint:
    xyz = None if point.xyz_m is None else tuple(float(value) for value in point.xyz_m)
    return CameraPoint(
        u=float(point.u),
        v=float(point.v),
        z_m=None if point.z_m is None else float(point.z_m),
        xyz_m=xyz,
        source="finger_lock",
        confidence=float(point.confidence),
    )


def _pbvs_tip_observation(visual, depth_m, K, virtual_xyz):
    """Camera-frame tip for PBVS. Real jaw depth wins; FK is only the fallback."""
    if visual is not None and depth_m is not None:
        z_m = sample_depth(depth_m, visual.u, visual.v, radius=10, percentile=5.0)
        if z_m is not None:
            return [float(v) for v in K.backproject(visual.u, visual.v, z_m)], "visual_depth"
    if virtual_xyz is None:
        return None, ""
    return [float(v) for v in virtual_xyz], "fk"


def _pbvs_tip_xyz(visual, depth_m, K, virtual_xyz):
    """Compatibility wrapper used by older callers/tests."""
    return _pbvs_tip_observation(visual, depth_m, K, virtual_xyz)[0]


def _side_range_m(depth_m, tip_uv, target_uv) -> float | None:
    """Range from the left camera to the jaw. The panel edge covers holes in the black jaw."""
    if depth_m is None:
        return None
    for uv in (tip_uv, target_uv):
        if uv is None:
            continue
        z = sample_depth(depth_m, float(uv[0]), float(uv[1]), radius=12, percentile=50.0)
        if z is not None:
            return z
    return None


def _debounce_side_contact(
    measured: bool,
    contact: bool,
    since: float | None,
    now: float,
    confirm_s: float = 0.30,
) -> tuple[bool, float | None]:
    """A single bad side frame must not reverse or stop the right arm."""
    if not measured or not contact:
        return False, None
    started = float(now) if since is None else float(since)
    return float(now) - started >= float(confirm_s), started


def _side_status(touch, seg_corners, tip) -> str:
    if touch is None:
        return "side yolo off"
    masks = keep_side_masks(touch.masks)
    parts = [f"{mask.name} {mask.conf:.2f}" for mask in masks]
    if not parts and seg_corners is not None:
        parts.append("panel")
    if tip is not None:
        parts.append("tip axis" if tip.source == "axis" else f"tip {tip.confidence:.2f}")
    return "side " + (" ".join(parts) if parts else "no gripper")


def _model_status_lines(touch, *, seg: bool = True) -> list[str]:
    """Keep segmentation and pose diagnostics in their own top-right panel."""
    if touch is None:
        if not seg:
            return ["POSE unavailable"]
        return ["SEG unavailable", "POSE unavailable"]
    best_masks: dict[str, float] = {}
    for mask in touch.masks:
        best_masks[mask.name] = max(best_masks.get(mask.name, 0.0), float(mask.conf))
    if touch.panel_corners:
        best_masks["panel"] = max(best_masks.get("panel", 0.0), float(touch.panel_conf))
    seg_text = " ".join(f"{name}:{best_masks[name]:.2f}" for name in ("panel", "gripper", "screen") if name in best_masks)
    pose_names = {
        "gripper_tip": "tip",
        "button_1": "b1",
        "button_2": "b2",
        "button_3": "b3",
        "FingerTip": "finger",
    }
    pose = " ".join(
        f"{pose_names[name]}:{float(point.conf):.2f}"
        for name, point in touch.points.items()
        if name in pose_names
    )
    if not seg:
        return [f"POSE {pose or 'none'}"]
    return [f"SEG {seg_text or 'none'}", f"POSE {pose or 'none'}"]


def _chest_gripper_uv(touch) -> list[float] | None:
    """Lowest image point of the chest gripper mask. That edge has to meet the buttons."""
    gripper = next(
        (
            mask
            for mask in keep_side_masks([] if touch is None else touch.masks, gripper_min=0.3)
            if mask.name == "gripper"
        ),
        None,
    )
    if gripper is None:
        return None
    poly = np.asarray(gripper.polygon, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None
    index = int(np.argmax(poly[:, 1]))
    return [round(float(poly[index, 0]), 1), round(float(poly[index, 1]), 1)]


def _cover_polygons(touch):
    """Gripper and screen hide the face. Button depth comes from the panel that is still visible."""
    if touch is None:
        return []
    return [mask.polygon for mask in keep_side_masks(touch.masks) if mask.name in {"gripper", "screen"}]


def _seg_panel_corners(touch, shape, conf_min: float):
    if touch is None or not touch.panel_corners or float(touch.panel_conf) < float(conf_min):
        return None
    quad = np.asarray(touch.panel_corners, dtype=np.float32).reshape(-1, 2)
    if quad.shape[0] != 4:
        return None
    height, width = int(shape[0]), int(shape[1])
    box_w = float(quad[:, 0].max() - quad[:, 0].min())
    box_h = float(quad[:, 1].max() - quad[:, 1].min())
    area = box_w * box_h
    if box_w < 40 or box_h < 20 or not (0.03 * width * height <= area <= 0.95 * width * height):
        return None
    return [(float(x), float(y)) for x, y in quad]


def _chest_raw_tip(touch, panel_corners, conf_min: float, finger: bool = False):
    """Tip is the trained pose keypoint. The gripper region is the seg mask.

    Revo2 uses FingerTip and does not fall back to the Dex1 jaw mask.

    A pose point that lands on the buttons while the jaw mask is somewhere
    else is not the tip; use the mask edge that faces the panel. When the
    mask is missing this frame, the pose keypoint still stands. Color is not
    a tip.
    """
    if finger:
        tip = pose_finger_tip(touch, conf_min)
        if tip is None:
            tip = pose_finger_tip(touch, 0.15)
        return tip
    gripper = next(
        (mask for mask in keep_side_masks([] if touch is None else touch.masks, gripper_min=0.3) if mask.name == "gripper"),
        None,
    )
    tip = pose_tip(touch, conf_min)
    if tip is None:
        tip = pose_tip(touch, 0.15)
    if tip is not None and gripper is not None and not tip_on_mask(tip, gripper, margin_px=100.0):
        tip = None
    if tip is not None:
        return tip
    if gripper is None:
        return None
    return gripper_toward_panel_tip(gripper, panel_corners)


def _on_panel_border(point, panel_corners, limit_px: float = 40.0) -> bool:
    """The closed jaw overlaps the green quad. A point on that border is still the jaw."""
    if not panel_corners:
        return False
    polygon = np.asarray(panel_corners, dtype=np.float32).reshape(-1, 1, 2)
    signed = float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True))
    return -4.0 <= signed <= float(limit_px)


def _chest_visual_tip(raw, tracked, manual: bool, panel_corners) -> CameraPoint | None:
    """Real image tip only. FK stays on the overlay and never closes the loop."""
    if raw is not None and getattr(raw, "source", "") in {"pose", "finger", "axis", "face"}:
        return raw
    chosen = None
    source = ""
    if manual and tracked is not None and (
        not point_inside_panel((tracked.u, tracked.v), panel_corners)
        or _on_panel_border((tracked.u, tracked.v), panel_corners)
    ):
        chosen = tracked
        source = "chest_click"
    elif raw is not None and (
        not point_inside_panel((raw.u, raw.v), panel_corners)
        or _on_panel_border((raw.u, raw.v), panel_corners)
    ):
        chosen = raw
        source = "chest_visual"
    if chosen is None:
        return None
    return CameraPoint(
        u=chosen.u,
        v=chosen.v,
        z_m=chosen.z_m,
        xyz_m=chosen.xyz_m,
        source=source,
        confidence=chosen.confidence,
    )


_PREVIEW_EXACT = {
    "等待胸视看到面板": "wait: chest has not seen the panel",
    "胸视没有真实尖端，FK 不驱动": "no visual tip; FK does not drive",
    "看不到夹爪，右臂保持，左臂找角度": "gripper not seen; right arm holds, left arm seeks an angle",
    "胸口没看到夹爪，侧视尖端过近，先退开": "chest lost the jaw; side tip is close, backing off",
    "胸视 IBVS：对准左右/上下": "chest IBVS: align left/right and up/down",
    "胸视对准左右/上下，侧视间隙同时前进": "chest aligns left/down; side gap moves forward",
    "侧视过近，先退开一小段": "side too close; backing off",
    "左手侧视：已接触按键": "side view: contact",
    "胸视已对准，等待左手侧视间隙": "aligned; waiting for side gap",
    "侧视未看到面板": "side: panel not in view",
    "侧视未看到面板，先退开": "side: no panel; backing off",
    "侧视未看到右夹爪": "side: gripper not in view",
    "侧视接触": "side: contact",
    "侧视接触确认中": "side: confirming contact",
    "侧视未抬起，不检测夹爪": "side idle until Raise; gripper search off",
    "右腕已看到按键，还没看到夹爪尖": "wrist: button seen, tip missing",
    "右腕相机还没看到按键": "wrist: button not in view",
    "指尖实时检测": "finger live",
    "指尖锁定是关的，正在用每帧检测": "finger live",
    "等待重新检测指尖": "waiting for next fingertip",
    "等待第一次指尖": "waiting for fingertip",
    "右腕相机：夹爪已碰到面板": "wrist: jaw on the button",
    "右腕接触确认中": "wrist: confirming contact",
}


def _preview_line(text: str) -> str:
    """OpenCV Hershey fonts cannot draw Chinese; those glyphs become '?'."""
    raw = str(text or "")
    mapped = _PREVIEW_EXACT.get(raw)
    if mapped:
        return mapped
    if raw.startswith("右腕已看到按键"):
        tail = "".join(ch for ch in raw if ch.isdigit())
        if "没有深度" in raw:
            return f"wrist: button seen, gap {tail}px, no depth"
        if tail:
            return f"wrist: button seen, tip gap {tail}px"
        return "wrist: button seen"
    if raw.startswith("侧视间隙") or raw.startswith("侧视伺服"):
        tail = "".join(ch for ch in raw if ch.isdigit())
        return f"side IBVS gap {tail}px" if tail else "side IBVS"
    if raw.startswith("侧视 IBVS"):
        tail = "".join(ch for ch in raw if ch.isdigit())
        extra = f" {tail}px" if tail else ""
        return f"side IBVS{extra}; chest +Z"
    if raw.isascii():
        return raw
    kept = " ".join("".join(ch if ord(ch) < 128 else " " for ch in raw).split())
    return kept or "see status text"


def _wrap_text(text: str, font: int, scale: float, thick: int, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    rows: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thick)
        if tw <= max_w:
            current = trial
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


def _hud(
    bgr: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (8, 8),
    max_width: int | None = None,
) -> np.ndarray:
    vis = bgr.copy()
    height, width = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    x = max(4, min(int(origin[0]), width - 8))
    y0 = max(4, int(origin[1]))
    max_w = max(40, min(width - x - 4, int(max_width or (width - x - 4))))
    chunks: list[str] = []
    for text in lines:
        text = _preview_line(text)
        if not text:
            continue
        chunks.extend(_wrap_text(text, font, scale, thick, max_w))
    if not chunks:
        return vis
    metrics = [cv2.getTextSize(chunk, font, scale, thick)[0] for chunk in chunks]
    line_step = max(th for _tw, th in metrics) + 6
    shown = min(len(chunks), max(1, (height - y0 - 5) // line_step))
    box_w = min(max_w, max(tw for tw, _th in metrics[:shown])) + 8
    box_h = shown * line_step + 4
    overlay = vis.copy()
    cv2.rectangle(
        overlay,
        (max(0, x - 4), max(0, y0 - 4)),
        (min(width - 1, x + box_w), min(height - 1, y0 + box_h)),
        (8, 8, 8),
        -1,
    )
    vis = cv2.addWeighted(overlay, 0.72, vis, 0.28, 0)
    y = y0
    for chunk, (_tw, th) in zip(chunks[:shown], metrics[:shown]):
        y += th + 6
        cv2.putText(vis, chunk, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(vis, chunk, (x, y), font, scale, (230, 240, 255), thick, cv2.LINE_AA)
    return vis
