"""Chest-camera pipeline: localize panel/buttons, then optional tip→button xyz."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.camera.d435i import D435iCamera, intrinsics_from_config
from ibvs_pad_touch.panel.render import draw_rect_on_image
from ibvs_pad_touch.platform import environment_name, gui_available
from ibvs_pad_touch.servo.controller import IBVSController
from ibvs_pad_touch.sim.camera_view import ChestScene
from ibvs_pad_touch.types import Intrinsics, Localization, ServoCommand


def resolve_show(requested: bool) -> bool:
    if not requested:
        return False
    if gui_available():
        return True
    print(f"[ibvs] env={environment_name()} has no display; skip --show, write --out only")
    return False


def maybe_show(title: str, frame, wait_ms: int = 20) -> bool:
    """Return True if the user asked to quit. Never crash on headless OpenCV."""
    try:
        cv2.imshow(title, frame)
        return cv2.waitKey(wait_ms) in (27, ord("q"))
    except cv2.error as exc:
        print(f"[ibvs] imshow unavailable: {exc}")
        return False


def overlay(bgr: np.ndarray, cmd: ServoCommand, loc: Localization | None = None) -> np.ndarray:
    vis = bgr.copy()
    corners = None if loc is None else loc.corners_px
    if corners:
        draw_rect_on_image(vis, corners, (80, 180, 80), "panel")
    if loc is not None:
        for name, point in loc.buttons.items():
            color = (0, 140, 255) if name == cmd.target else (0, 210, 235)
            cv2.circle(vis, (int(point.u), int(point.v)), 6, color, 2)
            cv2.putText(vis, name[-1], (int(point.u) + 6, int(point.v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if cmd.tip_uv:
        cv2.circle(vis, (int(cmd.tip_uv[0]), int(cmd.tip_uv[1])), 6, (0, 0, 255), -1)
        if cmd.feature_uv:
            cv2.arrowedLine(
                vis,
                (int(cmd.tip_uv[0]), int(cmd.tip_uv[1])),
                (int(cmd.feature_uv[0]), int(cmd.feature_uv[1])),
                (0, 80, 255),
                2,
                tipLength=0.15,
            )
    text = (
        f"{cmd.target} {cmd.phase}  d=({cmd.dx*1000:.1f},{cmd.dy*1000:.1f},{cmd.dz*1000:.1f})mm  "
        f"reached={int(cmd.reached)}  {cmd.message}"
    )
    cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def print_cmd(step: int, cmd: ServoCommand) -> None:
    print(
        f"[{step:04d}] {cmd.target:8s} {cmd.phase:10s}  "
        f"dx={cmd.dx:+.5f} dy={cmd.dy:+.5f} dz={cmd.dz:+.5f}  "
        f"xy={cmd.error_xy_norm:.4f} z={cmd.error_z_m:+.4f}  "
        f"reached={cmd.reached}  {cmd.message}"
    )


def run_synthetic(ctrl: IBVSController, target: str, steps: int, show: bool, out: Path | None, with_tip: bool) -> bool:
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    scene = ChestScene(ctrl.panel, K)
    reached = False
    last = None
    for i in range(steps):
        obs = scene.observe(target, include_tip=with_tip)
        cmd = ctrl.step(scene.render_bgr(target, draw_tip=with_tip), None, K, target=target, observation=obs)
        print_cmd(i, cmd)
        if with_tip:
            scene.apply_delta(cmd.dx, cmd.dy, cmd.dz)
        last = overlay(scene.render_bgr(target, draw_tip=with_tip), cmd, obs.localization)
        if show and maybe_show("IBVS_PAD_TOUCH", last, 20):
            break
        if cmd.reached:
            reached = True
            break
        if cmd.phase == "LOCALIZED" and not with_tip:
            break
    if out is not None and last is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), last)
    if show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return reached or cmd.phase == "LOCALIZED"


def run_realsense(ctrl: IBVSController, target: str, steps: int, show: bool) -> bool:
    cam = D435iCamera()
    cam.start()
    try:
        for i in range(steps):
            bgr, depth, K = cam.read()
            cmd = ctrl.step(bgr, depth, K, target=target)
            print_cmd(i, cmd)
            if show and maybe_show("IBVS_PAD_TOUCH", overlay(bgr, cmd, ctrl.last_localization), 1):
                break
            if cmd.phase == "LOCALIZED":
                return True
            time.sleep(1.0 / float(ctrl.cfg.get("control_hz", 20)))
    finally:
        cam.stop()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return False


def run_image(ctrl: IBVSController, target: str, image_path: Path, depth_path: Path | None, show: bool, out: Path | None) -> bool:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    depth = None if depth_path is None else np.load(depth_path)
    K = Intrinsics(fx=615.0, fy=615.0, cx=bgr.shape[1] / 2, cy=bgr.shape[0] / 2, width=bgr.shape[1], height=bgr.shape[0])
    try:
        cfg_k = intrinsics_from_config()
        if (cfg_k.width, cfg_k.height) == (bgr.shape[1], bgr.shape[0]):
            K = cfg_k
    except Exception:
        pass
    cmd = ctrl.step(bgr, depth, K, target=target)
    print_cmd(0, cmd)
    vis = overlay(bgr, cmd, ctrl.last_localization)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
    if show:
        maybe_show("IBVS_PAD_TOUCH", vis, 0)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return cmd.valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "realsense", "image"), default="synthetic")
    parser.add_argument("--target", default="button_1")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--depth", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Headless localize + short tip servo. For WSL/H2 CI.")
    parser.add_argument("--no-tip", action="store_true", help="Only localize panel/buttons, do not servo the tip.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ibvs_last.png")
    args = parser.parse_args()

    if args.smoke:
        args.source = "synthetic"
        args.no_tip = False
        args.steps = min(args.steps, 80)
        args.show = False
        print(f"[ibvs] smoke env={environment_name()}")

    ctrl = IBVSController.from_configs()
    ctrl.set_target(args.target)
    show = resolve_show(args.show)
    if args.source == "synthetic":
        ok = run_synthetic(ctrl, args.target, args.steps, show, args.out, with_tip=not args.no_tip)
    elif args.source == "realsense":
        ok = run_realsense(ctrl, args.target, args.steps, show)
    else:
        if args.image is None:
            parser.error("--image is required for source=image")
        ok = run_image(ctrl, args.target, args.image, args.depth, show, args.out)
    print(f"ok={ok} phase={ctrl.phase}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
