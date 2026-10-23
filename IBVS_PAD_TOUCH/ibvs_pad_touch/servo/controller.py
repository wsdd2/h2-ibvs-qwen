from __future__ import annotations

from typing import Optional

import numpy as np

from ibvs_pad_touch.config import load_ibvs_config
from ibvs_pad_touch.panel.model import PanelModel, load_panel
from ibvs_pad_touch.perception.features import sample_depth
from ibvs_pad_touch.perception.gripper_tip import detect_dex1_tip
from ibvs_pad_touch.perception.pipeline import PanelButtonLocalizer
from ibvs_pad_touch.servo.delta import (
    clip_delta,
    gate_xy_then_depth,
    image_xy_delta,
    tip_to_button_delta,
)
from ibvs_pad_touch.servo.reach import ReachGate
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization, Observation, ServoCommand


class IBVSController:
    """Chest-camera panel touch.

    1. Localize the panel, then buttons by scale (+ yellow refine). No motion yet.
    2. Drive camera X/Y (left-right / up-down) onto the switch. No depth yet.
    3. Then drive camera Z onto the switch panel, never the wall behind it.
    4. No hand-eye. Motion control consumes that xyz as robot Cartesian.
    """

    def __init__(
        self,
        panel: PanelModel | None = None,
        ibvs_cfg: dict | None = None,
        localizer: PanelButtonLocalizer | None = None,
    ):
        self.panel = panel or load_panel()
        self.cfg = ibvs_cfg or load_ibvs_config()
        self.localizer = localizer or PanelButtonLocalizer(self.panel)
        self.gate = ReachGate(
            xy_tol_norm=float(self.cfg["align_xy_tol_norm"]),
            z_tol_m=float(self.cfg["contact_z_tol_m"]),
            stable_frames=int(self.cfg["stable_frames"]),
            pixel_tol=float(self.cfg["pixel_tol"]),
        )
        self.phase = "SEEK_PANEL"
        self.target_name = "button_1"
        self.last_localization: Localization | None = None

    @classmethod
    def from_configs(cls, panel_path=None, ibvs_path=None) -> "IBVSController":
        return cls(panel=load_panel(panel_path), ibvs_cfg=load_ibvs_config(ibvs_path))

    def set_target(self, key: int | str) -> str:
        button = self.panel.button(key)
        if not button.poke:
            raise ValueError(f"{button.name} is not a stage-1 poke target")
        if button.name != self.target_name:
            self.gate.reset()
            self.phase = "SEEK_PANEL"
        self.target_name = button.name
        return self.target_name

    def lost_command(self, message: str) -> ServoCommand:
        self.gate.reset()
        self.phase = "LOST"
        return ServoCommand(
            dx=0.0,
            dy=0.0,
            dz=0.0,
            reached=False,
            valid=False,
            target=self.target_name,
            phase="LOST",
            error_xy_norm=float("inf"),
            error_z_m=float("inf"),
            message=message,
        )

    def localize(
        self,
        bgr: np.ndarray,
        depth_m: np.ndarray | None,
        K: Intrinsics,
        corners_px: Optional[list[tuple[float, float]]] = None,
    ) -> Localization:
        loc = self.localizer.localize(bgr, depth_m, K, corners_px=corners_px)
        self.last_localization = loc
        return loc

    def command_from_points(self, button: CameraPoint, tip: CameraPoint | None, K: Intrinsics) -> ServoCommand:
        if tip is None:
            self.phase = "LOCALIZED"
            self.gate.reset()
            return ServoCommand(
                dx=0.0,
                dy=0.0,
                dz=0.0,
                reached=False,
                valid=True,
                target=self.target_name,
                phase="LOCALIZED",
                error_xy_norm=float("inf"),
                error_z_m=float("inf"),
                feature_uv=button.uv,
                depth_m=button.z_m,
                button_xyz_m=button.xyz_m,
                message="panel_and_button_ok_need_tip",
                tip_source="",
            )

        raw, reason = tip_to_button_delta(button, tip, K)
        if reason in {"button_depth_missing", "tip_depth_missing"}:
            self.phase = "LOCALIZED"
            self.gate.reset()
            px_err = float(np.hypot(button.u - tip.u, button.v - tip.v))
            if px_err <= float(self.cfg["pixel_tol"]):
                delta = np.zeros(3, dtype=np.float64)
                message = "xy_aligned_need_depth"
            else:
                raw_2d = image_xy_delta(
                    button,
                    tip,
                    K,
                    float(self.cfg.get("image_servo_nominal_depth_m", 0.60)),
                )
                delta = clip_delta(
                    raw_2d,
                    float(self.cfg["max_delta_m"]),
                    gain=float(self.cfg.get("lambda_xyz", 0.35)),
                )
                self.phase = "ALIGN_XY"
                message = "align_switch_2d_no_depth"
            return ServoCommand(
                dx=float(delta[0]),
                dy=float(delta[1]),
                dz=0.0,
                reached=False,
                valid=True,
                target=self.target_name,
                phase=self.phase,
                error_xy_norm=float(np.hypot(delta[0], delta[1])),
                error_z_m=float("inf"),
                feature_uv=button.uv,
                tip_uv=tip.uv,
                depth_m=button.z_m,
                button_xyz_m=button.xyz_m,
                tip_xyz_m=tip.xyz_m,
                tip_source=tip.source,
                message=message,
            )
        if reason != "ok":
            return self.lost_command(reason)

        xy = float(np.hypot(raw[0], raw[1]))
        z_err = float(raw[2])
        px_err = float(np.hypot(button.u - tip.u, button.v - tip.v))
        xy_norm = xy
        aligned = px_err <= float(self.cfg["pixel_tol"]) and xy <= float(self.cfg["align_xy_tol_norm"])
        gain = float(self.cfg.get("lambda_xyz", 0.35))
        xy_tol = float(self.cfg["align_xy_tol_norm"])
        reached = self.gate.update(xy_norm, z_err, px_err, aligned)
        if reached:
            self.phase = "HOLD"
            delta = np.zeros(3)
            message = "hold"
        else:
            dx, dy, dz, stage = gate_xy_then_depth(
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
                xy_norm,
                xy_tol,
            )
            # Clip the active axes only. Depth must not start until the switch is in front.
            delta = clip_delta(np.array([dx, dy, dz], dtype=np.float64), float(self.cfg["max_delta_m"]), gain=gain)
            self.phase = stage
            message = "align_switch" if stage == "ALIGN_XY" else "panel_face"
        return ServoCommand(
            dx=float(delta[0]),
            dy=float(delta[1]),
            dz=float(delta[2]),
            reached=reached,
            valid=True,
            target=self.target_name,
            phase=self.phase,
            error_xy_norm=xy_norm,
            error_z_m=z_err,
            feature_uv=button.uv,
            tip_uv=tip.uv,
            depth_m=button.z_m,
            button_xyz_m=button.xyz_m,
            tip_xyz_m=tip.xyz_m,
            tip_source=tip.source,
            message=message,
        )

    def step(
        self,
        bgr: np.ndarray,
        depth_m: np.ndarray | None,
        K: Intrinsics,
        target: int | str | None = None,
        corners_px: Optional[list[tuple[float, float]]] = None,
        tip: CameraPoint | None = None,
        observation: Observation | None = None,
        detect_tip: bool = True,
    ) -> ServoCommand:
        if target is not None:
            self.set_target(target)
        if observation is not None and observation.lost:
            return self.lost_command(str((observation.extras or {}).get("reason", "lost")))

        loc = observation.localization if observation and observation.localization else None
        if loc is None:
            loc = self.localize(bgr, depth_m, K, corners_px=corners_px)
        if not loc.visible:
            return self.lost_command(loc.message or "panel_not_found")

        button = loc.button(self.target_name)
        if button is None and observation is not None and observation.feature is not None:
            feat = observation.feature
            button = CameraPoint(u=feat.u, v=feat.v, z_m=feat.z_m, xyz_m=K.backproject(feat.u, feat.v, feat.z_m), source="observation")
        if button is None:
            return self.lost_command("button_not_localized")

        tip_obs = tip
        if tip_obs is None and observation is not None:
            tip_obs = observation.tip
        if tip_obs is None and detect_tip:
            tip_obs = detect_dex1_tip(bgr, panel_corners=loc.corners_px)
        if tip_obs is not None and tip_obs.xyz_m is None and depth_m is not None:
            # Black Dex1 jaws often contain depth holes. In a larger patch,
            # choose the near foreground percentile instead of panel/background.
            z_tip = sample_depth(
                depth_m,
                tip_obs.u,
                tip_obs.v,
                radius=10,
                percentile=5.0,
            )
            if z_tip is not None:
                tip_obs = tip_obs.with_depth(K, z_tip)
        return self.command_from_points(button, tip_obs, K)
