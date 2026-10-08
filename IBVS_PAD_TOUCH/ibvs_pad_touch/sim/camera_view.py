from __future__ import annotations

import cv2
import numpy as np

from ibvs_pad_touch.panel.model import PanelModel, iter_named_rects
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization, Observation


def project_point(xyz_w: np.ndarray, camera_w: np.ndarray, K: Intrinsics) -> tuple[float, float, float]:
    """Panel frame: X right, Y down, Z toward the room. Chest camera looks toward -Z."""
    xyz_w = np.asarray(xyz_w, dtype=np.float64)
    camera_w = np.asarray(camera_w, dtype=np.float64)
    xyz_c = np.array(
        [
            xyz_w[0] - camera_w[0],
            xyz_w[1] - camera_w[1],
            camera_w[2] - xyz_w[2],
        ],
        dtype=np.float64,
    )
    z = float(xyz_c[2])
    if z <= 1e-6:
        raise ValueError("point is behind the camera")
    u = K.fx * (xyz_c[0] / z) + K.cx
    v = K.fy * (xyz_c[1] / z) + K.cy
    return float(u), float(v), z


class ChestScene:
    """Fixed chest camera. The closed Dex1 tip moves; the panel does not."""

    def __init__(
        self,
        model: PanelModel,
        K: Intrinsics,
        camera_w: np.ndarray | None = None,
        tip_w: np.ndarray | None = None,
    ):
        self.model = model
        self.K = K
        self.camera_w = (
            np.array([0.15, 0.065, 0.42], dtype=np.float64)
            if camera_w is None
            else np.asarray(camera_w, dtype=np.float64).copy()
        )
        self.tip_w = (
            np.array([0.20, 0.02, 0.12], dtype=np.float64)
            if tip_w is None
            else np.asarray(tip_w, dtype=np.float64).copy()
        )

    def _observe_xyz(self, xyz_w: np.ndarray, source: str) -> CameraPoint:
        u, v, z = project_point(xyz_w, self.camera_w, self.K)
        return CameraPoint(u=u, v=v, z_m=z, xyz_m=self.K.backproject(u, v, z), source=source)

    def corners_px(self) -> list[tuple[float, float]]:
        return [
            project_point(np.array([x, y, 0.0]), self.camera_w, self.K)[:2]
            for x, y in self.model.canonical_corners_m()
        ]

    def localize(self) -> Localization:
        buttons = {
            button.name: self._observe_xyz(np.array(button.rect.to_xyz(0.0)), "scale")
            for button in self.model.poke_buttons()
        }
        return Localization(visible=True, corners_px=self.corners_px(), buttons=buttons, message="ok")

    def observe(self, target: int | str, include_tip: bool = True) -> Observation:
        loc = self.localize()
        name = self.model.button(target).name
        button = loc.button(name)
        tip = self._observe_xyz(self.tip_w, "tip") if include_tip else None
        return Observation(
            corners_px=loc.corners_px,
            button_uv=None if button is None else button.uv,
            tip=tip,
            localization=loc,
        )

    def apply_delta(self, dx: float, dy: float, dz: float) -> None:
        # Camera frame: X right, Y down, Z forward (toward the panel = -world Z).
        self.tip_w[0] += float(dx)
        self.tip_w[1] += float(dy)
        self.tip_w[2] -= float(dz)

    def render_bgr(self, highlight: int | str | None = None, draw_tip: bool = True) -> np.ndarray:
        image = np.full((self.K.height, self.K.width, 3), 236, dtype=np.uint8)
        try:
            corners = np.array(self.corners_px(), dtype=np.int32)
        except ValueError:
            return image
        cv2.fillPoly(image, [corners], (226, 220, 214))
        cv2.polylines(image, [corners], True, (80, 180, 80), 2, cv2.LINE_AA)
        target_name = None if highlight is None else self.model.button(highlight).name
        for name, rect, meta in iter_named_rects(self.model):
            try:
                pts = np.array(
                    [project_point(np.array([x, y, 0.0]), self.camera_w, self.K)[:2] for x, y in rect.corners_m()],
                    dtype=np.int32,
                )
            except ValueError:
                continue
            if name == "display":
                cv2.fillPoly(image, [pts], (28, 28, 28))
                continue
            color = (0, 210, 235) if str(meta.get("color")) == "yellow" else (40, 40, 200) if str(meta.get("color")) == "red" else (90, 90, 90)
            if name == target_name:
                color = (0, 140, 255)
            cv2.fillPoly(image, [pts], color)
            if meta.get("poke"):
                cv2.polylines(image, [pts], True, (30, 30, 30), 1, cv2.LINE_AA)
                label = str(meta.get("label", name))
                cv2.putText(image, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        if draw_tip:
            try:
                tip = self._observe_xyz(self.tip_w, "tip")
                cv2.circle(image, (int(tip.u), int(tip.v)), 6, (0, 0, 255), -1)
                cv2.circle(image, (int(tip.u), int(tip.v)), 9, (20, 20, 20), 1)
            except ValueError:
                pass
        return image


# Keep the old name for any leftover imports.
SyntheticEye = ChestScene
