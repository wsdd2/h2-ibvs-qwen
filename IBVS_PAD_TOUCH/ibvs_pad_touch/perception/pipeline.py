from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ibvs_pad_touch.panel.model import PanelModel
from ibvs_pad_touch.perception.features import exposed_panel_depth, sample_depth
from ibvs_pad_touch.perception.localizer import (
    PanelLocalizer,
    project_xy,
    refine_yellow_centroid,
)
from ibvs_pad_touch.perception.yellow_buttons import assign_poke_buttons
from ibvs_pad_touch.types import CameraPoint, Intrinsics, Localization


class PanelButtonLocalizer:
    """Chest-camera pipeline: find the panel, then map buttons by the tape-measure model.

    Color blobs only refine a scale-model prior. They do not invent button identity.
    """

    def __init__(self, model: PanelModel, refine_color: bool = True):
        self.model = model
        self.localizer = PanelLocalizer(model)
        self.refine_color = refine_color

    def localize(
        self,
        bgr: np.ndarray,
        depth_m: np.ndarray | None,
        K: Intrinsics,
        corners_px: Optional[list[tuple[float, float]]] = None,
    ) -> Localization:
        if corners_px is not None:
            pose = self.localizer.from_corners(corners_px)
        else:
            pose = self.localizer.detect(bgr)
        yellow = assign_poke_buttons(bgr) if self.refine_color else None
        if pose is None and yellow is None:
            return Localization(visible=False, message="panel_not_found")

        plane_z = None
        if pose is not None and depth_m is not None:
            du, dv = display_center_uv(self.localizer, pose)
            plane_z = sample_depth(depth_m, du, dv, radius=8, percentile=50.0)
        pnp_xyz = (
            panel_button_xyz_from_corners(self.model, pose["corners_px"], K)
            if pose is not None
            else {}
        )

        buttons: dict[str, CameraPoint] = {}
        used_scale_only = False
        for button in self.model.poke_buttons():
            source = "scale"
            u = v = None
            if pose is not None:
                u, v = self.localizer.button_uv(pose, button.name)
                if self.refine_color and yellow is not None:
                    refined = refine_yellow_centroid(bgr, (u, v), half_wh=(22, 14))
                    if refined is not None:
                        jump = ((refined[0] - u) ** 2 + (refined[1] - v) ** 2) ** 0.5
                        if jump <= 18.0:
                            u, v = refined
                            source = "scale+yellow"
            if yellow is not None and button.name in yellow:
                yu, yv = yellow[button.name]
                if pose is None:
                    u, v = yu, yv
                    source = "yellow"
                elif ((yu - float(u)) ** 2 + (yv - float(v)) ** 2) ** 0.5 <= 24.0:
                    u, v = yu, yv
                    source = "scale+yellow"
            if u is None or v is None:
                continue
            point = CameraPoint(u=u, v=v, source=source)
            if source == "scale":
                used_scale_only = True
                z_m = plane_z
            else:
                z_m = sample_depth(depth_m, u, v) if depth_m is not None else None
            if z_m is None and depth_m is not None:
                z_m = sample_depth(depth_m, u, v)
            pnp_point = pnp_xyz.get(button.name)
            depth_disagrees = (
                z_m is not None
                and pnp_point is not None
                and abs(float(z_m) - float(pnp_point[2])) > 0.15
            )
            if z_m is not None and not depth_disagrees:
                point = point.with_depth(K, z_m)
            elif pnp_point is not None:
                xyz = pnp_point
                point = CameraPoint(
                    u=point.u,
                    v=point.v,
                    z_m=float(xyz[2]),
                    xyz_m=xyz,
                    source=f"{point.source}+pnp",
                    confidence=point.confidence,
                )
            buttons[button.name] = point
        if pose is not None and len(buttons) >= 3:
            return Localization(
                visible=True,
                corners_px=pose["corners_px"],
                buttons=buttons,
                message="scale_occluded" if used_scale_only and yellow is None else "ok",
            )
        if len(buttons) < 3:
            return Localization(
                visible=False,
                corners_px=None if pose is None else pose["corners_px"],
                buttons=buttons,
                message="buttons_incomplete",
            )
        return Localization(
            visible=True,
            corners_px=None if pose is None else pose["corners_px"],
            buttons=buttons,
            message="ok",
        )


def align_buttons_to_exposed_panel(
    loc: Localization,
    depth_m: np.ndarray | None,
    K: Intrinsics,
    exclude_polygons: Sequence[Sequence[Sequence[float]]] = (),
) -> Localization:
    """Keep button pixels even when the gripper covers them. Their depth is the visible face."""
    if depth_m is None or not loc.buttons or not loc.corners_px:
        return loc
    face_z = exposed_panel_depth(depth_m, loc.corners_px, exclude_polygons)
    if face_z is None and exclude_polygons:
        face_z = exposed_panel_depth(depth_m, loc.corners_px)
    if face_z is None:
        return loc
    buttons = {}
    for name, point in loc.buttons.items():
        if (
            "pnp" in str(point.source)
            and point.z_m is not None
            and abs(float(face_z) - float(point.z_m)) > 0.15
        ):
            buttons[name] = point
        else:
            buttons[name] = point.with_depth(K, face_z)
    return Localization(
        visible=loc.visible,
        corners_px=loc.corners_px,
        buttons=buttons,
        message=loc.message,
    )


def display_center_uv(localizer: PanelLocalizer, pose: dict) -> tuple[float, float]:
    display = localizer.model.display
    if display is None:
        corners = pose["corners_px"]
        return (
            0.25 * sum(pt[0] for pt in corners),
            0.25 * sum(pt[1] for pt in corners),
        )
    return project_xy((display.cx, display.cy), pose["H_meter"])


def panel_button_xyz_from_corners(
    model: PanelModel,
    corners_px: Sequence[Sequence[float]],
    K: Intrinsics,
) -> dict[str, tuple[float, float, float]]:
    """Monocular planar PnP fallback when the chest D435 exposes color only."""
    obj = np.asarray(
        [[x, y, 0.0] for x, y in model.canonical_corners_m()],
        dtype=np.float64,
    )
    image = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    camera = np.asarray(
        [[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        image,
        camera,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return {}
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera, None)
    rms = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image) ** 2, axis=1)))
    )
    if not np.isfinite(rms) or rms > 25.0:
        return {}
    result: dict[str, tuple[float, float, float]] = {}
    for button in model.poke_buttons():
        point = rotation @ np.asarray(button.rect.to_xyz(0.0), dtype=np.float64) + tvec.reshape(3)
        if not np.all(np.isfinite(point)) or not (0.08 <= float(point[2]) <= 2.5):
            return {}
        result[button.name] = (float(point[0]), float(point[1]), float(point[2]))
    return result
