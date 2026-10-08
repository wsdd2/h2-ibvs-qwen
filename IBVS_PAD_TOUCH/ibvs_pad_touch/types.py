from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def normalize(self, u: float, v: float) -> tuple[float, float]:
        return (float(u) - self.cx) / self.fx, (float(v) - self.cy) / self.fy

    def denormalize(self, x: float, y: float) -> tuple[float, float]:
        return x * self.fx + self.cx, y * self.fy + self.cy

    def backproject(self, u: float, v: float, z_m: float) -> tuple[float, float, float]:
        x, y = self.normalize(u, v)
        z = float(z_m)
        return x * z, y * z, z


@dataclass
class CameraPoint:
    u: float
    v: float
    z_m: Optional[float] = None
    xyz_m: Optional[tuple[float, float, float]] = None
    source: str = ""
    confidence: float = 1.0

    @property
    def uv(self) -> tuple[float, float]:
        return self.u, self.v

    def with_depth(self, K: Intrinsics, z_m: float) -> "CameraPoint":
        return CameraPoint(
            u=self.u,
            v=self.v,
            z_m=float(z_m),
            xyz_m=K.backproject(self.u, self.v, z_m),
            source=self.source,
            confidence=self.confidence,
        )


@dataclass
class Feature:
    u: float
    v: float
    z_m: float
    x_norm: float
    y_norm: float
    confidence: float = 1.0

    @property
    def uv(self) -> tuple[float, float]:
        return self.u, self.v


@dataclass
class Localization:
    """Panel-first result. Buttons come from the scale model, then optional color refine."""

    visible: bool
    corners_px: Optional[list[tuple[float, float]]] = None
    buttons: dict[str, CameraPoint] = field(default_factory=dict)
    message: str = ""

    def button(self, name: str) -> Optional[CameraPoint]:
        return self.buttons.get(name)


@dataclass
class ServoCommand:
    dx: float
    dy: float
    dz: float
    reached: bool
    valid: bool
    target: str
    phase: str
    error_xy_norm: float
    error_z_m: float
    feature_uv: Optional[tuple[float, float]] = None
    tip_uv: Optional[tuple[float, float]] = None
    depth_m: Optional[float] = None
    button_xyz_m: Optional[tuple[float, float, float]] = None
    tip_xyz_m: Optional[tuple[float, float, float]] = None
    tip_source: str = ""
    message: str = ""

    @property
    def delta_xyz(self) -> tuple[float, float, float]:
        return self.dx, self.dy, self.dz

    def as_dict(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "reached": self.reached,
            "valid": self.valid,
            "target": self.target,
            "phase": self.phase,
            "error_xy_norm": self.error_xy_norm,
            "error_z_m": self.error_z_m,
            "feature_uv": self.feature_uv,
            "tip_uv": self.tip_uv,
            "depth_m": self.depth_m,
            "button_xyz_m": self.button_xyz_m,
            "tip_xyz_m": self.tip_xyz_m,
            "tip_source": self.tip_source,
            "message": self.message,
        }


@dataclass
class Observation:
    feature: Optional[Feature] = None
    corners_px: Optional[list[tuple[float, float]]] = None
    button_uv: Optional[tuple[float, float]] = None
    tip: Optional[CameraPoint] = None
    localization: Optional[Localization] = None
    lost: bool = False
    extras: dict = field(default_factory=dict)
