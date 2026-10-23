from __future__ import annotations


class ReachGate:
    """Latch poke-reached after the feature stays inside xy/z gates."""

    def __init__(self, xy_tol_norm: float, z_tol_m: float, stable_frames: int, pixel_tol: float):
        self.xy_tol_norm = float(xy_tol_norm)
        self.z_tol_m = float(z_tol_m)
        self.stable_frames = max(1, int(stable_frames))
        self.pixel_tol = float(pixel_tol)
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def update(
        self,
        error_xy_norm: float,
        error_z_m: float,
        error_px: float,
        aligned: bool,
    ) -> bool:
        del aligned, error_px
        inside = error_xy_norm <= self.xy_tol_norm and abs(error_z_m) <= self.z_tol_m
        self._count = self._count + 1 if inside else 0
        return self._count >= self.stable_frames
