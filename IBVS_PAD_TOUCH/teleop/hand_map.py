"""Map a fingertip seen by a webcam onto a small H2 torso step.

The camera looks at the operator. Image +u is right, +v is down. A larger
hand span means the hand moved toward the camera. That approach is robot +X
(toward the panel). Image right is robot +Y (operator left, robot left).
Image up is robot +Z.

World hand landmarks stay inside the palm, so they cannot see the arm
translate. The control point is the index tip in the image, and the
wrist-to-knuckle span supplies monocular distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Wrist to index-finger knuckle on an adult hand, used only to turn pixels into metres.
NOMINAL_SPAN_M = 0.10


@dataclass(frozen=True)
class TipSample:
    """Index tip in pixels, plus the wrist-to-index-knuckle length in pixels.

    depth_m is the D405 range at the fingertip. None falls back to the hand span.
    """

    u: float
    v: float
    span_px: float
    depth_m: float | None = None


def _measured_depth(sample: TipSample) -> float | None:
    depth = sample.depth_m
    if depth is None:
        return None
    try:
        value = float(depth)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.05 or value > 0.70:
        return None
    return value


def torso_from_tips(
    anchor: TipSample,
    sample: TipSample,
    *,
    scale: float,
    range_m: float,
    flip_y: bool = False,
) -> tuple[float, float, float]:
    """Robot metres for the motion from anchor to sample. scale 0.1 is a ten-to-one reduction."""
    span = max(float(sample.span_px), 1.0)
    meters_per_px = NOMINAL_SPAN_M / span
    du = float(sample.u) - float(anchor.u)
    dv = float(sample.v) - float(anchor.v)
    anchor_depth = _measured_depth(anchor)
    sample_depth = _measured_depth(sample)
    if anchor_depth is not None and sample_depth is not None:
        closer_m = anchor_depth - sample_depth
    else:
        closer_m = float(range_m) * (1.0 - float(anchor.span_px) / span)
    robot_x = closer_m * float(scale)
    robot_y = du * meters_per_px * float(scale)
    robot_z = -dv * meters_per_px * float(scale)
    if flip_y:
        robot_y = -robot_y
    return (robot_x, robot_y, robot_z)


def clamp_step(
    delta: tuple[float, float, float],
    *,
    min_step_m: float,
    max_step_m: float,
) -> tuple[float, float, float] | None:
    """Drop axes smaller than the deadband. Cap each axis so one jog cannot exceed the arm limit."""
    command: list[float] = []
    moved = False
    for value in delta:
        number = float(value)
        if not math.isfinite(number) or abs(number) < float(min_step_m):
            command.append(0.0)
            continue
        command.append(max(-float(max_step_m), min(float(max_step_m), number)))
        moved = True
    if not moved:
        return None
    return (command[0], command[1], command[2])


class TipTeleop:
    """Hold still until engaged. Each command consumes only the metres it sent."""

    def __init__(
        self,
        *,
        scale: float = 0.1,
        range_m: float = 0.45,
        min_step_m: float = 0.002,
        max_step_m: float = 0.008,
        max_jump_px: float = 120.0,
        flip_y: bool = False,
    ) -> None:
        self.scale = float(scale)
        self.range_m = float(range_m)
        self.min_step_m = float(min_step_m)
        self.max_step_m = float(max_step_m)
        self.max_jump_px = float(max_jump_px)
        self.flip_y = bool(flip_y)
        self.anchor: TipSample | None = None
        self._committed_from: TipSample | None = None

    @property
    def engaged(self) -> bool:
        return self.anchor is not None

    def engage(self, sample: TipSample) -> None:
        self.anchor = sample

    def release(self) -> None:
        self.anchor = None

    def offer(self, sample: TipSample) -> tuple[float, float, float] | None:
        """One jog, or None when the motion is still inside the deadband."""
        anchor = self.anchor
        if anchor is None or sample.span_px < 5.0 or anchor.span_px < 5.0:
            return None
        jump = math.hypot(sample.u - anchor.u, sample.v - anchor.v)
        if jump > self.max_jump_px:
            self.anchor = sample
            return None
        raw = torso_from_tips(
            anchor,
            sample,
            scale=self.scale,
            range_m=self.range_m,
            flip_y=self.flip_y,
        )
        command = clamp_step(raw, min_step_m=self.min_step_m, max_step_m=self.max_step_m)
        if command is None:
            return None
        self._committed_from = anchor
        self.anchor = _anchor_after(
            anchor,
            sample,
            raw,
            command,
            scale=self.scale,
            range_m=self.range_m,
        )
        return command

    def undo(self) -> None:
        """Put back the anchor when a jog was computed but not accepted."""
        if self._committed_from is not None:
            self.anchor = self._committed_from
            self._committed_from = None


def _anchor_after(
    anchor: TipSample,
    sample: TipSample,
    raw: tuple[float, float, float],
    command: tuple[float, float, float],
    *,
    scale: float,
    range_m: float,
) -> TipSample:
    """Leave the unsent remainder between the new anchor and the current tip."""
    remain = tuple(raw[index] - command[index] for index in range(3))
    span = max(float(sample.span_px), 1.0)
    meters_per_px = NOMINAL_SPAN_M / span
    gain = meters_per_px * float(scale)
    du = 0.0 if gain == 0.0 else remain[1] / gain
    dv = 0.0 if gain == 0.0 else -remain[2] / gain
    sample_depth = _measured_depth(sample)
    if _measured_depth(anchor) is not None and sample_depth is not None and scale != 0.0:
        depth = sample_depth + remain[0] / float(scale)
        anchor_span = span
    else:
        depth_gain = float(range_m) * float(scale)
        ratio = 0.0 if depth_gain == 0.0 else remain[0] / depth_gain
        anchor_span = span * (1.0 - ratio)
        depth = sample.depth_m
    if anchor_span < 5.0:
        anchor_span = span
    return TipSample(float(sample.u) - du, float(sample.v) - dv, float(anchor_span), depth)
