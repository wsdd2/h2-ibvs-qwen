"""Shadow observer for ViT-VS (arXiv:2503.04545).

The observer matches patch descriptors, estimates a camera velocity, and
records it. It never returns a robot command. ``applied`` and
``rotation_applied`` stay false so a caller cannot mistake the report for
something the wrist loop should execute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROTATIONS_DEG = (0, 90, 180, -90)


@dataclass(frozen=True)
class ShadowView:
    """Intrinsics of the image the descriptors were extracted from."""

    fx: float
    fy: float
    cx: float
    cy: float
    stride: float = 14.0


@dataclass(frozen=True)
class PatchMatch:
    desired_rc: tuple[int, int]
    current_rc: tuple[int, int]
    similarity: float
    cyclical_distance: float


@dataclass(frozen=True)
class VitShadowReport:
    """Diagnostic only. Both apply flags are fixed false."""

    applied: bool
    rotation_applied: bool
    recommended_rotation_deg: int
    correspondences: tuple[PatchMatch, ...]
    mean_similarity: float
    velocity_m_s: tuple[float, float, float, float, float, float] | None
    shadow_step_m: tuple[float, float, float] | None
    message: str

    def __post_init__(self) -> None:
        if self.applied or self.rotation_applied:
            raise ValueError("ViT shadow reports cannot be marked applied")


class VitShadowObserver:
    """Compute a ViT-VS step beside the live controller and keep it."""

    def __init__(
        self,
        *,
        k: int = 24,
        cutoff: float = 1.0,
        gain: float = 0.8,
        alpha: float = 0.8,
        damping: float = 1e-6,
        dt_s: float = 0.25,
        max_step_m: float = 0.004,
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.gain = float(gain)
        self.alpha = float(alpha)
        self.damping = float(damping)
        self.dt_s = float(dt_s)
        self.max_step_m = float(max_step_m)
        self._rng = np.random.default_rng(seed)
        self._velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._velocity = None

    def observe(
        self,
        desired: np.ndarray,
        current: np.ndarray,
        depths_m: np.ndarray,
        view: ShadowView,
        rotated_currents: dict[int, np.ndarray] | None = None,
    ) -> VitShadowReport:
        """Match ``current`` to ``desired`` and record an unused velocity.

        ``rotated_currents`` holds descriptors of the current image after it
        has been rotated and run through the same extractor. Those maps only
        vote for ``recommended_rotation_deg``. The velocity always comes from
        the unrotated ``current`` map.
        """
        desired_map = _as_descriptors(desired, "desired")
        current_map = _as_descriptors(current, "current")
        if desired_map.shape != current_map.shape:
            raise ValueError("desired and current descriptors must share a shape")
        depths = np.asarray(depths_m, dtype=np.float64)
        if depths.shape != current_map.shape[:2]:
            raise ValueError("depths must align with the current patch grid")
        recommendation = self._recommend_rotation(desired_map, current_map, rotated_currents)
        eligible = best_buddy_matches(desired_map, current_map, self.cutoff)
        chosen = select_matches(eligible, self.k, self._rng)
        usable = _with_depth(chosen, depths)
        if not usable:
            return VitShadowReport(
                applied=False,
                rotation_applied=False,
                recommended_rotation_deg=recommendation,
                correspondences=tuple(chosen),
                mean_similarity=_mean_similarity(chosen),
                velocity_m_s=None,
                shadow_step_m=None,
                message="no_usable_correspondence",
            )
        raw = _camera_velocity(usable, depths, view, self.gain, self.damping)
        smoothed = self._smooth(raw)
        step = _clip_norm(smoothed[:3] * self.dt_s, self.max_step_m)
        return VitShadowReport(
            applied=False,
            rotation_applied=False,
            recommended_rotation_deg=recommendation,
            correspondences=tuple(chosen),
            mean_similarity=_mean_similarity(chosen),
            velocity_m_s=tuple(float(value) for value in smoothed),
            shadow_step_m=(float(step[0]), float(step[1]), float(step[2])),
            message="shadow_only",
        )

    def _recommend_rotation(
        self,
        desired: np.ndarray,
        current: np.ndarray,
        rotated_currents: dict[int, np.ndarray] | None,
    ) -> int:
        candidates = {0: current}
        for angle, features in (rotated_currents or {}).items():
            if int(angle) not in ROTATIONS_DEG:
                raise ValueError(f"rotation {angle} is outside {ROTATIONS_DEG}")
            if int(angle) == 0:
                continue
            rotated = _as_descriptors(features, f"rotated_{angle}")
            if rotated.shape != desired.shape:
                raise ValueError("rotated descriptors must match the desired shape")
            candidates[int(angle)] = rotated
        scores = {
            angle: _rotation_score(desired, features, self.cutoff, self.k)
            for angle, features in candidates.items()
        }
        return max(scores, key=lambda angle: (scores[angle], -abs(int(angle))))

    def _smooth(self, raw: np.ndarray) -> np.ndarray:
        if self._velocity is None:
            smoothed = raw
        else:
            smoothed = self.alpha * raw + (1.0 - self.alpha) * self._velocity
        self._velocity = smoothed
        return smoothed


def bin_patch_features(descriptors: np.ndarray, beta: int) -> np.ndarray:
    """Average each patch with ``beta`` rings of neighbors. Shape is unchanged."""
    features = _as_descriptors(descriptors, "descriptors")
    rings = int(beta)
    if rings <= 0:
        return features.copy()
    padded = np.pad(features, ((rings, rings), (rings, rings), (0, 0)), mode="constant")
    height, width = features.shape[:2]
    total = 0.0
    accum = np.zeros_like(features)
    for row_shift in range(-rings, rings + 1):
        for col_shift in range(-rings, rings + 1):
            row_from = rings + row_shift
            col_from = rings + col_shift
            accum += padded[row_from : row_from + height, col_from : col_from + width]
            total += 1.0
    return accum / total


def best_buddy_matches(
    desired: np.ndarray,
    current: np.ndarray,
    cutoff: float,
) -> list[PatchMatch]:
    """Keep mutual nearest neighbors whose return distance is within ``cutoff``."""
    desired_map = _as_descriptors(desired, "desired")
    current_map = _as_descriptors(current, "current")
    if desired_map.shape != current_map.shape:
        raise ValueError("desired and current descriptors must share a shape")
    desired_flat = _normalize(desired_map.reshape(-1, desired_map.shape[-1]))
    current_flat = _normalize(current_map.reshape(-1, current_map.shape[-1]))
    desired_to_current = np.argmax(desired_flat @ current_flat.T, axis=1)
    current_to_desired = np.argmax(current_flat @ desired_flat.T, axis=1)
    height, width = desired_map.shape[:2]
    matches: list[PatchMatch] = []
    for desired_index, current_index in enumerate(desired_to_current):
        returned = int(current_to_desired[int(current_index)])
        desired_rc = divmod(int(desired_index), width)
        returned_rc = divmod(returned, width)
        distance = float(np.hypot(desired_rc[0] - returned_rc[0], desired_rc[1] - returned_rc[1]))
        if distance > float(cutoff):
            continue
        similarity = float(desired_flat[desired_index] @ current_flat[int(current_index)])
        current_rc = divmod(int(current_index), width)
        matches.append(
            PatchMatch(
                desired_rc=(int(desired_rc[0]), int(desired_rc[1])),
                current_rc=(int(current_rc[0]), int(current_rc[1])),
                similarity=similarity,
                cyclical_distance=distance,
            )
        )
    return matches


def select_matches(matches: list[PatchMatch], k: int, rng: np.random.Generator) -> list[PatchMatch]:
    """Draw up to ``k`` matches so features do not all sit on one patch."""
    if len(matches) <= k:
        return list(matches)
    ranked = sorted(
        matches,
        key=lambda item: (item.cyclical_distance, -item.similarity, item.desired_rc[0], item.desired_rc[1]),
    )
    picked = rng.choice(len(ranked), size=int(k), replace=False)
    return [ranked[int(index)] for index in sorted(int(index) for index in picked)]


def _rotation_score(desired: np.ndarray, current: np.ndarray, cutoff: float, k: int) -> float:
    matches = best_buddy_matches(desired, current, cutoff)
    if not matches:
        return -1.0
    ranked = sorted(matches, key=lambda item: (item.cyclical_distance, -item.similarity, item.desired_rc))
    return _mean_similarity(ranked[:k])


def _with_depth(matches: list[PatchMatch], depths: np.ndarray) -> list[PatchMatch]:
    usable = []
    for item in matches:
        depth = float(depths[item.current_rc])
        if np.isfinite(depth) and 0.02 < depth < 2.5:
            usable.append(item)
    return usable


def _camera_velocity(
    matches: list[PatchMatch],
    depths: np.ndarray,
    view: ShadowView,
    gain: float,
    damping: float,
) -> np.ndarray:
    rows = []
    errors = []
    for item in matches:
        current_xy = _normalized(*item.current_rc, view)
        desired_xy = _normalized(*item.desired_rc, view)
        depth = float(depths[item.current_rc])
        rows.append(_interaction_row(current_xy[0], current_xy[1], depth))
        errors.append([current_xy[0] - desired_xy[0], current_xy[1] - desired_xy[1]])
    interaction = np.vstack(rows)
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    gram = interaction.T @ interaction + float(damping) * np.eye(6)
    return -float(gain) * np.linalg.solve(gram, interaction.T @ error)


def _interaction_row(x: float, y: float, z_m: float) -> np.ndarray:
    z = float(z_m)
    return np.array(
        [
            [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
            [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
        ],
        dtype=np.float64,
    )


def _normalized(row: int, col: int, view: ShadowView) -> tuple[float, float]:
    u = (float(col) + 0.5) * float(view.stride)
    v = (float(row) + 0.5) * float(view.stride)
    return (u - float(view.cx)) / float(view.fx), (v - float(view.cy)) / float(view.fy)


def _mean_similarity(matches: list[PatchMatch] | tuple[PatchMatch, ...]) -> float:
    if not matches:
        return 0.0
    return float(np.mean([item.similarity for item in matches]))


def _as_descriptors(value: np.ndarray, name: str) -> np.ndarray:
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < 1 or features.shape[2] < 1:
        raise ValueError(f"{name} descriptors must have shape (rows, cols, channels)")
    return features


def _normalize(flat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    scaled = flat / np.maximum(norms, 1e-12)
    scaled[norms[:, 0] < 1e-12] = 0.0
    return scaled


def comparison_payload(
    actual_step: tuple[float, float, float] | list[float] | None,
    report: VitShadowReport | None,
    *,
    enabled: bool,
    reference: bool,
    extractor: str,
    message: str | None = None,
) -> dict[str, object]:
    """JSON-ready comparison. ``applied`` stays false."""
    actual = _step3(actual_step)
    shadow = None if report is None else _step3(report.shadow_step_m)
    delta = None if actual is None or shadow is None else tuple(shadow[i] - actual[i] for i in range(3))
    text = message
    if text is None and report is not None:
        text = "shadow 已算出步长，未发给手臂" if report.message == "shadow_only" else report.message
    if text is None:
        text = "对照未开始"
    if actual is None and shadow is not None:
        text = f"{text}；当前方案这一拍没有腕部步长"
    return {
        "applied": False,
        "rotation_applied": False,
        "enabled": bool(enabled),
        "reference": bool(reference),
        "extractor": extractor,
        "actual_step_m": _round_step(actual, 6),
        "shadow_step_m": _round_step(shadow, 6),
        "delta_step_m": _round_step(delta, 6),
        "actual_step_mm": _round_step(None if actual is None else tuple(v * 1000.0 for v in actual), 2),
        "shadow_step_mm": _round_step(None if shadow is None else tuple(v * 1000.0 for v in shadow), 2),
        "delta_step_mm": _round_step(None if delta is None else tuple(v * 1000.0 for v in delta), 2),
        "matches": 0 if report is None else len(report.correspondences),
        "mean_similarity": None if report is None else round(float(report.mean_similarity), 4),
        "recommended_rotation_deg": 0 if report is None else int(report.recommended_rotation_deg),
        "message": text,
    }


def _step3(step: tuple[float, float, float] | list[float] | None) -> tuple[float, float, float] | None:
    if step is None:
        return None
    values = tuple(float(value) for value in step)
    if len(values) != 3 or not all(np.isfinite(values)):
        return None
    return values


def _round_step(step: tuple[float, float, float] | None, digits: int) -> list[float] | None:
    if step is None:
        return None
    return [round(value, digits) for value in step]


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > float(limit) > 0.0:
        return vector * (float(limit) / norm)
    return vector
