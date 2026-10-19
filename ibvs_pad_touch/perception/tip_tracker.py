from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ibvs_pad_touch.types import CameraPoint


class StableTipTracker:
    """Confirm automatic detections and track a user-seeded tip with LK flow."""

    def __init__(
        self,
        *,
        confirm_frames: int = 3,
        max_jump_px: float = 45.0,
        max_flow_error: float = 30.0,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_jump_px = float(max_jump_px)
        self.max_flow_error = float(max_flow_error)
        self._confirmed: Optional[np.ndarray] = None
        self._pending: Optional[np.ndarray] = None
        self._pending_count = 0
        self._previous_gray: Optional[np.ndarray] = None
        self._manual = False
        self._last_reason = "uninitialized"

    def seed(self, u: float, v: float) -> None:
        self._confirmed = np.asarray([float(u), float(v)], dtype=np.float32)
        self._pending = None
        self._pending_count = 0
        self._previous_gray = None
        self._manual = True
        self._last_reason = "manual_seed"

    def clear(self) -> None:
        self._confirmed = None
        self._pending = None
        self._pending_count = 0
        self._previous_gray = None
        self._manual = False
        self._last_reason = "cleared"

    def status(self) -> dict:
        return {
            "mode": "manual_flow" if self._manual else "auto_stable",
            "confirmed": self._confirmed is not None,
            "reason": self._last_reason,
            "pending_frames": self._pending_count,
        }

    def update(
        self,
        bgr: np.ndarray,
        candidate: CameraPoint | None,
    ) -> CameraPoint | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detected = (
            None
            if candidate is None
            else np.asarray([candidate.u, candidate.v], dtype=np.float32)
        )
        if self._manual:
            point = self._update_manual(gray, detected)
        else:
            point = self._update_auto(detected)
        self._previous_gray = gray
        if point is None:
            return None
        return CameraPoint(
            u=float(point[0]),
            v=float(point[1]),
            source="dex1_tip_tracked",
            confidence=1.0,
        )

    def _update_manual(
        self,
        gray: np.ndarray,
        detected: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if self._confirmed is None:
            self._last_reason = "manual_seed_missing"
            return None
        if self._previous_gray is None:
            self._last_reason = "manual_seed_confirmed"
            return self._confirmed.copy()
        old = self._confirmed.reshape(1, 1, 2)
        new, status, error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            old,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        if new is None or status is None or not bool(status.reshape(-1)[0]):
            self._last_reason = "flow_lost"
            return None
        tracked = new.reshape(-1, 2)[0]
        flow_error = float("inf") if error is None else float(error.reshape(-1)[0])
        jump = float(np.linalg.norm(tracked - self._confirmed))
        h, w = gray.shape[:2]
        if (
            flow_error > self.max_flow_error
            or jump > self.max_jump_px
            or not (2.0 <= tracked[0] < w - 2.0 and 2.0 <= tracked[1] < h - 2.0)
        ):
            self._last_reason = "flow_rejected"
            return None
        if detected is not None and float(np.linalg.norm(detected - tracked)) <= 0.5 * self.max_jump_px:
            tracked = 0.75 * tracked + 0.25 * detected
        self._confirmed = tracked.astype(np.float32)
        self._last_reason = "flow_ok"
        return self._confirmed.copy()

    def _update_auto(self, detected: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if detected is None:
            self._pending = None
            self._pending_count = 0
            self._last_reason = "auto_lost"
            return None
        if self._confirmed is not None:
            jump = float(np.linalg.norm(detected - self._confirmed))
            if jump > self.max_jump_px:
                self._confirmed = None
                self._pending = detected
                self._pending_count = 1
                self._last_reason = "auto_jump_rejected"
                return None
            self._confirmed = (0.65 * self._confirmed + 0.35 * detected).astype(np.float32)
            self._last_reason = "auto_ok"
            return self._confirmed.copy()
        if self._pending is None or float(np.linalg.norm(detected - self._pending)) > 0.5 * self.max_jump_px:
            self._pending = detected
            self._pending_count = 1
            self._last_reason = "auto_pending"
            return None
        count = self._pending_count + 1
        self._pending = ((self._pending * self._pending_count + detected) / count).astype(np.float32)
        self._pending_count = count
        if count < self.confirm_frames:
            self._last_reason = "auto_pending"
            return None
        self._confirmed = self._pending.copy()
        self._last_reason = "auto_confirmed"
        return self._confirmed.copy()
