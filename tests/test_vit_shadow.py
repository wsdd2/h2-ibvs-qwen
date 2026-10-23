from pathlib import Path

import numpy as np

from ibvs_pad_touch.servo.interaction import eye_in_hand_ibvs_step
from ibvs_pad_touch.servo.vit_shadow import (
    ShadowView,
    VitShadowObserver,
    VitShadowReport,
    best_buddy_matches,
    bin_patch_features,
    comparison_payload,
)
from ibvs_pad_touch.servo.vit_shadow_feed import ShadowMonitor


ROOT = Path(__file__).resolve().parents[1]
CONTROL_FILES = (
    "ibvs_pad_touch/web/arm_motion.py",
    "ibvs_pad_touch/web/live_ibvs.py",
    "ibvs_pad_touch/servo/controller.py",
    "ibvs_pad_touch/servo/interaction.py",
    "ibvs_pad_touch/servo/__init__.py",
)


def _one_hot(rows: int, cols: int) -> np.ndarray:
    features = np.zeros((rows, cols, rows * cols), dtype=np.float64)
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        features[row, col, index] = 1.0
    return features


def _view_for(features: np.ndarray) -> ShadowView:
    rows, cols = features.shape[:2]
    stride = 14.0
    return ShadowView(
        fx=200.0,
        fy=200.0,
        cx=0.5 * cols * stride,
        cy=0.5 * rows * stride,
        stride=stride,
    )


def test_control_path_does_not_reference_the_shadow_observer():
    for relative in CONTROL_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "vit_shadow" not in text
        assert "VitShadow" not in text


def test_shadow_report_cannot_be_marked_applied():
    try:
        VitShadowReport(
            applied=True,
            rotation_applied=False,
            recommended_rotation_deg=0,
            correspondences=(),
            mean_similarity=0.0,
            velocity_m_s=None,
            shadow_step_m=None,
            message="no",
        )
    except ValueError as exc:
        assert "applied" in str(exc)
    else:
        raise AssertionError("an applied shadow report was accepted")


def test_best_buddy_rejects_a_one_way_match():
    desired = np.zeros((1, 3, 2), dtype=np.float64)
    current = np.zeros((1, 3, 2), dtype=np.float64)
    desired[0, 0] = (1.0, 0.0)
    desired[0, 1] = (0.0, 1.0)
    desired[0, 2] = (0.0, 0.2)
    current[0, 0] = (0.8, 0.6)
    current[0, 1] = (0.0, 1.0)
    current[0, 2] = (1.0, 0.0)
    matches = best_buddy_matches(desired, current, cutoff=1.0)
    paired = {(item.desired_rc, item.current_rc) for item in matches}
    assert ((0, 0), (0, 0)) not in paired
    assert ((0, 1), (0, 1)) in paired


def test_identical_descriptors_record_a_zero_step_without_applying_it():
    features = _one_hot(4, 4)
    observer = VitShadowObserver(k=8, seed=1)
    report = observer.observe(features, features.copy(), np.full((4, 4), 0.20), _view_for(features))
    assert report.applied is False
    assert report.rotation_applied is False
    assert report.message == "shadow_only"
    assert report.shadow_step_m is not None
    assert np.linalg.norm(report.shadow_step_m) < 1e-9
    assert report.recommended_rotation_deg == 0


def test_rightward_feature_shift_is_recorded_and_not_sent_to_ibvs():
    desired = _one_hot(4, 6)
    current = np.zeros_like(desired)
    current[:, 1:] = desired[:, :-1]
    current[:, 0] = _one_hot(4, 6)[:, 0]
    # The first column above is not orthogonal to the desired basis. Replace it.
    current[:, 0, :] = 0.0
    current[:, 0, -1] = 1.0
    depths = np.full(desired.shape[:2], 0.20)
    classical_before = eye_in_hand_ibvs_step(
        (382.0, 240.0),
        (320.0, 240.0),
        0.20,
        0.20,
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    report = VitShadowObserver(k=12, seed=2).observe(desired, current, depths, _view_for(desired))
    classical_after = eye_in_hand_ibvs_step(
        (382.0, 240.0),
        (320.0, 240.0),
        0.20,
        0.20,
        fx=615.0,
        fy=615.0,
        cx=320.0,
        cy=240.0,
    )
    assert report.applied is False
    assert report.shadow_step_m is not None
    assert report.shadow_step_m[0] > 0.0
    assert abs(report.shadow_step_m[1]) < abs(report.shadow_step_m[0])
    assert classical_after["camera_step_m"] == classical_before["camera_step_m"]
    assert report.shadow_step_m != classical_before["camera_step_m"]


def test_rotation_recommendation_stays_diagnostic():
    desired = np.zeros((4, 4, 32), dtype=np.float64)
    current = np.zeros_like(desired)
    for index in range(16):
        row, col = divmod(index, 4)
        desired[row, col, index] = 1.0
        current[row, col, 16 + index] = 1.0
    rotated = {90: desired.copy(), 180: current.copy(), -90: current.copy()}
    report = VitShadowObserver(k=16, seed=3).observe(
        desired,
        current,
        np.full((4, 4), 0.20),
        _view_for(desired),
        rotated_currents=rotated,
    )
    assert report.recommended_rotation_deg == 90
    assert report.rotation_applied is False
    assert report.applied is False
    assert report.correspondences
    assert all(item.similarity == 0.0 for item in report.correspondences)
    assert all(item.current_rc == (0, 0) for item in report.correspondences)


def test_velocity_smoothing_uses_the_paper_update_and_still_does_not_apply():
    desired = _one_hot(4, 5)
    shifted = np.zeros_like(desired)
    shifted[:, 1:] = desired[:, :-1]
    shifted[:, 0, :] = 0.0
    shifted[:, 0, -1] = 1.0
    depths = np.full(desired.shape[:2], 0.25)
    view = _view_for(desired)
    observer = VitShadowObserver(k=8, alpha=0.8, seed=4)
    first = observer.observe(desired, shifted, depths, view)
    second = observer.observe(desired, desired.copy(), depths, view)
    assert first.velocity_m_s is not None
    assert second.velocity_m_s is not None
    assert second.applied is False
    expected = 0.2 * np.asarray(first.velocity_m_s)
    assert np.allclose(second.velocity_m_s, expected, atol=1e-9)


def test_missing_depth_produces_no_step():
    features = _one_hot(3, 3)
    report = VitShadowObserver(k=4, seed=5).observe(
        features,
        features.copy(),
        np.full((3, 3), np.nan),
        _view_for(features),
    )
    assert report.applied is False
    assert report.shadow_step_m is None
    assert report.velocity_m_s is None
    assert report.message == "no_usable_correspondence"


class _ShiftExtractor:
    name = "fake"

    def describe(self, bgr: np.ndarray) -> np.ndarray:
        base = np.zeros((4, 5, 20), dtype=np.float64)
        for index in range(20):
            row, col = divmod(index, 5)
            base[row, col, index] = 1.0
        if float(bgr.mean()) < 1.0:
            return base
        moved = np.zeros_like(base)
        moved[:, 1:] = base[:, :-1]
        moved[:, 0, -1] = 1.0
        return moved


def test_comparison_payload_reports_the_difference_without_applying_it():
    report = VitShadowObserver(k=4, seed=6).observe(
        _one_hot(3, 3),
        _one_hot(3, 3),
        np.full((3, 3), 0.2),
        _view_for(_one_hot(3, 3)),
    )
    payload = comparison_payload((0.001, 0.0, 0.0), report, enabled=True, reference=True, extractor="fake")
    assert payload["applied"] is False
    assert payload["rotation_applied"] is False
    assert payload["actual_step_mm"] == [1.0, 0.0, 0.0]
    assert payload["shadow_step_mm"] == [0.0, 0.0, 0.0]
    assert payload["delta_step_mm"] == [-1.0, 0.0, 0.0]


def test_monitor_shows_shadow_beside_the_live_step():
    current = {"frame": np.zeros((32, 40, 3), dtype=np.uint8)}
    command = {"step": (0.001, 0.0, 0.0)}
    monitor = ShadowMonitor(
        frame_fn=lambda: current["frame"],
        depth_fn=lambda: None,
        actual_fn=lambda: command["step"],
        intrinsics_fn=lambda: {
            "fx": 200.0,
            "fy": 200.0,
            "cx": 20.0,
            "cy": 16.0,
            "plane_z_m": 0.2,
            "depth_assumed": 1.0,
        },
        extractor=_ShiftExtractor(),
    )
    monitor.capture_reference(np.zeros((32, 40, 3), dtype=np.uint8))
    current["frame"] = np.full((32, 40, 3), 255, dtype=np.uint8)
    payload = monitor.refresh()
    assert payload["applied"] is False
    assert payload["actual_step_mm"][0] == 1.0
    assert payload["shadow_step_mm"][0] > 0.0
    assert payload["delta_step_mm"][0] != 0.0
    assert "标称深度" in str(payload["message"])
    assert command["step"] == (0.001, 0.0, 0.0)
    monitor.close()


def test_feature_binning_averages_a_neighborhood_without_changing_shape():
    features = np.zeros((7, 7, 1), dtype=np.float64)
    features[3, 3, 0] = 9.0
    binned = bin_patch_features(features, 1)
    assert binned.shape == features.shape
    assert binned[3, 3, 0] == 1.0
    assert binned[0, 0, 0] == 0.0
    assert np.allclose(bin_patch_features(np.ones((4, 4, 2)), 0), 1.0)
