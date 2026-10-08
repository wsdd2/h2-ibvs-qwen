import cv2
import numpy as np

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.localizer import detect_lcd_corners, detect_panel_corners, panel_quad_acceptable
from ibvs_pad_touch.perception.pipeline import PanelButtonLocalizer, panel_button_xyz_from_corners
from ibvs_pad_touch.perception.yellow_buttons import assign_poke_buttons
from ibvs_pad_touch.sim.camera_view import ChestScene
from ibvs_pad_touch.types import Intrinsics


def test_scale_model_projects_three_buttons_in_order():
    panel = load_panel()
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    scene = ChestScene(panel, K)
    loc = scene.localize()
    assert loc.visible
    b1, b2, b3 = loc.button("button_1"), loc.button("button_2"), loc.button("button_3")
    assert b1 is not None and b2 is not None and b3 is not None
    assert b1.u < b2.u
    assert b3.v > b1.v
    assert abs(b3.u - 0.5 * (b1.u + b2.u)) < 8


def test_localizer_uses_supplied_corners():
    panel = load_panel()
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    scene = ChestScene(panel, K)
    image = scene.render_bgr(draw_tip=False)
    loc = PanelButtonLocalizer(panel, refine_color=False).localize(image, None, K, corners_px=scene.corners_px())
    assert loc.visible
    assert set(loc.buttons) == {"button_1", "button_2", "button_3"}
    assert loc.buttons["button_1"].source == "scale+pnp"
    assert loc.buttons["button_1"].xyz_m is not None


def test_planar_pnp_recovers_button_xyz_without_depth():
    panel = load_panel()
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    obj = np.asarray([[x, y, 0.0] for x, y in panel.canonical_corners_m()], dtype=np.float64)
    rvec = np.asarray([0.08, -0.12, 0.03], dtype=np.float64)
    tvec = np.asarray([0.02, -0.03, 0.62], dtype=np.float64)
    camera = np.asarray([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]])
    corners, _ = cv2.projectPoints(obj, rvec, tvec, camera, None)
    found = panel_button_xyz_from_corners(panel, corners.reshape(-1, 2), K)
    rotation, _ = cv2.Rodrigues(rvec)
    expected = rotation @ np.asarray(panel.poke_point_m("button_1")) + tvec
    assert np.allclose(found["button_1"], expected, atol=2e-4)


def test_pnp_rejects_a_background_depth_far_behind_the_panel():
    panel = load_panel()
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    obj = np.asarray([[x, y, 0.0] for x, y in panel.canonical_corners_m()], dtype=np.float64)
    camera = np.asarray([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]])
    corners, _ = cv2.projectPoints(
        obj,
        np.asarray([0.02, -0.08, 0.01]),
        np.asarray([0.01, -0.02, 0.60]),
        camera,
        None,
    )
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    background_depth = np.full((480, 640), 1.40, dtype=np.float32)
    loc = PanelButtonLocalizer(panel, refine_color=False).localize(
        image,
        background_depth,
        K,
        corners_px=corners.reshape(-1, 2),
    )
    button = loc.button("button_1")
    assert button is not None
    assert button.source == "scale+pnp"
    assert 0.50 < button.z_m < 0.70


def _occluded_panel_no_yellow() -> np.ndarray:
    """White wall, visible housing + LCD, gripper covering the yellow keys."""
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (160, 180), (460, 310), (208, 218, 214), -1)
    cv2.rectangle(image, (195, 208), (370, 280), (32, 36, 38), -1)
    cv2.rectangle(image, (400, 190), (540, 420), (228, 228, 228), -1)
    cv2.circle(image, (455, 255), 28, (18, 18, 18), -1)
    return image


def test_button_behind_the_gripper_keeps_its_pixel_and_the_exposed_panel_depth():
    from ibvs_pad_touch.perception.pipeline import align_buttons_to_exposed_panel
    from ibvs_pad_touch.types import CameraPoint, Localization

    K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    depth = np.full((480, 640), 0.82, dtype=np.float32)
    depth[80:200, 430:560] = 0.41
    corners = [(80.0, 60.0), (560.0, 60.0), (560.0, 280.0), (80.0, 280.0)]
    gripper = [(430.0, 80.0), (560.0, 80.0), (560.0, 200.0), (430.0, 200.0)]
    loc = Localization(
        visible=True,
        corners_px=corners,
        buttons={
            "button_1": CameraPoint(u=480.0, v=120.0, z_m=0.41, source="scale"),
        },
    )
    out = align_buttons_to_exposed_panel(loc, depth, K, [gripper])
    button = out.button("button_1")
    assert button.u == 480.0 and button.v == 120.0
    assert abs(button.z_m - 0.82) < 0.02
    assert button.source == "scale"


def test_occluded_yellow_keys_use_panel_outline_and_scale():
    panel = load_panel()
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    image = _occluded_panel_no_yellow()
    assert assign_poke_buttons(image) is None
    loc = PanelButtonLocalizer(panel, refine_color=True).localize(image, None, K)
    assert loc.visible
    assert loc.corners_px is not None
    assert loc.message in {"ok", "scale_occluded"}
    b1, b2, b3 = loc.button("button_1"), loc.button("button_2"), loc.button("button_3")
    assert b1 is not None and b2 is not None and b3 is not None
    assert b1.u < b2.u
    assert b3.v > b1.v
    assert b1.source == "scale+pnp"


def test_lower_housing_band_does_not_hide_the_screen():
    panel = load_panel()
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (160, 180), (460, 310), (208, 218, 214), -1)
    cv2.rectangle(image, (195, 208), (370, 280), (32, 36, 38), -1)
    cv2.rectangle(image, (160, 268), (460, 360), (232, 234, 230), -1)
    corners = detect_panel_corners(image, panel)
    assert corners is not None
    assert float(np.min(corners[:, 1])) < 200.0
    center = (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))
    assert 180.0 < center[1] < 320.0


def test_lcd_expands_to_full_panel_by_scale():
    from ibvs_pad_touch.perception.localizer import panel_corners_from_lcd

    panel = load_panel()
    lcd_px = [(195.0, 208.0), (370.0, 208.0), (370.0, 280.0), (195.0, 280.0)]
    corners = panel_corners_from_lcd(lcd_px, panel)
    assert corners is not None
    xs, ys = corners[:, 0], corners[:, 1]
    assert abs(float(xs.min()) - 160.0) < 3.0
    assert abs(float(xs.max()) - 460.0) < 3.0
    assert abs(float(ys.min()) - 180.0) < 3.0
    assert abs(float(ys.max()) - 310.0) < 3.0


def test_blank_beige_frame_is_not_a_panel():
    frame = np.full((480, 640, 3), (198, 208, 206), dtype=np.uint8)
    assert detect_panel_corners(frame) is None
    K = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    loc = PanelButtonLocalizer(load_panel(), refine_color=True).localize(frame, None, K)
    assert not loc.visible
    assert loc.corners_px is None


def test_full_page_quad_is_rejected():
    h, w = 480, 640
    corners = np.array([[2, 2], [w - 2, 3], [w - 3, h - 2], [3, h - 3]], dtype=np.float32)
    assert not panel_quad_acceptable(corners, (h, w, 3))


def test_partial_panel_three_yellow_bars_are_rotation_invariant():
    image = np.full((320, 420, 3), 205, dtype=np.uint8)
    yellow = (0, 205, 245)
    cv2.rectangle(image, (75, 70), (100, 180), yellow, -1)   # split rocker
    cv2.rectangle(image, (135, 80), (160, 190), yellow, -1)  # OK
    cv2.rectangle(image, (270, 110), (298, 225), yellow, -1)  # RESET landmark
    for frame in (
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ):
        found = assign_poke_buttons(frame)
        assert found is not None
        assert set(found) == {"button_1", "button_2", "button_3"}
        b1, b2, ok = found["button_1"], found["button_2"], found["button_3"]
        assert np.hypot(b1[0] - b2[0], b1[1] - b2[1]) > 25
        rocker_mid = (0.5 * (b1[0] + b2[0]), 0.5 * (b1[1] + b2[1]))
        assert np.hypot(rocker_mid[0] - ok[0], rocker_mid[1] - ok[1]) < 100
