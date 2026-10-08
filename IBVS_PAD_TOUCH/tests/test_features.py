import numpy as np

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.localizer import homography_from_meter_corners, project_xy


def test_meter_homography_projects_button3_center():
    panel = load_panel()
    corners = [(10, 20), (310, 20), (310, 150), (10, 150)]
    H = homography_from_meter_corners(panel, corners)
    u, v = project_xy((panel.button(3).rect.cx, panel.button(3).rect.cy), H)
    expected_u = 10 + (0.2475 / 0.300) * 300
    expected_v = 20 + (0.050 / 0.130) * 130
    assert abs(u - expected_u) < 1e-3
    assert abs(v - expected_v) < 1e-3
    assert np.linalg.det(H) != 0
