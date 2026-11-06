import math

import numpy as np

from ibvs_pad_touch.web.arm_motion import DIRECT_EE_RPY_DEG, RIGHT_EE_OFFSET_WRIST_M
from ibvs_pad_touch.web.revo2_index_pad import (
    REACHABLE_PRESS_RPY_DEG,
    index_pad_axes_in_wrist,
    index_pad_in_wrist,
    index_pad_tool_rpy_deg,
    index_tip_in_wrist,
    index_tip_tool_rpy_deg,
)


def test_official_urdf_tip_is_the_poke_point():
    tip = np.asarray(index_tip_in_wrist(), dtype=float)
    pad = np.asarray(index_pad_in_wrist(), dtype=float)
    assert np.allclose(tip, (0.227024, 0.033819, -0.019981), atol=1e-5)
    assert np.allclose(pad, (0.217060, 0.032194, -0.022765), atol=1e-5)
    assert float(np.linalg.norm(tip - pad)) > 0.008
    assert float(tip[0]) > float(pad[0])
    assert index_tip_tool_rpy_deg() == REACHABLE_PRESS_RPY_DEG
    assert RIGHT_EE_OFFSET_WRIST_M == (0.205, 0.0, 0.0)


def test_pad_pose_tilts_toward_the_panel_without_demanding_parallel():
    from ibvs_pad_touch.servo.virtual_tip import rpy_to_matrix

    palm, _finger = index_pad_axes_in_wrist()
    pose = index_pad_tool_rpy_deg()
    rotation = rpy_to_matrix(*(math.radians(value) for value in pose))
    forward = float((rotation @ palm)[0])
    gripper = rpy_to_matrix(*(math.radians(value) for value in DIRECT_EE_RPY_DEG))
    gripper_forward = float((gripper @ palm)[0])
    assert forward > gripper_forward
    assert forward < 0.95
