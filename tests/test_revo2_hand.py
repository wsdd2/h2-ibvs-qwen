from ibvs_pad_touch.web.revo2_hand import INDEX, JOINTS, index_point_q, open_q


def test_index_point_opens_only_the_index_finger():
    pose = index_point_q()
    assert len(pose) == len(JOINTS)
    assert pose[INDEX] == 0.0
    assert all(value == 1.0 for i, value in enumerate(pose) if i != INDEX)


def test_open_pose_is_all_zeros():
    assert open_q() == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
