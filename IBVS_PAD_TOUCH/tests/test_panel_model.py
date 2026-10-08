from ibvs_pad_touch.panel.model import load_panel


def test_panel_size_and_poke_set():
    panel = load_panel()
    assert abs(panel.width_m - 0.300) < 1e-9
    assert abs(panel.height_m - 0.130) < 1e-9
    assert [b.button_id for b in panel.poke_buttons()] == [1, 2, 3]


def test_button_labels_match_photo():
    panel = load_panel()
    assert panel.button(1).label == "<"
    assert panel.button(2).label == ">"
    assert panel.button(3).label == "OK"
    assert panel.button("Button 1").name == "button_1"
    assert panel.button("OK").name == "button_3"


def test_pair_measured_from_right_and_top():
    panel = load_panel()
    b1, b2, b3 = panel.button(1), panel.button(2), panel.button(3)
    assert abs((panel.width_m - b2.rect.right_m) - 0.035) < 1e-6
    assert abs((panel.width_m - b1.rect.left_m) - 0.070) < 1e-6
    assert abs(b1.rect.cy - 0.030) < 1e-6
    assert abs(b2.rect.cy - 0.030) < 1e-6
    assert abs(b1.rect.width_m + b2.rect.width_m - 0.035) < 1e-6
    assert abs(b1.rect.right_m - b2.rect.left_m) < 1e-6
    assert b1.rect.cx < b2.rect.cx
    assert abs(b3.rect.cy - b1.rect.cy - 0.020) < 1e-6
    assert abs(b3.rect.cx - 0.5 * (b1.rect.left_m + b2.rect.right_m)) < 1e-6
    assert abs(b3.rect.width_m - 0.035) < 1e-6
