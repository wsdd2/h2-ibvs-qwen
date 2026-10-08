from pathlib import Path

from ibvs_pad_touch import lite_yaml
from ibvs_pad_touch.config import load_camera_config, load_ibvs_config, load_panel_config, load_robot_config


def test_bundled_configs_load_without_pyyaml():
    panel = load_panel_config()
    camera = load_camera_config()
    ibvs = load_ibvs_config()
    robot = load_robot_config()
    assert panel["buttons"]["button_2"]["center_m"][0] == 0.25625
    assert camera["camera"]["serial_number"] == "347622074029"
    assert camera["camera"]["known_serials"]["d435i"] == "233722073797"
    assert ibvs["control_hz"] == 20
    assert robot["hand_eye_calibration"] is False


def test_lite_yaml_matches_pyyaml_on_bundled_files():
    yaml = __import__("yaml")
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("panel.yaml", "ibvs.yaml", "camera_d435i.yaml", "robot_h2.yaml"):
        text = (root / name).read_text(encoding="utf-8")
        assert lite_yaml.safe_load(text) == yaml.safe_load(text)
