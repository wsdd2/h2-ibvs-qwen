import numpy as np

from ibvs_pad_touch.camera.devices import claimed_serials, pick_serial, write_claim
from ibvs_pad_touch.camera.image_io import pick_sidecar, read_image, write_bgr, write_ppm


def test_pick_preferred_when_present():
    devices = [{"serial": "347622074029", "name": "D435"}, {"serial": "349622074791", "name": "D435"}]
    assert pick_serial("347622074029", devices, set()) == "347622074029"


def test_missing_preferred_serial_does_not_take_the_other_camera():
    devices = [{"serial": "233722073797", "name": "D435I"}]
    assert pick_serial("347622074029", devices, set()) is None
    assert pick_serial("233722073797", devices, set()) == "233722073797"


def test_do_not_steal_claimed_camera():
    devices = [{"serial": "347622074029", "name": "D435"}]
    assert pick_serial("233722073797", devices, {"347622074029"}) is None


def test_claim_files_exclude_self(tmp_path):
    write_claim("chest", "347622074029", tmp_path)
    write_claim("d435i", "349622074791", tmp_path)
    assert claimed_serials("d435i", tmp_path) == {"347622074029"}
    assert claimed_serials("chest", tmp_path) == {"349622074791"}


def test_bgr_roundtrip(tmp_path):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:] = (10, 40, 200)
    path = write_bgr(tmp_path / "cam.jpg", frame)
    got = read_image(path)
    assert path.suffix == ".bgr"
    assert got is not None
    assert got.shape == (12, 16, 3)
    assert int(got[0, 0, 2]) == 200


def test_ppm_roundtrip(tmp_path):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:] = (10, 40, 200)
    path = write_ppm(tmp_path / "cam.jpg", frame)
    got = read_image(path)
    assert got is not None
    assert got.shape == (12, 16, 3)
    assert int(got[0, 0, 2]) == 200


def test_pick_sidecar_ignores_stale(tmp_path):
    stale = tmp_path / "old.jpg"
    stale.write_bytes(b"x" * 300)
    import os

    os.utime(stale, (1, 1))
    assert pick_sidecar([stale], max_age_s=4.0) is None
    fresh = write_ppm(tmp_path / "new.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert pick_sidecar([stale, fresh], max_age_s=4.0) == fresh
