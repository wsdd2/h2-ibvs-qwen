from ibvs_pad_touch.camera.sidecar_env import drop_foreign_site_dirs, is_conda_python


def test_conda_python_detects_env_path():
    assert is_conda_python("/home/unitree/anaconda3/envs/unitree/bin/python", "")
    assert is_conda_python("/usr/bin/python3", "/home/unitree/anaconda3/envs/unitree")
    assert not is_conda_python("/usr/bin/python3", "")


def test_conda_drops_ubuntu_numpy_dir():
    paths = [
        "/home/unitree/anaconda3/envs/unitree/lib/python3.8/site-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3/dist-packages",
        "/home/unitree/.local/lib/python3.10/site-packages",
        "/home/unitree/IBVS_PAD_TOUCH",
    ]
    cleaned = drop_foreign_site_dirs(paths, conda=True)
    assert cleaned == [
        "/home/unitree/anaconda3/envs/unitree/lib/python3.8/site-packages",
        "/home/unitree/IBVS_PAD_TOUCH",
    ]
    assert drop_foreign_site_dirs(paths, conda=False) == paths
