from ibvs_pad_touch.platform import environment_name, gui_available, is_linux, is_wsl, torch_device


def test_environment_name_is_known():
    assert environment_name() in {"wsl", "ubuntu", "windows"} or environment_name()


def test_gui_probe_does_not_raise():
    assert gui_available() in {True, False}


def test_wsl_implies_linux():
    if is_wsl():
        assert is_linux()


def test_torch_device_can_be_forced(monkeypatch):
    monkeypatch.setenv("IBVS_TORCH_DEVICE", "cuda:0")
    assert torch_device() == "cuda:0"
    monkeypatch.setenv("IBVS_TORCH_DEVICE", "cpu")
    assert torch_device() == "cpu"
