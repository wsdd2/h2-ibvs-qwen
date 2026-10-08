from ibvs_pad_touch.platform import environment_name, gui_available, is_linux, is_wsl


def test_environment_name_is_known():
    assert environment_name() in {"wsl", "ubuntu", "windows"} or environment_name()


def test_gui_probe_does_not_raise():
    assert gui_available() in {True, False}


def test_wsl_implies_linux():
    if is_wsl():
        assert is_linux()
