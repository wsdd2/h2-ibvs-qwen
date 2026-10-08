from scripts.camera_sidecar import (
    _lock_current_exposure,
    _prepare_auto_exposure,
    _set_manual_exposure,
)


class _Option:
    exposure = "exposure"
    gain = "gain"
    enable_auto_exposure = "auto"
    auto_exposure_priority = "priority"


class _Rs:
    option = _Option()


class _Sensor:
    def __init__(self):
        self.values = {
            "exposure": 166.0,
            "gain": 64.0,
            "auto": 1.0,
            "priority": 1.0,
        }

    def supports(self, option):
        return option in self.values

    def get_option(self, option):
        return self.values[option]

    def set_option(self, option, value):
        self.values[option] = float(value)


def test_auto_exposure_settles_then_locks_current_values():
    sensor = _Sensor()
    _prepare_auto_exposure(_Rs, sensor)
    assert sensor.values["auto"] == 1.0
    assert sensor.values["priority"] == 0.0
    exposure, gain = _lock_current_exposure(_Rs, sensor)
    assert exposure == 166.0
    assert gain == 64.0
    assert sensor.values["auto"] == 0.0
    assert sensor.values["exposure"] == 166.0
    assert sensor.values["gain"] == 64.0


def test_manual_exposure_uses_bounded_chest_values():
    sensor = _Sensor()
    exposure, gain = _set_manual_exposure(_Rs, sensor, 80.0, 32.0)
    assert exposure == 80.0
    assert gain == 32.0
    assert sensor.values["auto"] == 0.0
