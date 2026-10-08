"""Chest-camera panel touch. Heavy imports stay lazy so joints/web can start without cv2."""

from __future__ import annotations

__all__ = [
    "IBVSController",
    "PanelButtonLocalizer",
    "PanelModel",
    "ServoCommand",
    "environment_name",
    "load_camera_config",
    "load_ibvs_config",
    "load_panel",
    "load_panel_config",
    "load_robot_config",
]


def __getattr__(name: str):
    if name in {"load_camera_config", "load_ibvs_config", "load_panel_config", "load_robot_config"}:
        from ibvs_pad_touch import config as _config

        return getattr(_config, name)
    if name in {"PanelModel", "load_panel"}:
        from ibvs_pad_touch.panel.model import PanelModel, load_panel

        return {"PanelModel": PanelModel, "load_panel": load_panel}[name]
    if name == "environment_name":
        from ibvs_pad_touch.platform import environment_name

        return environment_name
    if name == "PanelButtonLocalizer":
        from ibvs_pad_touch.perception.pipeline import PanelButtonLocalizer

        return PanelButtonLocalizer
    if name in {"IBVSController", "ServoCommand"}:
        from ibvs_pad_touch.servo.controller import IBVSController, ServoCommand

        return {"IBVSController": IBVSController, "ServoCommand": ServoCommand}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
