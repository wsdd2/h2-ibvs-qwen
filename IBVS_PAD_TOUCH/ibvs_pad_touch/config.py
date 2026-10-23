from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # H2 often cannot reach PyPI
    from ibvs_pad_touch import lite_yaml as yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_config(path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return CONFIG_DIR / default_name
    candidate = Path(path)
    if candidate.exists():
        return candidate
    bundled = CONFIG_DIR / candidate
    if bundled.exists():
        return bundled
    raise FileNotFoundError(path)


def load_panel_config(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(resolve_config(path, "panel.yaml"))


def load_ibvs_config(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(resolve_config(path, "ibvs.yaml"))


def load_camera_config(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(resolve_config(path, "camera_d435i.yaml"))


def camera_views(cfg: dict | None = None) -> list[dict[str, Any]]:
    cam = (cfg or load_camera_config())["camera"]
    views = cam.get("views")
    if views:
        return [dict(item) for item in views]
    known = cam.get("known_serials") or {}
    if known:
        return [
            {"name": str(name), "label": str(name), "serial_number": str(serial), "model": cam.get("model", "")}
            for name, serial in known.items()
        ]
    serial = str(cam.get("serial_number") or "")
    return [{"name": "chest", "label": "chest", "serial_number": serial, "model": cam.get("model", "")}]


def load_robot_config(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(resolve_config(path, "robot_h2.yaml"))
