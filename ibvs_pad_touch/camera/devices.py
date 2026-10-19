"""Pick a live RealSense serial. A requested serial is never replaced by the other camera."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


CLAIM_DIR = Path("/tmp")


def list_realsense(rs) -> list[dict[str, str]]:
    if rs is None or not hasattr(rs, "context"):
        return []
    out: list[dict[str, str]] = []
    try:
        for dev in rs.context().query_devices():
            serial = str(dev.get_info(rs.camera_info.serial_number))
            name = ""
            if hasattr(dev, "supports") and dev.supports(rs.camera_info.name):
                name = str(dev.get_info(rs.camera_info.name))
            out.append({"serial": serial, "name": name})
    except Exception:
        return []
    return out


def claim_path(name: str, directory: Path | None = None) -> Path:
    return (directory or CLAIM_DIR) / f"ibvs_cam_{name}.serial"


def claimed_serials(except_name: str = "", directory: Path | None = None) -> set[str]:
    folder = directory or CLAIM_DIR
    used: set[str] = set()
    for path in folder.glob("ibvs_cam_*.serial"):
        if except_name and path.name == f"ibvs_cam_{except_name}.serial":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            used.add(text)
    return used


def write_claim(name: str, serial: str, directory: Path | None = None) -> Path:
    path = claim_path(name, directory)
    path.write_text(str(serial).strip() + "\n", encoding="utf-8")
    return path


def pick_serial(
    preferred: str,
    devices: list[dict[str, str]] | list[str],
    claimed: set[str] | None = None,
) -> Optional[str]:
    serials: list[str] = []
    for item in devices:
        serial = item["serial"] if isinstance(item, dict) else str(item)
        if serial and serial not in serials:
            serials.append(serial)
    taken = set(claimed or ())
    want = str(preferred or "").strip()
    if want:
        if want in serials and want not in taken:
            return want
        return None
    for serial in serials:
        if serial not in taken:
            return serial
    return None


def write_seen(devices: list[dict[str, str]], directory: Path | None = None) -> Path:
    path = (directory or CLAIM_DIR) / "ibvs_cam_seen.txt"
    path.write_text(format_device_line(devices) + "\n", encoding="utf-8")
    return path


def read_seen(directory: Path | None = None) -> str:
    path = (directory or CLAIM_DIR) / "ibvs_cam_seen.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def format_device_line(devices: list[dict[str, str]]) -> str:
    if not devices:
        return "(none)"
    return ", ".join(f"{item.get('name') or 'RealSense'} {item['serial']}" for item in devices)
