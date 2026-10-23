from __future__ import annotations

import os
import sys
from pathlib import Path


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_wsl() -> bool:
    if not is_linux():
        return False
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def is_native_ubuntu() -> bool:
    return is_linux() and not is_wsl()


def gui_available() -> bool:
    """WSL / H2 often have no usable display. Headless OpenCV is the default."""
    flag = os.environ.get("IBVS_HEADLESS", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return False
    if not is_linux():
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def environment_name() -> str:
    if is_wsl():
        return "wsl"
    if is_linux():
        return "ubuntu"
    if sys.platform.startswith("win"):
        return "windows"
    return sys.platform
