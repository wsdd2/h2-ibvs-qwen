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


_GPU_DEVICE: str | None = None


def torch_device() -> str:
    """Orin GPU when CUDA is visible. IBVS_TORCH_DEVICE overrides it."""
    forced = os.environ.get("IBVS_TORCH_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
        return "cuda:0"
    return "cpu"


def use_gpu() -> bool:
    return torch_device().startswith("cuda")


def gpu_name() -> str:
    if not use_gpu():
        return "cpu"
    try:
        import torch

        return str(torch.cuda.get_device_name(0))
    except Exception:
        return torch_device()


def prepare_torch_gpu() -> str:
    """Turn on cuDNN autotune once, and keep CPU thread pools off the control cores."""
    global _GPU_DEVICE
    if _GPU_DEVICE is not None:
        return _GPU_DEVICE
    device = torch_device()
    if not device.startswith("cuda"):
        _GPU_DEVICE = device
        return device
    try:
        import torch

        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        _GPU_DEVICE = "cpu"
        return "cpu"
    _GPU_DEVICE = device
    return device


def onnx_cuda() -> bool:
    """True when this process can actually run ONNX on the Orin GPU."""
    if not use_gpu():
        return False
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in set(ort.get_available_providers())
