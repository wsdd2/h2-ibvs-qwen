"""Load a RealSense binding that actually has rs.pipeline().

rs-enumerate-devices can see the D435 while a pip/user-site stub named
pyrealsense2 imports fine but has no pipeline. Prefer apt/librealsense.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

_CACHED: Any = None
_TRIED = False


def _is_usable(mod: Any) -> bool:
    return bool(mod) and hasattr(mod, "pipeline") and hasattr(mod, "config") and hasattr(mod, "stream")


def _user_site_path() -> Optional[str]:
    try:
        import site

        user = site.getusersitepackages()
    except Exception:
        return None
    return user if user else None


def _strip_user_site() -> Optional[str]:
    """Temporarily drop ~/.local so pip's empty pyrealsense2 stub cannot win."""
    user = _user_site_path()
    if not user:
        return None
    sys.path[:] = [item for item in sys.path if item != user]
    return user


def _restore_user_site(user: Optional[str]) -> None:
    """Put user site back so Flask / cyclonedds._clayer still import."""
    if user and user not in sys.path:
        sys.path.append(user)


def _extra_bind_dirs() -> list[Path]:
    home = Path.home()
    dirs: list[Path] = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/local/lib/python3/dist-packages"),
    ]
    for root in (Path("/usr/lib"), Path("/usr/local/lib"), home / "anaconda3", home / "miniconda3", Path("/opt")):
        dirs.extend(sorted(root.glob("python3.*/dist-packages")))
        dirs.extend(sorted(root.glob("python3.*/site-packages")))
        dirs.extend(sorted(root.glob("envs/*/lib/python*/site-packages")))
        dirs.extend(sorted(root.glob("lib/python*/site-packages")))
    try:
        import shutil

        binary = shutil.which("rs-enumerate-devices")
        if binary:
            prefix = Path(binary).resolve().parent.parent
            dirs.append(prefix / "lib" / "python3" / "dist-packages")
            dirs.extend(sorted((prefix / "lib").glob("python3.*/dist-packages")))
    except Exception:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for folder in dirs:
        key = str(folder)
        if key in seen or not folder.is_dir():
            continue
        if not (folder / "pyrealsense2").exists() and not list(folder.glob("pyrealsense2*.so")):
            continue
        seen.add(key)
        out.append(folder)
    return out


def _purge() -> None:
    for name in list(sys.modules):
        if name == "pyrealsense2" or name.startswith("pyrealsense2."):
            del sys.modules[name]


def _inner_binding(mod: Any) -> Any:
    """Apt librealsense on this H2 exposes pipeline on pyrealsense2.pyrealsense2."""
    if mod is None:
        return None
    try:
        inner = importlib.import_module("pyrealsense2.pyrealsense2")
    except Exception:
        return None
    return inner if _is_usable(inner) else None


def _import_from(folder: Path | None = None) -> Any:
    if folder is not None:
        path = str(folder)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    _purge()
    import pyrealsense2 as rs  # noqa: WPS433

    return _inner_binding(rs) or rs


def usable_realsense() -> Optional[Any]:
    """Return pyrealsense2 with pipeline, or None. Safe to call often."""
    global _CACHED, _TRIED
    if _TRIED:
        return _CACHED
    _TRIED = True
    user = _strip_user_site()
    try:
        try:
            import pyrealsense2 as rs
        except ImportError:
            rs = None
        inner = _inner_binding(rs)
        if inner is not None:
            print(
                f"[ibvs-cam] using pyrealsense2.pyrealsense2 {getattr(inner, '__file__', '')}",
                flush=True,
            )
            _CACHED = inner
            return inner
        if _is_usable(rs):
            _CACHED = rs
            return rs
        stub = getattr(rs, "__file__", None) if rs is not None else None
        if stub:
            print(f"[ibvs-cam] skip stub pyrealsense2 at {stub} (no pipeline)", flush=True)
        for folder in _extra_bind_dirs():
            try:
                cand = _import_from(folder)
            except Exception:
                continue
            if _is_usable(cand):
                print(f"[ibvs-cam] using librealsense bind {folder}", flush=True)
                _CACHED = cand
                return cand
        _CACHED = None
        return None
    finally:
        _restore_user_site(user)
