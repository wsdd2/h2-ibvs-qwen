"""Make unitree_sdk2py + cyclonedds._clayer importable in the Flask process.

The web python often has flask/cv2 but not the CycloneDDS C extension.
Hand-eye Terminal A works with:
  ~/.local/lib/python3.10/site-packages
  /usr/lib/python3/dist-packages
  /home/unitree/MscapeTech/unitree_sdk2_python
"""
from __future__ import annotations

import sys
from pathlib import Path


SDK_ROOTS = (
    Path("/home/unitree/MscapeTech/unitree_sdk2_python"),
    Path("/home/unitree/unitree_sdk2_python"),
    Path("/opt/unitree_sdk2_python"),
)


def _purge(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _candidate_site_dirs() -> list[Path]:
    home = Path.home()
    dirs = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/local/lib/python3/dist-packages"),
        home / ".local" / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages",
        home / ".local" / "lib" / "python3.10" / "site-packages",
        home / "anaconda3" / "envs" / "unitree" / "lib" / "python3.10" / "site-packages",
        home / "anaconda3" / "envs" / "h1_arm" / "lib" / "python3.10" / "site-packages",
    ]
    for root in (home / "anaconda3", home / "miniconda3", Path("/opt")):
        dirs.extend(sorted(root.glob("envs/*/lib/python*/site-packages")))
        dirs.extend(sorted(root.glob("lib/python*/site-packages")))
    dirs.extend(sorted((home / ".local" / "lib").glob("python3.*/site-packages")))
    return dirs


def cyclonedds_search_dirs() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for folder in _candidate_site_dirs():
        key = str(folder)
        if key in seen or not folder.is_dir():
            continue
        pkg = folder / "cyclonedds"
        if not pkg.is_dir():
            continue
        if not list(pkg.glob("_clayer*")):
            continue
        seen.add(key)
        out.append(folder)
    return out


def _sdk_search_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [
        here.parents[3] / "unitree_sdk2_python",
        Path.home() / "MscapeTech" / "unitree_sdk2_python",
        *SDK_ROOTS,
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _try_import_clayer() -> bool:
    try:
        import cyclonedds._clayer  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_cyclonedds_clayer() -> None:
    if _try_import_clayer():
        return
    tried: list[str] = []
    for folder in cyclonedds_search_dirs():
        path = str(folder)
        tried.append(path)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        _purge("cyclonedds")
        if _try_import_clayer():
            return
    raise ImportError(
        "No module named 'cyclonedds._clayer'. "
        "Flask 进程缺 CycloneDDS 原生扩展。在 H2 上:\n"
        "  export PYTHONPATH=/home/unitree/.local/lib/python3.10/site-packages:"
        "/usr/lib/python3/dist-packages:/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH\n"
        f"  已扫到含 _clayer 的目录: {tried or '无'}"
    )


def ensure_unitree_sdk2py() -> None:
    ensure_cyclonedds_clayer()
    try:
        import unitree_sdk2py  # noqa: F401
        return
    except ImportError:
        pass
    tried: list[str] = []
    for root in _sdk_search_roots():
        tried.append(str(root))
        if (root / "unitree_sdk2py").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import unitree_sdk2py  # noqa: F401
    except ImportError as exc:
        existing = [path for path in tried if (Path(path) / "unitree_sdk2py").is_dir()]
        raise ImportError(
            "找不到 unitree_sdk2py。\n"
            f"  实际报错: {exc}\n"
            f"  已存在的 SDK 目录: {existing or '无'}\n"
            "  export PYTHONPATH=/home/unitree/.local/lib/python3.10/site-packages:"
            "/usr/lib/python3/dist-packages:/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH"
        ) from exc
