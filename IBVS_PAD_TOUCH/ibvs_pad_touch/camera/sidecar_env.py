"""Keep conda camera sidecars off Ubuntu / Python 3.10 site-packages."""
from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence


FOREIGN_SITE_MARKERS = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
    "/.local/lib/python3.10/",
)


def is_conda_python(executable: str = "", conda_prefix: str | None = None) -> bool:
    exe = (executable or sys.executable).replace("\\", "/")
    prefix = os.environ.get("CONDA_PREFIX", "") if conda_prefix is None else conda_prefix
    return bool(prefix) or "/envs/" in exe or "/anaconda" in exe or "/miniconda" in exe


def drop_foreign_site_dirs(paths: Sequence[str], *, conda: bool) -> list[str]:
    if not conda:
        return list(paths)
    out: list[str] = []
    for item in paths:
        norm = item.replace("\\", "/")
        if any(marker in norm for marker in FOREIGN_SITE_MARKERS):
            continue
        out.append(item)
    return out


def apply_conda_path_guard(paths: Iterable[str] | None = None) -> list[str]:
    """Drop system numpy/flask dirs when this process is conda."""
    cleaned = drop_foreign_site_dirs(
        list(paths if paths is not None else sys.path),
        conda=is_conda_python(),
    )
    sys.path[:] = cleaned
    env_parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    os.environ["PYTHONPATH"] = os.pathsep.join(drop_foreign_site_dirs(env_parts, conda=is_conda_python()))
    return cleaned
