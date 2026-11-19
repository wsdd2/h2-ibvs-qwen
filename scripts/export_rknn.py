#!/usr/bin/env python3
"""Convert the two YOLO ONNX files to RK3588 FP16 RKNN.

Run this on an x86 Linux machine, not on the board. RKNN-Toolkit2 2.5 needs
its own environment because it requires torch<=2.2 and numpy<=1.26:

    python3 -m venv ~/venv-rknn
    source ~/venv-rknn/bin/activate
    python -m pip install -U pip
    python -m pip install rknn-toolkit2==2.5.0 "setuptools<80.1"
    python scripts/export_rknn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEG_ONNX = ROOT / "outputs" / "runs" / "yolo12_seg_20261014_1045" / "weights" / "best.onnx"
POSE_ONNX = ROOT / "outputs" / "runs" / "yolo26s_pose_20261103_1130" / "weights" / "best.onnx"


def _convert(onnx_path: Path) -> Path:
    from rknn.api import RKNN

    out = onnx_path.with_suffix(".rknn")
    rknn = RKNN(verbose=True)
    # FP16. The end-to-end head stores 0-1 confidence beside 0-640 coordinates.
    # INT8 uses one scale for that tensor and rounds every confidence to 0.
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit(f"load_onnx failed: {onnx_path}")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit(f"build failed: {onnx_path}")
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit(f"export failed: {out}")
    rknn.release()
    print(f"wrote {out}")
    return out


def main() -> None:
    if not SEG_ONNX.is_file() or not POSE_ONNX.is_file():
        raise SystemExit("best.onnx is missing; export ONNX before RKNN")
    _convert(POSE_ONNX)
    _convert(SEG_ONNX)


if __name__ == "__main__":
    sys.exit(main())
