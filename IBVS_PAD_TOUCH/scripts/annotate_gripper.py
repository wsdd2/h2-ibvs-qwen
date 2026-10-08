"""Mark the closed Dex1 poke tip on the handheld gripper photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.perception.gripper_tip import detect_dex1_tip


def annotate_one(path: Path, out_dir: Path) -> bool:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"skip {path}")
        return False
    tip = detect_dex1_tip(bgr)
    vis = bgr.copy()
    ok = tip is not None
    if tip is not None:
        u, v = int(round(tip.u)), int(round(tip.v))
        cv2.arrowedLine(vis, (u, v + 70), (u, v + 8), (0, 0, 255), 3, tipLength=0.35)
        cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.circle(vis, (u, v), 10, (0, 0, 255), 2, cv2.LINE_AA)
        label = f"tip ({u},{v})"
        cv2.putText(vis, label, (u + 14, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (u + 14, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"{path.name:10s} tip=({tip.u:.1f},{tip.v:.1f})")
    else:
        print(f"{path.name:10s} MISS")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_tip.png"
    cv2.imwrite(str(out), vis)
    print(f"  wrote {out}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gripper")
    args = parser.parse_args()
    paths = [Path(p) for p in args.images] if args.images else sorted((ROOT / "data" / "gripper").glob("*.png"))
    ok = True
    for path in paths:
        ok = annotate_one(path, args.out) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
