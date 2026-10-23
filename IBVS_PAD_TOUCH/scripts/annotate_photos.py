"""Label Button 1/2/3 on the field photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.perception.overlay import draw_localization
from ibvs_pad_touch.perception.pipeline import PanelButtonLocalizer
from ibvs_pad_touch.types import Intrinsics


def default_photos() -> list[Path]:
    return sorted((ROOT / "data" / "photos").glob("*.png"))


def annotate_one(path: Path, out_dir: Path, localizer: PanelButtonLocalizer) -> bool:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"skip unreadable {path}")
        return False
    h, w = bgr.shape[:2]
    K = Intrinsics(fx=0.8 * w, fy=0.8 * w, cx=w / 2, cy=h / 2, width=w, height=h)
    loc = localizer.localize(bgr, None, K)
    vis = draw_localization(bgr, loc)
    out = out_dir / f"{path.stem}_annot.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    status = "OK" if loc.visible else loc.message
    print(f"{path.name:16s} {status}")
    for name in ("button_1", "button_2", "button_3"):
        point = loc.button(name)
        if point is None:
            print(f"  {name}: missing")
            continue
        print(f"  {name}: u={point.u:.1f} v={point.v:.1f}  {point.source}")
    print(f"  wrote {out}")
    return loc.visible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "annotate")
    args = parser.parse_args()
    paths = [Path(p) for p in args.images] if args.images else default_photos()
    if not paths:
        raise SystemExit("no photos in data/photos")
    localizer = PanelButtonLocalizer(load_panel(), refine_color=True)
    ok = all(annotate_one(path, args.out, localizer) for path in paths)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
