"""Draw the tape-measure scale model and write it to outputs/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.panel.model import load_panel
from ibvs_pad_touch.panel.render import render_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="button_1", help="button_1 / button_2 / button_3")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "panel_scale.png")
    args = parser.parse_args()

    model = load_panel()
    image = render_panel(model, highlight=args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    b1, b2, b3 = model.button(1), model.button(2), model.button(3)
    print(f"panel {model.width_m * 1000:.0f} x {model.height_m * 1000:.0f} mm")
    print(f"Button 1  <   center=({b1.rect.cx * 1000:.2f}, {b1.rect.cy * 1000:.2f}) mm")
    print(f"Button 2  >   center=({b2.rect.cx * 1000:.2f}, {b2.rect.cy * 1000:.2f}) mm")
    print(f"Button 3  OK  center=({b3.rect.cx * 1000:.2f}, {b3.rect.cy * 1000:.2f}) mm")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
