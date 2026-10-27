#!/usr/bin/env python3
"""Hold the left Revo2 with the index extended and the other fingers closed.

The right hand is not commanded. Ctrl+C stops publishing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.web.revo2_hand import LeftRevo2


def main() -> None:
    hand = LeftRevo2()
    print(hand.point_index(), flush=True)
    print(hand.status(), flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(hand.stop(), flush=True)


if __name__ == "__main__":
    main()
