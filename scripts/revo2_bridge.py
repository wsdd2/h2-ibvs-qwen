#!/usr/bin/env python3
"""Run the right Revo2 bridge on the computer that owns the hand USB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.web.revo2_bridge import DEFAULT_PORT, serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    print("[revo2-bridge] 占用右手串口。brainco_hand_service 不要同时打开这只手。", flush=True)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
