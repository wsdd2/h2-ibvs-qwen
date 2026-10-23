#!/usr/bin/env python3
"""Write H2 upper-body LowState to /tmp/ibvs_joints.json for the Flask page."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibvs_pad_touch.web.joints import UpperBodyJointReader

OUT = Path("/tmp/ibvs_joints.json")


def main() -> None:
    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    reader = UpperBodyJointReader(iface=iface, demo=False)
    first = reader.snapshot()
    print(f"[ibvs-joints] source={first.source} ok={first.ok} {first.message}", flush=True)
    while True:
        snap = reader.snapshot().as_dict()
        snap["written_at"] = time.time()
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap), encoding="utf-8")
        tmp.replace(OUT)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
