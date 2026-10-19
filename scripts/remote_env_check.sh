#!/usr/bin/env bash
set -u
echo "PATH=$PATH"
echo "HOME=$HOME"
ls -d "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python /opt/*/bin/python 2>/dev/null
echo "--- pythons ---"
for py in \
  /usr/bin/python3 \
  "$(command -v python3 || true)" \
  "$HOME/anaconda3/envs/unitree/bin/python" \
  "$HOME/anaconda3/envs/h1_arm/bin/python" \
  "$HOME"/anaconda3/envs/*/bin/python \
  "$HOME/miniconda3/envs/rmb/bin/python"
do
  [ -n "$py" ] && [ -x "$py" ] || continue
  echo "== $py =="
  "$py" - <<'PY'
import importlib
for name in ("flask", "cv2", "pyrealsense2", "unitree_sdk2py"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name}: {getattr(mod, '__file__', 'ok')}")
    except Exception as exc:
        print(f"  {name}: FAIL {exc}")
try:
    import pyrealsense2 as rs
    serials = []
    for dev in rs.context().query_devices():
        serials.append(dev.get_info(rs.camera_info.serial_number))
    print("  realsense serials:", serials)
except Exception as exc:
    print("  realsense list FAIL", exc)
PY
done
echo "--- listeners ---"
ss -lntp 2>/dev/null | sed -n '1,40p' || netstat -lntp 2>/dev/null | sed -n '1,40p'
echo "--- find unitree_sdk2py ---"
find "$HOME" /opt /usr/local -name 'unitree_sdk2py' -type d 2>/dev/null | head
echo "--- camera procs ---"
ps -eo pid,cmd | grep -Ei 'realsense|run_web|dual_camera|capture_handeye|ibvs' | grep -v grep || true
