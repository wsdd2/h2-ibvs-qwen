#!/usr/bin/env bash
# WSL2 Ubuntu and native Ubuntu 22.04 (H2). No X11 required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/scripts/env.sh"

want_apt=0
want_robot=0
for arg in "$@"; do
  case "$arg" in
    --apt) want_apt=1 ;;
    --robot) want_robot=1 ;;
  esac
done

if [ "$want_apt" -eq 1 ]; then
  if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv || true
  else
    echo "[ibvs] skip apt (need: ./scripts/setup_ubuntu.sh --apt  and passwordless sudo)"
  fi
else
  echo "[ibvs] skip apt (pass --apt only if you really need system packages)"
fi

"$PY" -m pip install -q -U pip
"$PY" -m pip install -q -r "$ROOT/requirements.txt"

if [ "$want_robot" -eq 1 ]; then
  echo "[ibvs] install optional RealSense extra"
  "$PY" -m pip install -q -r "$ROOT/requirements-robot.txt" || {
    echo "[ibvs] pyrealsense2 pip failed; on H2 use the existing librealsense / conda env" >&2
  }
fi

"$PY" -m pytest tests -q
"$PY" scripts/preview_panel.py
"$PY" scripts/run_ibvs.py --smoke --source synthetic --target button_1
echo
echo "WSL:  ./scripts/run_ibvs.sh --source synthetic --target button_1 --no-tip"
echo "H2:   ./scripts/setup_ubuntu.sh --robot"
echo "      ./scripts/run_web.sh"
echo "      ./scripts/run_ibvs.sh --source realsense --target button_1"
echo "Python override: export IBVS_PYTHON=/path/to/python"
