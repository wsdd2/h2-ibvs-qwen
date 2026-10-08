# shellcheck shell=bash
# Source:  . scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export IBVS_HEADLESS="${IBVS_HEADLESS:-1}"
# Camera sidecars (conda 3.8) must not see Ubuntu/3.10 site-packages.
export IBVS_CAM_PYTHONPATH="$ROOT"
# Web/joints (system python3) need flask + cyclonedds._clayer.
IBVS_WEB_PYTHONPATH="$ROOT"
# Later entries are prepended, so the user site must stay last and win over apt sympy.
for _extra in \
  /usr/local/lib/python3/dist-packages \
  /usr/lib/python3/dist-packages \
  "${HOME}/.local/lib/python3.10/site-packages"
do
  if [ -d "${_extra}" ]; then
    IBVS_WEB_PYTHONPATH="${_extra}:${IBVS_WEB_PYTHONPATH}"
  fi
done

for _sdk in \
    "${HOME}/unitree_sdk2_python" \
    "${HOME}/MscapeTech/unitree_sdk2_python" \
    "${HOME}/lj_ws/unitree_sdk2_python" \
    "${HOME}/H2_joint_cartesian/third_party/unitree_sdk2_python" \
    "${HOME}/IBVS_PAD_TOUCH/third_party/unitree_sdk2_python"
do
  if [ -d "${_sdk}/unitree_sdk2py" ]; then
    export PYTHONPATH="${_sdk}${PYTHONPATH:+:$PYTHONPATH}"
    IBVS_WEB_PYTHONPATH="${_sdk}:${IBVS_WEB_PYTHONPATH}"
    break
  fi
done
export IBVS_WEB_PYTHONPATH

iter_ibvs_pythons() {
  printf '%s\n' \
    "${IBVS_WEB_PYTHON:-}" \
    "${IBVS_PYTHON:-}" \
    "${MT_ACT_PYTHON:-}" \
    /usr/bin/python3 \
    "$(command -v python3 2>/dev/null || true)" \
    "${HOME}/anaconda3/envs/h1_arm/bin/python" \
    "${HOME}/anaconda3/envs/unitree/bin/python" \
    "${HOME}/miniconda3/envs/rmb/bin/python"
  ls -1d \
    "${HOME}"/anaconda3/envs/*/bin/python \
    "${HOME}"/miniconda3/envs/*/bin/python \
    /opt/miniconda*/envs/*/bin/python \
    /opt/anaconda*/envs/*/bin/python \
    2>/dev/null || true
}

pick_python_importing() {
  local code="$1"
  local candidate seen=" "
  while IFS= read -r candidate; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    case "$seen" in
      *" $candidate "*) continue ;;
    esac
    seen="${seen}${candidate} "
    if "$candidate" -c "$code" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(iter_ibvs_pythons)
  return 1
}

pick_ibvs_python() {
  pick_python_importing "import flask, cv2; import cyclonedds._clayer" \
    || pick_python_importing "import flask, cv2" \
    || pick_python_importing "import cv2" \
    || pick_python_importing "import flask" \
    || pick_python_importing "import sys"
}

if ! PY="$(pick_ibvs_python)"; then
  echo "no python3 found; set IBVS_PYTHON" >&2
  return 1 2>/dev/null || exit 1
fi

echo "[ibvs] ROOT=$ROOT"
echo "[ibvs] PYTHON=$PY"
"$PY" - <<'PY' || true
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
try:
    from ibvs_pad_touch.platform import environment_name, gui_available
    print(f"[ibvs] env={environment_name()} gui={int(gui_available())}")
except Exception as exc:
    print(f"[ibvs] env probe skipped: {exc}")
PY
