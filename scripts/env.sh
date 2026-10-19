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

# The conda unitree binding segfaults inside query_devices. This H2's working
# camera stack is system python3 plus apt pyrealsense2.pyrealsense2.
CAMERA_QUERY='import pyrealsense2 as rs; assert hasattr(rs, "pipeline"); list(rs.context().query_devices())'
SYSTEM_CAMERA_QUERY='import pyrealsense2.pyrealsense2 as rs; assert hasattr(rs, "pipeline"); list(rs.context().query_devices())'

_system_camera_pythonpath() {
  local dist="/usr/lib/python3/dist-packages"
  local user="${HOME}/.local/lib/python3.10/site-packages"
  local path="$dist"
  if [ -d "$user" ]; then
    path="${dist}:${user}"
  fi
  printf '%s\n' "${path}:${IBVS_CAM_PYTHONPATH}"
}

_camera_query_ok() {
  local py="$1"
  local mode="$2"
  local lib="${3:-}"
  local query="${4:-$CAMERA_QUERY}"
  local pypath="${5:-$IBVS_CAM_PYTHONPATH}"
  local limit=() status=1 errfile detail
  if command -v timeout >/dev/null 2>&1; then
    limit=(timeout 8)
  fi
  errfile="$(mktemp)"
  if [ "$mode" = "clean" ]; then
    "${limit[@]}" env -u LD_LIBRARY_PATH PYTHONPATH="${pypath}" "$py" -c "$query" >/dev/null 2>"$errfile" && status=0 || status=$?
  elif [ "$mode" = "inherit" ]; then
    "${limit[@]}" env PYTHONPATH="${pypath}" "$py" -c "$query" >/dev/null 2>"$errfile" && status=0 || status=$?
  else
    "${limit[@]}" env LD_LIBRARY_PATH="${lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="${pypath}" "$py" -c "$query" >/dev/null 2>"$errfile" && status=0 || status=$?
  fi
  if [ "$status" -ne 0 ]; then
    detail="$(tr '\n' ' ' < "$errfile")"
    echo "[ibvs] camera probe ${py} ${mode} exit=${status} ${detail:-no stderr}" >&2
  fi
  rm -f "$errfile"
  return "$status"
}

pick_camera_python() {
  local candidate seen=" " prefix lib pypath
  pypath="$(_system_camera_pythonpath)"
  if [ -x /usr/bin/python3 ]; then
    if _camera_query_ok /usr/bin/python3 clean "" "$SYSTEM_CAMERA_QUERY" "$pypath"; then
      printf '%s\nclean\n%s\n' /usr/bin/python3 "$pypath"
      return 0
    fi
    if _camera_query_ok /usr/bin/python3 inherit "" "$SYSTEM_CAMERA_QUERY" "$pypath"; then
      printf '%s\ninherit\n%s\n' /usr/bin/python3 "$pypath"
      return 0
    fi
  fi
  while IFS= read -r candidate; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    case "$seen" in
      *" $candidate "*) continue ;;
    esac
    seen="${seen}${candidate} "
    if _camera_query_ok "$candidate" clean; then
      printf '%s\nclean\n%s\n' "$candidate" "$IBVS_CAM_PYTHONPATH"
      return 0
    fi
    prefix="$(cd "$(dirname "$candidate")/.." && pwd)"
    lib="${prefix}/lib"
    if [ -d "$lib" ] && compgen -G "${lib}/librealsense2.so*" >/dev/null; then
      if _camera_query_ok "$candidate" prefix "$lib"; then
        printf '%s\n%s\n%s\n' "$candidate" "${lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$IBVS_CAM_PYTHONPATH"
        return 0
      fi
    fi
    echo "[ibvs] camera probe skip ${candidate}" >&2
  done < <(iter_ibvs_pythons)
  return 1
}

_note_ultralytics() {
  local py="/usr/bin/python3"
  local err="" site
  [ -x "$py" ] || return 0
  if PYTHONPATH="${IBVS_WEB_PYTHONPATH}" "$py" -c "import ultralytics" >/dev/null 2>&1; then
    echo "[ibvs] ultralytics ok  py=${py}"
    return 0
  fi
  for site in \
    "${HOME}/.local/lib/python3.10/site-packages" \
    /usr/local/lib/python3.10/dist-packages \
    "${HOME}/anaconda3/envs/"*/lib/python3.10/site-packages \
    "${HOME}/miniconda3/envs/"*/lib/python3.10/site-packages
  do
    if [ -d "${site}/ultralytics" ]; then
      IBVS_WEB_PYTHONPATH="${site}:${IBVS_WEB_PYTHONPATH}"
      export IBVS_WEB_PYTHONPATH
      echo "[ibvs] ultralytics from ${site}"
      return 0
    fi
  done
  err="$(PYTHONPATH="${IBVS_WEB_PYTHONPATH}" "$py" -c "import ultralytics" 2>&1 >/dev/null || true)"
  echo "[ibvs] WARN: ${py} cannot import ultralytics (${err:-import failed})"
}

pick_ibvs_python() {
  pick_python_importing "import flask, cv2; import cyclonedds._clayer" \
    || pick_python_importing "import flask, cv2" \
    || pick_python_importing "import cv2" \
    || pick_python_importing "import flask" \
    || pick_python_importing "import sys"
}

_note_ultralytics

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
