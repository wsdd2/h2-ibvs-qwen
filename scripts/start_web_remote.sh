#!/usr/bin/env bash
# Foreground Flask + joint sidecar. h2_arm_service_node is not started here.
# The Service button starts that node; the arm_sdk button enables arm_sdk.
# Ctrl+C lowers the arm, then stops a service node if one is up. Use --detach only for SSH deploy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/scripts/env.sh"

DETACH=0
if [ "${1:-}" = "--detach" ]; then
  DETACH=1
  shift
fi

PORT="${1:-8080}"
EXTRA=()
shift || true
while [ "$#" -gt 0 ]; do
  EXTRA+=("$1")
  shift
done

# Robot PC2 keeps eth0 (192.168.123.162). Orin has no eth0; its link to the
# robot PC2 Wi-Fi is wlan0 (192.168.1.72).
IFACE="eth0"
for i in "${!EXTRA[@]}"; do
  if [ "${EXTRA[$i]}" = "--iface" ] && [ $((i + 1)) -lt ${#EXTRA[@]} ]; then
    IFACE="${EXTRA[$((i + 1))]}"
  fi
done
if ! ip link show "$IFACE" >/dev/null 2>&1; then
  if ip link show wlan0 >/dev/null 2>&1; then
    echo "[ibvs] ${IFACE} is not on this machine; DDS uses wlan0"
    replaced=0
    for i in "${!EXTRA[@]}"; do
      if [ "${EXTRA[$i]}" = "--iface" ] && [ $((i + 1)) -lt ${#EXTRA[@]} ]; then
        EXTRA[$((i + 1))]="wlan0"
        replaced=1
      fi
    done
    if [ "$replaced" -eq 0 ]; then
      EXTRA+=("--iface" "wlan0")
    fi
    IFACE="wlan0"
  else
    echo "[ibvs] WARN: DDS iface ${IFACE} is not on this machine" >&2
  fi
fi
echo "[ibvs] dds iface=${IFACE}"

if [ "$DETACH" -eq 0 ]; then
  # shellcheck source=session_log.sh
  . "$ROOT/scripts/session_log.sh"
  session_log_begin "${IBVS_LOG_DIR:-$ROOT/logs}"
  trap session_log_finish EXIT
fi

if [ -r /proc/version ] && grep -qi microsoft /proc/version; then
  echo "[ibvs] this is WSL, not the H2. ssh unitree@192.168.1.3 then run this script." >&2
fi

# Login shells export this from ~/.bashrc. SSH detach does not, so read the same line.
if [ -z "${ROS_DOMAIN_ID:-}" ] && [ -f "${HOME}/.bashrc" ]; then
  _domain_line="$(grep -E '^[[:space:]]*export[[:space:]]+ROS_DOMAIN_ID=[0-9]+' "${HOME}/.bashrc" | tail -n 1 || true)"
  _domain_value="${_domain_line##*=}"
  _domain_value="${_domain_value//\"/}"
  _domain_value="${_domain_value//\'/}"
  if [[ "${_domain_value}" =~ ^[0-9]+$ ]]; then
    export ROS_DOMAIN_ID="${_domain_value}"
    echo "[ibvs] ROS_DOMAIN_ID was unset; using ${_domain_value} from ~/.bashrc"
  fi
  unset _domain_line _domain_value
fi

stop_pidfile() {
  local file="$1"
  if [ -f "$file" ]; then
    local old
    old="$(cat "$file" || true)"
    if [ -n "${old}" ] && kill -0 "${old}" 2>/dev/null; then
      kill "${old}" 2>/dev/null || true
      sleep 0.2
    fi
    rm -f "$file"
  fi
}

h2_arm_service_alive() {
  # The kernel keeps 15 characters of the process name.
  pgrep -x h2_arm_service_ >/dev/null 2>&1 && return 0
  pgrep -f '[h]2_arm_service_node' >/dev/null 2>&1 && return 0
  return 1
}

stop_h2_arm_service() {
  local pid=""
  if [ -f /tmp/ibvs_h2_arm_service.pid ]; then
    pid="$(cat /tmp/ibvs_h2_arm_service.pid 2>/dev/null || true)"
  fi
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    # setsid makes this pid the group leader; the node is in that group.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    sleep 0.3
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  # comm is truncated to 15 characters: h2_arm_service_
  if h2_arm_service_alive; then
    pkill -x h2_arm_service_ >/dev/null 2>&1 || true
    pkill -f '[h]2_arm_service_node' >/dev/null 2>&1 || true
    sleep 0.2
    if h2_arm_service_alive; then
      pkill -9 -x h2_arm_service_ >/dev/null 2>&1 || true
      pkill -9 -f '[h]2_arm_service_node' >/dev/null 2>&1 || true
    fi
  fi
  rm -f /tmp/ibvs_h2_arm_service.pid
}

# Kept for a manual recovery shell. Boot does not call this.
start_h2_arm_service() {
  if [ ! -f /opt/ros/humble/setup.bash ] || [ ! -f "${HOME}/h2arm_control_ws/install/setup.bash" ]; then
    echo "[ibvs] WARN: skip h2_arm_service_node; ROS humble or ~/h2arm_control_ws/install is missing" >&2
    return 0
  fi
  if [ -z "${ROS_DOMAIN_ID:-}" ]; then
    echo "[ibvs] WARN: ROS_DOMAIN_ID unset; h2_arm_service_node and the web will use domain 0" >&2
  else
    echo "[ibvs] h2_arm_service_node ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
  fi
  echo "[ibvs] Service link uses this node. Do not Start on arm_sdk while it is running."
  rm -f /tmp/ibvs_h2_arm_service.log /tmp/ibvs_h2_arm_service.pid
  # Own session so Ctrl+C can lower the arm through the service before cleanup stops it.
  setsid bash -c '
    echo $$ > /tmp/ibvs_h2_arm_service.pid
    set +u
    source /opt/ros/humble/setup.bash
    source "$HOME/h2arm_control_ws/install/setup.bash"
    if [ -n "${ROS_DOMAIN_ID:-}" ]; then export ROS_DOMAIN_ID; fi
    if [ -n "${ROS_LOCALHOST_ONLY:-}" ]; then export ROS_LOCALHOST_ONLY; fi
    if [ -n "${RMW_IMPLEMENTATION:-}" ]; then export RMW_IMPLEMENTATION; fi
    exec ros2 run h2arm_control h2_arm_service_node
  ' > /tmp/ibvs_h2_arm_service.log 2>&1 < /dev/null &
}

wait_h2_arm_service() {
  local i pid=""
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if [ -f /tmp/ibvs_h2_arm_service.pid ]; then
      pid="$(cat /tmp/ibvs_h2_arm_service.pid 2>/dev/null || true)"
    fi
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null && h2_arm_service_alive; then
      echo "[ibvs] h2_arm_service_node pid=${pid}  log=/tmp/ibvs_h2_arm_service.log"
      return 0
    fi
    sleep 0.4
  done
  echo "[ibvs] WARN: h2_arm_service_node did not stay up. See /tmp/ibvs_h2_arm_service.log" >&2
  tail -n 20 /tmp/ibvs_h2_arm_service.log >&2 || true
  return 0
}

stop_pidfile /tmp/ibvs_web.pid
stop_pidfile /tmp/ibvs_joints.pid
stop_pidfile /tmp/ibvs_cam_chest.pid
stop_pidfile /tmp/ibvs_cam_d435i.pid
echo "[ibvs] h2_arm_service_node is not started at boot. Stopping a leftover node from the last run."
stop_h2_arm_service

JOINT_PY="$PY"
if SDK_PY="$(pick_python_importing "import unitree_sdk2py")"; then
  JOINT_PY="$SDK_PY"
elif [ -x "${HOME}/anaconda3/envs/unitree/bin/python" ]; then
  JOINT_PY="${HOME}/anaconda3/envs/unitree/bin/python"
fi
if WEB_PY="$(pick_python_importing "import flask, cv2")"; then
  PY="$WEB_PY"
elif WEB_PY="$(pick_python_importing "import cv2")"; then
  PY="$WEB_PY"
  echo "[ibvs] web python has cv2 but may miss flask: $PY" >&2
else
  echo "[ibvs] no python with cv2. Scan:" >&2
  while IFS= read -r candidate; do
    [ -x "$candidate" ] || continue
    echo -n "  $candidate  " >&2
    "$candidate" -c "import cv2,flask; print('cv2+flask', cv2.__version__)" 2>/dev/null \
      || "$candidate" -c "import cv2; print('cv2', cv2.__version__)" 2>/dev/null \
      || echo "no cv2" >&2
  done < <(iter_ibvs_pythons)
  echo "[ibvs] old H2 used /usr/bin/python3 (apt opencv) for the page, h1_arm only for joints." >&2
  exit 1
fi

JOINT_PID=""
WEB_PID=""
CAM_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  if [ -n "${WEB_PID}" ] && kill -0 "${WEB_PID}" 2>/dev/null; then
    echo "[ibvs] Ctrl+C: wait for web to lower the right arm..."
    # Ctrl+C reaches the foreground Python too. Let its SIGINT handler lower
    # the arm; an immediate SIGTERM used to interrupt that sequence.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
      kill -0 "${WEB_PID}" 2>/dev/null || break
      sleep 0.6
    done
    if kill -0 "${WEB_PID}" 2>/dev/null; then
      echo "[ibvs] web still running after lower timeout, send TERM"
      kill -TERM "${WEB_PID}" 2>/dev/null || true
      sleep 2
    fi
    if kill -0 "${WEB_PID}" 2>/dev/null; then
      echo "[ibvs] web still running after TERM, kill -9"
      kill -9 "${WEB_PID}" 2>/dev/null || true
    fi
  fi
  if [ -n "${JOINT_PID}" ] && kill -0 "${JOINT_PID}" 2>/dev/null; then
    kill "${JOINT_PID}" 2>/dev/null || true
  fi
  local pid
  for pid in "${CAM_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  local _cam_wait
  for _cam_wait in 1 2 3 4 5 6 7 8 9 10; do
    local _cam_alive=0
    for pid in "${CAM_PIDS[@]:-}"; do
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        _cam_alive=1
      fi
    done
    [ "${_cam_alive}" -eq 0 ] && break
    sleep 0.4
  done
  for pid in "${CAM_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[ibvs] camera ${pid} did not stop, kill -9"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  wait "${WEB_PID}" "${JOINT_PID}" "${CAM_PIDS[@]:-}" 2>/dev/null || true
  # After the web lowers the arm. The node must stay up during that sequence.
  stop_h2_arm_service
  rm -f /tmp/ibvs_web.pid /tmp/ibvs_joints.pid /tmp/ibvs_cam_chest.pid /tmp/ibvs_cam_d435i.pid
  rm -f /tmp/ibvs_cam_chest.serial /tmp/ibvs_cam_d435i.serial \
        /tmp/ibvs_cam_chest.jpg /tmp/ibvs_cam_chest.ppm /tmp/ibvs_cam_chest.bgr \
        /tmp/ibvs_cam_d435i.jpg /tmp/ibvs_cam_d435i.ppm /tmp/ibvs_cam_d435i.bgr \
        /tmp/ibvs_cam_seen.txt /tmp/ibvs_depth_chest.bin /tmp/ibvs_depth_d435i.bin
  session_log_finish
}

clear_camera_files() {
  rm -f /tmp/ibvs_cam_chest.serial /tmp/ibvs_cam_d435i.serial \
        /tmp/ibvs_cam_chest.jpg /tmp/ibvs_cam_chest.ppm /tmp/ibvs_cam_chest.bgr \
        /tmp/ibvs_cam_d435i.jpg /tmp/ibvs_cam_d435i.ppm /tmp/ibvs_cam_d435i.bgr \
        /tmp/ibvs_cam_seen.txt /tmp/ibvs_depth_chest.bin /tmp/ibvs_depth_d435i.bin
}

launch_camera_sidecar() {
  local wrap=()
  if [ "${1:-}" = "--nohup" ]; then
    wrap=(nohup)
    shift
  fi
  local name="$1"
  local serial="$2"
  local out="$3"
  local depth="$4"
  shift 4
  case "${IBVS_CAM_LD_LIBRARY_PATH:-clean}" in
    inherit)
      "${wrap[@]}" env \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" \
        --name "$name" --serial "$serial" --out "$out" \
        --lock-exposure-after 0 --depth-out "$depth" "$@"
      ;;
    clean|"")
      "${wrap[@]}" env -u LD_LIBRARY_PATH \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" \
        --name "$name" --serial "$serial" --out "$out" \
        --lock-exposure-after 0 --depth-out "$depth" "$@"
      ;;
    *)
      "${wrap[@]}" env LD_LIBRARY_PATH="${IBVS_CAM_LD_LIBRARY_PATH}" \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" \
        --name "$name" --serial "$serial" --out "$out" \
        --lock-exposure-after 0 --depth-out "$depth" "$@"
      ;;
  esac
}

launch_camera_pair() {
  local wrap=()
  if [ "${1:-}" = "--nohup" ]; then
    wrap=(nohup)
    shift
  fi
  local py_args=(--dual)
  local spec
  for spec in "$@"; do
    py_args+=(--cam "$spec")
  done
  case "${IBVS_CAM_LD_LIBRARY_PATH:-clean}" in
    inherit)
      exec "${wrap[@]}" env \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" "${py_args[@]}"
      ;;
    clean|"")
      exec "${wrap[@]}" env -u LD_LIBRARY_PATH \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" "${py_args[@]}"
      ;;
    *)
      exec "${wrap[@]}" env LD_LIBRARY_PATH="${IBVS_CAM_LD_LIBRARY_PATH}" \
        PYTHONPATH="${IBVS_CAM_PYTHONPATH}" PYTHONUNBUFFERED=1 \
        "$CAM_PY" "$ROOT/scripts/camera_sidecar.py" "${py_args[@]}"
      ;;
  esac
}

start_camera_sidecars() {
  if [ -z "${CAM_PY:-}" ]; then
    echo "[ibvs] WARN: cameras not started; no pyrealsense2 survived query_devices" >&2
    return 0
  fi
  echo "[ibvs] camera sidecar py=${CAM_PY}  PYTHONPATH=${IBVS_CAM_PYTHONPATH}"
  case "${IBVS_CAM_LD_LIBRARY_PATH:-clean}" in
    inherit) echo "[ibvs] camera LD_LIBRARY_PATH inherited" ;;
    clean|"") echo "[ibvs] camera LD_LIBRARY_PATH cleared" ;;
    *) echo "[ibvs] camera LD_LIBRARY_PATH=${IBVS_CAM_LD_LIBRARY_PATH}" ;;
  esac
  clear_camera_files
  # One process opens both cameras. A second process sees usb=(none)
  # after the first pipeline.start() on this USB hub.
  launch_camera_pair \
    "chest:347622074029:/tmp/ibvs_cam_chest.jpg:/tmp/ibvs_depth_chest.bin" \
    "d435i:233722073797:/tmp/ibvs_cam_d435i.jpg:/tmp/ibvs_depth_d435i.bin" \
    > >(sed -u 's/^/[cam] /') 2>&1 &
  CAM_PIDS+=("$!")
  echo $! > /tmp/ibvs_cam_chest.pid
  echo $! > /tmp/ibvs_cam_d435i.pid
}

CAM_PY=""
IBVS_CAM_LD_LIBRARY_PATH=""
pkill -TERM -f '[c]amera_sidecar.py' >/dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8; do
  pgrep -f '[c]amera_sidecar.py' >/dev/null 2>&1 || break
  sleep 0.4
done
if _cam_pick="$(pick_camera_python)"; then
  CAM_PY="$(printf '%s\n' "$_cam_pick" | head -n 1)"
  IBVS_CAM_LD_LIBRARY_PATH="$(printf '%s\n' "$_cam_pick" | sed -n '2p')"
  _cam_pypath="$(printf '%s\n' "$_cam_pick" | sed -n '3p')"
  if [ -n "${_cam_pypath}" ]; then
    IBVS_CAM_PYTHONPATH="${_cam_pypath}"
  fi
  unset _cam_pypath
else
  echo "[ibvs] WARN: every pyrealsense2 crashed in rs.context().query_devices" >&2
  echo "[ibvs] Close any hand-eye or realsense process still holding the cameras, then start again." >&2
fi
unset _cam_pick

if [ "$DETACH" -eq 1 ]; then
  echo "[ibvs] h2_arm_service_node stays down until the Service button."
  PYTHONPATH="${IBVS_WEB_PYTHONPATH}" nohup "$JOINT_PY" "$ROOT/scripts/joint_sidecar.py" "${IFACE}" > /tmp/ibvs_joints.log 2>&1 &
  echo $! > /tmp/ibvs_joints.pid
  if [ -n "${CAM_PY:-}" ]; then
    clear_camera_files
    launch_camera_pair --nohup \
      "chest:347622074029:/tmp/ibvs_cam_chest.jpg:/tmp/ibvs_depth_chest.bin" \
      "d435i:233722073797:/tmp/ibvs_cam_d435i.jpg:/tmp/ibvs_depth_d435i.bin" \
      > /tmp/ibvs_cam_chest.log 2>&1 &
    echo $! > /tmp/ibvs_cam_chest.pid
    echo $! > /tmp/ibvs_cam_d435i.pid
  fi
  PYTHONPATH="${IBVS_WEB_PYTHONPATH}" nohup "$PY" "$ROOT/scripts/run_web.py" --host 0.0.0.0 --port "${PORT}" --confirm-arm-motion "${EXTRA[@]}" \
    > /tmp/ibvs_web.log 2>&1 &
  echo $! > /tmp/ibvs_web.pid
  echo "detached web=$(cat /tmp/ibvs_web.pid) port=${PORT}"
  sleep 1.2
  tail -n 20 /tmp/ibvs_web.log || true
  exit 0
fi

trap cleanup EXIT INT TERM
echo "[ibvs] foreground  Ctrl+C stops web + joints + cameras, and a Service node if one was started"
echo "[ibvs] joints py=${JOINT_PY}"
echo "[ibvs] web    py=${PY}  http://0.0.0.0:${PORT}/"
if PYTHONPATH="${IBVS_WEB_PYTHONPATH}" "$PY" -c "import cyclonedds._clayer" >/dev/null 2>&1; then
  echo "[ibvs] cyclonedds._clayer ok  py=${PY}"
else
  echo "[ibvs] WARN: ${PY} has no cyclonedds._clayer; Start cannot raise the arm" >&2
  echo "[ibvs] export PYTHONPATH=/home/unitree/.local/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:/home/unitree/MscapeTech/unitree_sdk2_python:\$PYTHONPATH" >&2
fi
echo "[ibvs] arm_sdk and h2_arm_service_node stay down until their buttons. AI stand first."
echo "[ibvs] Ctrl+C lowers the right arm (~8s), then stops h2_arm_service_node if Service was used."
start_camera_sidecars

PYTHONPATH="${IBVS_WEB_PYTHONPATH}" PYTHONUNBUFFERED=1 "$JOINT_PY" "$ROOT/scripts/joint_sidecar.py" "${IFACE}" \
  > >(sed -u 's/^/[joints] /') 2>&1 &
JOINT_PID=$!
echo $! > /tmp/ibvs_joints.pid

PYTHONPATH="${IBVS_WEB_PYTHONPATH}" PYTHONUNBUFFERED=1 "$PY" "$ROOT/scripts/run_web.py" --host 0.0.0.0 --port "${PORT}" --confirm-arm-motion "${EXTRA[@]}" &
WEB_PID=$!
echo $! > /tmp/ibvs_web.pid
wait "${WEB_PID}"
