#!/usr/bin/env bash
# Copy this tree to the H2 Ubuntu 22.04 machine. Does not install anything on the robot.
# Paths are relative to the *robot* login home. Do not let this WSL user expand ~.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${IBVS_ROBOT_HOST:-}"
DEST_IN="${IBVS_ROBOT_DIR:-IBVS_PAD_TOUCH}"

if [ -z "$HOST" ]; then
  echo "set IBVS_ROBOT_HOST, e.g. export IBVS_ROBOT_HOST=unitree@192.168.1.3" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required" >&2
  exit 1
fi

remote_dir() {
  local dest="$1"
  if [ -n "${HOME:-}" ] && [ "$dest" = "$HOME/IBVS_PAD_TOUCH" ]; then
    dest="IBVS_PAD_TOUCH"
  elif [ -n "${HOME:-}" ] && [[ "$dest" == "$HOME/"* ]]; then
    dest="${dest#"$HOME/"}"
  fi
  if [[ "$dest" == ~/* ]]; then
    dest="${dest:2}"
  elif [ "$dest" = "~" ]; then
    dest="."
  fi
  printf '%s' "$dest"
}

DEST="$(remote_dir "$DEST_IN")"

# New robot: create the folder as unitree, not as this WSL user.
ssh -o StrictHostKeyChecking=accept-new "$HOST" "mkdir -p -- $(printf %q "$DEST")"

rsync -az --info=progress2 \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude '.git/' \
  --exclude '*.pyc' \
  "$ROOT/" "$HOST:$DEST/"

# Arm motion imports these sibling modules at runtime. Keep them in lockstep
# with the dashboard so new IK entry points and dq feedforward cannot be missing.
ssh "$HOST" "mkdir -p ~/MscapeTech/Hand_Eye_Calib/handeye_calib"
for name in h2_ee_ik.py h2_arm_sdk.py; do
  local_path="$ROOT/../Hand_Eye_Calib/handeye_calib/$name"
  if [ -f "$local_path" ]; then
    rsync -az --info=progress2 \
      "$local_path" \
      "$HOST:MscapeTech/Hand_Eye_Calib/handeye_calib/$name"
  fi
done

# WSL /mnt/e is a Windows drive: shebang becomes bash\r unless we strip CR on the robot.
ssh "$HOST" "find $(printf %q "$DEST") -type f \\( -name '*.sh' -o -name '*.py' \\) -print0 | xargs -0 sed -i 's/\\r\$//'"
ssh "$HOST" "sed -i 's/\\r\$//' ~/MscapeTech/Hand_Eye_Calib/handeye_calib/h2_ee_ik.py ~/MscapeTech/Hand_Eye_Calib/handeye_calib/h2_arm_sdk.py 2>/dev/null || true"

echo
echo "uploaded to $HOST:~/$DEST"
echo "on robot:"
echo "  cd ~/$DEST"
echo "  chmod +x scripts/*.sh"
echo "  ./scripts/setup_ubuntu.sh --robot"
echo "  ./scripts/start_web_remote.sh 8080"
