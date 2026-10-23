# Session log for a foreground start_web_remote.sh run.
# The file is ibvs_<start>_running.log while the task is up, then renamed to
# ibvs_<start>_<end>.log. Both stamps are YYYYMMDD_HHMMSS.

session_log_prune() {
  local dir="$1"
  local keep="$2"
  local files=()
  local path
  [ -d "$dir" ] || return 0
  shopt -s nullglob
  local list=( "$dir"/ibvs_*.log )
  shopt -u nullglob
  if [ "${#list[@]}" -eq 0 ]; then
    return 0
  fi
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    files+=("$path")
  done < <(ls -1tr "${list[@]}")
  local count="${#files[@]}"
  local overflow=$((count - keep))
  local i
  if [ "$overflow" -le 0 ]; then
    return 0
  fi
  for ((i = 0; i < overflow; i++)); do
    rm -f -- "${files[$i]}"
  done
  echo "[ibvs] 日志目录超过 50 条，已删除 ${overflow} 条最旧日志"
}

session_log_begin() {
  local dir="$1"
  mkdir -p "$dir"
  # Leave room for this run so the directory holds at most 50 logs.
  session_log_prune "$dir" 49
  local start
  start="$(date +%Y%m%d_%H%M%S)"
  local file="${dir}/ibvs_${start}_running.log"
  export IBVS_SESSION_LOG_DIR="$dir"
  export IBVS_SESSION_START="$start"
  export IBVS_SESSION_LOG="$file"
  export IBVS_SESSION_LOG_FINISHED=0
  : > "$file"
  # tee ignores Ctrl+C so the exit line is still written.
  exec > >(trap '' INT TERM; tee -a "$file") 2>&1
  echo "[ibvs] 任务开始 ${start}  日志先写入 ${file}"
}

session_log_finish() {
  if [ "${IBVS_SESSION_LOG_FINISHED:-0}" = 1 ]; then
    return 0
  fi
  export IBVS_SESSION_LOG_FINISHED=1
  local src="${IBVS_SESSION_LOG:-}"
  local start="${IBVS_SESSION_START:-}"
  local dir="${IBVS_SESSION_LOG_DIR:-}"
  if [ -z "$src" ] || [ -z "$start" ] || [ -z "$dir" ]; then
    return 0
  fi
  local end dest
  end="$(date +%Y%m%d_%H%M%S)"
  dest="${dir}/ibvs_${start}_${end}.log"
  if [ -e "$dest" ]; then
    dest="${dir}/ibvs_${start}_${end}_$$.log"
  fi
  if [ -f "$src" ] && [ "$src" != "$dest" ]; then
    mv -f -- "$src" "$dest" 2>/dev/null || dest="$src"
  elif [ ! -f "$src" ]; then
    dest="$src"
  fi
  echo "[ibvs] 本次日志: ${dest}"
}
