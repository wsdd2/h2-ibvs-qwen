#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/scripts/env.sh"
exec "$PY" scripts/run_ibvs.py "$@"
