#!/bin/zsh
# Start the local Foundation service on demand (macOS).
#
# Studio launchers call this before they start. When the service already
# answers /health nothing is started. Otherwise the service is started detached
# from this repository's .venv, and the script waits until /health answers.
# The single-instance lock in runtime_foundation keeps concurrent callers from
# starting a second service; the loser exits and its caller sees the winner.
set -euo pipefail

ROOT_DIR="${0:A:h:h:h}"
PYTHON="$ROOT_DIR/.venv/bin/python"
HOST="${RUNTIME_FOUNDATION_HOST:-127.0.0.1}"
PORT="${RUNTIME_FOUNDATION_PORT:-8765}"
HEALTH_URL="http://${HOST}:${PORT}/health"
LOG_DIR="${HOME}/Library/Logs/local-llm-runtime-foundation"
LOG_FILE="${LOG_DIR}/service.log"
TIMEOUT_SECONDS="${FOUNDATION_START_TIMEOUT_SECONDS:-60}"

healthy() {
  curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1
}

if healthy; then
  echo "Foundation is already running: http://${HOST}:${PORT}"
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Foundation .venv is not available: $PYTHON" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting Foundation on ${HOST}:${PORT}" >>"$LOG_FILE"
(
  cd "$ROOT_DIR"
  RUNTIME_FOUNDATION_HOST="$HOST" RUNTIME_FOUNDATION_PORT="$PORT" \
    nohup "$PYTHON" -m runtime_foundation >>"$LOG_FILE" 2>&1 </dev/null &
)

for _ in $(seq 1 $((TIMEOUT_SECONDS * 4))); do
  if healthy; then
    echo "Foundation started: http://${HOST}:${PORT}"
    exit 0
  fi
  sleep 0.25
done

echo "Foundation did not become ready within ${TIMEOUT_SECONDS}s. Log: ${LOG_FILE}" >&2
exit 1
