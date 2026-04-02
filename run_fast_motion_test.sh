#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

SOURCE="${MVP_SOURCE:-$ROOT_DIR/data/football.mp4}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  SOURCE="$1"
  shift
fi

PORT="${MVP_PORT:-8765}"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--port" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    PORT="${ARGS[i + 1]}"
  elif [[ "${ARGS[i]}" == --port=* ]]; then
    PORT="${ARGS[i]#--port=}"
  fi
done

find_listen_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null || true
    return
  fi
}

PORT_PIDS="$(find_listen_pids "$PORT")"
if [[ -n "${PORT_PIDS// /}" ]]; then
  kill $PORT_PIDS 2>/dev/null || true
  sleep 1
  PORT_PIDS="$(find_listen_pids "$PORT")"
  if [[ -n "${PORT_PIDS// /}" ]]; then
    kill -9 $PORT_PIDS 2>/dev/null || true
    sleep 1
  fi
fi

exec "$PYTHON_BIN" -m src.server \
  --source "$SOURCE" \
  --profile "${MVP_PROFILE:-auto}" \
  --fps "${MVP_FPS:-30}" \
  --yolo-conf "${MVP_YOLO_CONF:-0.20}" \
  --track-buffer "${MVP_TRACK_BUFFER:-30}" \
  --bytetrack-track-activation-threshold "${MVP_BYTETRACK_TRACK_ACTIVATION_THRESHOLD:-0.20}" \
  --bytetrack-kalman-position-weight "${MVP_BYTETRACK_KALMAN_POSITION_WEIGHT:-0.06}" \
  --bytetrack-kalman-velocity-weight "${MVP_BYTETRACK_KALMAN_VELOCITY_WEIGHT:-0.01125}" \
  --decode-buffer-size "${MVP_DECODE_BUFFER_SIZE:-4}" \
  --decode-drop-policy "${MVP_DECODE_DROP_POLICY:-drop_oldest}" \
  --prefer-latest-frame \
  --smooth-alpha "${MVP_SMOOTH_ALPHA:-0.35}" \
  --reid-ttl-ms "${MVP_REID_TTL_MS:-1500}" \
  --reid-distance-px "${MVP_REID_DISTANCE_PX:-85.0}" \
  --host "${MVP_HOST:-0.0.0.0}" \
  --port "$PORT" \
  --log-level "${MVP_LOG_LEVEL:-info}" \
  "$@"
