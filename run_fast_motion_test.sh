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

BALL_MODEL_DEFAULT=""
if [[ -f "$ROOT_DIR/models/yolov5m-football-best.pt" ]]; then
  BALL_MODEL_DEFAULT="$ROOT_DIR/models/yolov5m-football-best.pt"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
mkdir -p "$MPLCONFIGDIR"

CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "$CUDA_ROOT" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-13.1 /usr/local/cuda-12.9; do
    if [[ -d "$candidate" ]]; then
      CUDA_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -n "$CUDA_ROOT" && -d "$CUDA_ROOT" ]]; then
  export CUDA_HOME="$CUDA_ROOT"
  export CUDA_PATH="$CUDA_ROOT"
  export PATH="$CUDA_ROOT/bin:$PATH"

  CUDA_LIB_DIRS=()
  if [[ -d "$CUDA_ROOT/lib64" ]]; then
    CUDA_LIB_DIRS+=("$CUDA_ROOT/lib64")
  fi
  if [[ -d "$CUDA_ROOT/targets/x86_64-linux/lib" ]]; then
    CUDA_LIB_DIRS+=("$CUDA_ROOT/targets/x86_64-linux/lib")
  fi
  if [[ -d "/usr/lib/wsl/lib" ]]; then
    CUDA_LIB_DIRS=("/usr/lib/wsl/lib" "${CUDA_LIB_DIRS[@]}")
  fi

  if [[ ${#CUDA_LIB_DIRS[@]} -gt 0 ]]; then
    CUDA_LIB_PATH="$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")"
    export LD_LIBRARY_PATH="${CUDA_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

BALL_HALF_VALUE="${MVP_BALL_YOLO_HALF:-false}"
BALL_HALF_FLAG="--no-ball-yolo-half"
case "${BALL_HALF_VALUE,,}" in
  1|true|yes|on)
    BALL_HALF_FLAG="--ball-yolo-half"
    ;;
esac

SERVER_ARGS=(
  --source "$SOURCE"
  --profile "${MVP_PROFILE:-auto}"
  --fps "${MVP_FPS:-30}"
  --yolo-conf "${MVP_YOLO_CONF:-0.20}"
  --ball-yolo-model "${MVP_BALL_YOLO_MODEL:-$BALL_MODEL_DEFAULT}"
  --ball-yolo-device "${MVP_BALL_YOLO_DEVICE:-cuda:0}"
  "$BALL_HALF_FLAG"
  --ball-yolo-backend "${MVP_BALL_YOLO_BACKEND:-yolov5}"
  --ball-yolo-conf "${MVP_BALL_YOLO_CONF:-0.12}"
  --ball-yolo-imgsz "${MVP_BALL_YOLO_IMGSZ:-1536}"
  --ball-min-area "${MVP_BALL_MIN_AREA:-6.0}"
  --ball-max-area "${MVP_BALL_MAX_AREA:-3000.0}"
  --ball-max-aspect-ratio "${MVP_BALL_MAX_ASPECT_RATIO:-2.5}"
  --ball-max-detections "${MVP_BALL_MAX_DETECTIONS:-1}"
  --track-buffer "${MVP_TRACK_BUFFER:-30}"
  --bytetrack-track-activation-threshold "${MVP_BYTETRACK_TRACK_ACTIVATION_THRESHOLD:-0.20}"
  --bytetrack-kalman-position-weight "${MVP_BYTETRACK_KALMAN_POSITION_WEIGHT:-0.06}"
  --bytetrack-kalman-velocity-weight "${MVP_BYTETRACK_KALMAN_VELOCITY_WEIGHT:-0.01125}"
  --decode-buffer-size "${MVP_DECODE_BUFFER_SIZE:-4}"
  --decode-drop-policy "${MVP_DECODE_DROP_POLICY:-drop_oldest}"
  --prefer-latest-frame
  --smooth-alpha "${MVP_SMOOTH_ALPHA:-0.35}"
  --reid-ttl-ms "${MVP_REID_TTL_MS:-1500}"
  --reid-distance-px "${MVP_REID_DISTANCE_PX:-85.0}"
  --host "${MVP_HOST:-0.0.0.0}"
  --port "$PORT"
  --log-level "${MVP_LOG_LEVEL:-info}"
)

exec "$PYTHON_BIN" -m src.server "${SERVER_ARGS[@]}" "$@"
