cd /home/chenyu/workplace/soccer

ROOT_DIR="$(pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
mkdir -p "$MPLCONFIGDIR" "$ROOT_DIR/runs"

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

echo "=== GPU check ==="
nvidia-smi || exit 1

"$PYTHON_BIN" -u - <<'PY'
import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import cv2
import torch

from src.pipeline import _track_bbox
from src.runtime import build_detector, build_postprocessor, build_tracker
from src.server import (
    _add_decode_args,
    _add_detection_args,
    _add_postprocess_args,
    _add_team_args,
    _add_tracker_args,
    build_runtime_settings,
)

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，请先确认宿主机终端里 torch.cuda.is_available() 为 True")

logging.basicConfig(level=logging.WARNING)

root = Path("/home/chenyu/workplace/soccer")
source = root / "data" / "football.mp4"
output_dir = root / "runs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "football_overlay_full_gpu.avi"
summary_path = output_dir / "football_overlay_full_gpu.summary.json"

if not source.exists():
    raise SystemExit(f"找不到源视频: {source}")

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=str(source))
parser.add_argument("--fps", type=int, default=None)
parser.add_argument("--profile", choices=["auto", "nvidia", "cpu"], default="nvidia")
parser.add_argument("--record-path", default=None)
parser.add_argument("--log-level", default="warning")
_add_detection_args(parser)
_add_tracker_args(parser)
_add_decode_args(parser)
_add_team_args(parser)
_add_postprocess_args(parser)

args = parser.parse_args([
    "--source", str(source),
    "--profile", "nvidia",
    "--fps", "30",
    "--yolo-device", "cuda:0",
    "--ball-yolo-device", "cuda:0",
    "--yolo-half",
    "--no-ball-yolo-half",
    "--ball-yolo-model", str(root / "models" / "yolov5m-football-best.pt"),
    "--ball-yolo-backend", "yolov5",
    "--yolo-conf", "0.20",
    "--ball-yolo-conf", "0.12",
    "--ball-yolo-imgsz", "1536",
    "--track-buffer", "30",
    "--bytetrack-track-activation-threshold", "0.20",
    "--bytetrack-kalman-position-weight", "0.06",
    "--bytetrack-kalman-velocity-weight", "0.01125",
    "--decode-buffer-size", "4",
    "--decode-drop-policy", "drop_oldest",
    "--prefer-latest-frame",
    "--smooth-alpha", "0.35",
    "--reid-ttl-ms", "1500",
    "--reid-distance-px", "85.0",
])

settings = build_runtime_settings(args)

if not str(settings.yolo_device).startswith("cuda"):
    raise SystemExit(f"主检测器没有落到 GPU: {settings.yolo_device}")
if settings.ball_yolo_model and not str(settings.ball_yolo_device).startswith("cuda"):
    raise SystemExit(f"足球检测器没有落到 GPU: {settings.ball_yolo_device}")

print("runtime_settings:")
print(json.dumps({
    "effective_profile": settings.effective_profile,
    "yolo_device": settings.yolo_device,
    "yolo_half": settings.yolo_half,
    "ball_yolo_device": settings.ball_yolo_device,
    "ball_yolo_half": settings.ball_yolo_half,
    "yolo_model": settings.yolo_model,
    "ball_yolo_model": settings.ball_yolo_model,
}, ensure_ascii=False, indent=2))

detector = build_detector(settings)
tracker = build_tracker(settings)
postprocessor = build_postprocessor(settings)

cap = cv2.VideoCapture(str(source))
if not cap.isOpened():
    raise SystemExit(f"无法打开视频: {source}")

input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
if input_fps <= 0:
    input_fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

if width <= 0 or height <= 0:
    raise SystemExit("视频宽高无效")

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"XVID"),
    input_fps,
    (width, height),
)
if not writer.isOpened():
    raise SystemExit(f"无法创建输出视频: {output_path}")

team_colors = {
    "A": (66, 135, 245),
    "B": (52, 199, 89),
    "red": (76, 76, 255),
    "blue": (255, 140, 66),
    "unknown": (255, 255, 255),
}
kind_colors = {
    "ball": (33, 189, 255),
    "player": (90, 220, 255),
}

def track_color(track):
    if int(getattr(track, "missed_frames", 0)) > 0:
        return (0, 165, 255)
    team = str(getattr(track, "team", "unknown") or "unknown")
    kind = str(getattr(track, "kind", "player") or "player")
    return team_colors.get(team, kind_colors.get(kind, (255, 255, 255)))

def draw_label(frame, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = max(0, min(int(x1), frame.shape[1] - tw - 8))
    ty = max(0, int(y1) - th - baseline - 8)
    cv2.rectangle(frame, (tx, ty), (tx + tw + 8, ty + th + baseline + 6), (15, 18, 20), -1)
    cv2.putText(frame, text, (tx + 4, ty + th + 1), font, scale, color, thickness, cv2.LINE_AA)

def draw_overlays(frame, detections, tracks):
    overlay = frame.copy()

    for det in detections:
        x1 = int(det.x)
        y1 = int(det.y)
        x2 = int(det.x + det.w)
        y2 = int(det.y + det.h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (87, 224, 255), 1)
        draw_label(overlay, x1, y1, f"det {det.kind} {det.confidence:.2f}", (255, 255, 255))

    for track in tracks:
        x, y, w, h = _track_bbox(track)
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        color = track_color(track)
        thickness = 3 if getattr(track, "kind", "") == "ball" else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        parts = [
            str(getattr(track, "kind", "track")),
            f"#{int(getattr(track, 'track_id', -1))}",
        ]
        team = str(getattr(track, "team", "unknown") or "unknown")
        if team != "unknown":
            parts.append(team)
        parts.append(f"{float(getattr(track, 'confidence', 0.0)):.2f}")
        if int(getattr(track, "missed_frames", 0)) > 0:
            parts.append("pred")

        draw_label(overlay, x1, y1, " ".join(parts), color)

    return overlay

batch_size = max(1, int(getattr(detector, "batch_size", 1)))
frames = []
frame_indices = []
processed = 0

detect_ms_values = []
track_ms_values = []
post_ms_values = []
tracks_per_frame = []
detections_per_frame = []

start = time.perf_counter()

def flush_batch():
    global processed
    if not frames:
        return

    t0 = time.perf_counter()
    detections_batch = detector.detect_many(frames)
    detect_ms_per_frame = ((time.perf_counter() - t0) * 1000.0) / max(1, len(frames))

    for frame, frame_idx, detections in zip(frames, frame_indices, detections_batch):
        ts_ms = int(round(frame_idx * 1000.0 / input_fps))

        t1 = time.perf_counter()
        raw_tracks = tracker.update(detections, ts_ms, frame)
        t2 = time.perf_counter()
        tracks = postprocessor.update(raw_tracks, ts_ms)
        t3 = time.perf_counter()

        overlay = draw_overlays(frame, detections, tracks)
        writer.write(overlay)

        detect_ms_values.append(detect_ms_per_frame)
        track_ms_values.append((t2 - t1) * 1000.0)
        post_ms_values.append((t3 - t2) * 1000.0)
        tracks_per_frame.append(len(tracks))
        detections_per_frame.append(len(detections))

        processed += 1
        if processed % 30 == 0 or processed == frame_count:
            elapsed = time.perf_counter() - start
            fps = processed / elapsed if elapsed > 0 else 0.0
            print(f"processed {processed}/{frame_count} frames | render_fps={fps:.3f}")

    frames.clear()
    frame_indices.clear()

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
    frame_indices.append(frame_idx)
    frame_idx += 1
    if len(frames) >= batch_size:
        flush_batch()

flush_batch()
cap.release()
writer.release()

wall_time_s = time.perf_counter() - start

summary = {
    "source": str(source),
    "output": str(output_path),
    "summary": str(summary_path),
    "effective_profile": settings.effective_profile,
    "yolo_device": settings.yolo_device,
    "ball_yolo_device": settings.ball_yolo_device,
    "input_fps": round(input_fps, 3),
    "frames": processed,
    "video_duration_s": round(processed / input_fps, 3) if input_fps > 0 else 0.0,
    "wall_time_s": round(wall_time_s, 3),
    "offline_render_fps": round(processed / wall_time_s, 3) if wall_time_s > 0 else 0.0,
    "detect_ms_avg": round(statistics.fmean(detect_ms_values), 3) if detect_ms_values else 0.0,
    "track_ms_avg": round(statistics.fmean(track_ms_values), 3) if track_ms_values else 0.0,
    "post_ms_avg": round(statistics.fmean(post_ms_values), 3) if post_ms_values else 0.0,
    "detections_per_frame_avg": round(statistics.fmean(detections_per_frame), 3) if detections_per_frame else 0.0,
    "tracks_per_frame_avg": round(statistics.fmean(tracks_per_frame), 3) if tracks_per_frame else 0.0,
}

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nfinished")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY