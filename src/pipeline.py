from __future__ import annotations

import asyncio
import time
from typing import Any

from .detector import DetectorProtocol, HeuristicDetector
from .postprocess import TrackPostProcessor
from .projector import PITCH_LENGTH_M, PITCH_WIDTH_M, LinearProjector, ProjectorProtocol
from .recorder import JsonlRecorder
from .tracker import NearestTracker, TrackerProtocol
from .video_reader import BufferedVideoReader, DecodeConfig

SCHEMA_VERSION = "1.3"


def _fallback_bbox(kind: str, x_px: float, y_px: float) -> tuple[float, float, float, float]:
    if kind == "ball":
        size = 10.0
        return (x_px - size / 2.0, y_px - size / 2.0, size, size)

    width = 24.0
    height = 52.0
    return (x_px - width / 2.0, y_px - height, width, height)


class StreamProcessor:
    def __init__(
        self,
        source: str,
        target_fps: int = 15,
        reconnect_sleep_s: float = 2.0,
        detector: DetectorProtocol | None = None,
        tracker: TrackerProtocol | None = None,
        projector: ProjectorProtocol | None = None,
        decode_config: DecodeConfig | None = None,
        postprocessor: TrackPostProcessor | None = None,
        recorder: JsonlRecorder | None = None,
        prefer_latest_frame: bool = True,
    ) -> None:
        self.source_arg = source
        self.target_fps = target_fps
        self.reconnect_sleep_s = reconnect_sleep_s
        self.detector = detector or HeuristicDetector()
        self.tracker = tracker or NearestTracker()
        self.projector = projector or LinearProjector()
        self.reader = BufferedVideoReader(
            source=source,
            config=decode_config,
            reconnect_sleep_s=reconnect_sleep_s,
        )
        self.postprocessor = postprocessor or TrackPostProcessor()
        self.recorder = recorder
        self.prefer_latest_frame = prefer_latest_frame
        self.running = True
        self.seq = 1

    def stop(self) -> None:
        self.running = False

    async def run(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        min_interval_s = 1.0 / max(1, self.target_fps)
        last_emit_perf = 0.0
        self.reader.start()

        try:
            while self.running:
                packet = await asyncio.to_thread(
                    self.reader.pop_frame,
                    1.0,
                    self.prefer_latest_frame,
                )
                if packet is None:
                    continue

                now_perf = time.perf_counter()
                if now_perf - last_emit_perf < min_interval_s:
                    continue
                last_emit_perf = now_perf

                frame = packet.frame
                frame_height, frame_width = frame.shape[:2]
                self.projector.update(frame_width, frame_height)

                process_ts_ms = int(time.time() * 1000)
                frame_ts_ms = packet.source_ts_ms if packet.source_ts_ms is not None else packet.capture_ts_ms

                t0 = time.perf_counter()
                detections = await asyncio.to_thread(self.detector.detect, frame)
                t1 = time.perf_counter()
                raw_tracks = await asyncio.to_thread(self.tracker.update, detections, process_ts_ms)
                t2 = time.perf_counter()
                tracks = await asyncio.to_thread(self.postprocessor.update, raw_tracks, process_ts_ms)
                t3 = time.perf_counter()

                detection_payload = [
                    {
                        "type": det.kind,
                        "team": det.team or "unknown",
                        "x": int(det.x),
                        "y": int(det.y),
                        "w": int(det.w),
                        "h": int(det.h),
                        "conf": round(float(det.confidence), 3),
                    }
                    for det in detections
                ]

                entities: list[dict[str, Any]] = []
                track_payload: list[dict[str, Any]] = []
                for track in tracks:
                    x_m, y_m = self.projector.to_pitch(track.x_px, track.y_px)
                    vx_m, vy_m = self.projector.velocity_to_pitch(
                        track.x_px,
                        track.y_px,
                        track.vx_px,
                        track.vy_px,
                    )
                    entities.append(
                        {
                            "id": track.track_id,
                            "type": track.kind,
                            "team": track.team,
                            "x": round(x_m, 2),
                            "y": round(y_m, 2),
                            "vx": round(vx_m, 2),
                            "vy": round(vy_m, 2),
                            "conf": round(float(track.confidence), 3),
                        }
                    )
                    if (
                        track.bbox_x is None
                        or track.bbox_y is None
                        or track.bbox_w is None
                        or track.bbox_h is None
                    ):
                        bbox_x, bbox_y, bbox_w, bbox_h = _fallback_bbox(track.kind, track.x_px, track.y_px)
                    else:
                        bbox_x = track.bbox_x
                        bbox_y = track.bbox_y
                        bbox_w = track.bbox_w
                        bbox_h = track.bbox_h
                    track_payload.append(
                        {
                            "id": track.track_id,
                            "type": track.kind,
                            "team": track.team,
                            "x": round(float(bbox_x), 2),
                            "y": round(float(bbox_y), 2),
                            "w": round(float(max(1.0, bbox_w)), 2),
                            "h": round(float(max(1.0, bbox_h)), 2),
                            "conf": round(float(track.confidence), 3),
                        }
                    )

                decode_stats = self.reader.stats()
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "seq": self.seq,
                    "frame_ts_ms": frame_ts_ms,
                    "frame": {
                        "width": frame_width,
                        "height": frame_height,
                        "index": packet.frame_index,
                        "source_ts_ms": packet.source_ts_ms,
                        "capture_ts_ms": packet.capture_ts_ms,
                    },
                    "pitch": {
                        "length_m": PITCH_LENGTH_M,
                        "width_m": PITCH_WIDTH_M,
                    },
                    "detections": detection_payload,
                    "tracks": track_payload,
                    "entities": entities,
                    "meta": {
                        "detector": getattr(self.detector, "name", self.detector.__class__.__name__),
                        "tracker": getattr(self.tracker, "name", self.tracker.__class__.__name__),
                        "projector": getattr(self.projector, "name", self.projector.__class__.__name__),
                        "detect_ms": round((t1 - t0) * 1000.0, 2),
                        "track_ms": round((t2 - t1) * 1000.0, 2),
                        "post_ms": round((t3 - t2) * 1000.0, 2),
                        "process_ms": round((t3 - t0) * 1000.0, 2),
                        "decode_backend": decode_stats.get("backend"),
                        "decode_buffered": decode_stats.get("buffered_frames", 0),
                        "decode_dropped": decode_stats.get("dropped_frames", 0),
                        "decode_reconnects": decode_stats.get("reconnects", 0),
                        "capture_ts_ms": packet.capture_ts_ms,
                        "frame_index": packet.frame_index,
                    },
                }
                self.seq += 1

                if self.recorder is not None:
                    self.recorder.write(payload)

                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(payload)
        finally:
            self.reader.stop()
            if self.recorder is not None:
                self.recorder.close()
