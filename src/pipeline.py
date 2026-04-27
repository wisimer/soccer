from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .detector import DetectorProtocol, YoloDetector
from .game_engine import GameEngine
from .postprocess import TrackPostProcessor
from .projector import PITCH_LENGTH_M, PITCH_WIDTH_M, LinearProjector, ProjectorProtocol
from .recorder import JsonlRecorder
from .tracker import ByteTrackAdapter, TrackerProtocol
from .video_reader import BufferedVideoReader, DecodeConfig

SCHEMA_VERSION = "1.5"


def _fallback_bbox(kind: str, x_px: float, y_px: float) -> tuple[float, float, float, float]:
    """Build a conservative fallback box when tracker output lacks bbox data."""

    if kind == "ball":
        size = 10.0
        return (x_px - size / 2.0, y_px - size / 2.0, size, size)

    width = 24.0
    height = 52.0
    return (x_px - width / 2.0, y_px - height, width, height)


def _track_bbox(track: Any) -> tuple[float, float, float, float]:
    if track.bbox_x is None or track.bbox_y is None or track.bbox_w is None or track.bbox_h is None:
        return _fallback_bbox(track.kind, track.x_px, track.y_px)
    return (
        float(track.bbox_x),
        float(track.bbox_y),
        float(track.bbox_w),
        float(track.bbox_h),
    )


@dataclass(slots=True)
class _ContinuityState:
    predicted_ratio: float = 0.0
    id_switch_rate: float = 0.0
    last_ball_seen_ms: int | None = None
    last_player_ids: set[int] | None = None


class StreamProcessor:
    def __init__(
        self,
        source: str,
        target_fps: int = 30,
        reconnect_sleep_s: float = 2.0,
        detector: DetectorProtocol | None = None,
        tracker: TrackerProtocol | None = None,
        projector: ProjectorProtocol | None = None,
        decode_config: DecodeConfig | None = None,
        postprocessor: TrackPostProcessor | None = None,
        recorder: JsonlRecorder | None = None,
        game_engine: GameEngine | None = None,
        prefer_latest_frame: bool = True,
        continuity_window_ms: int = 1200,
    ) -> None:
        self.source_arg = source
        self.target_fps = target_fps
        self.reconnect_sleep_s = reconnect_sleep_s
        self.detector = detector or YoloDetector(allowed_kinds=("ball",))
        self.tracker = tracker or ByteTrackAdapter(frame_rate=target_fps)
        self.projector = projector or LinearProjector()
        self.reader = BufferedVideoReader(
            source=source,
            config=decode_config,
            reconnect_sleep_s=reconnect_sleep_s,
        )
        self.postprocessor = postprocessor or TrackPostProcessor()
        self.recorder = recorder
        self.game_engine = game_engine or GameEngine()
        self.prefer_latest_frame = prefer_latest_frame
        self.continuity_window_ms = max(200, int(continuity_window_ms))
        self.running = True
        self.seq = 1
        self._continuity = _ContinuityState()
        self._last_ball_entity: dict[str, Any] | None = None
        self._last_ball_track: dict[str, Any] | None = None
        self._last_projector_shape: tuple[int, int] | None = None

    def stop(self) -> None:
        """Request the processing loop to stop."""

        self.running = False

    def _serialize_detection(self, detection: Any) -> dict[str, Any]:
        return {
            "type": detection.kind,
            "team": detection.team or "unknown",
            "x": int(detection.x),
            "y": int(detection.y),
            "w": int(detection.w),
            "h": int(detection.h),
            "conf": round(float(detection.confidence), 3),
        }

    def _serialize_track(self, track: Any) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float, float, float]]:
        bbox_x, bbox_y, bbox_w, bbox_h = _track_bbox(track)
        x_m, y_m = self.projector.to_pitch(track.x_px, track.y_px)
        vx_m, vy_m = self.projector.velocity_to_pitch(
            track.x_px,
            track.y_px,
            track.vx_px,
            track.vy_px,
        )
        entity_item = {
            "id": track.track_id,
            "type": track.kind,
            "team": track.team,
            "x": round(x_m, 2),
            "y": round(y_m, 2),
            "vx": round(vx_m, 2),
            "vy": round(vy_m, 2),
            "conf": round(float(track.confidence), 3),
        }
        track_item = {
            "id": track.track_id,
            "type": track.kind,
            "team": track.team,
            "x": round(float(bbox_x), 2),
            "y": round(float(bbox_y), 2),
            "w": round(float(max(1.0, bbox_w)), 2),
            "h": round(float(max(1.0, bbox_h)), 2),
            "conf": round(float(track.confidence), 3),
        }
        return entity_item, track_item, (bbox_x, bbox_y, bbox_w, bbox_h)

    def _remember_ball(self, track: Any, entity_item: dict[str, Any], bbox: tuple[float, float, float, float]) -> None:
        _, _, bbox_w, bbox_h = bbox
        self._last_ball_entity = dict(entity_item)
        self._last_ball_track = {
            "id": track.track_id,
            "team": track.team,
            "center_x": float(track.x_px),
            "center_y": float(track.y_px),
            "vx_px": float(track.vx_px),
            "vy_px": float(track.vy_px),
            "w": float(max(1.0, bbox_w)),
            "h": float(max(1.0, bbox_h)),
            "conf": float(track.confidence),
        }

    def _serialize_tracks_payload(
        self,
        tracks: list[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        track_payload: list[dict[str, Any]] = []
        for track in tracks:
            entity_item, track_item, bbox = self._serialize_track(track)
            entities.append(entity_item)
            track_payload.append(track_item)
            if track.kind == "ball":
                self._remember_ball(track, entity_item, bbox)
        return entities, track_payload

    def _frame_payload(self, packet: Any, frame_width: int, frame_height: int) -> dict[str, Any]:
        return {
            "width": frame_width,
            "height": frame_height,
            "index": packet.frame_index,
            "source_ts_ms": packet.source_ts_ms,
            "capture_ts_ms": packet.capture_ts_ms,
        }

    def _build_payload(
        self,
        packet: Any,
        frame_width: int,
        frame_height: int,
        frame_ts_ms: int,
        detection_payload: list[dict[str, Any]],
        track_payload: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        game_payload: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "seq": self.seq,
            "frame_ts_ms": frame_ts_ms,
            "frame": self._frame_payload(packet, frame_width, frame_height),
            "pitch": {
                "length_m": PITCH_LENGTH_M,
                "width_m": PITCH_WIDTH_M,
            },
            "detections": detection_payload,
            "tracks": track_payload,
            "entities": entities,
            **game_payload,
            "meta": meta,
        }

    def _build_meta(
        self,
        packet: Any,
        detect_ms_per_frame: float,
        track_ms: float,
        post_ms: float,
        decode_stats: dict[str, Any],
        detect_batch_size: int,
    ) -> dict[str, Any]:
        return {
            "detector": getattr(self.detector, "name", self.detector.__class__.__name__),
            "tracker": getattr(self.tracker, "name", self.tracker.__class__.__name__),
            "projector": getattr(self.projector, "name", self.projector.__class__.__name__),
            "detect_ms": round(detect_ms_per_frame, 2),
            "track_ms": round(track_ms, 2),
            "post_ms": round(post_ms, 2),
            "process_ms": round(detect_ms_per_frame + track_ms + post_ms, 2),
            "decode_backend": decode_stats.get("backend"),
            "decode_buffered": decode_stats.get("buffered_frames", 0),
            "decode_dropped": decode_stats.get("dropped_frames", 0),
            "decode_reconnects": decode_stats.get("reconnects", 0),
            "capture_ts_ms": packet.capture_ts_ms,
            "frame_index": packet.frame_index,
            "detect_batch_size": detect_batch_size,
        }

    def _estimate_continuity(self, tracks: list[Any], ts_ms: int) -> dict[str, Any]:
        """Estimate short-horizon tracking continuity health metrics."""

        has_ball = any(getattr(track, "kind", "") == "ball" for track in tracks)
        if has_ball:
            self._continuity.last_ball_seen_ms = ts_ms

        gap_ms = 0
        if self._continuity.last_ball_seen_ms is None:
            gap_ms = self.continuity_window_ms
        else:
            gap_ms = max(0, ts_ms - self._continuity.last_ball_seen_ms)

        predicted_count = sum(1 for track in tracks if int(getattr(track, "missed_frames", 0)) > 0)
        instant_predicted_ratio = (predicted_count / len(tracks)) if tracks else 0.0
        self._continuity.predicted_ratio = (
            self._continuity.predicted_ratio * 0.82 + instant_predicted_ratio * 0.18
        )

        player_ids = {
            int(getattr(track, "track_id"))
            for track in tracks
            if getattr(track, "kind", "") == "player"
        }
        if self._continuity.last_player_ids is not None:
            union = player_ids | self._continuity.last_player_ids
            instant_switch = (len(player_ids ^ self._continuity.last_player_ids) / len(union)) if union else 0.0
            self._continuity.id_switch_rate = (
                self._continuity.id_switch_rate * 0.8 + instant_switch * 0.2
            )
        self._continuity.last_player_ids = player_ids

        health = "good"
        if (
            gap_ms > self.continuity_window_ms
            or self._continuity.predicted_ratio > 0.55
            or self._continuity.id_switch_rate > 0.45
        ):
            health = "bad"
        elif (
            gap_ms > int(self.continuity_window_ms * 0.35)
            or self._continuity.predicted_ratio > 0.25
            or self._continuity.id_switch_rate > 0.22
        ):
            health = "warn"

        return {
            "gap_ms": int(gap_ms),
            "predicted_ratio": round(float(self._continuity.predicted_ratio), 3),
            "id_switch_rate": round(float(self._continuity.id_switch_rate), 3),
            "health": health,
        }

    def _apply_ball_continuity(
        self,
        entities: list[dict[str, Any]],
        track_payload: list[dict[str, Any]],
        ts_ms: int,
    ) -> None:
        """Inject a short-lived predicted ball when detections temporarily disappear."""

        if any(item.get("type") == "ball" for item in entities):
            return
        if self._continuity.last_ball_seen_ms is None:
            return
        if self._last_ball_entity is None:
            return

        gap_ms = max(0, ts_ms - self._continuity.last_ball_seen_ms)
        if gap_ms > self.continuity_window_ms:
            return

        dt_s = gap_ms / 1000.0
        decay = max(0.2, 1.0 - gap_ms / max(1, self.continuity_window_ms))

        ghost = dict(self._last_ball_entity)
        ghost["x"] = round(float(ghost.get("x", 0.0)) + float(ghost.get("vx", 0.0)) * dt_s, 2)
        ghost["y"] = round(float(ghost.get("y", 0.0)) + float(ghost.get("vy", 0.0)) * dt_s, 2)
        ghost["x"] = max(0.0, min(float(PITCH_LENGTH_M), float(ghost["x"])))
        ghost["y"] = max(0.0, min(float(PITCH_WIDTH_M), float(ghost["y"])))
        ghost["conf"] = round(max(0.05, float(ghost.get("conf", 0.4)) * decay), 3)
        ghost["predicted"] = True
        entities.append(ghost)

        if self._last_ball_track is not None:
            last_track = self._last_ball_track
            center_x = float(last_track["center_x"]) + float(last_track.get("vx_px", 0.0)) * dt_s
            center_y = float(last_track["center_y"]) + float(last_track.get("vy_px", 0.0)) * dt_s
            width = float(last_track["w"])
            height = float(last_track["h"])
            track_payload.append(
                {
                    "id": last_track["id"],
                    "type": "ball",
                    "team": last_track.get("team", "unknown"),
                    "x": round(center_x - width * 0.5, 2),
                    "y": round(center_y - height * 0.5, 2),
                    "w": round(width, 2),
                    "h": round(height, 2),
                    "conf": round(max(0.05, float(last_track.get("conf", 0.4)) * decay), 3),
                    "predicted": True,
                }
            )

    async def run(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Run the end-to-end streaming pipeline and publish payloads."""

        min_interval_s = 1.0 / max(1, self.target_fps)
        detect_batch_size = max(1, int(getattr(self.detector, "batch_size", 1)))
        last_emit_perf = 0.0
        self.reader.start()

        try:
            while self.running:
                if last_emit_perf > 0.0:
                    wait_s = min_interval_s - (time.perf_counter() - last_emit_perf)
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                packets = await asyncio.to_thread(
                    self.reader.pop_frames,
                    detect_batch_size,
                    1.0,
                    self.prefer_latest_frame,
                )
                if not packets:
                    continue

                t0 = time.perf_counter()
                frames = [packet.frame for packet in packets]
                detections_batch = await asyncio.to_thread(self.detector.detect_many, frames)
                t1 = time.perf_counter()
                detect_ms_per_frame = ((t1 - t0) * 1000.0) / max(1, len(packets))
                last_emit_perf = t1

                for packet, detections in zip(packets, detections_batch):
                    frame = packet.frame
                    frame_height, frame_width = frame.shape[:2]
                    shape = (frame_width, frame_height)
                    if self._last_projector_shape != shape:
                        self.projector.update(frame_width, frame_height)
                        self._last_projector_shape = shape

                    process_ts_ms = int(time.time() * 1000)
                    frame_ts_ms = packet.source_ts_ms if packet.source_ts_ms is not None else packet.capture_ts_ms

                    t2 = time.perf_counter()
                    raw_tracks = await asyncio.to_thread(self.tracker.update, detections, process_ts_ms, frame)
                    t3 = time.perf_counter()
                    tracks = await asyncio.to_thread(self.postprocessor.update, raw_tracks, process_ts_ms)
                    t4 = time.perf_counter()

                    detection_payload = [self._serialize_detection(det) for det in detections]
                    entities, track_payload = self._serialize_tracks_payload(tracks)

                    continuity_health = self._estimate_continuity(tracks, process_ts_ms)
                    self._apply_ball_continuity(entities, track_payload, process_ts_ms)
                    game_payload = self.game_engine.update_from_frame(
                        entities=entities,
                        ts_ms=process_ts_ms,
                        continuity_health=continuity_health,
                    )

                    decode_stats = self.reader.stats()
                    track_ms = (t3 - t2) * 1000.0
                    post_ms = (t4 - t3) * 1000.0
                    meta = self._build_meta(
                        packet=packet,
                        detect_ms_per_frame=detect_ms_per_frame,
                        track_ms=track_ms,
                        post_ms=post_ms,
                        decode_stats=decode_stats,
                        detect_batch_size=len(packets),
                    )
                    payload = self._build_payload(
                        packet=packet,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        frame_ts_ms=frame_ts_ms,
                        detection_payload=detection_payload,
                        track_payload=track_payload,
                        entities=entities,
                        game_payload=game_payload,
                        meta=meta,
                    )
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
