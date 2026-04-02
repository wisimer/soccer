from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import Detection, TrackState

BYTETRACK_BASE_KALMAN_POSITION_WEIGHT = 1.0 / 20.0
BYTETRACK_BASE_KALMAN_VELOCITY_WEIGHT = 1.0 / 160.0
DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT = BYTETRACK_BASE_KALMAN_POSITION_WEIGHT * 1.2
DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT = BYTETRACK_BASE_KALMAN_VELOCITY_WEIGHT * 1.8


class TrackerProtocol(Protocol):
    name: str

    def update(self, detections: list[Detection], ts_ms: int) -> list[TrackState]: ...


class ByteTrackAdapter:
    """Thin wrapper around supervision.ByteTrack with TrackState outputs."""

    name = "bytetrack"

    def __init__(
        self,
        track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        kalman_position_weight: float = DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT,
        kalman_velocity_weight: float = DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "supervision is not installed. Install with `pip install supervision`."
            ) from exc

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._kalman_position_weight = max(1e-4, float(kalman_position_weight))
        self._kalman_velocity_weight = max(1e-4, float(kalman_velocity_weight))
        self._apply_kalman_tuning(self._tracker.kalman_filter)
        self._apply_kalman_tuning(self._tracker.shared_kalman)
        self._state_by_id: dict[int, TrackState] = {}
        self._max_missed_frames = max(8, track_buffer // 2)
        self._visible_missed_frames = min(self._max_missed_frames, max(3, frame_rate // 5))

    def _apply_kalman_tuning(self, kalman_filter: object) -> None:
        if hasattr(kalman_filter, "_std_weight_position"):
            setattr(kalman_filter, "_std_weight_position", self._kalman_position_weight)
        if hasattr(kalman_filter, "_std_weight_velocity"):
            setattr(kalman_filter, "_std_weight_velocity", self._kalman_velocity_weight)

    def update(self, detections: list[Detection], ts_ms: int) -> list[TrackState]:
        if detections:
            xyxy = np.array(
                [[d.x, d.y, d.x + d.w, d.y + d.h] for d in detections],
                dtype=np.float32,
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([0 if d.kind == "player" else 1 for d in detections], dtype=np.int32)
            team = np.array([(d.team or "unknown") for d in detections], dtype=object)
            kind = np.array([d.kind for d in detections], dtype=object)
            sv_dets = self._sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_id,
                data={"team": team, "kind": kind},
            )
        else:
            sv_dets = self._sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                data={},
            )

        tracked = self._tracker.update_with_detections(sv_dets)
        tracker_ids = getattr(tracked, "tracker_id", None)

        now_ids: set[int] = set()
        if tracker_ids is not None and len(tracker_ids) > 0:
            for i, raw_track_id in enumerate(tracker_ids):
                if raw_track_id is None:
                    continue
                track_id = int(raw_track_id)
                if track_id < 0:
                    continue

                box = tracked.xyxy[i]
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                cx = (x1 + x2) / 2.0

                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                kind = "player" if cls_id == 0 else "ball"
                tracked_data = tracked.data or {}
                if "kind" in tracked_data and len(tracked_data["kind"]) > i:
                    kind = str(tracked_data["kind"][i])
                cy = y2 if kind == "player" else (y1 + y2) / 2.0

                team = "unknown"
                if "team" in tracked_data and len(tracked_data["team"]) > i:
                    team = str(tracked_data["team"][i])

                prev = self._state_by_id.get(track_id)
                if prev is None:
                    state = TrackState(
                        track_id=track_id,
                        kind=kind,
                        team=team,
                        x_px=cx,
                        y_px=cy,
                        vx_px=0.0,
                        vy_px=0.0,
                        confidence=conf,
                        last_ts_ms=ts_ms,
                        missed_frames=0,
                        bbox_x=x1,
                        bbox_y=y1,
                        bbox_w=max(1.0, x2 - x1),
                        bbox_h=max(1.0, y2 - y1),
                    )
                else:
                    dt_s = max((ts_ms - prev.last_ts_ms) / 1000.0, 1e-3)
                    state = TrackState(
                        track_id=track_id,
                        kind=kind,
                        team=team if team != "unknown" else prev.team,
                        x_px=cx,
                        y_px=cy,
                        vx_px=(cx - prev.x_px) / dt_s,
                        vy_px=(cy - prev.y_px) / dt_s,
                        confidence=conf,
                        last_ts_ms=ts_ms,
                        missed_frames=0,
                        bbox_x=x1,
                        bbox_y=y1,
                        bbox_w=max(1.0, x2 - x1),
                        bbox_h=max(1.0, y2 - y1),
                    )

                self._state_by_id[track_id] = state
                now_ids.add(track_id)

        for track_id, state in list(self._state_by_id.items()):
            if track_id in now_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames > self._max_missed_frames:
                self._state_by_id.pop(track_id, None)

        return [
            state for state in self._state_by_id.values() if state.missed_frames <= self._visible_missed_frames
        ]
