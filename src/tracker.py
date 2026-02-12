from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import Detection, TrackState


class TrackerProtocol(Protocol):
    name: str

    def update(self, detections: list[Detection], ts_ms: int) -> list[TrackState]: ...


class NearestTracker:
    name = "nearest"

    def __init__(self, max_missed_frames: int = 8) -> None:
        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}
        self.max_missed_frames = max_missed_frames

    def update(self, detections: list[Detection], ts_ms: int) -> list[TrackState]:
        used_track_ids: set[int] = set()
        assignments: dict[int, int] = {}
        new_track_ids: set[int] = set()

        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_distance = float("inf")
            det_x, det_y = det.foot_point if det.kind == "player" else det.center
            distance_gate = 90.0 if det.kind == "ball" else 70.0

            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                if track.kind != det.kind:
                    continue

                distance = float(np.hypot(det_x - track.x_px, det_y - track.y_px))
                if distance < best_distance and distance <= distance_gate:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                assignments[det_idx] = best_track_id
                used_track_ids.add(best_track_id)

        for det_idx, det in enumerate(detections):
            det_x, det_y = det.foot_point if det.kind == "player" else det.center
            if det_idx in assignments:
                track = self._tracks[assignments[det_idx]]
                dt_s = max((ts_ms - track.last_ts_ms) / 1000.0, 1e-3)
                track.vx_px = (det_x - track.x_px) / dt_s
                track.vy_px = (det_y - track.y_px) / dt_s
                track.x_px = det_x
                track.y_px = det_y
                track.last_ts_ms = ts_ms
                track.confidence = det.confidence
                track.team = det.team or track.team
                track.missed_frames = 0
            else:
                track_id = self._next_id
                self._tracks[track_id] = TrackState(
                    track_id=track_id,
                    kind=det.kind,
                    team=det.team or "unknown",
                    x_px=det_x,
                    y_px=det_y,
                    vx_px=0.0,
                    vy_px=0.0,
                    confidence=det.confidence,
                    last_ts_ms=ts_ms,
                )
                new_track_ids.add(track_id)
                self._next_id += 1

        matched_ids = set(assignments.values())
        remove_ids: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_ids:
                continue
            if track_id in new_track_ids:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                remove_ids.append(track_id)

        for track_id in remove_ids:
            self._tracks.pop(track_id, None)

        return [track for track in self._tracks.values() if track.missed_frames <= 1]


class ByteTrackAdapter:
    """Thin wrapper around supervision.ByteTrack with TrackState outputs."""

    name = "bytetrack"

    def __init__(
        self,
        track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
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
        self._state_by_id: dict[int, TrackState] = {}
        self._max_missed_frames = max(8, track_buffer // 2)

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
                    )

                self._state_by_id[track_id] = state
                now_ids.add(track_id)

        for track_id, state in list(self._state_by_id.items()):
            if track_id in now_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames > self._max_missed_frames:
                self._state_by_id.pop(track_id, None)

        return [state for state in self._state_by_id.values() if state.missed_frames <= 1]
