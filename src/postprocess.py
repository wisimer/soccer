from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import TrackState


@dataclass
class PostprocessConfig:
    smooth_alpha: float = 0.35
    reid_ttl_ms: int = 1500
    reid_distance_px: float = 85.0


class TrackPostProcessor:
    """Stabilize trajectories and remap short-lived ID breaks."""

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        self.config = config or PostprocessConfig()
        self._next_stable_id = 1
        self._raw_to_stable: dict[int, int] = {}
        self._stable_state: dict[int, TrackState] = {}
        self._inactive: dict[int, TrackState] = {}

    def update(self, tracks: list[TrackState], ts_ms: int) -> list[TrackState]:
        active_raw_ids = {track.track_id for track in tracks}

        for raw_id in list(self._raw_to_stable.keys()):
            if raw_id in active_raw_ids:
                continue
            stable_id = self._raw_to_stable.pop(raw_id)
            prev_state = self._stable_state.get(stable_id)
            if prev_state is not None:
                self._inactive[stable_id] = prev_state

        self._purge_inactive(ts_ms)

        outputs: list[TrackState] = []
        for observed in tracks:
            stable_id = self._resolve_stable_id(observed, ts_ms)
            smoothed = self._smooth(stable_id, observed, ts_ms)
            self._stable_state[stable_id] = smoothed
            outputs.append(smoothed)

        return outputs

    def _resolve_stable_id(self, observed: TrackState, ts_ms: int) -> int:
        raw_id = observed.track_id
        if raw_id in self._raw_to_stable:
            return self._raw_to_stable[raw_id]

        matched_stable = self._match_inactive(observed, ts_ms)
        if matched_stable is not None:
            self._raw_to_stable[raw_id] = matched_stable
            self._inactive.pop(matched_stable, None)
            return matched_stable

        stable_id = self._next_stable_id
        self._next_stable_id += 1
        self._raw_to_stable[raw_id] = stable_id
        return stable_id

    def _match_inactive(self, observed: TrackState, ts_ms: int) -> int | None:
        best_id: int | None = None
        best_distance = float("inf")

        for stable_id, candidate in self._inactive.items():
            age_ms = ts_ms - candidate.last_ts_ms
            if age_ms > self.config.reid_ttl_ms:
                continue
            if candidate.kind != observed.kind:
                continue
            if (
                observed.team not in ("", "unknown")
                and candidate.team not in ("", "unknown")
                and observed.team != candidate.team
            ):
                continue

            distance = float(np.hypot(observed.x_px - candidate.x_px, observed.y_px - candidate.y_px))
            if distance < self.config.reid_distance_px and distance < best_distance:
                best_id = stable_id
                best_distance = distance

        return best_id

    def _smooth(self, stable_id: int, observed: TrackState, ts_ms: int) -> TrackState:
        alpha = float(np.clip(self.config.smooth_alpha, 0.0, 1.0))
        prev = self._stable_state.get(stable_id)
        if prev is None:
            prev = self._inactive.get(stable_id)

        if prev is None:
            return TrackState(
                track_id=stable_id,
                kind=observed.kind,
                team=observed.team,
                x_px=observed.x_px,
                y_px=observed.y_px,
                vx_px=observed.vx_px,
                vy_px=observed.vy_px,
                confidence=observed.confidence,
                last_ts_ms=ts_ms,
                bbox_x=observed.bbox_x,
                bbox_y=observed.bbox_y,
                bbox_w=observed.bbox_w,
                bbox_h=observed.bbox_h,
                missed_frames=0,
            )

        dt_s = max((ts_ms - prev.last_ts_ms) / 1000.0, 1e-3)
        x = alpha * observed.x_px + (1.0 - alpha) * prev.x_px
        y = alpha * observed.y_px + (1.0 - alpha) * prev.y_px
        vx = (x - prev.x_px) / dt_s
        vy = (y - prev.y_px) / dt_s
        bbox_x = self._blend_optional(alpha, observed.bbox_x, prev.bbox_x)
        bbox_y = self._blend_optional(alpha, observed.bbox_y, prev.bbox_y)
        bbox_w = self._blend_optional(alpha, observed.bbox_w, prev.bbox_w)
        bbox_h = self._blend_optional(alpha, observed.bbox_h, prev.bbox_h)

        return TrackState(
            track_id=stable_id,
            kind=observed.kind,
            team=observed.team if observed.team != "unknown" else prev.team,
            x_px=x,
            y_px=y,
            vx_px=vx,
            vy_px=vy,
            confidence=alpha * observed.confidence + (1.0 - alpha) * prev.confidence,
            last_ts_ms=ts_ms,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            missed_frames=0,
        )

    @staticmethod
    def _blend_optional(alpha: float, current: float | None, previous: float | None) -> float | None:
        if current is None:
            return previous
        if previous is None:
            return current
        return alpha * current + (1.0 - alpha) * previous

    def _purge_inactive(self, ts_ms: int) -> None:
        to_remove = [
            stable_id
            for stable_id, state in self._inactive.items()
            if ts_ms - state.last_ts_ms > self.config.reid_ttl_ms
        ]
        for stable_id in to_remove:
            self._inactive.pop(stable_id, None)
            self._stable_state.pop(stable_id, None)
