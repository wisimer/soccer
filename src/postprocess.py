from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import TrackState


@dataclass
class PostprocessConfig:
    smooth_alpha: float = 0.35
    reid_ttl_ms: int = 1500
    reid_distance_px: float = 85.0
    ball_smooth_alpha: float = 0.68
    fast_motion_px: float = 20.0
    fast_motion_alpha: float = 0.9
    max_prediction_dt_s: float = 0.25
    ball_prediction_damping: float = 0.9
    player_prediction_damping: float = 0.78
    ball_confidence_decay: float = 0.97
    player_confidence_decay: float = 0.93


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

            age_s = age_ms / 1000.0
            pred_dt = min(age_s, max(1e-3, float(self.config.max_prediction_dt_s)))
            damping = (
                float(np.clip(self.config.ball_prediction_damping, 0.0, 1.0))
                if observed.kind == "ball"
                else float(np.clip(self.config.player_prediction_damping, 0.0, 1.0))
            )
            pred_x = candidate.x_px + candidate.vx_px * pred_dt * damping
            pred_y = candidate.y_px + candidate.vy_px * pred_dt * damping

            distance = float(np.hypot(observed.x_px - pred_x, observed.y_px - pred_y))
            distance_gate = self.config.reid_distance_px * (1.5 if observed.kind == "ball" else 1.0)
            if distance < distance_gate and distance < best_distance:
                best_id = stable_id
                best_distance = distance

        return best_id

    def _smooth(self, stable_id: int, observed: TrackState, ts_ms: int) -> TrackState:
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
        if observed.missed_frames > 0:
            if observed.last_ts_ms >= ts_ms:
                pred_x = observed.x_px
                pred_y = observed.y_px
                pred_vx = observed.vx_px
                pred_vy = observed.vy_px
                pred_conf = observed.confidence
                pred_bbox_x = observed.bbox_x
                pred_bbox_y = observed.bbox_y
                pred_bbox_w = observed.bbox_w
                pred_bbox_h = observed.bbox_h
            else:
                (
                    pred_x,
                    pred_y,
                    pred_vx,
                    pred_vy,
                    pred_conf,
                    pred_bbox_x,
                    pred_bbox_y,
                    pred_bbox_w,
                    pred_bbox_h,
                ) = self._predict_missing(prev, observed, dt_s)

            return TrackState(
                track_id=stable_id,
                kind=observed.kind,
                team=observed.team if observed.team != "unknown" else prev.team,
                x_px=pred_x,
                y_px=pred_y,
                vx_px=pred_vx,
                vy_px=pred_vy,
                confidence=pred_conf,
                last_ts_ms=ts_ms,
                bbox_x=pred_bbox_x,
                bbox_y=pred_bbox_y,
                bbox_w=pred_bbox_w,
                bbox_h=pred_bbox_h,
                missed_frames=observed.missed_frames,
            )

        alpha = self._adaptive_alpha(prev, observed)
        x = alpha * observed.x_px + (1.0 - alpha) * prev.x_px
        y = alpha * observed.y_px + (1.0 - alpha) * prev.y_px
        obs_vx = (observed.x_px - prev.x_px) / dt_s
        obs_vy = (observed.y_px - prev.y_px) / dt_s
        vx = alpha * obs_vx + (1.0 - alpha) * prev.vx_px
        vy = alpha * obs_vy + (1.0 - alpha) * prev.vy_px
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

    def _adaptive_alpha(self, prev: TrackState, observed: TrackState) -> float:
        base_alpha = float(np.clip(self.config.smooth_alpha, 0.0, 1.0))
        fast_alpha = float(np.clip(self.config.fast_motion_alpha, 0.0, 1.0))
        if observed.kind == "ball":
            base_alpha = max(base_alpha, float(np.clip(self.config.ball_smooth_alpha, 0.0, 1.0)))
            fast_alpha = max(fast_alpha, base_alpha)

        innovation = float(np.hypot(observed.x_px - prev.x_px, observed.y_px - prev.y_px))
        motion_gate = max(1.0, float(self.config.fast_motion_px))
        if innovation >= motion_gate:
            return fast_alpha

        ratio = float(np.clip(innovation / motion_gate, 0.0, 1.0))
        ease = ratio * ratio
        return base_alpha + (fast_alpha - base_alpha) * ease

    def _predict_missing(
        self,
        prev: TrackState,
        observed: TrackState,
        dt_s: float,
    ) -> tuple[float, float, float, float, float, float | None, float | None, float | None, float | None]:
        pred_dt = min(dt_s, max(1e-3, float(self.config.max_prediction_dt_s)))
        if observed.kind == "ball":
            damping = float(np.clip(self.config.ball_prediction_damping, 0.0, 1.0))
            conf_decay = float(np.clip(self.config.ball_confidence_decay, 0.0, 1.0))
        else:
            damping = float(np.clip(self.config.player_prediction_damping, 0.0, 1.0))
            conf_decay = float(np.clip(self.config.player_confidence_decay, 0.0, 1.0))

        missed = max(1, observed.missed_frames)
        decay = damping**missed
        vx = prev.vx_px * decay
        vy = prev.vy_px * decay
        x = prev.x_px + vx * pred_dt
        y = prev.y_px + vy * pred_dt

        dx = x - prev.x_px
        dy = y - prev.y_px
        base_bbox_x = prev.bbox_x if prev.bbox_x is not None else observed.bbox_x
        base_bbox_y = prev.bbox_y if prev.bbox_y is not None else observed.bbox_y
        bbox_x = base_bbox_x + dx if base_bbox_x is not None else None
        bbox_y = base_bbox_y + dy if base_bbox_y is not None else None
        bbox_w = prev.bbox_w if prev.bbox_w is not None else observed.bbox_w
        bbox_h = prev.bbox_h if prev.bbox_h is not None else observed.bbox_h
        confidence = max(0.05, prev.confidence * (conf_decay**missed))

        return (x, y, vx, vy, confidence, bbox_x, bbox_y, bbox_w, bbox_h)

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
