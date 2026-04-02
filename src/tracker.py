from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .models import Detection, TrackState

BYTETRACK_BASE_KALMAN_POSITION_WEIGHT = 1.0 / 20.0
BYTETRACK_BASE_KALMAN_VELOCITY_WEIGHT = 1.0 / 160.0
DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT = BYTETRACK_BASE_KALMAN_POSITION_WEIGHT * 1.2
DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT = BYTETRACK_BASE_KALMAN_VELOCITY_WEIGHT * 1.8


@dataclass
class GlobalMotionConfig:
    enabled: bool = True
    method: str = "sparseOptFlow"
    downscale: float = 2.0
    min_points: int = 12
    motion_deadband_px: float = 1.0
    max_translation_px: float = 80.0


class TrackerProtocol(Protocol):
    name: str

    def update(
        self,
        detections: list[Detection],
        ts_ms: int,
        frame: np.ndarray | None = None,
    ) -> list[TrackState]: ...


class _SparseOptFlowMotionCompensator:
    def __init__(self, config: GlobalMotionConfig) -> None:
        self._config = config
        self._previous_gray: np.ndarray | None = None

    def estimate(self, frame: np.ndarray | None) -> tuple[float, float]:
        if frame is None or frame.size == 0:
            return (0.0, 0.0)

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, float(self._config.downscale))
        if scale > 1.0:
            width = max(32, int(round(current_gray.shape[1] / scale)))
            height = max(32, int(round(current_gray.shape[0] / scale)))
            current_gray = cv2.resize(current_gray, (width, height), interpolation=cv2.INTER_AREA)

        previous_gray = self._previous_gray
        self._previous_gray = current_gray
        if (
            not self._config.enabled
            or str(self._config.method).strip().lower() == "off"
            or previous_gray is None
            or previous_gray.shape != current_gray.shape
        ):
            return (0.0, 0.0)

        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < self._config.min_points:
            return (0.0, 0.0)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            return (0.0, 0.0)

        valid = status.reshape(-1) == 1
        if int(np.count_nonzero(valid)) < self._config.min_points:
            return (0.0, 0.0)

        previous_valid = previous_points[valid].reshape(-1, 2)
        current_valid = current_points[valid].reshape(-1, 2)
        affine, inliers = cv2.estimateAffinePartial2D(
            previous_valid,
            current_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is None:
            return (0.0, 0.0)

        if inliers is not None and int(np.count_nonzero(inliers)) < self._config.min_points:
            return (0.0, 0.0)

        dx = float(affine[0, 2]) * scale
        dy = float(affine[1, 2]) * scale
        magnitude = float(np.hypot(dx, dy))
        if magnitude < float(self._config.motion_deadband_px):
            return (0.0, 0.0)
        max_translation = max(1.0, float(self._config.max_translation_px))
        if magnitude > max_translation:
            scale_ratio = max_translation / magnitude
            dx *= scale_ratio
            dy *= scale_ratio
        return (dx, dy)


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
        gmc_config: GlobalMotionConfig | None = None,
    ) -> None:
        try:
            import supervision as sv
            from supervision.detection.utils.iou_and_nms import box_iou_batch
            from supervision.tracker.byte_tracker import matching
            from supervision.tracker.byte_tracker.core import (
                joint_tracks,
                remove_duplicate_tracks,
                sub_tracks,
            )
            from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState as SVTrackState
        except ImportError as exc:
            raise RuntimeError(
                "supervision is not installed. Install with `pip install supervision`."
            ) from exc

        self._sv = sv
        self._box_iou_batch = box_iou_batch
        self._matching = matching
        self._joint_tracks = joint_tracks
        self._sub_tracks = sub_tracks
        self._remove_duplicate_tracks = remove_duplicate_tracks
        self._STrack = STrack
        self._SVTrackState = SVTrackState
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
        self._gmc = _SparseOptFlowMotionCompensator(gmc_config or GlobalMotionConfig())

    def _apply_kalman_tuning(self, kalman_filter: object) -> None:
        if hasattr(kalman_filter, "_std_weight_position"):
            setattr(kalman_filter, "_std_weight_position", self._kalman_position_weight)
        if hasattr(kalman_filter, "_std_weight_velocity"):
            setattr(kalman_filter, "_std_weight_velocity", self._kalman_velocity_weight)

    @staticmethod
    def _apply_translation_to_tracks(tracks: list[object], dx: float, dy: float) -> None:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        for track in tracks:
            mean = getattr(track, "mean", None)
            if mean is None or len(mean) < 2:
                continue
            mean[0] += dx
            mean[1] += dy

    @staticmethod
    def _apply_translation_to_state(state: TrackState, dx: float, dy: float) -> None:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        state.x_px += dx
        state.y_px += dy
        if state.bbox_x is not None:
            state.bbox_x += dx
        if state.bbox_y is not None:
            state.bbox_y += dy

    def _update_with_detections(
        self,
        detections: object,
        motion: tuple[float, float],
    ) -> object:
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        tracks = self._update_with_tensors(tensors=tensors, motion=motion)

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = self._box_iou_batch(detection_bounding_boxes, track_bounding_boxes)
            iou_costs = 1 - ious
            matches, _, _ = self._matching.linear_assignment(iou_costs, 0.5)
            detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                detections.tracker_id[i_detection] = int(tracks[i_track].external_track_id)

            return detections[detections.tracker_id != -1]

        detections = self._sv.Detections.empty()
        detections.tracker_id = np.array([], dtype=int)
        return detections

    def _update_with_tensors(self, tensors: np.ndarray, motion: tuple[float, float]) -> list[object]:
        self._tracker.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if tensors.size == 0:
            tensors = np.empty((0, 5), dtype=np.float32)

        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self._tracker.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self._tracker.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            detections = [
                self._STrack(
                    self._STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self._tracker.minimum_consecutive_frames,
                    self._tracker.shared_kalman,
                    self._tracker.internal_id_counter,
                    self._tracker.external_id_counter,
                )
                for (tlbr, score_keep) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        unconfirmed = []
        tracked_tracks = []

        for track in self._tracker.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        strack_pool = self._joint_tracks(tracked_tracks, self._tracker.lost_tracks)
        self._STrack.multi_predict(strack_pool, self._tracker.shared_kalman)
        dx, dy = motion
        self._apply_translation_to_tracks(strack_pool, dx, dy)
        self._apply_translation_to_tracks(unconfirmed, dx, dy)

        dists = self._matching.iou_distance(strack_pool, detections)
        dists = self._matching.fuse_score(dists, detections)
        matches, u_track, u_detection = self._matching.linear_assignment(
            dists,
            thresh=self._tracker.minimum_matching_threshold,
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == self._SVTrackState.Tracked:
                track.update(detections[idet], self._tracker.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self._tracker.frame_id)
                refind_tracks.append(track)

        if len(dets_second) > 0:
            detections_second = [
                self._STrack(
                    self._STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self._tracker.minimum_consecutive_frames,
                    self._tracker.shared_kalman,
                    self._tracker.internal_id_counter,
                    self._tracker.external_id_counter,
                )
                for (tlbr, score_second) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == self._SVTrackState.Tracked
        ]
        dists = self._matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = self._matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == self._SVTrackState.Tracked:
                track.update(det, self._tracker.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self._tracker.frame_id)
                refind_tracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != self._SVTrackState.Lost:
                track.state = self._SVTrackState.Lost
                lost_tracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self._matching.iou_distance(unconfirmed, detections)
        dists = self._matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = self._matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self._tracker.frame_id)
            activated_tracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = self._SVTrackState.Removed
            removed_tracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self._tracker.det_thresh:
                continue
            track.activate(self._tracker.kalman_filter, self._tracker.frame_id)
            activated_tracks.append(track)

        for track in self._tracker.lost_tracks:
            if self._tracker.frame_id - track.frame_id > self._tracker.max_time_lost:
                track.state = self._SVTrackState.Removed
                removed_tracks.append(track)

        self._tracker.tracked_tracks = [
            track for track in self._tracker.tracked_tracks if track.state == self._SVTrackState.Tracked
        ]
        self._tracker.tracked_tracks = self._joint_tracks(self._tracker.tracked_tracks, activated_tracks)
        self._tracker.tracked_tracks = self._joint_tracks(self._tracker.tracked_tracks, refind_tracks)
        self._tracker.lost_tracks = self._sub_tracks(self._tracker.lost_tracks, self._tracker.tracked_tracks)
        self._tracker.lost_tracks.extend(lost_tracks)
        self._tracker.lost_tracks = self._sub_tracks(self._tracker.lost_tracks, self._tracker.removed_tracks)
        self._tracker.removed_tracks = removed_tracks
        self._tracker.tracked_tracks, self._tracker.lost_tracks = self._remove_duplicate_tracks(
            self._tracker.tracked_tracks,
            self._tracker.lost_tracks,
        )
        return [track for track in self._tracker.tracked_tracks if track.is_activated]

    def update(
        self,
        detections: list[Detection],
        ts_ms: int,
        frame: np.ndarray | None = None,
    ) -> list[TrackState]:
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

        motion = self._gmc.estimate(frame)
        tracked = self._update_with_detections(sv_dets, motion=motion)
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
            self._apply_translation_to_state(state, motion[0], motion[1])
            state.missed_frames += 1
            if state.missed_frames > self._max_missed_frames:
                self._state_by_id.pop(track_id, None)

        return [
            state for state in self._state_by_id.values() if state.missed_frames <= self._visible_missed_frames
        ]
