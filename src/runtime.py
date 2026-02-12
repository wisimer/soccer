from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detector import DetectorProtocol, HeuristicDetector, YoloDetector
from .postprocess import PostprocessConfig, TrackPostProcessor
from .projector import HomographyProjector, LinearProjector, ProjectorProtocol
from .recorder import JsonlRecorder
from .tracker import ByteTrackAdapter, NearestTracker, TrackerProtocol
from .video_reader import DecodeConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSettings:
    performance_profile: str = "auto"
    effective_profile: str = "cpu"
    detector_backend: str = "heuristic"
    tracker_backend: str = "nearest"
    calibration_path: str | None = None
    yolo_model: str = "yolov8n.pt"
    yolo_device: str = "cpu"
    yolo_half: bool = False
    yolo_conf: float = 0.25
    yolo_imgsz: int = 1280
    track_buffer: int = 30
    bytetrack_track_activation_threshold: float = 0.25
    decode_backend: str = "opencv"
    decode_buffer_size: int = 8
    decode_drop_policy: str = "drop_oldest"
    prefer_latest_frame: bool = True
    smooth_alpha: float = 0.35
    reid_ttl_ms: int = 1500
    reid_distance_px: float = 85.0
    record_path: str | None = None


def detect_accelerators() -> tuple[bool, bool]:
    try:
        import torch
    except Exception:
        return (False, False)

    cuda = bool(torch.cuda.is_available())
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return (cuda, mps)


def resolve_effective_profile(requested_profile: str) -> str:
    requested = (requested_profile or "auto").strip().lower()
    cuda_available, mps_available = detect_accelerators()
    if requested == "custom":
        return "custom"
    if requested == "cpu":
        return "cpu"
    if requested == "nvidia":
        if cuda_available:
            return "nvidia"
        if mps_available:
            return "apple"
        return "cpu"
    if requested == "apple":
        if mps_available:
            return "apple"
        if cuda_available:
            return "nvidia"
        return "cpu"
    if cuda_available:
        return "nvidia"
    if mps_available:
        return "apple"
    return "cpu"


def profile_defaults(profile: str) -> dict[str, Any]:
    normalized = (profile or "cpu").strip().lower()
    common: dict[str, Any] = {
        "yolo_conf": 0.20,
        "bytetrack_track_activation_threshold": 0.20,
        "prefer_latest_frame": True,
    }

    if normalized == "nvidia":
        common.update(
            {
                "yolo_device": "cuda:0",
                "yolo_half": True,
                "yolo_imgsz": 1280,
                "decode_backend": "pyav",
            }
        )
    elif normalized == "apple":
        common.update(
            {
                "yolo_device": "mps",
                "yolo_half": False,
                "yolo_imgsz": 960,
                "decode_backend": "opencv",
            }
        )
    elif normalized == "cpu":
        common.update(
            {
                "yolo_device": "cpu",
                "yolo_half": False,
                "yolo_imgsz": 640,
                "decode_backend": "opencv",
            }
        )

    return common


def build_detector(settings: RuntimeSettings) -> DetectorProtocol:
    backend = settings.detector_backend.lower()
    if backend == "yolo":
        try:
            detector = YoloDetector(
                model_path=settings.yolo_model,
                device=settings.yolo_device,
                confidence_threshold=settings.yolo_conf,
                image_size=settings.yolo_imgsz,
                use_half=settings.yolo_half,
            )
            logger.info(
                "Detector backend: yolo (%s, device=%s, half=%s, conf=%s, imgsz=%s, profile=%s/%s)",
                settings.yolo_model,
                settings.yolo_device,
                settings.yolo_half,
                settings.yolo_conf,
                settings.yolo_imgsz,
                settings.performance_profile,
                settings.effective_profile,
            )
            return detector
        except Exception as exc:
            logger.warning("YOLO unavailable, fallback to heuristic detector: %s", exc)

    detector = HeuristicDetector()
    logger.info("Detector backend: heuristic")
    return detector


def build_tracker(settings: RuntimeSettings) -> TrackerProtocol:
    backend = settings.tracker_backend.lower()
    if backend == "bytetrack":
        try:
            tracker = ByteTrackAdapter(
                track_buffer=settings.track_buffer,
                track_activation_threshold=settings.bytetrack_track_activation_threshold,
            )
            logger.info("Tracker backend: bytetrack")
            return tracker
        except Exception as exc:
            logger.warning("ByteTrack unavailable, fallback to nearest tracker: %s", exc)

    tracker = NearestTracker(max_missed_frames=max(8, settings.track_buffer // 2))
    logger.info("Tracker backend: nearest")
    return tracker


def build_projector(settings: RuntimeSettings) -> ProjectorProtocol:
    if settings.calibration_path:
        calibration = Path(settings.calibration_path)
        if calibration.exists():
            try:
                projector = HomographyProjector.from_json(calibration)
                logger.info("Projector backend: homography (%s)", calibration)
                return projector
            except Exception as exc:
                logger.warning("Failed to load calibration '%s': %s", calibration, exc)
        else:
            logger.warning("Calibration file not found: %s", calibration)

    projector = LinearProjector()
    logger.info("Projector backend: linear")
    return projector


def build_decode_config(settings: RuntimeSettings) -> DecodeConfig:
    config = DecodeConfig(
        backend=settings.decode_backend,
        buffer_size=settings.decode_buffer_size,
        drop_policy=settings.decode_drop_policy,
    )
    logger.info(
        "Decode config: backend=%s, buffer=%s, drop_policy=%s, prefer_latest=%s",
        config.backend,
        config.buffer_size,
        config.drop_policy,
        settings.prefer_latest_frame,
    )
    return config


def build_postprocessor(settings: RuntimeSettings) -> TrackPostProcessor:
    config = PostprocessConfig(
        smooth_alpha=settings.smooth_alpha,
        reid_ttl_ms=settings.reid_ttl_ms,
        reid_distance_px=settings.reid_distance_px,
    )
    logger.info(
        "Postprocess config: smooth_alpha=%s, reid_ttl_ms=%s, reid_distance_px=%s",
        config.smooth_alpha,
        config.reid_ttl_ms,
        config.reid_distance_px,
    )
    return TrackPostProcessor(config=config)


def build_recorder(settings: RuntimeSettings) -> JsonlRecorder | None:
    if not settings.record_path:
        return None
    recorder = JsonlRecorder(settings.record_path)
    logger.info("Recorder enabled: %s", settings.record_path)
    return recorder
