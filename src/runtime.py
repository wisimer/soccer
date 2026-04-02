from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detector import DetectorProtocol, YoloDetector
from .postprocess import PostprocessConfig, TrackPostProcessor
from .projector import LinearProjector, ProjectorProtocol
from .recorder import JsonlRecorder
from .tracker import (
    DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT,
    DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT,
    ByteTrackAdapter,
    GlobalMotionConfig,
    TrackerProtocol,
)
from .video_reader import DecodeConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSettings:
    performance_profile: str = "auto"
    effective_profile: str = "cpu"
    target_fps: int = 30
    yolo_model: str = "yolov8n.pt"
    yolo_device: str = "cpu"
    yolo_half: bool = False
    yolo_conf: float = 0.25
    yolo_imgsz: int = 1280
    track_buffer: int = 30
    bytetrack_track_activation_threshold: float = 0.25
    bytetrack_kalman_position_weight: float = DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT
    bytetrack_kalman_velocity_weight: float = DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT
    gmc_enabled: bool = True
    gmc_method: str = "sparseOptFlow"
    gmc_downscale: float = 2.0
    gmc_min_points: int = 12
    gmc_motion_deadband_px: float = 1.0
    gmc_max_translation_px: float = 80.0
    decode_backend: str = "opencv"
    decode_buffer_size: int = 8
    decode_drop_policy: str = "drop_oldest"
    prefer_latest_frame: bool = True
    smooth_alpha: float = 0.35
    reid_ttl_ms: int = 1500
    reid_distance_px: float = 85.0
    record_path: str | None = None


def resolve_preferred_yolo_model(explicit: str | None = None) -> str:
    if explicit is not None:
        stripped = str(explicit).strip()
        if stripped:
            return stripped

    base_dir = Path(__file__).resolve().parents[1]
    candidates = (
        base_dir / "models" / "yolo-v8-football-players-best.pt",
        base_dir / "yolov8n.pt",
        base_dir / "yolo11n.pt",
        base_dir / "yolo11s.pt",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return "yolov8n.pt"


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def normalize_runtime_settings(settings: RuntimeSettings) -> RuntimeSettings:
    settings.target_fps = max(1, int(settings.target_fps))
    settings.yolo_model = resolve_preferred_yolo_model(settings.yolo_model)
    settings.yolo_conf = _clamp_float(settings.yolo_conf, 0.01, 0.95)
    settings.yolo_imgsz = max(320, int(settings.yolo_imgsz))
    settings.yolo_imgsz = max(32, (settings.yolo_imgsz // 32) * 32)
    settings.track_buffer = max(8, int(settings.track_buffer))
    settings.bytetrack_track_activation_threshold = _clamp_float(
        settings.bytetrack_track_activation_threshold,
        0.05,
        0.95,
    )
    settings.bytetrack_kalman_position_weight = _clamp_float(
        settings.bytetrack_kalman_position_weight,
        1e-4,
        1.0,
    )
    settings.bytetrack_kalman_velocity_weight = _clamp_float(
        settings.bytetrack_kalman_velocity_weight,
        1e-4,
        1.0,
    )
    settings.gmc_method = str(settings.gmc_method).strip() or "off"
    if settings.gmc_method.lower() != "sparseoptflow":
        settings.gmc_method = "off"
    settings.gmc_downscale = _clamp_float(settings.gmc_downscale, 1.0, 8.0)
    settings.gmc_min_points = max(4, int(settings.gmc_min_points))
    settings.gmc_motion_deadband_px = _clamp_float(settings.gmc_motion_deadband_px, 0.0, 10.0)
    settings.gmc_max_translation_px = _clamp_float(settings.gmc_max_translation_px, 4.0, 400.0)
    settings.decode_backend = str(settings.decode_backend).strip().lower()
    if settings.decode_backend not in {"opencv", "pyav"}:
        settings.decode_backend = "opencv"
    settings.decode_buffer_size = max(1, int(settings.decode_buffer_size))
    settings.decode_drop_policy = str(settings.decode_drop_policy).strip().lower()
    if settings.decode_drop_policy not in {"drop_oldest", "drop_newest"}:
        settings.decode_drop_policy = "drop_oldest"
    settings.smooth_alpha = _clamp_float(settings.smooth_alpha, 0.0, 1.0)
    settings.reid_ttl_ms = max(200, int(settings.reid_ttl_ms))
    settings.reid_distance_px = max(1.0, float(settings.reid_distance_px))
    if settings.yolo_half and not str(settings.yolo_device).lower().startswith("cuda"):
        settings.yolo_half = False
    return settings


def detect_accelerators() -> tuple[bool, bool]:
    try:
        import torch
    except Exception:
        return (False, False)

    cuda = bool(torch.cuda.is_available())
    return (cuda, False)


def resolve_effective_profile(requested_profile: str) -> str:
    requested = (requested_profile or "auto").strip().lower()
    cuda_available, _ = detect_accelerators()
    if requested == "cpu":
        return "cpu"
    if requested == "nvidia":
        if cuda_available:
            return "nvidia"
        return "cpu"
    if cuda_available:
        return "nvidia"
    return "cpu"


def profile_defaults(profile: str) -> dict[str, Any]:
    normalized = (profile or "cpu").strip().lower()
    common: dict[str, Any] = {
        "yolo_conf": 0.20,
        "bytetrack_track_activation_threshold": 0.20,
        "bytetrack_kalman_position_weight": DEFAULT_BYTETRACK_KALMAN_POSITION_WEIGHT,
        "bytetrack_kalman_velocity_weight": DEFAULT_BYTETRACK_KALMAN_VELOCITY_WEIGHT,
        "gmc_enabled": True,
        "gmc_method": "sparseOptFlow",
        "gmc_downscale": 2.0,
        "gmc_min_points": 12,
        "gmc_motion_deadband_px": 1.0,
        "gmc_max_translation_px": 80.0,
        "prefer_latest_frame": True,
    }

    if normalized == "nvidia":
        common.update(
            {
                "yolo_device": "cuda:0",
                "yolo_half": True,
                "yolo_imgsz": 768,
                "decode_backend": "pyav",
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


def build_tracker(settings: RuntimeSettings) -> TrackerProtocol:
    tracker = ByteTrackAdapter(
        track_buffer=settings.track_buffer,
        track_activation_threshold=settings.bytetrack_track_activation_threshold,
        frame_rate=settings.target_fps,
        kalman_position_weight=settings.bytetrack_kalman_position_weight,
        kalman_velocity_weight=settings.bytetrack_kalman_velocity_weight,
        gmc_config=GlobalMotionConfig(
            enabled=settings.gmc_enabled,
            method=settings.gmc_method,
            downscale=settings.gmc_downscale,
            min_points=settings.gmc_min_points,
            motion_deadband_px=settings.gmc_motion_deadband_px,
            max_translation_px=settings.gmc_max_translation_px,
        ),
    )
    logger.info(
        "Tracker backend: bytetrack (buffer=%s, activation=%s, fps=%s, kalman_pos=%s, kalman_vel=%s, gmc_enabled=%s, gmc_method=%s, gmc_downscale=%s)",
        settings.track_buffer,
        settings.bytetrack_track_activation_threshold,
        settings.target_fps,
        settings.bytetrack_kalman_position_weight,
        settings.bytetrack_kalman_velocity_weight,
        settings.gmc_enabled,
        settings.gmc_method,
        settings.gmc_downscale,
    )
    return tracker


def build_projector() -> ProjectorProtocol:
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
