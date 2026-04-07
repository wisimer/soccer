from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detector import BallDetectionConfig, DetectorProtocol, HybridDetector, TeamClassifierConfig, YoloDetector
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
    """Resolved runtime settings after CLI, env, YAML, and profile merge."""

    performance_profile: str = "auto"
    effective_profile: str = "cpu"
    target_fps: int = 30
    yolo_model: str = "yolov8n.pt"
    yolo_device: str = "cpu"
    yolo_half: bool = False
    yolo_conf: float = 0.25
    yolo_imgsz: int = 1280
    yolo_batch_size: int = 1
    ball_yolo_model: str | None = None
    ball_yolo_device: str = "cpu"
    ball_yolo_half: bool = False
    ball_yolo_conf: float = 0.12
    ball_yolo_imgsz: int = 1536
    ball_yolo_batch_size: int = 1
    ball_yolo_backend: str = "auto"
    ball_min_area: float = 6.0
    ball_max_area: float = 3000.0
    ball_max_aspect_ratio: float = 2.5
    ball_max_detections: int = 1
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
    decode_opencv_buffer_size: int = 2
    decode_drop_policy: str = "drop_oldest"
    decode_reconnect_sleep_s: float = 2.0
    prefer_latest_frame: bool = True
    team_classification_mode: str = "auto"
    team_a_primary_color: str = "red"
    team_b_primary_color: str = "blue"
    team_color_saturation_threshold: int = 45
    team_color_min_ratio: float = 0.08
    smooth_alpha: float = 0.35
    ball_smooth_alpha: float = 0.68
    fast_motion_px: float = 20.0
    fast_motion_alpha: float = 0.9
    max_prediction_dt_s: float = 0.25
    ball_prediction_damping: float = 0.9
    player_prediction_damping: float = 0.78
    ball_confidence_decay: float = 0.97
    player_confidence_decay: float = 0.93
    reid_ttl_ms: int = 1500
    reid_distance_px: float = 85.0
    reid_enabled: bool = True
    reid_max_inactive_tracks: int = 256
    record_path: str | None = None


def resolve_preferred_yolo_model(explicit: str | None = None) -> str:
    """Pick an explicit YOLO model or the best available local fallback."""

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


def resolve_preferred_ball_yolo_model(explicit: str | None = None) -> str | None:
    if explicit is not None:
        stripped = str(explicit).strip()
        if stripped:
            return stripped

    base_dir = Path(__file__).resolve().parents[1]
    candidates = (
        base_dir / "models" / "yolov5m-football-best.pt",
        base_dir / "models" / "football-ball-yolov5m.pt",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float into the requested range."""

    return max(minimum, min(maximum, float(value)))


def _normalize_team_color(value: str, fallback: str) -> str:
    """Normalize a configured team color name."""

    normalized = str(value or "").strip().lower()
    if normalized:
        return normalized
    return fallback


def normalize_runtime_settings(settings: RuntimeSettings) -> RuntimeSettings:
    """Clamp runtime settings to safe, internally consistent ranges."""

    settings.target_fps = max(1, int(settings.target_fps))
    settings.yolo_model = resolve_preferred_yolo_model(settings.yolo_model)
    settings.yolo_conf = _clamp_float(settings.yolo_conf, 0.01, 0.95)
    settings.yolo_imgsz = max(320, int(settings.yolo_imgsz))
    settings.yolo_imgsz = max(32, (settings.yolo_imgsz // 32) * 32)
    settings.yolo_batch_size = max(1, int(settings.yolo_batch_size))
    settings.ball_yolo_model = resolve_preferred_ball_yolo_model(settings.ball_yolo_model)
    settings.ball_yolo_conf = _clamp_float(settings.ball_yolo_conf, 0.01, 0.95)
    settings.ball_yolo_imgsz = max(320, int(settings.ball_yolo_imgsz))
    settings.ball_yolo_imgsz = max(32, (settings.ball_yolo_imgsz // 32) * 32)
    settings.ball_yolo_batch_size = max(1, int(settings.ball_yolo_batch_size))
    settings.ball_yolo_backend = str(settings.ball_yolo_backend).strip().lower() or "auto"
    if settings.ball_yolo_backend not in {"auto", "ultralytics", "yolov5"}:
        settings.ball_yolo_backend = "auto"
    if settings.ball_yolo_backend == "auto":
        if "yolov5" in str(settings.ball_yolo_model).strip().lower():
            settings.ball_yolo_backend = "yolov5"
        else:
            settings.ball_yolo_backend = "ultralytics"
    settings.ball_min_area = max(1.0, float(settings.ball_min_area))
    settings.ball_max_area = max(settings.ball_min_area, float(settings.ball_max_area))
    settings.ball_max_aspect_ratio = _clamp_float(settings.ball_max_aspect_ratio, 1.0, 10.0)
    settings.ball_max_detections = max(1, int(settings.ball_max_detections))
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
    settings.decode_opencv_buffer_size = max(1, int(settings.decode_opencv_buffer_size))
    settings.decode_drop_policy = str(settings.decode_drop_policy).strip().lower()
    if settings.decode_drop_policy not in {"drop_oldest", "drop_newest"}:
        settings.decode_drop_policy = "drop_oldest"
    settings.decode_reconnect_sleep_s = _clamp_float(settings.decode_reconnect_sleep_s, 0.1, 30.0)
    settings.team_classification_mode = str(settings.team_classification_mode).strip().lower() or "auto"
    if settings.team_classification_mode not in {"auto", "manual"}:
        settings.team_classification_mode = "auto"
    settings.team_a_primary_color = _normalize_team_color(settings.team_a_primary_color, "red")
    settings.team_b_primary_color = _normalize_team_color(settings.team_b_primary_color, "blue")
    settings.team_color_saturation_threshold = max(0, min(255, int(settings.team_color_saturation_threshold)))
    settings.team_color_min_ratio = _clamp_float(settings.team_color_min_ratio, 0.0, 1.0)
    settings.smooth_alpha = _clamp_float(settings.smooth_alpha, 0.0, 1.0)
    settings.ball_smooth_alpha = _clamp_float(settings.ball_smooth_alpha, 0.0, 1.0)
    settings.fast_motion_px = max(1.0, float(settings.fast_motion_px))
    settings.fast_motion_alpha = _clamp_float(settings.fast_motion_alpha, 0.0, 1.0)
    settings.max_prediction_dt_s = _clamp_float(settings.max_prediction_dt_s, 0.01, 2.0)
    settings.ball_prediction_damping = _clamp_float(settings.ball_prediction_damping, 0.0, 1.0)
    settings.player_prediction_damping = _clamp_float(settings.player_prediction_damping, 0.0, 1.0)
    settings.ball_confidence_decay = _clamp_float(settings.ball_confidence_decay, 0.0, 1.0)
    settings.player_confidence_decay = _clamp_float(settings.player_confidence_decay, 0.0, 1.0)
    settings.reid_ttl_ms = max(200, int(settings.reid_ttl_ms))
    settings.reid_distance_px = max(1.0, float(settings.reid_distance_px))
    settings.reid_max_inactive_tracks = max(16, int(settings.reid_max_inactive_tracks))
    yolo_backend_hint = str(settings.yolo_model).strip().lower()
    ball_yolo_backend_hint = str(settings.ball_yolo_backend).strip().lower()
    ball_model_hint = str(settings.ball_yolo_model).strip().lower()
    if settings.yolo_half and "yolov5" in yolo_backend_hint:
        settings.yolo_half = False
    if settings.ball_yolo_half and (
        ball_yolo_backend_hint == "yolov5" or "yolov5" in ball_model_hint
    ):
        settings.ball_yolo_half = False
    if settings.yolo_half and not str(settings.yolo_device).lower().startswith("cuda"):
        settings.yolo_half = False
    if settings.ball_yolo_half and not str(settings.ball_yolo_device).lower().startswith("cuda"):
        settings.ball_yolo_half = False
    return settings


def detect_accelerators() -> bool:
    try:
        import torch
    except Exception:
        return False

    return bool(torch.cuda.is_available())


def resolve_effective_profile(requested_profile: str) -> str:
    requested = (requested_profile or "auto").strip().lower()
    cuda_available = detect_accelerators()
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
    """Return hardware-aware defaults for the requested performance profile."""

    normalized = (profile or "cpu").strip().lower()
    common: dict[str, Any] = {
        "yolo_conf": 0.20,
        "ball_yolo_conf": 0.12,
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
        "yolo_batch_size": 1,
        "ball_yolo_batch_size": 1,
        "reid_enabled": True,
        "reid_max_inactive_tracks": 256,
    }

    if normalized == "nvidia":
        common.update(
            {
                "yolo_device": "cuda:0",
                "yolo_half": True,
                "yolo_imgsz": 768,
                "yolo_batch_size": 2,
                "ball_yolo_device": "cuda:0",
                "ball_yolo_half": True,
                "ball_yolo_imgsz": 1536,
                "decode_backend": "pyav",
                "decode_opencv_buffer_size": 1,
            }
        )
    elif normalized == "cpu":
        common.update(
            {
                "yolo_device": "cpu",
                "yolo_half": False,
                "yolo_imgsz": 640,
                "ball_yolo_device": "cpu",
                "ball_yolo_half": False,
                "ball_yolo_imgsz": 1280,
                "decode_backend": "opencv",
                "decode_opencv_buffer_size": 2,
            }
        )

    return common


def build_detector(settings: RuntimeSettings) -> DetectorProtocol:
    """Build the detector with merged runtime settings."""

    team_config = TeamClassifierConfig(
        mode=settings.team_classification_mode,
        team_a_primary_color=settings.team_a_primary_color,
        team_b_primary_color=settings.team_b_primary_color,
        saturation_threshold=settings.team_color_saturation_threshold,
        min_color_ratio=settings.team_color_min_ratio,
    )
    player_detector = YoloDetector(
        model_path=settings.yolo_model,
        device=settings.yolo_device,
        confidence_threshold=settings.yolo_conf,
        image_size=settings.yolo_imgsz,
        use_half=settings.yolo_half,
        batch_size=settings.yolo_batch_size,
        team_config=team_config,
        allowed_kinds=("player",) if settings.ball_yolo_model else None,
    )
    detector: DetectorProtocol = player_detector
    if settings.ball_yolo_model:
        ball_detector = YoloDetector(
            model_path=settings.ball_yolo_model,
            device=settings.ball_yolo_device,
            confidence_threshold=settings.ball_yolo_conf,
            image_size=settings.ball_yolo_imgsz,
            use_half=settings.ball_yolo_half,
            batch_size=settings.ball_yolo_batch_size,
            team_config=team_config,
            backend=settings.ball_yolo_backend,
            allowed_kinds=("ball",),
            ball_config=BallDetectionConfig(
                min_area=settings.ball_min_area,
                max_area=settings.ball_max_area,
                max_aspect_ratio=settings.ball_max_aspect_ratio,
                max_detections=settings.ball_max_detections,
            ),
        )
        detector = HybridDetector(player_detector=player_detector, ball_detector=ball_detector)
    logger.info(
        "Detector backend: %s (player_model=%s, player_device=%s, player_half=%s, player_conf=%s, player_imgsz=%s, player_batch=%s, ball_model=%s, ball_device=%s, ball_half=%s, ball_conf=%s, ball_imgsz=%s, ball_batch=%s, team_mode=%s, profile=%s/%s)",
        getattr(detector, "name", detector.__class__.__name__),
        settings.yolo_model,
        settings.yolo_device,
        settings.yolo_half,
        settings.yolo_conf,
        settings.yolo_imgsz,
        settings.yolo_batch_size,
        settings.ball_yolo_model,
        settings.ball_yolo_device,
        settings.ball_yolo_half,
        settings.ball_yolo_conf,
        settings.ball_yolo_imgsz,
        settings.ball_yolo_batch_size,
        settings.team_classification_mode,
        settings.performance_profile,
        settings.effective_profile,
    )
    return detector


def build_tracker(settings: RuntimeSettings) -> TrackerProtocol:
    """Build the tracker with GMC-aware runtime settings."""

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
    """Build the pitch projector implementation."""

    projector = LinearProjector()
    logger.info("Projector backend: linear")
    return projector


def build_decode_config(settings: RuntimeSettings) -> DecodeConfig:
    """Build decoder buffering settings."""

    config = DecodeConfig(
        backend=settings.decode_backend,
        buffer_size=settings.decode_buffer_size,
        opencv_buffer_size=settings.decode_opencv_buffer_size,
        drop_policy=settings.decode_drop_policy,
    )
    logger.info(
        "Decode config: backend=%s, buffer=%s, opencv_buffer=%s, drop_policy=%s, prefer_latest=%s, reconnect_sleep_s=%s",
        config.backend,
        config.buffer_size,
        config.opencv_buffer_size,
        config.drop_policy,
        settings.prefer_latest_frame,
        settings.decode_reconnect_sleep_s,
    )
    return config


def build_postprocessor(settings: RuntimeSettings) -> TrackPostProcessor:
    """Build the trajectory smoother and short-term re-identification stage."""

    config = PostprocessConfig(
        smooth_alpha=settings.smooth_alpha,
        ball_smooth_alpha=settings.ball_smooth_alpha,
        fast_motion_px=settings.fast_motion_px,
        fast_motion_alpha=settings.fast_motion_alpha,
        max_prediction_dt_s=settings.max_prediction_dt_s,
        ball_prediction_damping=settings.ball_prediction_damping,
        player_prediction_damping=settings.player_prediction_damping,
        ball_confidence_decay=settings.ball_confidence_decay,
        player_confidence_decay=settings.player_confidence_decay,
        reid_ttl_ms=settings.reid_ttl_ms,
        reid_distance_px=settings.reid_distance_px,
        reid_enabled=settings.reid_enabled,
        reid_max_inactive_tracks=settings.reid_max_inactive_tracks,
    )
    logger.info(
        "Postprocess config: smooth_alpha=%s, ball_smooth_alpha=%s, fast_motion_px=%s, fast_motion_alpha=%s, reid_enabled=%s, reid_ttl_ms=%s, reid_distance_px=%s, reid_max_inactive=%s",
        config.smooth_alpha,
        config.ball_smooth_alpha,
        config.fast_motion_px,
        config.fast_motion_alpha,
        config.reid_enabled,
        config.reid_ttl_ms,
        config.reid_distance_px,
        config.reid_max_inactive_tracks,
    )
    return TrackPostProcessor(config=config)


def build_recorder(settings: RuntimeSettings) -> JsonlRecorder | None:
    """Build optional JSONL recorder."""

    if not settings.record_path:
        return None
    recorder = JsonlRecorder(settings.record_path)
    logger.info("Recorder enabled: %s", settings.record_path)
    return recorder
