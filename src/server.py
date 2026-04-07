from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .game_engine import GameEngine
from .pipeline import SCHEMA_VERSION, StreamProcessor
from .runtime import (
    RuntimeSettings,
    build_decode_config,
    build_detector,
    normalize_runtime_settings,
    build_postprocessor,
    build_projector,
    build_recorder,
    build_tracker,
    profile_defaults,
    resolve_preferred_yolo_model,
    resolve_effective_profile,
)

T = TypeVar("T")


class ConnectionHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)

        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

    async def size(self) -> int:
        async with self._lock:
            return len(self._clients)


class JoinGameRequest(BaseModel):
    user_id: str | None = None
    team: str


class SkillActionRequest(BaseModel):
    user_id: str
    team: str
    skill: str
    client_event_id: str | None = None


async def _forward_loop(queue: asyncio.Queue[dict[str, Any]], hub: ConnectionHub) -> None:
    while True:
        payload = await queue.get()
        await hub.broadcast(payload)


def _resolve_local_video_file(video_source: str, base_dir: Path) -> Path | None:
    source = str(video_source).strip()
    if not source:
        return None
    if source.isdigit():
        return None
    if "://" in source:
        return None

    raw_path = Path(source).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(base_dir / raw_path)
        candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _video_preview_url(local_video_file: Path | None) -> str | None:
    if local_video_file is None:
        return None
    return "/api/video"


def _runtime_snapshot(
    runtime_settings: RuntimeSettings,
    detector: Any,
    tracker: Any,
    projector: Any,
    local_video_file: Path | None,
) -> dict[str, Any]:
    preview_url = _video_preview_url(local_video_file)
    return {
        "performance_profile": runtime_settings.performance_profile,
        "effective_profile": runtime_settings.effective_profile,
        "target_fps": runtime_settings.target_fps,
        "detector": getattr(detector, "name", detector.__class__.__name__),
        "tracker": getattr(tracker, "name", tracker.__class__.__name__),
        "projector": getattr(projector, "name", projector.__class__.__name__),
        "yolo_model": runtime_settings.yolo_model,
        "yolo_device": runtime_settings.yolo_device,
        "yolo_half": runtime_settings.yolo_half,
        "yolo_conf": runtime_settings.yolo_conf,
        "yolo_imgsz": runtime_settings.yolo_imgsz,
        "yolo_batch_size": runtime_settings.yolo_batch_size,
        "ball_yolo_model": runtime_settings.ball_yolo_model,
        "ball_yolo_device": runtime_settings.ball_yolo_device,
        "ball_yolo_half": runtime_settings.ball_yolo_half,
        "ball_yolo_conf": runtime_settings.ball_yolo_conf,
        "ball_yolo_imgsz": runtime_settings.ball_yolo_imgsz,
        "ball_yolo_batch_size": runtime_settings.ball_yolo_batch_size,
        "ball_yolo_backend": runtime_settings.ball_yolo_backend,
        "ball_min_area": runtime_settings.ball_min_area,
        "ball_max_area": runtime_settings.ball_max_area,
        "ball_max_aspect_ratio": runtime_settings.ball_max_aspect_ratio,
        "ball_max_detections": runtime_settings.ball_max_detections,
        "track_buffer": runtime_settings.track_buffer,
        "decode_backend": runtime_settings.decode_backend,
        "decode_buffer_size": runtime_settings.decode_buffer_size,
        "decode_opencv_buffer_size": runtime_settings.decode_opencv_buffer_size,
        "decode_drop_policy": runtime_settings.decode_drop_policy,
        "decode_reconnect_sleep_s": runtime_settings.decode_reconnect_sleep_s,
        "bytetrack_track_activation_threshold": runtime_settings.bytetrack_track_activation_threshold,
        "bytetrack_kalman_position_weight": runtime_settings.bytetrack_kalman_position_weight,
        "bytetrack_kalman_velocity_weight": runtime_settings.bytetrack_kalman_velocity_weight,
        "gmc_enabled": runtime_settings.gmc_enabled,
        "gmc_method": runtime_settings.gmc_method,
        "gmc_downscale": runtime_settings.gmc_downscale,
        "gmc_min_points": runtime_settings.gmc_min_points,
        "gmc_motion_deadband_px": runtime_settings.gmc_motion_deadband_px,
        "gmc_max_translation_px": runtime_settings.gmc_max_translation_px,
        "prefer_latest_frame": runtime_settings.prefer_latest_frame,
        "team_classification_mode": runtime_settings.team_classification_mode,
        "team_a_primary_color": runtime_settings.team_a_primary_color,
        "team_b_primary_color": runtime_settings.team_b_primary_color,
        "team_color_saturation_threshold": runtime_settings.team_color_saturation_threshold,
        "team_color_min_ratio": runtime_settings.team_color_min_ratio,
        "smooth_alpha": runtime_settings.smooth_alpha,
        "ball_smooth_alpha": runtime_settings.ball_smooth_alpha,
        "fast_motion_px": runtime_settings.fast_motion_px,
        "fast_motion_alpha": runtime_settings.fast_motion_alpha,
        "max_prediction_dt_s": runtime_settings.max_prediction_dt_s,
        "ball_prediction_damping": runtime_settings.ball_prediction_damping,
        "player_prediction_damping": runtime_settings.player_prediction_damping,
        "ball_confidence_decay": runtime_settings.ball_confidence_decay,
        "player_confidence_decay": runtime_settings.player_confidence_decay,
        "reid_enabled": runtime_settings.reid_enabled,
        "reid_ttl_ms": runtime_settings.reid_ttl_ms,
        "reid_distance_px": runtime_settings.reid_distance_px,
        "reid_max_inactive_tracks": runtime_settings.reid_max_inactive_tracks,
        "record_path": runtime_settings.record_path,
        "video_preview_url": preview_url,
        "ws_schema_version": SCHEMA_VERSION,
        "game_mode": "realtime-command-battle",
    }


def create_app(video_source: str, target_fps: int, runtime_settings: RuntimeSettings) -> FastAPI:
    base_dir = Path(__file__).resolve().parents[1]
    web_dir = base_dir / "web"
    local_video_file = _resolve_local_video_file(video_source, base_dir)
    preview_url = _video_preview_url(local_video_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5)
        hub = ConnectionHub()
        game_engine = GameEngine()

        detector = build_detector(runtime_settings)
        tracker = build_tracker(runtime_settings)
        projector = build_projector()
        decode_config = build_decode_config(runtime_settings)
        postprocessor = build_postprocessor(runtime_settings)
        recorder = build_recorder(runtime_settings)

        processor = StreamProcessor(
            source=video_source,
            target_fps=target_fps,
            reconnect_sleep_s=runtime_settings.decode_reconnect_sleep_s,
            detector=detector,
            tracker=tracker,
            projector=projector,
            decode_config=decode_config,
            postprocessor=postprocessor,
            recorder=recorder,
            game_engine=game_engine,
            prefer_latest_frame=runtime_settings.prefer_latest_frame,
        )

        producer_task = asyncio.create_task(processor.run(queue), name="stream-processor")
        forward_task = asyncio.create_task(_forward_loop(queue, hub), name="ws-forwarder")

        app.state.queue = queue
        app.state.hub = hub
        app.state.game_engine = game_engine
        app.state.processor = processor
        app.state.runtime = _runtime_snapshot(runtime_settings, detector, tracker, projector, local_video_file)

        try:
            yield
        finally:
            processor.stop()
            producer_task.cancel()
            forward_task.cancel()
            await asyncio.gather(producer_task, forward_task, return_exceptions=True)

    app = FastAPI(title="football-mvp", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "video_source": video_source,
            "video_preview_url": preview_url,
            "target_fps": target_fps,
            "runtime": app.state.runtime,
            "clients": await app.state.hub.size(),
        }

    @app.get("/api/video")
    async def video_preview() -> FileResponse:
        if local_video_file is None:
            raise HTTPException(status_code=404, detail="Current source is not a local video file.")
        return FileResponse(local_video_file)

    @app.post("/api/game/join")
    async def game_join(request: JoinGameRequest) -> dict[str, Any]:
        game_engine: GameEngine = app.state.game_engine
        user_id = (request.user_id or f"u_{uuid.uuid4().hex[:10]}").strip()
        team = str(request.team or "").strip().upper()
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if team not in ("A", "B"):
            raise HTTPException(status_code=400, detail="team must be A or B")

        try:
            join_result = game_engine.join(user_id=user_id, team=team)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "ok": True,
            "join": join_result,
            "state": game_engine.get_state(user_id=user_id),
        }

    @app.post("/api/game/action")
    async def game_action(request: SkillActionRequest) -> dict[str, Any]:
        game_engine: GameEngine = app.state.game_engine
        user_id = str(request.user_id or "").strip()
        team = str(request.team or "").strip().upper()
        skill = str(request.skill or "").strip()
        if not user_id or not skill:
            raise HTTPException(status_code=400, detail="user_id and skill are required")
        if team not in ("A", "B"):
            raise HTTPException(status_code=400, detail="team must be A or B")

        result = game_engine.submit_action(
            user_id=user_id,
            team=team,
            skill=skill,
            client_event_id=request.client_event_id,
        )
        result["state"] = game_engine.get_state(user_id=user_id)
        return result

    @app.get("/api/game/state")
    async def game_state(user_id: str | None = None) -> dict[str, Any]:
        game_engine: GameEngine = app.state.game_engine
        if user_id is not None:
            user_id = str(user_id).strip() or None
        return game_engine.get_state(user_id=user_id)

    @app.get("/api/game/result/latest")
    async def game_result_latest() -> dict[str, Any]:
        game_engine: GameEngine = app.state.game_engine
        result = game_engine.get_latest_result()
        if result is None:
            raise HTTPException(status_code=404, detail="No round result available yet.")
        return result

    @app.get("/api/game/highlight/latest")
    async def game_highlight_latest() -> dict[str, Any]:
        game_engine: GameEngine = app.state.game_engine
        highlight = game_engine.get_latest_highlight()
        if highlight is None:
            raise HTTPException(status_code=404, detail="No highlight available yet.")
        return highlight

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        hub: ConnectionHub = app.state.hub
        await hub.connect(ws)
        try:
            while True:
                # Keep the connection alive and detect client disconnect.
                await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await hub.disconnect(ws)

    return app


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _env_value(name: str, caster: Callable[[Any], T | None]) -> T | None:
    return caster(os.getenv(name))


def _resolve_value(
    explicit: Any,
    yaml_value: Any,
    profiled: Any,
    env_name: str,
    hard_default: T,
    caster: Callable[[Any], T | None],
) -> T:
    for candidate in (explicit, _env_value(env_name, caster), yaml_value, profiled, hard_default):
        resolved = caster(candidate)
        if resolved is not None:
            return resolved
    return hard_default


def _resolve_bool(
    explicit: bool | None,
    yaml_value: Any,
    profiled: Any,
    env_name: str,
    hard_default: bool,
) -> bool:
    return _resolve_value(explicit, yaml_value, profiled, env_name, hard_default, _coerce_bool)


def _resolve_float(
    explicit: float | None,
    yaml_value: Any,
    profiled: Any,
    env_name: str,
    hard_default: float,
) -> float:
    return _resolve_value(explicit, yaml_value, profiled, env_name, hard_default, _coerce_float)


def _resolve_int(
    explicit: int | None,
    yaml_value: Any,
    profiled: Any,
    env_name: str,
    hard_default: int,
) -> int:
    return _resolve_value(explicit, yaml_value, profiled, env_name, hard_default, _coerce_int)


def _resolve_str(
    explicit: str | None,
    yaml_value: Any,
    profiled: Any,
    env_name: str,
    hard_default: str,
) -> str:
    return _resolve_value(explicit, yaml_value, profiled, env_name, hard_default, _coerce_str)


def _load_runtime_yaml() -> dict[str, Any]:
    config_path = Path(os.getenv("MVP_CONFIG_PATH", Path(__file__).resolve().parents[1] / "config" / "runtime.yaml"))
    if not config_path.exists() or not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return {}


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if isinstance(section, dict):
        return section
    return {}


def _runtime_config_sections(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracker_config = _config_section(config, "tracker")
    legacy_gmc_config = config.get("gmc") if isinstance(config.get("gmc"), dict) else {}
    return {
        "input": _config_section(config, "input"),
        "detection": _config_section(config, "detection"),
        "tracker": tracker_config,
        "team": _config_section(config, "team"),
        "postprocess": _config_section(config, "postprocess"),
        "gmc": _config_section(tracker_config, "gmc") or legacy_gmc_config,
    }


def build_runtime_settings(args: argparse.Namespace) -> RuntimeSettings:
    """Resolve CLI, env, profile, and YAML into runtime settings."""

    requested_profile = str(args.profile or os.getenv("MVP_PROFILE", "auto")).lower()
    effective_profile = resolve_effective_profile(requested_profile)
    profiled = profile_defaults(effective_profile)
    config = _runtime_config_sections(_load_runtime_yaml())
    input_config = config["input"]
    detection_config = config["detection"]
    tracker_config = config["tracker"]
    team_config = config["team"]
    postprocess_config = config["postprocess"]
    gmc_config = config["gmc"]
    arg = lambda name: getattr(args, name, None)

    yolo_device = _resolve_str(
        args.yolo_device,
        detection_config.get("yolo_device"),
        profiled.get("yolo_device"),
        "MVP_YOLO_DEVICE",
        "cpu",
    )
    yolo_half = _resolve_bool(
        args.yolo_half,
        detection_config.get("yolo_half"),
        profiled.get("yolo_half"),
        "MVP_YOLO_HALF",
        False,
    )
    if yolo_half and not yolo_device.lower().startswith("cuda"):
        yolo_half = False

    decode_backend = _resolve_str(
        args.decode_backend,
        input_config.get("decode_backend"),
        profiled.get("decode_backend"),
        "MVP_DECODE_BACKEND",
        "opencv",
    ).lower()
    if decode_backend not in ("opencv", "pyav"):
        decode_backend = "opencv"

    settings = RuntimeSettings(
        performance_profile=requested_profile,
        effective_profile=effective_profile,
        target_fps=_resolve_int(arg("fps"), input_config.get("fps"), None, "MVP_FPS", 30),
        yolo_model=resolve_preferred_yolo_model(
            _resolve_str(arg("yolo_model"), detection_config.get("yolo_model"), None, "MVP_YOLO_MODEL", "")
        ),
        yolo_device=yolo_device,
        yolo_half=yolo_half,
        yolo_conf=_resolve_float(
            arg("yolo_conf"),
            detection_config.get("yolo_conf"),
            profiled.get("yolo_conf"),
            "MVP_YOLO_CONF",
            0.25,
        ),
        yolo_imgsz=_resolve_int(
            arg("yolo_imgsz"),
            detection_config.get("yolo_imgsz"),
            profiled.get("yolo_imgsz"),
            "MVP_YOLO_IMGSZ",
            1280,
        ),
        yolo_batch_size=_resolve_int(
            arg("yolo_batch_size"),
            detection_config.get("yolo_batch_size"),
            profiled.get("yolo_batch_size"),
            "MVP_YOLO_BATCH_SIZE",
            1,
        ),
        ball_yolo_model=_resolve_str(
            arg("ball_yolo_model"),
            detection_config.get("ball_yolo_model"),
            None,
            "MVP_BALL_YOLO_MODEL",
            "",
        ),
        ball_yolo_device=_resolve_str(
            arg("ball_yolo_device"),
            detection_config.get("ball_yolo_device"),
            profiled.get("ball_yolo_device"),
            "MVP_BALL_YOLO_DEVICE",
            yolo_device,
        ),
        ball_yolo_half=_resolve_bool(
            arg("ball_yolo_half"),
            detection_config.get("ball_yolo_half"),
            profiled.get("ball_yolo_half"),
            "MVP_BALL_YOLO_HALF",
            yolo_half,
        ),
        ball_yolo_conf=_resolve_float(
            arg("ball_yolo_conf"),
            detection_config.get("ball_yolo_conf"),
            profiled.get("ball_yolo_conf"),
            "MVP_BALL_YOLO_CONF",
            0.12,
        ),
        ball_yolo_imgsz=_resolve_int(
            arg("ball_yolo_imgsz"),
            detection_config.get("ball_yolo_imgsz"),
            profiled.get("ball_yolo_imgsz"),
            "MVP_BALL_YOLO_IMGSZ",
            1536,
        ),
        ball_yolo_batch_size=_resolve_int(
            arg("ball_yolo_batch_size"),
            detection_config.get("ball_yolo_batch_size"),
            profiled.get("ball_yolo_batch_size"),
            "MVP_BALL_YOLO_BATCH_SIZE",
            1,
        ),
        ball_yolo_backend=_resolve_str(
            arg("ball_yolo_backend"),
            detection_config.get("ball_yolo_backend"),
            None,
            "MVP_BALL_YOLO_BACKEND",
            "auto",
        ),
        ball_min_area=_resolve_float(
            arg("ball_min_area"),
            detection_config.get("ball_min_area"),
            None,
            "MVP_BALL_MIN_AREA",
            6.0,
        ),
        ball_max_area=_resolve_float(
            arg("ball_max_area"),
            detection_config.get("ball_max_area"),
            None,
            "MVP_BALL_MAX_AREA",
            3000.0,
        ),
        ball_max_aspect_ratio=_resolve_float(
            arg("ball_max_aspect_ratio"),
            detection_config.get("ball_max_aspect_ratio"),
            None,
            "MVP_BALL_MAX_ASPECT_RATIO",
            2.5,
        ),
        ball_max_detections=_resolve_int(
            arg("ball_max_detections"),
            detection_config.get("ball_max_detections"),
            None,
            "MVP_BALL_MAX_DETECTIONS",
            1,
        ),
        track_buffer=_resolve_int(arg("track_buffer"), tracker_config.get("track_buffer"), None, "MVP_TRACK_BUFFER", 30),
        bytetrack_track_activation_threshold=_resolve_float(
            arg("bytetrack_track_activation_threshold"),
            tracker_config.get("track_activation_threshold"),
            profiled.get("bytetrack_track_activation_threshold"),
            "MVP_BYTETRACK_TRACK_ACTIVATION_THRESHOLD",
            0.25,
        ),
        bytetrack_kalman_position_weight=_resolve_float(
            arg("bytetrack_kalman_position_weight"),
            tracker_config.get("kalman_position_weight"),
            profiled.get("bytetrack_kalman_position_weight"),
            "MVP_BYTETRACK_KALMAN_POSITION_WEIGHT",
            0.06,
        ),
        bytetrack_kalman_velocity_weight=_resolve_float(
            arg("bytetrack_kalman_velocity_weight"),
            tracker_config.get("kalman_velocity_weight"),
            profiled.get("bytetrack_kalman_velocity_weight"),
            "MVP_BYTETRACK_KALMAN_VELOCITY_WEIGHT",
            0.01125,
        ),
        gmc_enabled=_resolve_bool(
            arg("gmc_enabled"),
            gmc_config.get("enabled"),
            profiled.get("gmc_enabled"),
            "MVP_GMC_ENABLED",
            True,
        ),
        gmc_method=_resolve_str(
            arg("gmc_method"),
            gmc_config.get("method"),
            profiled.get("gmc_method"),
            "MVP_GMC_METHOD",
            "sparseOptFlow",
        ),
        gmc_downscale=_resolve_float(
            arg("gmc_downscale"),
            gmc_config.get("downscale"),
            profiled.get("gmc_downscale"),
            "MVP_GMC_DOWNSCALE",
            2.0,
        ),
        gmc_min_points=_resolve_int(
            arg("gmc_min_points"),
            gmc_config.get("min_points"),
            profiled.get("gmc_min_points"),
            "MVP_GMC_MIN_POINTS",
            12,
        ),
        gmc_motion_deadband_px=_resolve_float(
            arg("gmc_motion_deadband_px"),
            gmc_config.get("motion_deadband_px"),
            profiled.get("gmc_motion_deadband_px"),
            "MVP_GMC_MOTION_DEADBAND_PX",
            1.0,
        ),
        gmc_max_translation_px=_resolve_float(
            arg("gmc_max_translation_px"),
            gmc_config.get("max_translation_px"),
            profiled.get("gmc_max_translation_px"),
            "MVP_GMC_MAX_TRANSLATION_PX",
            80.0,
        ),
        decode_backend=decode_backend,
        decode_buffer_size=_resolve_int(
            arg("decode_buffer_size"),
            input_config.get("decode_buffer_size"),
            None,
            "MVP_DECODE_BUFFER_SIZE",
            8,
        ),
        decode_opencv_buffer_size=_resolve_int(
            arg("decode_opencv_buffer_size"),
            input_config.get("opencv_buffer_size"),
            profiled.get("decode_opencv_buffer_size"),
            "MVP_DECODE_OPENCV_BUFFER_SIZE",
            2,
        ),
        decode_drop_policy=_resolve_str(
            arg("decode_drop_policy"),
            input_config.get("decode_drop_policy"),
            None,
            "MVP_DECODE_DROP_POLICY",
            "drop_oldest",
        ),
        decode_reconnect_sleep_s=_resolve_float(
            arg("decode_reconnect_sleep_s"),
            input_config.get("reconnect_sleep_s"),
            None,
            "MVP_DECODE_RECONNECT_SLEEP_S",
            2.0,
        ),
        prefer_latest_frame=_resolve_bool(
            arg("prefer_latest_frame"),
            input_config.get("prefer_latest_frame"),
            profiled.get("prefer_latest_frame"),
            "MVP_PREFER_LATEST_FRAME",
            True,
        ),
        team_classification_mode=_resolve_str(
            arg("team_classification_mode"),
            team_config.get("classification_mode"),
            None,
            "MVP_TEAM_CLASSIFICATION_MODE",
            "auto",
        ),
        team_a_primary_color=_resolve_str(
            arg("team_a_primary_color"),
            team_config.get("team_a_primary_color"),
            None,
            "MVP_TEAM_A_PRIMARY_COLOR",
            "red",
        ),
        team_b_primary_color=_resolve_str(
            arg("team_b_primary_color"),
            team_config.get("team_b_primary_color"),
            None,
            "MVP_TEAM_B_PRIMARY_COLOR",
            "blue",
        ),
        team_color_saturation_threshold=_resolve_int(
            arg("team_color_saturation_threshold"),
            team_config.get("saturation_threshold"),
            None,
            "MVP_TEAM_COLOR_SATURATION_THRESHOLD",
            45,
        ),
        team_color_min_ratio=_resolve_float(
            arg("team_color_min_ratio"),
            team_config.get("min_color_ratio"),
            None,
            "MVP_TEAM_COLOR_MIN_RATIO",
            0.08,
        ),
        smooth_alpha=_resolve_float(
            arg("smooth_alpha"),
            postprocess_config.get("smooth_alpha"),
            None,
            "MVP_SMOOTH_ALPHA",
            0.35,
        ),
        ball_smooth_alpha=_resolve_float(
            arg("ball_smooth_alpha"),
            postprocess_config.get("ball_smooth_alpha"),
            None,
            "MVP_BALL_SMOOTH_ALPHA",
            0.68,
        ),
        fast_motion_px=_resolve_float(
            arg("fast_motion_px"),
            postprocess_config.get("fast_motion_px"),
            None,
            "MVP_FAST_MOTION_PX",
            20.0,
        ),
        fast_motion_alpha=_resolve_float(
            arg("fast_motion_alpha"),
            postprocess_config.get("fast_motion_alpha"),
            None,
            "MVP_FAST_MOTION_ALPHA",
            0.9,
        ),
        max_prediction_dt_s=_resolve_float(
            arg("max_prediction_dt_s"),
            postprocess_config.get("max_prediction_dt_s"),
            None,
            "MVP_MAX_PREDICTION_DT_S",
            0.25,
        ),
        ball_prediction_damping=_resolve_float(
            arg("ball_prediction_damping"),
            postprocess_config.get("ball_prediction_damping"),
            None,
            "MVP_BALL_PREDICTION_DAMPING",
            0.9,
        ),
        player_prediction_damping=_resolve_float(
            arg("player_prediction_damping"),
            postprocess_config.get("player_prediction_damping"),
            None,
            "MVP_PLAYER_PREDICTION_DAMPING",
            0.78,
        ),
        ball_confidence_decay=_resolve_float(
            arg("ball_confidence_decay"),
            postprocess_config.get("ball_confidence_decay"),
            None,
            "MVP_BALL_CONFIDENCE_DECAY",
            0.97,
        ),
        player_confidence_decay=_resolve_float(
            arg("player_confidence_decay"),
            postprocess_config.get("player_confidence_decay"),
            None,
            "MVP_PLAYER_CONFIDENCE_DECAY",
            0.93,
        ),
        reid_ttl_ms=_resolve_int(
            arg("reid_ttl_ms"),
            postprocess_config.get("reid_ttl_ms"),
            None,
            "MVP_REID_TTL_MS",
            1500,
        ),
        reid_distance_px=_resolve_float(
            arg("reid_distance_px"),
            postprocess_config.get("reid_distance_px"),
            None,
            "MVP_REID_DISTANCE_PX",
            85.0,
        ),
        reid_enabled=_resolve_bool(
            arg("reid_enabled"),
            postprocess_config.get("reid_enabled"),
            profiled.get("reid_enabled"),
            "MVP_REID_ENABLED",
            True,
        ),
        reid_max_inactive_tracks=_resolve_int(
            arg("reid_max_inactive_tracks"),
            postprocess_config.get("reid_max_inactive_tracks"),
            profiled.get("reid_max_inactive_tracks"),
            "MVP_REID_MAX_INACTIVE_TRACKS",
            256,
        ),
        record_path=_resolve_str(arg("record_path"), input_config.get("record_path"), None, "MVP_RECORD_PATH", ""),
    )
    return normalize_runtime_settings(settings)


def _default_video_source() -> str:
    """Return the preferred default local test clip when available."""

    env_source = os.getenv("MVP_VIDEO_SOURCE")
    if env_source:
        return env_source

    preferred = Path(__file__).resolve().parents[1] / "data" / "football.mp4"
    if preferred.exists() and preferred.is_file():
        return str(preferred)
    return "0"


def _add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-batch-size", type=int, default=None)
    parser.add_argument("--ball-yolo-model", default=None)
    parser.add_argument("--ball-yolo-device", default=None)
    parser.add_argument("--ball-yolo-half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ball-yolo-conf", type=float, default=None)
    parser.add_argument("--ball-yolo-imgsz", type=int, default=None)
    parser.add_argument("--ball-yolo-batch-size", type=int, default=None)
    parser.add_argument("--ball-yolo-backend", choices=["auto", "ultralytics", "yolov5"], default=None)
    parser.add_argument("--ball-min-area", type=float, default=None)
    parser.add_argument("--ball-max-area", type=float, default=None)
    parser.add_argument("--ball-max-aspect-ratio", type=float, default=None)
    parser.add_argument("--ball-max-detections", type=int, default=None)


def _add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track-buffer", type=int, default=None)
    parser.add_argument("--bytetrack-track-activation-threshold", type=float, default=None)
    parser.add_argument(
        "--bytetrack-kalman-position-weight",
        type=float,
        default=None,
        help="ByteTrack Kalman position process noise weight",
    )
    parser.add_argument(
        "--bytetrack-kalman-velocity-weight",
        type=float,
        default=None,
        help="ByteTrack Kalman velocity process noise weight",
    )
    parser.add_argument(
        "--gmc-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable camera motion compensation before ByteTrack association",
    )
    parser.add_argument(
        "--gmc-method",
        choices=["off", "sparseOptFlow"],
        default=None,
        help="Camera motion compensation method",
    )
    parser.add_argument("--gmc-downscale", type=float, default=None, help="Downscale factor for GMC feature tracking")
    parser.add_argument("--gmc-min-points", type=int, default=None, help="Minimum tracked feature points for GMC")
    parser.add_argument(
        "--gmc-motion-deadband-px",
        type=float,
        default=None,
        help="Ignore GMC translation smaller than this pixel magnitude",
    )
    parser.add_argument(
        "--gmc-max-translation-px",
        type=float,
        default=None,
        help="Clamp per-frame GMC translation to this magnitude",
    )


def _add_decode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "pyav"],
        default=None,
    )
    parser.add_argument(
        "--decode-buffer-size",
        type=int,
        default=None,
        help="Decoder frame buffer size",
    )
    parser.add_argument(
        "--decode-opencv-buffer-size",
        type=int,
        default=None,
        help="OpenCV capture buffer size for network streams",
    )
    parser.add_argument(
        "--decode-drop-policy",
        choices=["drop_oldest", "drop_newest"],
        default=None,
        help="Buffer full strategy",
    )
    parser.add_argument(
        "--decode-reconnect-sleep-s",
        type=float,
        default=None,
        help="Reconnect backoff in seconds after decoder errors",
    )
    parser.add_argument(
        "--prefer-latest-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer newest frame to reduce latency",
    )


def _add_team_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--team-classification-mode",
        choices=["auto", "manual"],
        default=None,
        help="Auto infer jersey colors or use configured team primary colors",
    )
    parser.add_argument("--team-a-primary-color", default=None)
    parser.add_argument("--team-b-primary-color", default=None)
    parser.add_argument("--team-color-saturation-threshold", type=int, default=None)
    parser.add_argument("--team-color-min-ratio", type=float, default=None)


def _add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=None,
        help="EMA smoothing alpha [0,1]",
    )
    parser.add_argument("--ball-smooth-alpha", type=float, default=None)
    parser.add_argument("--fast-motion-px", type=float, default=None)
    parser.add_argument("--fast-motion-alpha", type=float, default=None)
    parser.add_argument("--max-prediction-dt-s", type=float, default=None)
    parser.add_argument("--ball-prediction-damping", type=float, default=None)
    parser.add_argument("--player-prediction-damping", type=float, default=None)
    parser.add_argument("--ball-confidence-decay", type=float, default=None)
    parser.add_argument("--player-confidence-decay", type=float, default=None)
    parser.add_argument("--reid-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--reid-ttl-ms",
        type=int,
        default=None,
        help="Short-term re-identification TTL in milliseconds",
    )
    parser.add_argument(
        "--reid-distance-px",
        type=float,
        default=None,
        help="Max pixel distance for short-term re-identification",
    )
    parser.add_argument(
        "--reid-max-inactive-tracks",
        type=int,
        default=None,
        help="Maximum inactive track states kept for short-term re-identification",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the server process."""

    parser = argparse.ArgumentParser(description="Football stream MVP server")
    parser.add_argument(
        "--source",
        default=_default_video_source(),
        help="RTSP/HLS URL, video file path, or camera index (default: local test clip if present, else 0)",
    )
    parser.add_argument("--fps", type=int, default=None, help="Processing FPS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("MVP_PORT", "8765")))

    parser.add_argument(
        "--profile",
        choices=["auto", "nvidia", "cpu"],
        default=os.getenv("MVP_PROFILE", "auto"),
        help="Performance profile: auto detect or force hardware-specific defaults",
    )
    _add_detection_args(parser)
    _add_tracker_args(parser)
    _add_decode_args(parser)
    _add_team_args(parser)
    _add_postprocess_args(parser)
    parser.add_argument(
        "--record-path",
        default=None,
        help="Optional JSONL output path for payload recording",
    )
    parser.add_argument("--log-level", default=os.getenv("MVP_LOG_LEVEL", "info"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime_settings = build_runtime_settings(args)
    app = create_app(
        video_source=str(args.source),
        target_fps=runtime_settings.target_fps,
        runtime_settings=runtime_settings,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
