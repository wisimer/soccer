from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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


def create_app(video_source: str, target_fps: int, runtime_settings: RuntimeSettings) -> FastAPI:
    base_dir = Path(__file__).resolve().parents[1]
    web_dir = base_dir / "web"
    local_video_file = _resolve_local_video_file(video_source, base_dir)

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
        app.state.runtime = {
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
            "track_buffer": runtime_settings.track_buffer,
            "decode_backend": runtime_settings.decode_backend,
            "decode_buffer_size": runtime_settings.decode_buffer_size,
            "decode_drop_policy": runtime_settings.decode_drop_policy,
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
            "smooth_alpha": runtime_settings.smooth_alpha,
            "reid_ttl_ms": runtime_settings.reid_ttl_ms,
            "reid_distance_px": runtime_settings.reid_distance_px,
            "record_path": runtime_settings.record_path,
            "video_preview_url": "/api/video" if local_video_file is not None else None,
            "ws_schema_version": SCHEMA_VERSION,
            "game_mode": "realtime-command-battle",
        }

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
            "video_preview_url": "/api/video" if local_video_file is not None else None,
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


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = str(raw).strip()
    return stripped or None


def _resolve_bool(
    explicit: bool | None,
    profiled: Any,
    env_name: str,
    hard_default: bool,
) -> bool:
    if explicit is not None:
        return bool(explicit)
    if profiled is not None:
        return bool(profiled)
    env_val = _env_bool(env_name)
    if env_val is not None:
        return env_val
    return hard_default


def _resolve_float(
    explicit: float | None,
    profiled: Any,
    env_name: str,
    hard_default: float,
) -> float:
    if explicit is not None:
        return float(explicit)
    if profiled is not None:
        return float(profiled)
    env_val = _env_float(env_name)
    if env_val is not None:
        return float(env_val)
    return hard_default


def _resolve_int(
    explicit: int | None,
    profiled: Any,
    env_name: str,
    hard_default: int,
) -> int:
    if explicit is not None:
        return int(explicit)
    if profiled is not None:
        return int(profiled)
    env_val = _env_int(env_name)
    if env_val is not None:
        return int(env_val)
    return hard_default


def _resolve_str(
    explicit: str | None,
    profiled: Any,
    env_name: str,
    hard_default: str,
) -> str:
    if explicit is not None:
        return str(explicit)
    if profiled is not None:
        return str(profiled)
    env_val = _env_str(env_name)
    if env_val is not None:
        return env_val
    return hard_default


def _load_runtime_yaml() -> dict[str, Any]:
    config_path = Path(os.getenv("MVP_CONFIG_PATH", Path(__file__).resolve().parents[1] / "config" / "runtime.yaml"))
    if not config_path.exists() or not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return {}


def build_runtime_settings(args: argparse.Namespace) -> RuntimeSettings:
    requested_profile = str(args.profile or os.getenv("MVP_PROFILE", "auto")).lower()
    effective_profile = resolve_effective_profile(requested_profile)
    profiled = profile_defaults(effective_profile)
    yaml_config = _load_runtime_yaml()
    gmc_config = yaml_config.get("gmc") if isinstance(yaml_config.get("gmc"), dict) else {}

    yolo_device = _resolve_str(args.yolo_device, profiled.get("yolo_device"), "MVP_YOLO_DEVICE", "cpu")
    yolo_half = _resolve_bool(args.yolo_half, profiled.get("yolo_half"), "MVP_YOLO_HALF", False)
    if yolo_half and not yolo_device.lower().startswith("cuda"):
        yolo_half = False

    decode_backend = _resolve_str(
        args.decode_backend,
        profiled.get("decode_backend"),
        "MVP_DECODE_BACKEND",
        "opencv",
    ).lower()
    if decode_backend not in ("opencv", "pyav"):
        decode_backend = "opencv"

    settings = RuntimeSettings(
        performance_profile=requested_profile,
        effective_profile=effective_profile,
        target_fps=args.fps,
        yolo_model=resolve_preferred_yolo_model(args.yolo_model),
        yolo_device=yolo_device,
        yolo_half=yolo_half,
        yolo_conf=_resolve_float(args.yolo_conf, profiled.get("yolo_conf"), "MVP_YOLO_CONF", 0.25),
        yolo_imgsz=_resolve_int(args.yolo_imgsz, profiled.get("yolo_imgsz"), "MVP_YOLO_IMGSZ", 1280),
        track_buffer=args.track_buffer,
        bytetrack_track_activation_threshold=_resolve_float(
            args.bytetrack_track_activation_threshold,
            profiled.get("bytetrack_track_activation_threshold"),
            "MVP_BYTETRACK_TRACK_ACTIVATION_THRESHOLD",
            0.25,
        ),
        bytetrack_kalman_position_weight=_resolve_float(
            args.bytetrack_kalman_position_weight,
            profiled.get("bytetrack_kalman_position_weight"),
            "MVP_BYTETRACK_KALMAN_POSITION_WEIGHT",
            0.06,
        ),
        bytetrack_kalman_velocity_weight=_resolve_float(
            args.bytetrack_kalman_velocity_weight,
            profiled.get("bytetrack_kalman_velocity_weight"),
            "MVP_BYTETRACK_KALMAN_VELOCITY_WEIGHT",
            0.01125,
        ),
        gmc_enabled=_resolve_bool(
            args.gmc_enabled,
            gmc_config.get("enabled"),
            "MVP_GMC_ENABLED",
            True,
        ),
        gmc_method=_resolve_str(
            args.gmc_method,
            gmc_config.get("method"),
            "MVP_GMC_METHOD",
            "sparseOptFlow",
        ),
        gmc_downscale=_resolve_float(
            args.gmc_downscale,
            gmc_config.get("downscale"),
            "MVP_GMC_DOWNSCALE",
            2.0,
        ),
        gmc_min_points=_resolve_int(
            args.gmc_min_points,
            gmc_config.get("min_points"),
            "MVP_GMC_MIN_POINTS",
            12,
        ),
        gmc_motion_deadband_px=_resolve_float(
            args.gmc_motion_deadband_px,
            gmc_config.get("motion_deadband_px"),
            "MVP_GMC_MOTION_DEADBAND_PX",
            1.0,
        ),
        gmc_max_translation_px=_resolve_float(
            args.gmc_max_translation_px,
            gmc_config.get("max_translation_px"),
            "MVP_GMC_MAX_TRANSLATION_PX",
            80.0,
        ),
        decode_backend=decode_backend,
        decode_buffer_size=args.decode_buffer_size,
        decode_drop_policy=args.decode_drop_policy,
        prefer_latest_frame=_resolve_bool(
            args.prefer_latest_frame,
            profiled.get("prefer_latest_frame"),
            "MVP_PREFER_LATEST_FRAME",
            True,
        ),
        smooth_alpha=args.smooth_alpha,
        reid_ttl_ms=args.reid_ttl_ms,
        reid_distance_px=args.reid_distance_px,
        record_path=args.record_path,
    )
    return normalize_runtime_settings(settings)


def _default_video_source() -> str:
    env_source = os.getenv("MVP_VIDEO_SOURCE")
    if env_source:
        return env_source

    preferred = Path(__file__).resolve().parents[1] / "data" / "football.mp4"
    if preferred.exists() and preferred.is_file():
        return str(preferred)
    return "0"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football stream MVP server")
    parser.add_argument(
        "--source",
        default=_default_video_source(),
        help="RTSP/HLS URL, video file path, or camera index (default: local test clip if present, else 0)",
    )
    parser.add_argument("--fps", type=int, default=int(os.getenv("MVP_FPS", "30")), help="Processing FPS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("MVP_PORT", "8765")))

    parser.add_argument(
        "--profile",
        choices=["auto", "nvidia", "cpu"],
        default=os.getenv("MVP_PROFILE", "auto"),
        help="Performance profile: auto detect or force hardware-specific defaults",
    )
    parser.add_argument(
        "--yolo-model",
        default=resolve_preferred_yolo_model(os.getenv("MVP_YOLO_MODEL")),
    )
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--track-buffer", type=int, default=int(os.getenv("MVP_TRACK_BUFFER", "30")))
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
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "pyav"],
        default=None,
    )
    parser.add_argument(
        "--decode-buffer-size",
        type=int,
        default=int(os.getenv("MVP_DECODE_BUFFER_SIZE", "8")),
        help="Decoder frame buffer size",
    )
    parser.add_argument(
        "--decode-drop-policy",
        choices=["drop_oldest", "drop_newest"],
        default=os.getenv("MVP_DECODE_DROP_POLICY", "drop_oldest"),
        help="Buffer full strategy",
    )
    parser.add_argument(
        "--prefer-latest-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer newest frame to reduce latency",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=float(os.getenv("MVP_SMOOTH_ALPHA", "0.35")),
        help="EMA smoothing alpha [0,1]",
    )
    parser.add_argument(
        "--reid-ttl-ms",
        type=int,
        default=int(os.getenv("MVP_REID_TTL_MS", "1500")),
        help="Short-term re-identification TTL in milliseconds",
    )
    parser.add_argument(
        "--reid-distance-px",
        type=float,
        default=float(os.getenv("MVP_REID_DISTANCE_PX", "85.0")),
        help="Max pixel distance for short-term re-identification",
    )
    parser.add_argument(
        "--record-path",
        default=os.getenv("MVP_RECORD_PATH"),
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
        target_fps=int(args.fps),
        runtime_settings=runtime_settings,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
