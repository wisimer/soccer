from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from .models import Detection


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TeamClassifierConfig:
    mode: str = "auto"
    team_a_primary_color: str = "red"
    team_b_primary_color: str = "blue"
    saturation_threshold: int = 45
    min_color_ratio: float = 0.08


@dataclass(frozen=True, slots=True)
class BallDetectionConfig:
    min_area: float = 6.0
    max_area: float = 3000.0
    max_aspect_ratio: float = 2.5
    max_detections: int = 1


_HUE_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 12), (168, 179)),
    "orange": ((13, 24),),
    "yellow": ((25, 38),),
    "green": ((39, 84),),
    "cyan": ((85, 99),),
    "blue": ((100, 136),),
    "purple": ((137, 167),),
}

_COLOR_ALIASES = {
    "burgundy": "red",
    "crimson": "red",
    "maroon": "red",
    "pink": "red",
    "gold": "yellow",
    "lime": "green",
    "teal": "cyan",
    "aqua": "cyan",
    "skyblue": "cyan",
    "navy": "blue",
    "violet": "purple",
    "grey": "white",
    "gray": "white",
    "dark": "black",
}


class DetectorProtocol(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def detect_many(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...


def _normalize_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {i: str(name) for i, name in enumerate(list(raw_names))}
    return {}


def _empty_prediction() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
    )


class _UltralyticsModelRunner:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float,
        image_size: int,
        use_half: bool,
        batch_size: int,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_path)
        self._patch_legacy_model_signatures()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.use_half = bool(use_half) and str(device).lower().startswith("cuda")
        self.batch_size = max(1, int(batch_size))
        self.class_names = _normalize_class_names(getattr(self.model.model, "names", {}) or {})

    def predict(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not frames:
            return []

        results = self.model.predict(
            source=frames if len(frames) > 1 else frames[0],
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            half=self.use_half,
            batch=min(self.batch_size, len(frames)),
            verbose=False,
        )
        predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                predictions.append(_empty_prediction())
                continue
            predictions.append(
                (
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                )
            )
        return predictions

    def _patch_legacy_model_signatures(self) -> None:
        model_impl = getattr(self.model, "model", None)
        if model_impl is None:
            return
        self._patch_method_kwargs(model_impl, "fuse")
        self._patch_method_kwargs(model_impl, "forward")

    @staticmethod
    def _patch_method_kwargs(model_impl: Any, method_name: str) -> None:
        method = getattr(model_impl, method_name, None)
        if method is None:
            return
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return

        allowed_kwargs = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        original_method = method

        def _compat_method(*args: Any, **kwargs: Any) -> Any:
            filtered_kwargs = {name: value for name, value in kwargs.items() if name in allowed_kwargs}
            return original_method(*args, **filtered_kwargs)

        setattr(model_impl, method_name, _compat_method)


class _Yolov5ModelRunner:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float,
        image_size: int,
        use_half: bool,
        batch_size: int,
    ) -> None:
        try:
            import torch
            from yolov5.models.common import AutoShape, DetectMultiBackend
            from yolov5.models.yolo import ClassificationModel, SegmentationModel
            from yolov5.utils.torch_utils import select_device
        except ImportError as exc:
            raise RuntimeError(
                "yolov5 is not installed. Install with `pip install yolov5`."
            ) from exc

        original_torch_load = torch.load

        def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        try:
            resolved_device = select_device(device)
            backend_model = DetectMultiBackend(
                model_path,
                device=resolved_device,
                fp16=bool(use_half) and str(device).lower().startswith("cuda"),
                fuse=False,
            )
        finally:
            torch.load = original_torch_load

        if backend_model.pt and not isinstance(backend_model.model, (ClassificationModel, SegmentationModel)):
            self.model = AutoShape(backend_model, verbose=False)
        else:
            self.model = backend_model

        self.device = str(resolved_device)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        requested_half = bool(use_half) and str(device).lower().startswith("cuda")
        self.use_half = False
        self.batch_size = max(1, int(batch_size))
        self.model.conf = confidence_threshold
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 300
        self._half_enabled = False
        if requested_half:
            logger.info("YOLOv5 backend keeps FP32 inference on CUDA for stability with custom checkpoints")
        raw_names = getattr(self.model, "names", None)
        if raw_names is None and hasattr(self.model, "model"):
            raw_names = getattr(self.model.model, "names", None)
        self.class_names = _normalize_class_names(raw_names or {})

    def predict(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not frames:
            return []

        try:
            results = self.model(frames, size=self.image_size)
        except Exception as exc:
            should_retry_without_half = (
                self._half_enabled
                and str(self.device).lower().startswith("cuda")
                and "cublas_status_invalid_value" in str(exc).lower()
            )
            if not should_retry_without_half:
                raise
            logger.warning("YOLOv5 FP16 inference failed on %s, retrying with FP32: %s", self.device, exc)
            self.use_half = False
            self._half_enabled = False
            if hasattr(self.model, "model"):
                self.model.model.float()
            results = self.model(frames, size=self.image_size)
        raw_predictions = list(getattr(results, "pred", []) or [])
        predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for index in range(len(frames)):
            if index >= len(raw_predictions) or raw_predictions[index] is None or len(raw_predictions[index]) == 0:
                predictions.append(_empty_prediction())
                continue
            pred = raw_predictions[index]
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
            else:
                pred = np.asarray(pred)
            predictions.append(
                (
                    pred[:, :4].astype(np.float32, copy=False),
                    pred[:, 4].astype(np.float32, copy=False),
                    pred[:, 5].astype(np.int32, copy=False),
                )
            )
        return predictions


def _normalize_color_name(value: str) -> str:
    """Normalize user-supplied team color labels into canonical names."""

    normalized = str(value or "").strip().lower().replace(" ", "").replace("-", "")
    return _COLOR_ALIASES.get(normalized, normalized)


def _team_color_ratio(hsv_roi: np.ndarray, color_name: str, saturation_threshold: int) -> float:
    """Measure how much of the ROI matches the requested jersey color."""

    if hsv_roi.size == 0:
        return 0.0

    color = _normalize_color_name(color_name)
    sat = hsv_roi[:, :, 1]
    val = hsv_roi[:, :, 2]
    hue = hsv_roi[:, :, 0]

    if color == "white":
        mask = (sat <= 40) & (val >= 170)
    elif color == "black":
        mask = val <= 70
    else:
        ranges = _HUE_RANGES.get(color)
        if not ranges:
            return 0.0
        hue_mask = np.zeros(hue.shape, dtype=bool)
        for start, end in ranges:
            hue_mask |= (hue >= start) & (hue <= end)
        mask = hue_mask & (sat >= saturation_threshold)

    return float(np.count_nonzero(mask)) / float(mask.size)


def infer_team_from_bbox(
    hsv_frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    config: TeamClassifierConfig | None = None,
) -> str:
    """Infer player team label from jersey color in the upper half of the box."""

    team_config = config or TeamClassifierConfig()
    top = y
    bottom = y + max(1, h // 2)
    left = x
    right = x + w
    roi = hsv_frame[top:bottom, left:right]
    if roi.size == 0:
        return "unknown"

    if team_config.mode == "manual":
        score_a = _team_color_ratio(roi, team_config.team_a_primary_color, team_config.saturation_threshold)
        score_b = _team_color_ratio(roi, team_config.team_b_primary_color, team_config.saturation_threshold)
        threshold = float(team_config.min_color_ratio)
        if max(score_a, score_b) < threshold or abs(score_a - score_b) < 1e-6:
            return "unknown"
        return "A" if score_a > score_b else "B"

    sat = roi[:, :, 1]
    hue = roi[:, :, 0]
    valid = sat > team_config.saturation_threshold
    if not np.any(valid):
        return "unknown"
    mean_hue = float(np.mean(hue[valid]))
    if mean_hue < 20 or mean_hue > 160:
        return "A"
    if 85 <= mean_hue <= 140:
        return "B"
    return "unknown"


class YoloDetector:
    """YOLO detector for mainstream production-grade object detection."""

    name = "yolo"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        image_size: int = 1280,
        use_half: bool = False,
        batch_size: int = 1,
        team_config: TeamClassifierConfig | None = None,
        backend: str = "auto",
        allowed_kinds: tuple[str, ...] | None = None,
        ball_config: BallDetectionConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.use_half = bool(use_half) and str(device).lower().startswith("cuda")
        self.batch_size = max(1, int(batch_size))
        self.team_config = team_config or TeamClassifierConfig()
        self.allowed_kinds = {str(kind).strip().lower() for kind in (allowed_kinds or ()) if str(kind).strip()}
        self.ball_config = ball_config or BallDetectionConfig()
        self.backend = self._resolve_backend(model_path=model_path, backend=backend)
        self.runner = self._build_runner(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            use_half=use_half,
            batch_size=batch_size,
            backend=self.backend,
        )
        self.model = self.runner.model
        self.class_names = self.runner.class_names
        self.name = f"yolo-{self.backend}"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection for a single frame."""

        return self.detect_many([frame])[0]

    def detect_many(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run batched detection for one or more frames."""

        if not frames:
            return []

        predictions = self.runner.predict(frames)
        return [self._build_detections(frame, xyxy, conf, cls) for frame, (xyxy, conf, cls) in zip(frames, predictions)]

    def _build_detections(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if xyxy.size == 0 or conf.size == 0 or cls.size == 0:
            return detections

        frame_h, frame_w = frame.shape[:2]
        hsv_frame: np.ndarray | None = None

        for box, score, cls_id in zip(xyxy, conf, cls):
            class_name = self.class_names.get(cls_id, "")
            kind = self._resolve_kind(cls_id, class_name)
            if kind is None:
                continue
            if self.allowed_kinds and kind not in self.allowed_kinds:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            if kind == "ball" and not self._accept_ball_bbox(w=w, h=h):
                continue

            team = None
            if kind == "player":
                if hsv_frame is None:
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                team = infer_team_from_bbox(hsv_frame, x1, y1, w, h, self.team_config)
            detections.append(
                Detection(
                    kind=kind,
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    confidence=float(score),
                    team=team,
                )
            )

        max_ball_detections = max(0, int(self.ball_config.max_detections))
        if max_ball_detections:
            player_detections = [item for item in detections if item.kind != "ball"]
            ball_detections = [item for item in detections if item.kind == "ball"]
            if len(ball_detections) > max_ball_detections:
                ball_detections = sorted(ball_detections, key=lambda item: item.confidence, reverse=True)[
                    :max_ball_detections
                ]
            detections = player_detections + ball_detections

        return detections

    def _accept_ball_bbox(self, w: int, h: int) -> bool:
        area = float(w * h)
        if area < float(self.ball_config.min_area) or area > float(self.ball_config.max_area):
            return False
        ratio = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
        return ratio <= float(self.ball_config.max_aspect_ratio)

    @staticmethod
    def _resolve_backend(model_path: str, backend: str) -> str:
        normalized_backend = str(backend or "auto").strip().lower()
        if normalized_backend in {"ultralytics", "yolov5"}:
            return normalized_backend
        if "yolov5" in str(model_path).strip().lower():
            return "yolov5"
        return "ultralytics"

    @staticmethod
    def _build_runner(
        model_path: str,
        device: str,
        confidence_threshold: float,
        image_size: int,
        use_half: bool,
        batch_size: int,
        backend: str,
    ) -> _UltralyticsModelRunner | _Yolov5ModelRunner:
        if backend == "yolov5":
            return _Yolov5ModelRunner(
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold,
                image_size=image_size,
                use_half=use_half,
                batch_size=batch_size,
            )
        return _UltralyticsModelRunner(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            use_half=use_half,
            batch_size=batch_size,
        )

    @staticmethod
    def _resolve_kind(cls_id: int, class_name: str) -> str | None:
        name = class_name.strip().lower()
        if "ball" in name or "football" in name:
            return "ball"

        player_tokens = (
            "person",
            "player",
            "goalkeeper",
            "keeper",
            "goalie",
            "referee",
            "ref",
        )
        if any(token in name for token in player_tokens):
            return "player"

        # Backward-compatible fallback for default COCO IDs.
        if cls_id == 32:
            return "ball"
        if cls_id == 0:
            return "player"
        return None


class HybridDetector:
    name = "hybrid-yolo"

    def __init__(self, player_detector: DetectorProtocol, ball_detector: DetectorProtocol) -> None:
        self.player_detector = player_detector
        self.ball_detector = ball_detector
        self.batch_size = max(
            1,
            int(getattr(player_detector, "batch_size", 1)),
            int(getattr(ball_detector, "batch_size", 1)),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_many([frame])[0]

    def detect_many(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []

        player_batch = self.player_detector.detect_many(frames)
        ball_batch = self.ball_detector.detect_many(frames)
        merged: list[list[Detection]] = []
        for players, balls in zip(player_batch, ball_batch):
            merged.append(list(players) + list(balls))
        return merged
