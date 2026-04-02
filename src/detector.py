from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .models import Detection


@dataclass(frozen=True, slots=True)
class TeamClassifierConfig:
    mode: str = "auto"
    team_a_primary_color: str = "red"
    team_b_primary_color: str = "blue"
    saturation_threshold: int = 45
    min_color_ratio: float = 0.08


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
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.use_half = bool(use_half) and str(device).lower().startswith("cuda")
        self.batch_size = max(1, int(batch_size))
        self.team_config = team_config or TeamClassifierConfig()
        raw_names = getattr(self.model.model, "names", {}) or {}
        if isinstance(raw_names, dict):
            self.class_names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self.class_names = {i: str(name) for i, name in enumerate(list(raw_names))}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection for a single frame."""

        return self.detect_many([frame])[0]

    def detect_many(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run batched detection for one or more frames."""

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

        return [self._build_detections(frame, result) for frame, result in zip(frames, results)]

    def _build_detections(self, frame: np.ndarray, result: object) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)

        frame_h, frame_w = frame.shape[:2]
        hsv_frame: np.ndarray | None = None

        for box, score, cls_id in zip(xyxy, conf, cls):
            class_name = self.class_names.get(cls_id, "")
            kind = self._resolve_kind(cls_id, class_name)
            if kind is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            if kind == "ball" and w * h > 3000:
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

        return detections

    @staticmethod
    def _resolve_kind(cls_id: int, class_name: str) -> str | None:
        name = class_name.strip().lower()
        if "ball" in name:
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
