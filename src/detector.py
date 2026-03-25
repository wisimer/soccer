from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .models import Detection


class DetectorProtocol(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def infer_team_from_bbox(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    top = y
    bottom = y + max(1, h // 2)
    left = x
    right = x + w
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return "unknown"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv_roi[:, :, 1]
    hue = hsv_roi[:, :, 0]
    valid = sat > 45
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
        raw_names = getattr(self.model.model, "names", {}) or {}
        if isinstance(raw_names, dict):
            self.class_names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self.class_names = {i: str(name) for i, name in enumerate(list(raw_names))}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)

        frame_h, frame_w = frame.shape[:2]

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

            team = infer_team_from_bbox(frame, x1, y1, w, h) if kind == "player" else None
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
