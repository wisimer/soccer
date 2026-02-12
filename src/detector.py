from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .models import Detection


class DetectorProtocol(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def infer_team_from_bbox(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    # Use upper body area to estimate jersey color.
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


class HeuristicDetector:
    """Simple detector for bootstrapping the full pipeline."""

    name = "heuristic"

    def __init__(self) -> None:
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=30,
            detectShadows=False,
        )
        self.last_ball_center: tuple[float, float] | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        field_mask = cv2.inRange(
            hsv,
            np.array([35, 20, 20], dtype=np.uint8),
            np.array([95, 255, 255], dtype=np.uint8),
        )

        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        moving_non_field = cv2.bitwise_and(fg_mask, cv2.bitwise_not(field_mask))
        contours, _ = cv2.findContours(
            moving_non_field,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180 or area > 7000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 18 or w < 8:
                continue
            if h / max(w, 1) < 1.0:
                continue

            team = infer_team_from_bbox(frame, x, y, w, h)
            detections.append(
                Detection(
                    kind="player",
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=0.55,
                    team=team,
                )
            )

        ball_detection = self._detect_ball(hsv)
        if ball_detection is not None:
            detections.append(ball_detection)

        return detections

    def _detect_ball(self, hsv: np.ndarray) -> Detection | None:
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 70, 255], dtype=np.uint8),
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            white_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best_score = float("inf")
        best_det: Detection | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or area > 220:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            if self.last_ball_center is None:
                dist = 0.0
            else:
                dx = cx - self.last_ball_center[0]
                dy = cy - self.last_ball_center[1]
                dist = float(np.hypot(dx, dy))

            score = dist - circularity * 15.0
            if score < best_score:
                best_score = score
                best_det = Detection(
                    kind="ball",
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=min(0.9, max(0.3, float(circularity))),
                    team=None,
                )

        if best_det is not None:
            self.last_ball_center = best_det.center
        return best_det


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
