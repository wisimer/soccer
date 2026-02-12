from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class ProjectorProtocol(Protocol):
    name: str

    def update(self, frame_width: int, frame_height: int) -> None: ...

    def to_pitch(self, x_px: float, y_px: float) -> tuple[float, float]: ...

    def velocity_to_pitch(
        self,
        x_px: float,
        y_px: float,
        vx_px: float,
        vy_px: float,
    ) -> tuple[float, float]: ...


@dataclass
class LinearProjector:
    name: str = "linear"
    frame_width: int = 1920
    frame_height: int = 1080

    def update(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = max(1, frame_width)
        self.frame_height = max(1, frame_height)

    def to_pitch(self, x_px: float, y_px: float) -> tuple[float, float]:
        x = x_px / self.frame_width * PITCH_LENGTH_M
        y = y_px / self.frame_height * PITCH_WIDTH_M
        return (float(x), float(y))

    def velocity_to_pitch(
        self,
        x_px: float,
        y_px: float,
        vx_px: float,
        vy_px: float,
    ) -> tuple[float, float]:
        vx = vx_px / self.frame_width * PITCH_LENGTH_M
        vy = vy_px / self.frame_height * PITCH_WIDTH_M
        return (float(vx), float(vy))


class HomographyProjector:
    name = "homography"

    def __init__(self, homography: np.ndarray) -> None:
        if homography.shape != (3, 3):
            raise ValueError("homography must be 3x3")
        self.homography = homography.astype(np.float64)
        self.frame_width = 1920
        self.frame_height = 1080

    @classmethod
    def from_json(cls, path: str | Path) -> "HomographyProjector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        if "homography" in data:
            matrix = np.array(data["homography"], dtype=np.float64)
            return cls(matrix)

        image_points = np.array(data["image_points"], dtype=np.float64)
        pitch_points = np.array(data["pitch_points"], dtype=np.float64)
        if image_points.shape[0] < 4 or pitch_points.shape[0] < 4:
            raise ValueError("image_points and pitch_points must contain at least 4 points")

        matrix, _ = cv2.findHomography(image_points, pitch_points, method=0)
        if matrix is None:
            raise ValueError("failed to estimate homography")
        return cls(matrix)

    def update(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = max(1, frame_width)
        self.frame_height = max(1, frame_height)

    def to_pitch(self, x_px: float, y_px: float) -> tuple[float, float]:
        pts = np.array([[[x_px, y_px]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pts, self.homography)
        x, y = projected[0, 0, 0], projected[0, 0, 1]
        x = float(np.clip(x, 0.0, PITCH_LENGTH_M))
        y = float(np.clip(y, 0.0, PITCH_WIDTH_M))
        return (x, y)

    def velocity_to_pitch(
        self,
        x_px: float,
        y_px: float,
        vx_px: float,
        vy_px: float,
    ) -> tuple[float, float]:
        x0, y0 = self.to_pitch(x_px, y_px)
        x1, y1 = self.to_pitch(x_px + vx_px, y_px + vy_px)
        return (float(x1 - x0), float(y1 - y0))
