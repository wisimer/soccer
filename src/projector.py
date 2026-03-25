from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
