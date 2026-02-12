from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    kind: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    team: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        # The foot point is typically closer to a player's true ground position.
        return (self.x + self.w / 2.0, self.y + self.h)


@dataclass
class TrackState:
    track_id: int
    kind: str
    team: str
    x_px: float
    y_px: float
    vx_px: float
    vy_px: float
    confidence: float
    last_ts_ms: int
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_w: float | None = None
    bbox_h: float | None = None
    missed_frames: int = 0
