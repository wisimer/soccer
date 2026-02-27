from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

TEAM_LABEL = {"A": "萌熊", "B": "猴子"}


@dataclass(slots=True)
class GameEvent:
    event_id: str
    type: str
    team: str
    confidence: float
    ts_ms: int
    text: str
    x: float | None = None
    y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "type": self.type,
            "team": self.team,
            "confidence": round(float(self.confidence), 3),
            "ts_ms": int(self.ts_ms),
            "text": self.text,
            "x": None if self.x is None else round(float(self.x), 2),
            "y": None if self.y is None else round(float(self.y), 2),
        }


class EventEngine:
    """Convert tracked entities to high-level entertainment events."""

    def __init__(
        self,
        possession_radius_m: float = 4.5,
        shot_speed_mps: float = 8.0,
        danger_speed_mps: float = 5.2,
    ) -> None:
        self.possession_radius_m = float(possession_radius_m)
        self.shot_speed_mps = float(shot_speed_mps)
        self.danger_speed_mps = float(danger_speed_mps)
        self._next_event_id = 1
        self._last_holder_team: str | None = None
        self._last_holder_id: int | None = None
        self._last_possession_team: str | None = None
        self._last_event_at: dict[tuple[str, str], int] = {}
        self._recent_events: deque[GameEvent] = deque(maxlen=240)
        self._cooldown_ms = {
            "possession_start": 700,
            "pass_complete": 900,
            "steal": 1000,
            "shot_attempt": 1700,
            "danger_moment": 2200,
        }

    def get_recent_events(self) -> list[GameEvent]:
        return list(self._recent_events)

    def update(self, entities: list[dict[str, Any]], ts_ms: int) -> list[GameEvent]:
        ball = self._pick_ball(entities)
        players = self._pick_players(entities)
        events: list[GameEvent] = []

        holder_team: str | None = None
        holder_id: int | None = None
        holder_conf = 0.0

        if ball is not None and players:
            holder_team, holder_id, holder_conf = self._resolve_possession(ball, players)

        if holder_team in ("A", "B"):
            if self._last_holder_team is None:
                evt = self._emit(
                    "possession_start",
                    holder_team,
                    min(0.95, 0.55 + holder_conf * 0.5),
                    ts_ms,
                    self._render_text("possession_start", holder_team),
                    ball["x"],
                    ball["y"],
                )
                if evt is not None:
                    events.append(evt)
            elif self._last_holder_team != holder_team:
                evt = self._emit(
                    "steal",
                    holder_team,
                    min(0.95, 0.6 + holder_conf * 0.4),
                    ts_ms,
                    self._render_text("steal", holder_team),
                    ball["x"],
                    ball["y"],
                )
                if evt is not None:
                    events.append(evt)

                evt = self._emit(
                    "possession_start",
                    holder_team,
                    min(0.95, 0.55 + holder_conf * 0.5),
                    ts_ms,
                    self._render_text("possession_start", holder_team),
                    ball["x"],
                    ball["y"],
                )
                if evt is not None:
                    events.append(evt)
            elif self._last_holder_id is not None and holder_id is not None and self._last_holder_id != holder_id:
                evt = self._emit(
                    "pass_complete",
                    holder_team,
                    min(0.95, 0.5 + holder_conf * 0.45),
                    ts_ms,
                    self._render_text("pass_complete", holder_team),
                    ball["x"],
                    ball["y"],
                )
                if evt is not None:
                    events.append(evt)

            self._last_holder_team = holder_team
            self._last_holder_id = holder_id
            self._last_possession_team = holder_team

        if ball is None:
            return events

        speed = float(np.hypot(ball["vx"], ball["vy"]))
        shot_team = holder_team or self._last_possession_team
        if shot_team in ("A", "B") and speed >= self.shot_speed_mps:
            toward_left_goal = ball["x"] <= 20.0 and ball["vx"] < -self.shot_speed_mps * 0.5
            toward_right_goal = ball["x"] >= 85.0 and ball["vx"] > self.shot_speed_mps * 0.5
            if toward_left_goal or toward_right_goal:
                evt = self._emit(
                    "shot_attempt",
                    shot_team,
                    min(0.97, max(0.35, ball["conf"])),
                    ts_ms,
                    self._render_text("shot_attempt", shot_team),
                    ball["x"],
                    ball["y"],
                )
                if evt is not None:
                    events.append(evt)

        danger_team = holder_team or self._last_possession_team
        in_danger_zone = ball["x"] <= 18.0 or ball["x"] >= 87.0
        if danger_team in ("A", "B") and in_danger_zone and speed >= self.danger_speed_mps:
            evt = self._emit(
                "danger_moment",
                danger_team,
                min(0.95, max(0.4, ball["conf"])),
                ts_ms,
                self._render_text("danger_moment", danger_team),
                ball["x"],
                ball["y"],
            )
            if evt is not None:
                events.append(evt)

        return events

    def _pick_ball(self, entities: list[dict[str, Any]]) -> dict[str, Any] | None:
        balls = [e for e in entities if str(e.get("type")) == "ball"]
        if not balls:
            return None
        balls.sort(key=lambda e: float(e.get("conf", 0.0)), reverse=True)
        return balls[0]

    def _pick_players(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            e
            for e in entities
            if str(e.get("type")) == "player" and str(e.get("team", "unknown")) in ("A", "B")
        ]

    def _resolve_possession(
        self,
        ball: dict[str, Any],
        players: list[dict[str, Any]],
    ) -> tuple[str | None, int | None, float]:
        bx = float(ball.get("x", 0.0))
        by = float(ball.get("y", 0.0))
        best: tuple[float, dict[str, Any]] | None = None

        for player in players:
            px = float(player.get("x", 0.0))
            py = float(player.get("y", 0.0))
            dist = float(np.hypot(px - bx, py - by))
            if best is None or dist < best[0]:
                best = (dist, player)

        if best is None:
            return (None, None, 0.0)

        dist, player = best
        if dist > self.possession_radius_m:
            return (None, None, 0.0)

        return (
            str(player.get("team", "unknown")),
            int(player.get("id", -1)) if player.get("id") is not None else None,
            float(player.get("conf", 0.0)),
        )

    def _emit(
        self,
        event_type: str,
        team: str,
        confidence: float,
        ts_ms: int,
        text: str,
        x: float | None = None,
        y: float | None = None,
    ) -> GameEvent | None:
        key = (event_type, team)
        cool_down = int(self._cooldown_ms.get(event_type, 1000))
        last_ts = self._last_event_at.get(key)
        if last_ts is not None and ts_ms - last_ts < cool_down:
            return None

        evt = GameEvent(
            event_id=f"evt_{self._next_event_id}",
            type=event_type,
            team=team,
            confidence=float(np.clip(confidence, 0.05, 0.99)),
            ts_ms=int(ts_ms),
            text=text,
            x=x,
            y=y,
        )
        self._next_event_id += 1
        self._last_event_at[key] = ts_ms
        self._recent_events.append(evt)
        return evt

    @staticmethod
    def _render_text(event_type: str, team: str) -> str:
        subject = TEAM_LABEL.get(team, "队伍")
        templates = {
            "possession_start": f"{subject}拿到球权！",
            "pass_complete": f"{subject}完成一次漂亮传导。",
            "steal": f"{subject}完成抢断反击！",
            "shot_attempt": f"{subject}发动了一脚射门尝试！",
            "danger_moment": f"{subject}制造了高压威胁！",
        }
        return templates.get(event_type, f"{subject}触发了事件：{event_type}")
