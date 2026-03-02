from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .events import GameEvent


@dataclass(frozen=True, slots=True)
class SkillConfig:
    name: str
    cooldown_ms: int
    energy_cost: int
    event_gate: tuple[str, ...]
    score_multiplier: float
    fx_style: str
    window_before_ms: int = 0
    window_after_ms: int = 1600


@dataclass(slots=True)
class SkillAction:
    action_id: str
    user_id: str
    team: str
    skill: str
    ts_ms: int
    client_event_id: str | None = None


@dataclass(slots=True)
class SkillResolution:
    action_id: str
    user_id: str
    team: str
    skill: str
    quality: str
    success: bool
    ts_ms: int
    score_delta: int
    fx_style: str
    message: str
    event_id: str | None = None
    event_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.action_id,
            "user_id": self.user_id,
            "team": self.team,
            "skill": self.skill,
            "quality": self.quality,
            "success": self.success,
            "ts_ms": self.ts_ms,
            "score_delta": self.score_delta,
            "fx_style": self.fx_style,
            "message": self.message,
            "event_id": self.event_id,
            "event_type": self.event_type,
        }


SKILL_CONFIGS: dict[str, SkillConfig] = {
    "BoostPass": SkillConfig(
        name="BoostPass",
        cooldown_ms=3800,
        energy_cost=16,
        event_gate=("pass_complete",),
        score_multiplier=1.25,
        fx_style="trail",
        window_after_ms=1800,
    ),
    "StealAura": SkillConfig(
        name="StealAura",
        cooldown_ms=5200,
        energy_cost=20,
        event_gate=("steal",),
        score_multiplier=1.45,
        fx_style="shock",
        window_after_ms=2000,
    ),
    "ShotBuff": SkillConfig(
        name="ShotBuff",
        cooldown_ms=6200,
        energy_cost=24,
        event_gate=("shot_attempt",),
        score_multiplier=1.65,
        fx_style="flare",
        window_before_ms=1200,
        window_after_ms=1500,
    ),
    "GoalGuard": SkillConfig(
        name="GoalGuard",
        cooldown_ms=7200,
        energy_cost=22,
        event_gate=("danger_moment", "shot_attempt"),
        score_multiplier=1.4,
        fx_style="shield",
        window_before_ms=800,
        window_after_ms=1500,
    ),
    "ComboLock": SkillConfig(
        name="ComboLock",
        cooldown_ms=7800,
        energy_cost=26,
        event_gate=("pass_complete", "steal", "danger_moment"),
        score_multiplier=1.85,
        fx_style="pulse",
        window_after_ms=1700,
    ),
    "HypeRoar": SkillConfig(
        name="HypeRoar",
        cooldown_ms=9800,
        energy_cost=28,
        event_gate=("danger_moment", "shot_attempt"),
        score_multiplier=2.1,
        fx_style="roar",
        window_before_ms=1000,
        window_after_ms=1800,
    ),
}
SKILL_CONFIGS_BY_LOWER: dict[str, SkillConfig] = {
    name.lower(): config for name, config in SKILL_CONFIGS.items()
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_skill_config(skill_name: str) -> SkillConfig | None:
    if skill_name in SKILL_CONFIGS:
        return SKILL_CONFIGS[skill_name]
    normalized = str(skill_name).strip().lower()
    if not normalized:
        return None
    return SKILL_CONFIGS_BY_LOWER.get(normalized)


def list_skills() -> list[dict[str, Any]]:
    return [
        {
            "name": cfg.name,
            "cooldown_ms": cfg.cooldown_ms,
            "energy_cost": cfg.energy_cost,
            "event_gate": list(cfg.event_gate),
            "score_multiplier": cfg.score_multiplier,
            "fx_style": cfg.fx_style,
            "window_before_ms": cfg.window_before_ms,
            "window_after_ms": cfg.window_after_ms,
        }
        for cfg in SKILL_CONFIGS.values()
    ]


def resolve_action_quality(
    action: SkillAction,
    config: SkillConfig,
    recent_events: list[GameEvent],
) -> tuple[str, GameEvent | None, int | None]:
    best_event: GameEvent | None = None
    best_cost = float("inf")
    best_delta: int | None = None

    for event in recent_events:
        if event.team != action.team:
            continue
        if event.type not in config.event_gate:
            continue

        delta = action.ts_ms - event.ts_ms
        if delta < -config.window_before_ms or delta > config.window_after_ms:
            continue

        cost = abs(delta)
        if cost < best_cost:
            best_cost = cost
            best_event = event
            best_delta = delta

    if best_event is None:
        return ("MISS", None, None)

    confidence = float(_clamp(float(best_event.confidence), 0.0, 1.0))
    max_window = max(config.window_before_ms, config.window_after_ms, 1)
    distance_score = 1.0 - min(1.0, abs(float(best_delta or 0)) / float(max_window))

    if confidence >= 0.65 and distance_score >= 0.62:
        return ("PERFECT", best_event, best_delta)
    if confidence >= 0.35:
        return ("GOOD", best_event, best_delta)
    return ("MISS", best_event, best_delta)


def quality_multiplier(quality: str) -> float:
    if quality == "PERFECT":
        return 1.25
    if quality == "GOOD":
        return 1.0
    return 0.0
