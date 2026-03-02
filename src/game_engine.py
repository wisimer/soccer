from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .events import EventEngine, GameEvent, TEAM_LABEL
from .highlights import build_round_highlight
from .skills import (
    SkillAction,
    SkillResolution,
    get_skill_config,
    list_skills,
    quality_multiplier,
    resolve_action_quality,
)


@dataclass(slots=True)
class PlayerState:
    user_id: str
    team: str
    energy: int
    last_action_ts_ms: int = 0
    cooldowns: dict[str, int] = field(default_factory=dict)
    consumed_event_ids: set[str] = field(default_factory=set)


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, value)))


class GameEngine:
    """Round state machine + skill actions + score system."""

    def __init__(
        self,
        lobby_ms: int = 10000,
        live_ms: int = 90000,
        result_ms: int = 15000,
        combo_window_ms: int = 8000,
        per_round_energy: int = 100,
        action_rate_limit_ms: int = 300,
    ) -> None:
        self.lobby_ms = int(lobby_ms)
        self.live_ms = int(live_ms)
        self.result_ms = int(result_ms)
        self.combo_window_ms = int(combo_window_ms)
        self.per_round_energy = int(per_round_energy)
        self.action_rate_limit_ms = int(action_rate_limit_ms)

        now = self._now_ms()
        self._lock = RLock()
        self._event_engine = EventEngine()
        self._round_id = 1
        self._phase = "LOBBY"
        self._phase_started_ms = now
        self._phase_duration_ms = self.lobby_ms
        self._next_action_id = 1

        self._players: dict[str, PlayerState] = {}
        self._score = {"A": 0, "B": 0}
        self._combo_count = {"A": 0, "B": 0}
        self._combo_mul = {"A": 1.0, "B": 1.0}
        self._last_combo_ts_ms: dict[str, int | None] = {"A": None, "B": None}
        self._mood = {"A": "calm", "B": "calm"}
        self._mascot_energy = {"A": 50, "B": 50}
        self._continuity_health = {
            "gap_ms": 0,
            "predicted_ratio": 0.0,
            "id_switch_rate": 0.0,
            "health": "good",
        }

        self._recent_events: deque[dict[str, Any]] = deque(maxlen=64)
        self._recent_skills: deque[dict[str, Any]] = deque(maxlen=64)
        self._latest_result: dict[str, Any] | None = None
        self._latest_highlight: dict[str, Any] | None = None

        self._event_points = {
            "possession_start": 1,
            "pass_complete": 1,
            "steal": 2,
            "shot_attempt": 3,
            "danger_moment": 5,
        }
        self._skill_base_points = {
            "BoostPass": 2,
            "StealAura": 3,
            "ShotBuff": 4,
            "GoalGuard": 3,
            "ComboLock": 4,
            "HypeRoar": 5,
        }

    def update_from_frame(
        self,
        entities: list[dict[str, Any]],
        ts_ms: int,
        continuity_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = int(ts_ms)
            self._tick(now)

            if continuity_health is not None:
                self._continuity_health = {
                    "gap_ms": int(continuity_health.get("gap_ms", 0)),
                    "predicted_ratio": round(float(continuity_health.get("predicted_ratio", 0.0)), 3),
                    "id_switch_rate": round(float(continuity_health.get("id_switch_rate", 0.0)), 3),
                    "health": str(continuity_health.get("health", "good")),
                }

            new_events = self._event_engine.update(entities, now)
            is_live = self._phase == "LIVE"
            for event in new_events:
                score_delta = self._apply_event_score(event) if is_live else 0
                event_item = event.to_dict()
                event_item["score_delta"] = score_delta
                self._recent_events.append(event_item)

            self._update_mascot_state()
            payload = self._snapshot(now)
            if self._phase == "RESULT" and self._latest_result is not None:
                payload["game_result"] = self._latest_result
                payload["game_highlight"] = self._latest_highlight
            else:
                payload["game_result"] = None
                payload["game_highlight"] = None
            return payload

    def join(self, user_id: str, team: str, ts_ms: int | None = None) -> dict[str, Any]:
        if team not in ("A", "B"):
            raise ValueError("team must be 'A' or 'B'")

        now = int(ts_ms if ts_ms is not None else self._now_ms())
        with self._lock:
            player = self._players.get(user_id)
            if player is None:
                player = PlayerState(user_id=user_id, team=team, energy=self.per_round_energy)
                self._players[user_id] = player
            else:
                player.team = team
                if self._phase != "LIVE":
                    player.energy = self.per_round_energy

            return {
                "ok": True,
                "user_id": user_id,
                "team": team,
                "round_id": self._round_id,
                "phase": self._phase,
                "joined_at_ms": now,
            }

    def submit_action(
        self,
        user_id: str,
        team: str,
        skill: str,
        ts_ms: int | None = None,
        client_event_id: str | None = None,
    ) -> dict[str, Any]:
        now = int(ts_ms if ts_ms is not None else self._now_ms())
        with self._lock:
            self._tick(now)

            player = self._players.get(user_id)
            if player is None:
                return self._action_fail(user_id, team, skill, now, "MISS", "请先加入阵营。")
            if player.team != team:
                return self._action_fail(user_id, team, skill, now, "MISS", "阵营不匹配。")

            config = get_skill_config(skill)
            if config is None:
                return self._action_fail(user_id, team, skill, now, "MISS", "未知技能。")

            if now - player.last_action_ts_ms < self.action_rate_limit_ms:
                return self._action_fail(user_id, team, config.name, now, "MISS", "操作太快，请稍后。")
            if self._phase != "LIVE":
                return self._action_fail(user_id, team, config.name, now, "MISS", "当前非对局阶段。")
            if player.energy < config.energy_cost:
                return self._action_fail(user_id, team, config.name, now, "MISS", "能量不足。")

            cool_until = int(player.cooldowns.get(config.name, 0))
            if now < cool_until:
                return self._action_fail(user_id, team, config.name, now, "MISS", "技能冷却中。")

            action_id = f"act_{self._next_action_id}"
            self._next_action_id += 1
            action = SkillAction(
                action_id=action_id,
                user_id=user_id,
                team=team,
                skill=config.name,
                ts_ms=now,
                client_event_id=client_event_id,
            )

            quality, matched_event, _ = resolve_action_quality(
                action=action,
                config=config,
                recent_events=self._event_engine.get_recent_events(),
            )

            player.last_action_ts_ms = now
            player.cooldowns[config.name] = now + config.cooldown_ms
            player.energy = max(0, player.energy - config.energy_cost)

            success = bool(matched_event is not None and quality in ("PERFECT", "GOOD"))
            score_delta = 0
            event_id: str | None = None
            event_type: str | None = None

            if success and matched_event is not None:
                if matched_event.event_id in player.consumed_event_ids:
                    success = False
                    quality = "MISS"
                else:
                    player.consumed_event_ids.add(matched_event.event_id)
                    base = float(self._skill_base_points.get(config.name, 2))
                    score_delta = int(round(base * config.score_multiplier * quality_multiplier(quality)))
                    self._score[team] += max(0, score_delta)
                    self._boost_combo(team, now)
                    event_id = matched_event.event_id
                    event_type = matched_event.type

            resolution = SkillResolution(
                action_id=action_id,
                user_id=user_id,
                team=team,
                skill=config.name,
                quality=quality,
                success=success,
                ts_ms=now,
                score_delta=score_delta,
                fx_style=config.fx_style,
                message=self._resolution_message(success=success, quality=quality),
                event_id=event_id,
                event_type=event_type,
            )
            self._recent_skills.append(resolution.to_dict())
            self._update_mascot_state()
            return {
                "ok": True,
                "resolution": resolution.to_dict(),
                "player": self._player_view(player, now),
                "game_score": self._score_view(),
            }

    def get_state(self, user_id: str | None = None, ts_ms: int | None = None) -> dict[str, Any]:
        now = int(ts_ms if ts_ms is not None else self._now_ms())
        with self._lock:
            self._tick(now)
            snapshot = self._snapshot(now)
            snapshot["skills_catalog"] = list_skills()
            if user_id and user_id in self._players:
                snapshot["player"] = self._player_view(self._players[user_id], now)
            else:
                snapshot["player"] = None
            return snapshot

    def get_latest_result(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_result

    def get_latest_highlight(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_highlight

    def _snapshot(self, now: int) -> dict[str, Any]:
        elapsed = max(0, now - self._phase_started_ms)
        remain_ms = max(0, self._phase_duration_ms - elapsed)
        return {
            "game_round": {
                "id": self._round_id,
                "phase": self._phase,
                "remain_ms": remain_ms,
                "started_ms": self._phase_started_ms,
                "duration_ms": self._phase_duration_ms,
            },
            "game_score": self._score_view(),
            "game_events": list(self._recent_events)[-8:],
            "game_skills": list(self._recent_skills)[-8:],
            "game_mascot": {
                "team_a": {"mood": self._mood["A"], "energy": self._mascot_energy["A"]},
                "team_b": {"mood": self._mood["B"], "energy": self._mascot_energy["B"]},
            },
            "continuity_health": self._continuity_health,
        }

    def _score_view(self) -> dict[str, Any]:
        return {
            "team_a": int(self._score["A"]),
            "team_b": int(self._score["B"]),
            "combo_a": round(float(self._combo_mul["A"]), 2),
            "combo_b": round(float(self._combo_mul["B"]), 2),
        }

    def _player_view(self, player: PlayerState, now: int) -> dict[str, Any]:
        return {
            "user_id": player.user_id,
            "team": player.team,
            "energy": int(player.energy),
            "cooldowns": {
                name: max(0, int(until - now))
                for name, until in player.cooldowns.items()
                if until > now
            },
        }

    def _tick(self, now: int) -> None:
        elapsed = now - self._phase_started_ms
        if elapsed < self._phase_duration_ms:
            return

        if self._phase == "LOBBY":
            self._start_live(now)
        elif self._phase == "LIVE":
            self._finish_round(now)
        elif self._phase == "RESULT":
            self._start_next_round(now)

    def _start_live(self, now: int) -> None:
        self._phase = "LIVE"
        self._phase_started_ms = now
        self._phase_duration_ms = self.live_ms
        self._reset_round_runtime(clear_results=True)

    def _finish_round(self, now: int) -> None:
        score_a = int(self._score["A"])
        score_b = int(self._score["B"])
        if score_a > score_b:
            winner = "A"
        elif score_b > score_a:
            winner = "B"
        else:
            winner = "DRAW"

        mvp_team = "A" if score_a >= score_b else "B"
        self._latest_result = {
            "round_id": self._round_id,
            "winner": winner,
            "score": {"A": score_a, "B": score_b},
            "mvp_team": mvp_team,
            "finished_at_ms": now,
            "winner_name": "平局" if winner == "DRAW" else TEAM_LABEL.get(winner, "队伍"),
        }
        self._latest_highlight = build_round_highlight(
            round_id=self._round_id,
            winner_team=winner,
            score_a=score_a,
            score_b=score_b,
            mvp_team=mvp_team,
            ts_ms=now,
        )

        self._phase = "RESULT"
        self._phase_started_ms = now
        self._phase_duration_ms = self.result_ms

    def _start_next_round(self, now: int) -> None:
        self._round_id += 1
        self._phase = "LOBBY"
        self._phase_started_ms = now
        self._phase_duration_ms = self.lobby_ms
        self._reset_round_runtime(clear_results=False)

    def _reset_round_runtime(self, clear_results: bool) -> None:
        self._score = {"A": 0, "B": 0}
        self._combo_count = {"A": 0, "B": 0}
        self._combo_mul = {"A": 1.0, "B": 1.0}
        self._last_combo_ts_ms = {"A": None, "B": None}
        self._recent_events.clear()
        self._recent_skills.clear()
        if clear_results:
            self._latest_result = None
            self._latest_highlight = None
        for player in self._players.values():
            player.energy = self.per_round_energy
            player.cooldowns.clear()
            player.consumed_event_ids.clear()

    def _apply_event_score(self, event: GameEvent) -> int:
        team = event.team
        if team not in ("A", "B"):
            return 0
        base = int(self._event_points.get(event.type, 1))
        self._boost_combo(team, event.ts_ms)
        score_delta = int(round(base * self._combo_mul[team]))
        self._score[team] += score_delta
        return score_delta

    def _boost_combo(self, team: str, ts_ms: int) -> None:
        last_ts = self._last_combo_ts_ms.get(team)
        if last_ts is None or ts_ms - last_ts > self.combo_window_ms:
            self._combo_count[team] = 1
        else:
            self._combo_count[team] += 1
        self._last_combo_ts_ms[team] = ts_ms
        self._combo_mul[team] = min(3.0, 1.0 + 0.25 * max(0, self._combo_count[team] - 1))

    def _update_mascot_state(self) -> None:
        diff = self._score["A"] - self._score["B"]
        self._mood["A"] = self._calc_mood(diff)
        self._mood["B"] = self._calc_mood(-diff)

        self._mascot_energy["A"] = _clip_int(45 + self._combo_count["A"] * 8 + max(0, diff) * 0.5, 0, 100)
        self._mascot_energy["B"] = _clip_int(
            45 + self._combo_count["B"] * 8 + max(0, -diff) * 0.5,
            0,
            100,
        )

    @staticmethod
    def _calc_mood(lead: int) -> str:
        if lead >= 20:
            return "hype"
        if lead >= 8:
            return "happy"
        if lead <= -20:
            return "tilt"
        if lead <= -8:
            return "tense"
        return "focus"

    def _action_fail(
        self,
        user_id: str,
        team: str,
        skill: str,
        ts_ms: int,
        quality: str,
        message: str,
    ) -> dict[str, Any]:
        resolution = SkillResolution(
            action_id=f"act_{self._next_action_id}",
            user_id=user_id,
            team=team,
            skill=skill,
            quality=quality,
            success=False,
            ts_ms=ts_ms,
            score_delta=0,
            fx_style="none",
            message=message,
        )
        self._next_action_id += 1
        return {"ok": False, "resolution": resolution.to_dict()}

    @staticmethod
    def _resolution_message(success: bool, quality: str) -> str:
        if not success:
            return "未命中事件窗口。"
        if quality == "PERFECT":
            return "完美命中！"
        return "技能命中！"

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
