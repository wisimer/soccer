from __future__ import annotations

from typing import Any

from .events import TEAM_LABEL


def build_round_highlight(
    round_id: int,
    winner_team: str,
    score_a: int,
    score_b: int,
    mvp_team: str,
    ts_ms: int,
) -> dict[str, Any]:
    winner_name = TEAM_LABEL.get(winner_team, "平局")
    mvp_name = TEAM_LABEL.get(mvp_team, winner_name)

    if score_a == score_b:
        title = "势均力敌！"
        subtitle = f"第 {round_id} 回合打成平局 {score_a}:{score_b}"
    else:
        title = f"{winner_name}拿下回合！"
        subtitle = f"第 {round_id} 回合比分 {score_a}:{score_b}"

    share_text = f"{title} {subtitle} MVP：{mvp_name}"
    return {
        "round_id": int(round_id),
        "winner_team": winner_team,
        "mvp_team": mvp_team,
        "title": title,
        "subtitle": subtitle,
        "share_text": share_text,
        "theme": "panda" if mvp_team == "A" else "monkey",
        "generated_at_ms": int(ts_ms),
    }
