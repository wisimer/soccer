const pitchLength = 105;
const pitchWidth = 68;

const videoEl = document.getElementById("sourceVideo");
const overlayCanvas = document.getElementById("overlay");
const overlayCtx = overlayCanvas.getContext("2d");

const pitchCanvas = document.getElementById("pitch");
const pitchCtx = pitchCanvas.getContext("2d");
const pitchViewportEl = document.getElementById("pitchViewport");

const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const videoHintEl = document.getElementById("videoHint");

const entities = new Map();
let latestSeq = 0;
let latestLatency = 0;
let latestBackend = "-";
let latestProcess = "-";
let latestDecode = "-";
let latestDetections = [];
let latestTracks = [];
let latestFrame = { width: 0, height: 0, source_ts_ms: null, capture_ts_ms: null };
let lastVideoSyncAt = 0;
let lastHardSeekAt = 0;
let videoRateUntil = 0;
let lastOverlayRenderAt = 0;
let lastPitchRenderAt = 0;

const isMobileDevice =
  window.matchMedia("(max-width: 900px)").matches ||
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
const videoSyncConfig = {
  minSyncIntervalMs: isMobileDevice ? 620 : 320,
  softDriftSec: isMobileDevice ? 0.55 : 0.35,
  hardSeekDriftSec: isMobileDevice ? 2.2 : 1.2,
  hardSeekCooldownMs: isMobileDevice ? 3000 : 1800,
  rateGain: isMobileDevice ? 0.09 : 0.12,
  maxRateDelta: isMobileDevice ? 0.06 : 0.1,
  minRate: isMobileDevice ? 0.94 : 0.9,
  maxRate: isMobileDevice ? 1.06 : 1.1,
  rateHoldMs: isMobileDevice ? 900 : 600,
};

const defaultPitchCamera = Object.freeze({
  yaw: -34,
  pitch: 34,
  distance: 120,
  targetX: pitchLength * 0.5,
  targetY: 0,
  targetZ: pitchWidth * 0.5,
});
const pitchCamera = { ...defaultPitchCamera };

let activePitchPointer = null;
const activeTouchPoints = new Map();
let pinchSnapshot = null;

const teamNameMap = { A: "萌熊", B: "猴子" };
const skillLabelMap = {
  BoostPass: "传导强化",
  StealAura: "抢断气场",
  ShotBuff: "射门增幅",
  GoalGuard: "守门怒吼",
  ComboLock: "连击锁定",
  HypeRoar: "全员狂热",
};
const moodLabelMap = {
  calm: "calm",
  focus: "focus",
  happy: "happy",
  hype: "hype",
  tense: "tense",
  tilt: "tilt",
};
const actionLabelMap = {
  attack: "闪电进攻",
  defend: "铜墙防守",
  hype: "狂热连击",
};
const actionPlanMap = {
  attack: ["ShotBuff", "BoostPass"],
  defend: ["GoalGuard", "StealAura"],
  hype: ["ComboLock", "HypeRoar"],
};
const actionMetaMap = {
  attack: "自动择优：射门/传导",
  defend: "自动择优：守门/抢断",
  hype: "自动择优：连击/狂热",
};
const eventHintMap = {
  possession_start: "attack",
  pass_complete: "attack",
  steal: "defend",
  shot_attempt: "attack",
  danger_moment: "hype",
};
const skillToActionMap = Object.fromEntries(
  Object.entries(actionPlanMap).flatMap(([action, skills]) => skills.map((skill) => [skill, action]))
);
const actionHintTTLms = 2300;

const joinAEl = document.getElementById("joinA");
const joinBEl = document.getElementById("joinB");
const playerBadgeEl = document.getElementById("playerBadge");
const scoreAEl = document.getElementById("scoreA");
const scoreBEl = document.getElementById("scoreB");
const comboAEl = document.getElementById("comboA");
const comboBEl = document.getElementById("comboB");
const roundPhaseEl = document.getElementById("roundPhase");
const roundRemainEl = document.getElementById("roundRemain");
const mascotMoodAEl = document.getElementById("mascotMoodA");
const mascotMoodBEl = document.getElementById("mascotMoodB");
const mascotEnergyAEl = document.getElementById("mascotEnergyA");
const mascotEnergyBEl = document.getElementById("mascotEnergyB");
const energyTextEl = document.getElementById("energyText");
const windowHintEl = document.getElementById("windowHint");
const streakTextEl = document.getElementById("streakText");
const eventFeedEl = document.getElementById("eventFeed");
const resultModalEl = document.getElementById("resultModal");
const resultTitleEl = document.getElementById("resultTitle");
const resultSubtitleEl = document.getElementById("resultSubtitle");
const resultScoreEl = document.getElementById("resultScore");
const resultMvpEl = document.getElementById("resultMvp");
const resultCloseEl = document.getElementById("resultClose");
const actionButtons = Array.from(document.querySelectorAll(".action-card"));
const actionViews = actionButtons.map((button) => ({
  button,
  action: button.dataset.action || "",
  costEl: button.querySelector(".action-cost"),
  metaEl: button.querySelector(".action-meta"),
  skillEl: button.querySelector(".action-skill"),
  cooldownEl: button.querySelector(".action-cooldown"),
}));

const defaultSkillCatalog = [
  { name: "BoostPass", cooldown_ms: 3800, energy_cost: 16 },
  { name: "StealAura", cooldown_ms: 5200, energy_cost: 20 },
  { name: "ShotBuff", cooldown_ms: 6200, energy_cost: 24 },
  { name: "GoalGuard", cooldown_ms: 7200, energy_cost: 22 },
  { name: "ComboLock", cooldown_ms: 7800, energy_cost: 26 },
  { name: "HypeRoar", cooldown_ms: 9800, energy_cost: 28 },
];
const skillCatalog = new Map(defaultSkillCatalog.map((item) => [item.name, item]));

const seenEventIds = new Set();
const seenSkillIds = new Set();
const eventTimeline = [];
const overlayFrameBuffer = [];
const maxOverlayFrameBuffer = isMobileDevice ? 320 : 260;
let overlaySyncDeltaMs = 0;
const maxFutureOverlayLeadMs = isMobileDevice ? 80 : 120;

const localUserId = (() => {
  const key = "soccer_game_user_id";
  try {
    const current = window.localStorage.getItem(key);
    if (current) {
      return current;
    }
    const created = `u_${Math.random().toString(36).slice(2, 12)}`;
    window.localStorage.setItem(key, created);
    return created;
  } catch (_error) {
    return `u_${Math.random().toString(36).slice(2, 12)}`;
  }
})();

let joinedTeam = null;
let pendingAction = false;
let localHitStreak = 0;
let lastHintEventType = "";
let lastHintEventTsMs = 0;
let shownResultRoundId = 0;
const gameView = {
  round: { id: 0, phase: "LOBBY", remain_ms: 0, started_ms: 0, duration_ms: 0 },
  score: { team_a: 0, team_b: 0, combo_a: 1, combo_b: 1 },
  mascot: {
    team_a: { mood: "focus", energy: 50 },
    team_b: { mood: "focus", energy: 50 },
  },
  continuity: { gap_ms: 0, predicted_ratio: 0, id_switch_rate: 0, health: "good" },
  player: null,
  roundSyncPerf: performance.now(),
};
document.body.dataset.streakLevel = "0";

function wsUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws`;
}

function trimSet(setObj, maxSize) {
  while (setObj.size > maxSize) {
    const first = setObj.values().next().value;
    if (first == null) {
      return;
    }
    setObj.delete(first);
  }
}

function resolveFrameSourceTsMs(payload) {
  const sourceTs = payload && payload.frame ? Number(payload.frame.source_ts_ms) : NaN;
  return Number.isFinite(sourceTs) && sourceTs > 0 ? sourceTs : null;
}

function pushOverlaySnapshot(payload) {
  const sourceTsMs = resolveFrameSourceTsMs(payload);
  overlayFrameBuffer.push({
    sourceTsMs,
    frame: payload.frame || null,
    detections: Array.isArray(payload.detections) ? payload.detections : [],
    tracks: Array.isArray(payload.tracks) ? payload.tracks : [],
  });
  if (overlayFrameBuffer.length > maxOverlayFrameBuffer) {
    overlayFrameBuffer.splice(0, overlayFrameBuffer.length - maxOverlayFrameBuffer);
  }
}

function pickOverlaySnapshotForVideoTime() {
  if (overlayFrameBuffer.length === 0) {
    overlaySyncDeltaMs = 0;
    return null;
  }
  if (!videoEl || !Number.isFinite(videoEl.currentTime) || videoEl.currentTime <= 0) {
    overlaySyncDeltaMs = 0;
    return overlayFrameBuffer[overlayFrameBuffer.length - 1];
  }

  const targetMs = videoEl.currentTime * 1000;
  let best = null;
  let bestCost = Number.POSITIVE_INFINITY;

  for (let i = overlayFrameBuffer.length - 1; i >= 0; i -= 1) {
    const item = overlayFrameBuffer[i];
    if (!Number.isFinite(item.sourceTsMs)) {
      continue;
    }
    const delta = item.sourceTsMs - targetMs;
    if (delta > maxFutureOverlayLeadMs) {
      continue;
    }
    const cost = Math.abs(delta) + (delta > 0 ? 240 : 0);
    if (cost < bestCost) {
      best = item;
      bestCost = cost;
      overlaySyncDeltaMs = Math.round(delta);
    }
    if (delta < -2200 && best !== null) {
      break;
    }
  }

  if (best === null) {
    overlaySyncDeltaMs = 0;
    return overlayFrameBuffer[overlayFrameBuffer.length - 1];
  }
  return best;
}

function pushEventLine(text, kind = "event") {
  eventTimeline.unshift({ text, kind });
  while (eventTimeline.length > 16) {
    eventTimeline.pop();
  }
  renderEventFeed();
}

function renderEventFeed() {
  if (!eventFeedEl) {
    return;
  }
  if (eventTimeline.length === 0) {
    eventFeedEl.innerHTML = '<div class="event-item">等待实时事件...</div>';
    return;
  }
  eventFeedEl.innerHTML = eventTimeline
    .map((item) => `<div class="event-item ${item.kind}">${item.text}</div>`)
    .join("");
}

function updateEntitiesFromPayload(payloadEntities, nowPerf) {
  const seen = new Set();
  for (const e of payloadEntities) {
    seen.add(e.id);
    const current = entities.get(e.id);
    if (!current) {
      entities.set(e.id, {
        id: e.id,
        type: e.type,
        team: e.team,
        x: e.x,
        y: e.y,
        targetX: e.x,
        targetY: e.y,
        updatedAt: nowPerf,
      });
      continue;
    }
    current.type = e.type;
    current.team = e.team;
    current.targetX = e.x;
    current.targetY = e.y;
    current.updatedAt = nowPerf;
  }
  return seen;
}

function pruneStaleEntities(seen, nowPerf, ttlMs = 1000) {
  for (const id of entities.keys()) {
    if (seen.has(id)) {
      continue;
    }
    const item = entities.get(id);
    if (item && nowPerf - item.updatedAt > ttlMs) {
      entities.delete(id);
    }
  }
}

function refreshMetaBar(continuity) {
  metaEl.innerHTML = `
      <span>seq: ${latestSeq}</span>
      <span>entities: ${entities.size}</span>
      <span>tracks: ${latestTracks.length}</span>
      <span>detections: ${latestDetections.length}</span>
      <span>latency: ${latestLatency} ms</span>
      <span>backend: ${latestBackend}</span>
      <span>process: ${latestProcess} ms</span>
      <span>decode: ${latestDecode}</span>
      <span>overlay_sync: ${overlaySyncDeltaMs} ms</span>
      <span>continuity: ${continuity.health} gap=${continuity.gap_ms}ms</span>
    `;
}

function applySkillsCatalog(items) {
  if (!Array.isArray(items)) {
    return;
  }
  for (const item of items) {
    if (!item || !item.name) {
      continue;
    }
    skillCatalog.set(item.name, {
      name: item.name,
      cooldown_ms: Number(item.cooldown_ms || 0),
      energy_cost: Number(item.energy_cost || 0),
      event_gate: Array.isArray(item.event_gate) ? item.event_gate : [],
      score_multiplier: Number(item.score_multiplier || 1),
    });
  }
}

function phaseLabel(phase) {
  if (phase === "LIVE") {
    return "LIVE";
  }
  if (phase === "RESULT") {
    return "RESULT";
  }
  return "LOBBY";
}

function formatRemainMs(ms) {
  return `${(Math.max(0, ms) / 1000).toFixed(1)}s`;
}

function apiJsonOptions(payload) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch (_error) {
    return null;
  }
}

function skillDisplayName(skill) {
  return skillLabelMap[skill] || skill || "未知技能";
}

function resolveActionChoice(action, player) {
  const fallback = {
    action,
    skill: null,
    cooldownMs: 0,
    energyCost: 0,
    display: "无可用技能",
  };
  const candidates = actionPlanMap[action];
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return fallback;
  }

  const cooldowns = (player && player.cooldowns) || {};
  const energy = Number((player && player.energy) || 0);
  const details = candidates
    .map((skill) => {
      const cfg = skillCatalog.get(skill);
      if (!cfg) {
        return null;
      }
      const energyCost = Number(cfg.energy_cost || 0);
      const cooldownMs = Math.max(0, Number(cooldowns[skill] || 0));
      return {
        action,
        skill,
        cooldownMs,
        energyCost,
        display: skillDisplayName(skill),
        ready: cooldownMs <= 0 && energy >= energyCost,
        energyGap: Math.max(0, energyCost - energy),
      };
    })
    .filter(Boolean);

  if (details.length === 0) {
    return fallback;
  }

  const ready = details.find((item) => item.ready);
  if (ready) {
    return ready;
  }

  details.sort((a, b) => {
    if (a.energyGap !== b.energyGap) {
      return a.energyGap - b.energyGap;
    }
    if (a.cooldownMs !== b.cooldownMs) {
      return a.cooldownMs - b.cooldownMs;
    }
    return a.energyCost - b.energyCost;
  });
  return details[0];
}

function deriveHintAction() {
  if (!lastHintEventType) {
    return null;
  }
  if (Date.now() - lastHintEventTsMs > actionHintTTLms) {
    return null;
  }
  return eventHintMap[lastHintEventType] || null;
}

function updateLocalHitStreak(quality) {
  if (quality === "PERFECT" || quality === "GOOD") {
    localHitStreak += 1;
  } else {
    localHitStreak = 0;
  }
  if (localHitStreak >= 6) {
    document.body.dataset.streakLevel = "3";
  } else if (localHitStreak >= 3) {
    document.body.dataset.streakLevel = "2";
  } else if (localHitStreak >= 1) {
    document.body.dataset.streakLevel = "1";
  } else {
    document.body.dataset.streakLevel = "0";
  }
}

async function loadVideoPreview() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`health status ${res.status}`);
    }
    const health = await res.json();
    const previewUrl = String(health.video_preview_url || "");
    if (previewUrl) {
      videoEl.src = previewUrl;
      videoEl.playbackRate = 1.0;
      videoHintEl.style.display = "none";
      videoEl.play().catch(() => {});
      return;
    }
  } catch (error) {
    console.warn("failed to load /api/health", error);
  }

  videoHintEl.style.display = "flex";
  videoHintEl.textContent = "当前数据源不是本地视频文件，页面仅显示映射与跟踪数据。";
}

function ingestGameEvents(items) {
  if (!Array.isArray(items)) {
    return;
  }
  for (const event of items) {
    const eventId = String(event.id || `${event.type}_${event.ts_ms}`);
    if (seenEventIds.has(eventId)) {
      continue;
    }
    seenEventIds.add(eventId);
    trimSet(seenEventIds, 160);
    const eventType = String(event.type || "");
    if (eventHintMap[eventType]) {
      lastHintEventType = eventType;
      lastHintEventTsMs = Number(event.ts_ms || Date.now());
    }
    const team = teamNameMap[event.team] || "队伍";
    const text = event.text || `${team} 触发 ${event.type}`;
    pushEventLine(text, "event");
  }
}

function ingestSkillEvents(items) {
  if (!Array.isArray(items)) {
    return;
  }
  for (const skill of items) {
    const skillId = String(skill.id || `${skill.skill}_${skill.ts_ms}`);
    if (seenSkillIds.has(skillId)) {
      continue;
    }
    seenSkillIds.add(skillId);
    trimSet(seenSkillIds, 160);
    const team = teamNameMap[skill.team] || "队伍";
    const quality = String(skill.quality || "MISS").toUpperCase();
    const type = quality === "MISS" ? "warn" : "skill";
    const action = skillToActionMap[skill.skill];
    const actionLabel = action ? actionLabelMap[action] : skillDisplayName(skill.skill);
    const scoreDelta = Number(skill.score_delta || 0);
    const suffix = scoreDelta > 0 ? ` +${scoreDelta}` : "";
    const text = `${team} ${actionLabel} ${quality}${suffix}`;
    pushEventLine(text, type);
  }
}

function showResultModal(result, highlight) {
  if (!result || !resultModalEl) {
    return;
  }
  const roundId = Number(result.round_id || 0);
  if (roundId <= 0 || roundId === shownResultRoundId) {
    return;
  }
  shownResultRoundId = roundId;

  const scoreA = result.score && Number.isFinite(result.score.A) ? result.score.A : 0;
  const scoreB = result.score && Number.isFinite(result.score.B) ? result.score.B : 0;
  const winnerLabel =
    result.winner === "DRAW" ? "平局" : `${teamNameMap[result.winner] || "队伍"} 胜出`;
  const mvpTeam = result.mvp_team || "A";

  resultTitleEl.textContent = highlight && highlight.title ? highlight.title : `第 ${roundId} 回合结算`;
  resultSubtitleEl.textContent =
    highlight && highlight.subtitle ? highlight.subtitle : `${winnerLabel} · 90 秒快局`;
  resultScoreEl.textContent = `${scoreA} : ${scoreB}`;
  resultMvpEl.textContent = `MVP: ${teamNameMap[mvpTeam] || "队伍"}`;
  resultModalEl.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function hideResultModal() {
  if (resultModalEl) {
    resultModalEl.classList.add("hidden");
  }
  document.body.classList.remove("modal-open");
}

function applyGameSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") {
    return;
  }
  if (snapshot.game_round) {
    gameView.round = { ...gameView.round, ...snapshot.game_round };
    gameView.roundSyncPerf = performance.now();
  }
  if (snapshot.game_score) {
    gameView.score = { ...gameView.score, ...snapshot.game_score };
  }
  if (snapshot.game_mascot) {
    gameView.mascot = {
      team_a: { ...gameView.mascot.team_a, ...(snapshot.game_mascot.team_a || {}) },
      team_b: { ...gameView.mascot.team_b, ...(snapshot.game_mascot.team_b || {}) },
    };
  }
  if (snapshot.continuity_health) {
    gameView.continuity = { ...gameView.continuity, ...snapshot.continuity_health };
  }
  if (snapshot.player !== undefined) {
    gameView.player = snapshot.player;
    joinedTeam = snapshot.player && snapshot.player.team ? snapshot.player.team : joinedTeam;
  }
  if (snapshot.skills_catalog) {
    applySkillsCatalog(snapshot.skills_catalog);
  }
  ingestGameEvents(snapshot.game_events);
  ingestSkillEvents(snapshot.game_skills);
  if (snapshot.game_result) {
    showResultModal(snapshot.game_result, snapshot.game_highlight || null);
  }
  renderGameHud();
}

function renderGameHud() {
  const round = gameView.round || {};
  const score = gameView.score || {};
  const mascot = gameView.mascot || {};
  const player = gameView.player;
  const nowPerf = performance.now();
  const elapsedSync = nowPerf - gameView.roundSyncPerf;
  const displayRemain = Math.max(0, Number(round.remain_ms || 0) - elapsedSync);
  document.body.dataset.phase = String(round.phase || "LOBBY").toLowerCase();

  scoreAEl.textContent = String(Number(score.team_a || 0));
  scoreBEl.textContent = String(Number(score.team_b || 0));
  comboAEl.textContent = `x${Number(score.combo_a || 1).toFixed(2)}`;
  comboBEl.textContent = `x${Number(score.combo_b || 1).toFixed(2)}`;
  roundPhaseEl.textContent = phaseLabel(round.phase || "LOBBY");
  roundRemainEl.textContent = formatRemainMs(displayRemain);

  mascotMoodAEl.textContent = moodLabelMap[(mascot.team_a && mascot.team_a.mood) || "focus"] || "focus";
  mascotMoodBEl.textContent = moodLabelMap[(mascot.team_b && mascot.team_b.mood) || "focus"] || "focus";
  mascotEnergyAEl.textContent = String(Number((mascot.team_a && mascot.team_a.energy) || 0));
  mascotEnergyBEl.textContent = String(Number((mascot.team_b && mascot.team_b.energy) || 0));

  if (player && player.team) {
    joinedTeam = player.team;
    playerBadgeEl.textContent = `已加入 ${teamNameMap[player.team]} · ${localUserId}`;
    energyTextEl.textContent = `能量 ${Number(player.energy || 0)} / 100`;
  } else if (joinedTeam) {
    playerBadgeEl.textContent = `已加入 ${teamNameMap[joinedTeam]} · ${localUserId}`;
    energyTextEl.textContent = "能量等待同步";
  } else {
    playerBadgeEl.textContent = "未加入阵营";
    energyTextEl.textContent = "能量 0 / 100";
  }
  document.body.dataset.team = joinedTeam ? String(joinedTeam).toLowerCase() : "none";

  joinAEl.disabled = joinedTeam === "A";
  joinBEl.disabled = joinedTeam === "B";

  const phase = round.phase || "LOBBY";
  const playerEnergy = Number((player && player.energy) || 0);
  for (const view of actionViews) {
    const { button, action, costEl, metaEl, skillEl, cooldownEl } = view;
    const choice = resolveActionChoice(action, player);
    const canPlay =
      Boolean(joinedTeam) &&
      phase === "LIVE" &&
      !pendingAction &&
      Boolean(choice.skill) &&
      playerEnergy >= Number(choice.energyCost || 0) &&
      Number(choice.cooldownMs || 0) <= 0;
    button.classList.toggle("disabled", !canPlay);
    button.classList.toggle("cooling", Number(choice.cooldownMs || 0) > 0);
    button.disabled = !canPlay;

    if (costEl) {
      costEl.textContent = String(Number(choice.energyCost || 0));
    }
    if (metaEl) {
      metaEl.textContent = actionMetaMap[action] || "自动择优";
    }
    if (skillEl) {
      skillEl.textContent = `技能: ${choice.display}`;
    }
    if (cooldownEl) {
      cooldownEl.textContent = choice.cooldownMs > 0 ? `${(choice.cooldownMs / 1000).toFixed(1)}s` : "";
    }
  }

  const hintAction = deriveHintAction();
  if (windowHintEl) {
    windowHintEl.textContent = hintAction
      ? `窗口提示：${actionLabelMap[hintAction]} 命中率更高`
      : "窗口提示：等待关键事件";
  }
  if (streakTextEl) {
    streakTextEl.textContent = `连中 ${localHitStreak}`;
  }
}

function markActionFeedback(action, quality) {
  const view = actionViews.find((item) => item.action === action);
  if (!view) {
    return;
  }
  const { button } = view;
  button.classList.remove("feedback-perfect", "feedback-good", "feedback-miss");
  if (quality === "PERFECT") {
    button.classList.add("feedback-perfect");
    document.body.classList.remove("fx-good", "fx-miss");
    document.body.classList.add("fx-perfect");
  } else if (quality === "GOOD") {
    button.classList.add("feedback-good");
    document.body.classList.remove("fx-perfect", "fx-miss");
    document.body.classList.add("fx-good");
  } else {
    button.classList.add("feedback-miss");
    document.body.classList.remove("fx-perfect", "fx-good");
    document.body.classList.add("fx-miss");
  }
  window.setTimeout(() => {
    button.classList.remove("feedback-perfect", "feedback-good", "feedback-miss");
    document.body.classList.remove("fx-perfect", "fx-good", "fx-miss");
  }, 420);
}

async function fetchGameState() {
  try {
    const res = await fetch(`/api/game/state?user_id=${encodeURIComponent(localUserId)}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      return;
    }
    const state = await res.json();
    applyGameSnapshot(state);
  } catch (error) {
    console.warn("failed to load game state", error);
  }
}

async function joinTeam(team) {
  try {
    const res = await fetch("/api/game/join", apiJsonOptions({ user_id: localUserId, team }));
    if (!res.ok) {
      return;
    }
    const payload = await res.json();
    joinedTeam = team;
    applyGameSnapshot(payload.state || {});
    pushEventLine(`你加入了${teamNameMap[team]}阵营`, "skill");
  } catch (error) {
    console.warn("join failed", error);
  }
}

async function triggerAction(action) {
  if (!joinedTeam || pendingAction) {
    return;
  }
  const choice = resolveActionChoice(action, gameView.player);
  if (!choice.skill) {
    pushEventLine("当前没有可触发技能", "warn");
    return;
  }

  pendingAction = true;
  renderGameHud();
  try {
    const res = await fetch(
      "/api/game/action",
      apiJsonOptions({
        user_id: localUserId,
        team: joinedTeam,
        skill: choice.skill,
      })
    );
    if (!res.ok) {
      pushEventLine("技能请求失败", "warn");
      return;
    }
    const payload = await res.json();
    if (payload.resolution) {
      const quality = String(payload.resolution.quality || "MISS").toUpperCase();
      markActionFeedback(action, quality);
      updateLocalHitStreak(quality);
      const qualityText = quality === "PERFECT" ? "完美命中" : quality === "GOOD" ? "命中" : "落空";
      const scoreDelta = Number(payload.resolution.score_delta || 0);
      const suffix = scoreDelta > 0 ? ` +${scoreDelta}` : "";
      const message = `${actionLabelMap[action]} · ${choice.display} ${qualityText}${suffix}`;
      pushEventLine(message, quality === "MISS" ? "warn" : "skill");
    }
    if (payload.state) {
      applyGameSnapshot(payload.state);
    } else {
      renderGameHud();
    }
  } catch (error) {
    console.warn("action failed", error);
    pushEventLine("技能请求异常", "warn");
  } finally {
    pendingAction = false;
    renderGameHud();
  }
}

function bindGameControls() {
  joinAEl.addEventListener("click", () => joinTeam("A"));
  joinBEl.addEventListener("click", () => joinTeam("B"));
  for (const { button, action } of actionViews) {
    button.addEventListener("click", () => {
      if (action) {
        triggerAction(action);
      }
    });
  }
  if (resultCloseEl) {
    resultCloseEl.addEventListener("click", hideResultModal);
  }
  if (resultModalEl) {
    resultModalEl.addEventListener("click", (event) => {
      if (event.target === resultModalEl) {
        hideResultModal();
      }
    });
  }
}

function connect() {
  const socket = new WebSocket(wsUrl());

  socket.addEventListener("open", () => {
    statusEl.classList.add("online");
    statusEl.textContent = "Connected";
    const timer = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send("ping");
      }
    }, 3000);
    socket.addEventListener("close", () => clearInterval(timer), { once: true });
  });

  socket.addEventListener("close", () => {
    statusEl.classList.remove("online");
    statusEl.textContent = "Disconnected, reconnecting...";
    setTimeout(connect, 1500);
  });

  socket.addEventListener("message", (event) => {
    const data = safeParseJson(event.data);
    if (!data || typeof data !== "object") {
      return;
    }
    latestSeq = Number(data.seq || 0);
    latestBackend = data.meta
      ? `${data.meta.detector}/${data.meta.tracker}/${data.meta.projector}`
      : "-";
    latestProcess = data.meta ? data.meta.process_ms : "-";
    latestDecode = data.meta
      ? `${data.meta.decode_backend} drop=${data.meta.decode_dropped} buf=${data.meta.decode_buffered}`
      : "-";

    latestDetections = Array.isArray(data.detections) ? data.detections : [];
    latestTracks = Array.isArray(data.tracks) ? data.tracks : [];
    latestFrame = data.frame || latestFrame;
    latestLatency =
      latestFrame && Number.isFinite(latestFrame.source_ts_ms)
        ? Math.max(0, Date.now() - Number(latestFrame.source_ts_ms))
        : 0;
    pushOverlaySnapshot(data);
    syncVideoTimeToFrame(latestFrame);

    const now = performance.now();
    const payloadEntities = Array.isArray(data.entities) ? data.entities : [];
    const seen = updateEntitiesFromPayload(payloadEntities, now);
    pruneStaleEntities(seen, now, 1000);

    applyGameSnapshot(data);
    refreshMetaBar(gameView.continuity || {});
  });
}

function syncVideoTimeToFrame(frameInfo) {
  if (!frameInfo || frameInfo.source_ts_ms == null) {
    return;
  }
  if (!videoEl || Number.isNaN(videoEl.duration) || !Number.isFinite(videoEl.duration) || videoEl.duration <= 0) {
    return;
  }
  const now = performance.now();
  if (now - lastVideoSyncAt < videoSyncConfig.minSyncIntervalMs) {
    return;
  }
  lastVideoSyncAt = now;

  const targetSeconds = Math.max(0, frameInfo.source_ts_ms / 1000);
  const safeTarget = Math.min(Math.max(0, videoEl.duration - 0.05), targetSeconds);
  const signedDrift = safeTarget - videoEl.currentTime;
  const driftAbs = Math.abs(signedDrift);

  if (driftAbs >= videoSyncConfig.hardSeekDriftSec) {
    if (now - lastHardSeekAt >= videoSyncConfig.hardSeekCooldownMs) {
      videoEl.currentTime = safeTarget;
      videoEl.playbackRate = 1.0;
      videoRateUntil = 0;
      lastHardSeekAt = now;
    }
    return;
  }

  if (driftAbs >= videoSyncConfig.softDriftSec) {
    const rateAdjust = clamp(
      signedDrift * videoSyncConfig.rateGain,
      -videoSyncConfig.maxRateDelta,
      videoSyncConfig.maxRateDelta
    );
    const nextRate = clamp(1.0 + rateAdjust, videoSyncConfig.minRate, videoSyncConfig.maxRate);
    if (Math.abs((videoEl.playbackRate || 1.0) - nextRate) > 0.01) {
      videoEl.playbackRate = nextRate;
    }
    videoRateUntil = now + videoSyncConfig.rateHoldMs;
    return;
  }

  if (videoRateUntil > 0 && now >= videoRateUntil) {
    if (Math.abs((videoEl.playbackRate || 1.0) - 1.0) > 0.01) {
      videoEl.playbackRate = 1.0;
    }
    videoRateUntil = 0;
  }
}

function syncOverlayCanvas() {
  const rect = overlayCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);

  if (overlayCanvas.width !== targetW || overlayCanvas.height !== targetH) {
    overlayCanvas.width = targetW;
    overlayCanvas.height = targetH;
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return { width: w, height: h };
}

function syncPitchCanvas() {
  const rect = pitchCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);

  if (pitchCanvas.width !== targetW || pitchCanvas.height !== targetH) {
    pitchCanvas.width = targetW;
    pitchCanvas.height = targetH;
    pitchCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return { width: w, height: h };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function wrapAngle(deg) {
  let angle = deg % 360;
  if (angle > 180) {
    angle -= 360;
  }
  if (angle < -180) {
    angle += 360;
  }
  return angle;
}

function toRadians(deg) {
  return (deg * Math.PI) / 180;
}

function dot3(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross3(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function normalize3(v) {
  const len = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function buildCameraBasis(viewportWidth, viewportHeight) {
  const yaw = toRadians(pitchCamera.yaw);
  const pitch = toRadians(pitchCamera.pitch);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const position = {
    x: pitchCamera.targetX + pitchCamera.distance * cosPitch * Math.cos(yaw),
    y: pitchCamera.targetY + pitchCamera.distance * sinPitch,
    z: pitchCamera.targetZ + pitchCamera.distance * cosPitch * Math.sin(yaw),
  };
  const target = {
    x: pitchCamera.targetX,
    y: pitchCamera.targetY + 2.6,
    z: pitchCamera.targetZ,
  };

  const forward = normalize3({
    x: target.x - position.x,
    y: target.y - position.y,
    z: target.z - position.z,
  });
  let right = normalize3(cross3(forward, { x: 0, y: 1, z: 0 }));
  if (Math.hypot(right.x, right.y, right.z) < 0.001) {
    right = { x: 1, y: 0, z: 0 };
  }
  const up = normalize3(cross3(right, forward));
  const fov = 46;
  const focal = (viewportHeight * 0.5) / Math.tan(toRadians(fov * 0.5));

  return {
    position,
    forward,
    right,
    up,
    focal,
    viewportWidth,
    viewportHeight,
  };
}

function projectWorldPoint(x, y, z, camera) {
  const rel = {
    x: x - camera.position.x,
    y: y - camera.position.y,
    z: z - camera.position.z,
  };
  const cx = dot3(rel, camera.right);
  const cy = dot3(rel, camera.up);
  const cz = dot3(rel, camera.forward);
  if (cz <= 0.15) {
    return null;
  }

  const scale = camera.focal / cz;
  return {
    x: camera.viewportWidth * 0.5 + cx * scale,
    y: camera.viewportHeight * 0.56 - cy * scale,
    depth: cz,
    scale,
  };
}

function panCameraByPixels(dx, dy) {
  const yaw = toRadians(pitchCamera.yaw);
  const right = { x: -Math.sin(yaw), z: Math.cos(yaw) };
  const forward = { x: Math.cos(yaw), z: Math.sin(yaw) };
  const dragScale = pitchCamera.distance * 0.0065;

  pitchCamera.targetX = clamp(
    pitchCamera.targetX + (-dx * right.x + dy * forward.x) * dragScale,
    -22,
    pitchLength + 22
  );
  pitchCamera.targetZ = clamp(
    pitchCamera.targetZ + (-dx * right.z + dy * forward.z) * dragScale,
    -18,
    pitchWidth + 18
  );
}

function resetPitchCamera() {
  pitchCamera.yaw = defaultPitchCamera.yaw;
  pitchCamera.pitch = defaultPitchCamera.pitch;
  pitchCamera.distance = defaultPitchCamera.distance;
  pitchCamera.targetX = defaultPitchCamera.targetX;
  pitchCamera.targetY = defaultPitchCamera.targetY;
  pitchCamera.targetZ = defaultPitchCamera.targetZ;
}

function createPinchSnapshot() {
  if (activeTouchPoints.size < 2) {
    return null;
  }
  const points = Array.from(activeTouchPoints.values());
  const a = points[0];
  const b = points[1];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return {
    distance: Math.max(1, Math.hypot(dx, dy)),
    centerX: (a.x + b.x) / 2,
    centerY: (a.y + b.y) / 2,
  };
}

function setupPitchCameraControls() {
  if (!pitchViewportEl) {
    return;
  }

  resetPitchCamera();
  pitchViewportEl.addEventListener("contextmenu", (event) => event.preventDefault());

  pitchViewportEl.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "touch" && event.button !== 0 && event.button !== 2) {
      return;
    }
    pitchViewportEl.setPointerCapture(event.pointerId);
    pitchViewportEl.classList.add("dragging");

    if (event.pointerType === "touch") {
      activeTouchPoints.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (activeTouchPoints.size >= 2) {
        activePitchPointer = null;
        pinchSnapshot = createPinchSnapshot();
      } else {
        activePitchPointer = {
          pointerId: event.pointerId,
          clientX: event.clientX,
          clientY: event.clientY,
          mode: "orbit",
        };
      }
      return;
    }

    activePitchPointer = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      mode: event.shiftKey || event.button === 2 ? "pan" : "orbit",
    };
  });

  pitchViewportEl.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") {
      if (!activeTouchPoints.has(event.pointerId)) {
        return;
      }
      activeTouchPoints.set(event.pointerId, { x: event.clientX, y: event.clientY });

      if (activeTouchPoints.size >= 2) {
        const nextSnapshot = createPinchSnapshot();
        if (pinchSnapshot && nextSnapshot) {
          const zoomFactor = clamp(nextSnapshot.distance / pinchSnapshot.distance, 0.85, 1.15);
          pitchCamera.distance = clamp(pitchCamera.distance / zoomFactor, 40, 250);
          panCameraByPixels(nextSnapshot.centerX - pinchSnapshot.centerX, nextSnapshot.centerY - pinchSnapshot.centerY);
        }
        pinchSnapshot = nextSnapshot;
        return;
      }
    }

    if (!activePitchPointer || activePitchPointer.pointerId !== event.pointerId) {
      return;
    }

    const dx = event.clientX - activePitchPointer.clientX;
    const dy = event.clientY - activePitchPointer.clientY;
    const mode = event.shiftKey ? "pan" : activePitchPointer.mode;

    if (mode === "pan") {
      panCameraByPixels(dx, dy);
    } else {
      pitchCamera.yaw = wrapAngle(pitchCamera.yaw + dx * 0.26);
      pitchCamera.pitch = clamp(pitchCamera.pitch - dy * 0.2, 10, 80);
    }

    activePitchPointer.clientX = event.clientX;
    activePitchPointer.clientY = event.clientY;
  });

  const releasePointer = (event) => {
    if (event.pointerType === "touch") {
      activeTouchPoints.delete(event.pointerId);
      if (activeTouchPoints.size < 2) {
        pinchSnapshot = null;
      }
      if (activeTouchPoints.size === 1) {
        const [id, point] = Array.from(activeTouchPoints.entries())[0];
        activePitchPointer = {
          pointerId: id,
          clientX: point.x,
          clientY: point.y,
          mode: "orbit",
        };
      }
    }

    if (activePitchPointer && activePitchPointer.pointerId === event.pointerId) {
      activePitchPointer = null;
    }

    if (!activePitchPointer && activeTouchPoints.size === 0) {
      pitchViewportEl.classList.remove("dragging");
    }
  };

  pitchViewportEl.addEventListener("pointerup", releasePointer);
  pitchViewportEl.addEventListener("pointercancel", releasePointer);
  pitchViewportEl.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const zoomFactor = Math.exp(event.deltaY * 0.00115);
      pitchCamera.distance = clamp(pitchCamera.distance * zoomFactor, 40, 250);
    },
    { passive: false }
  );
  pitchViewportEl.addEventListener("dblclick", (event) => {
    event.preventDefault();
    resetPitchCamera();
  });
}

function teamColor(team) {
  if (team === "A") {
    return "#ff5d4d";
  }
  if (team === "B") {
    return "#49a1ff";
  }
  return "#f5f7fa";
}

function drawVideoOverlay() {
  const { width: canvasW, height: canvasH } = syncOverlayCanvas();
  overlayCtx.clearRect(0, 0, canvasW, canvasH);

  const synced = pickOverlaySnapshotForVideoTime();
  const drawFrame = (synced && synced.frame) || latestFrame;
  const detections = (synced && synced.detections) || latestDetections;
  const tracks = (synced && synced.tracks) || latestTracks;

  const frameW = drawFrame.width || videoEl.videoWidth || 1;
  const frameH = drawFrame.height || videoEl.videoHeight || 1;
  const sx = canvasW / frameW;
  const sy = canvasH / frameH;

  overlayCtx.setLineDash([6, 4]);
  overlayCtx.lineWidth = 1.5;
  overlayCtx.strokeStyle = "rgba(255, 224, 87, 0.9)";
  for (const det of detections) {
    const x = det.x * sx;
    const y = det.y * sy;
    const w = det.w * sx;
    const h = det.h * sy;
    overlayCtx.strokeRect(x, y, w, h);
  }

  overlayCtx.setLineDash([]);
  overlayCtx.font = "12px sans-serif";
  overlayCtx.textBaseline = "top";
  for (const track of tracks) {
    const x = track.x * sx;
    const y = track.y * sy;
    const w = track.w * sx;
    const h = track.h * sy;
    const color = teamColor(track.team);

    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2.4;
    overlayCtx.strokeRect(x, y, w, h);

    if (isMobileDevice && track.type !== "ball") {
      continue;
    }

    const label = `${track.id} ${track.type} ${(track.conf * 100).toFixed(0)}%`;
    const textWidth = overlayCtx.measureText(label).width;
    const textX = Math.max(0, x);
    const textY = Math.max(0, y - 18);
    overlayCtx.fillStyle = "rgba(15, 18, 20, 0.78)";
    overlayCtx.fillRect(textX, textY, textWidth + 8, 16);
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(label, textX + 4, textY + 2);
  }
}

function drawWorldLine(a, b, stroke, width, camera) {
  const p0 = projectWorldPoint(a.x, a.y, a.z, camera);
  const p1 = projectWorldPoint(b.x, b.y, b.z, camera);
  if (!p0 || !p1) {
    return;
  }
  pitchCtx.strokeStyle = stroke;
  pitchCtx.lineWidth = width;
  pitchCtx.beginPath();
  pitchCtx.moveTo(p0.x, p0.y);
  pitchCtx.lineTo(p1.x, p1.y);
  pitchCtx.stroke();
}

function drawGroundPolyline(points, stroke, width, camera, closePath = false) {
  const projected = [];
  for (const p of points) {
    const pp = projectWorldPoint(p.x, 0, p.z, camera);
    if (!pp) {
      return;
    }
    projected.push(pp);
  }
  if (projected.length < 2) {
    return;
  }

  pitchCtx.strokeStyle = stroke;
  pitchCtx.lineWidth = width;
  pitchCtx.beginPath();
  pitchCtx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i += 1) {
    pitchCtx.lineTo(projected[i].x, projected[i].y);
  }
  if (closePath) {
    pitchCtx.lineTo(projected[0].x, projected[0].y);
  }
  pitchCtx.stroke();
}

function drawGroundRect(x0, z0, x1, z1, stroke, width, camera) {
  drawGroundPolyline(
    [
      { x: x0, z: z0 },
      { x: x1, z: z0 },
      { x: x1, z: z1 },
      { x: x0, z: z1 },
    ],
    stroke,
    width,
    camera,
    true
  );
}

function drawGroundCircle(cx, cz, radius, stroke, width, camera) {
  const points = [];
  const segments = 64;
  for (let i = 0; i <= segments; i += 1) {
    const t = (i / segments) * Math.PI * 2;
    points.push({ x: cx + Math.cos(t) * radius, z: cz + Math.sin(t) * radius });
  }
  drawGroundPolyline(points, stroke, width, camera, false);
}

function drawGoal(goalX, direction, camera) {
  const halfWidth = 3.66;
  const goalDepth = 2.4;
  const goalHeight = 2.44;
  const nearX = goalX;
  const farX = goalX + direction * goalDepth;
  const z0 = pitchWidth * 0.5 - halfWidth;
  const z1 = pitchWidth * 0.5 + halfWidth;
  const color = "rgba(242, 250, 255, 0.95)";

  drawWorldLine({ x: nearX, y: 0, z: z0 }, { x: nearX, y: goalHeight, z: z0 }, color, 2.1, camera);
  drawWorldLine({ x: nearX, y: 0, z: z1 }, { x: nearX, y: goalHeight, z: z1 }, color, 2.1, camera);
  drawWorldLine({ x: nearX, y: goalHeight, z: z0 }, { x: nearX, y: goalHeight, z: z1 }, color, 2.1, camera);

  drawWorldLine({ x: farX, y: 0, z: z0 }, { x: farX, y: goalHeight, z: z0 }, "rgba(220, 238, 248, 0.7)", 1.4, camera);
  drawWorldLine({ x: farX, y: 0, z: z1 }, { x: farX, y: goalHeight, z: z1 }, "rgba(220, 238, 248, 0.7)", 1.4, camera);
  drawWorldLine({ x: farX, y: goalHeight, z: z0 }, { x: farX, y: goalHeight, z: z1 }, "rgba(220, 238, 248, 0.7)", 1.4, camera);
  drawWorldLine({ x: nearX, y: goalHeight, z: z0 }, { x: farX, y: goalHeight, z: z0 }, color, 1.6, camera);
  drawWorldLine({ x: nearX, y: goalHeight, z: z1 }, { x: farX, y: goalHeight, z: z1 }, color, 1.6, camera);
  drawWorldLine({ x: nearX, y: 0, z: z0 }, { x: farX, y: 0, z: z0 }, "rgba(220, 238, 248, 0.6)", 1.2, camera);
  drawWorldLine({ x: nearX, y: 0, z: z1 }, { x: farX, y: 0, z: z1 }, "rgba(220, 238, 248, 0.6)", 1.2, camera);
}

function drawPitchSurface(camera, canvasW, canvasH) {
  const skyGradient = pitchCtx.createLinearGradient(0, 0, 0, canvasH);
  skyGradient.addColorStop(0, "#8fd1ff");
  skyGradient.addColorStop(0.52, "#d8f2ff");
  skyGradient.addColorStop(1, "#daf1c2");
  pitchCtx.fillStyle = skyGradient;
  pitchCtx.fillRect(0, 0, canvasW, canvasH);

  pitchCtx.fillStyle = "rgba(121, 165, 103, 0.45)";
  pitchCtx.fillRect(0, canvasH * 0.68, canvasW, canvasH * 0.32);

  const corners = [
    projectWorldPoint(0, 0, 0, camera),
    projectWorldPoint(pitchLength, 0, 0, camera),
    projectWorldPoint(pitchLength, 0, pitchWidth, camera),
    projectWorldPoint(0, 0, pitchWidth, camera),
  ];
  if (corners.some((p) => !p)) {
    return;
  }

  const drawFieldPath = () => {
    pitchCtx.beginPath();
    pitchCtx.moveTo(corners[0].x, corners[0].y);
    pitchCtx.lineTo(corners[1].x, corners[1].y);
    pitchCtx.lineTo(corners[2].x, corners[2].y);
    pitchCtx.lineTo(corners[3].x, corners[3].y);
    pitchCtx.closePath();
  };

  drawFieldPath();
  pitchCtx.fillStyle = "#51ac4c";
  pitchCtx.fill();

  pitchCtx.save();
  drawFieldPath();
  pitchCtx.clip();
  for (let i = 0; i < 12; i += 1) {
    const x0 = (pitchLength / 12) * i;
    const x1 = (pitchLength / 12) * (i + 1);
    const stripe = [
      projectWorldPoint(x0, 0, 0, camera),
      projectWorldPoint(x1, 0, 0, camera),
      projectWorldPoint(x1, 0, pitchWidth, camera),
      projectWorldPoint(x0, 0, pitchWidth, camera),
    ];
    if (stripe.some((p) => !p)) {
      continue;
    }
    pitchCtx.fillStyle = i % 2 === 0 ? "rgba(120, 195, 99, 0.2)" : "rgba(52, 138, 57, 0.16)";
    pitchCtx.beginPath();
    pitchCtx.moveTo(stripe[0].x, stripe[0].y);
    pitchCtx.lineTo(stripe[1].x, stripe[1].y);
    pitchCtx.lineTo(stripe[2].x, stripe[2].y);
    pitchCtx.lineTo(stripe[3].x, stripe[3].y);
    pitchCtx.closePath();
    pitchCtx.fill();
  }
  pitchCtx.restore();

  const line = "rgba(238, 250, 213, 0.96)";
  drawGroundRect(0, 0, pitchLength, pitchWidth, line, 2.2, camera);
  drawWorldLine({ x: pitchLength * 0.5, y: 0, z: 0 }, { x: pitchLength * 0.5, y: 0, z: pitchWidth }, line, 1.8, camera);
  drawGroundCircle(pitchLength * 0.5, pitchWidth * 0.5, 9.15, line, 1.8, camera);
  drawGroundRect(0, pitchWidth * 0.5 - 20.15, 16.5, pitchWidth * 0.5 + 20.15, line, 1.8, camera);
  drawGroundRect(
    pitchLength - 16.5,
    pitchWidth * 0.5 - 20.15,
    pitchLength,
    pitchWidth * 0.5 + 20.15,
    line,
    1.8,
    camera
  );
  drawGroundRect(0, pitchWidth * 0.5 - 9.16, 5.5, pitchWidth * 0.5 + 9.16, line, 1.6, camera);
  drawGroundRect(
    pitchLength - 5.5,
    pitchWidth * 0.5 - 9.16,
    pitchLength,
    pitchWidth * 0.5 + 9.16,
    line,
    1.6,
    camera
  );

  drawGoal(0, -1, camera);
  drawGoal(pitchLength, 1, camera);
}

function collectRenderableEntities(now) {
  const active = [];
  for (const e of entities.values()) {
    e.x += (e.targetX - e.x) * 0.22;
    e.y += (e.targetY - e.y) * 0.22;
    if (now - e.updatedAt > 1800) {
      entities.delete(e.id);
      continue;
    }
    active.push(e);
  }
  return active;
}

function drawPanda(entity, camera, now) {
  const phase = now * 0.007 + Number(entity.id || 0) * 0.65;
  const bob = Math.sin(phase) * 0.08;
  const base = projectWorldPoint(entity.x, 0, entity.y, camera);
  const top = projectWorldPoint(entity.x, 1.75 + bob, entity.y, camera);
  if (!base || !top) {
    return;
  }

  const pandaSize = clamp((base.y - top.y) * 1.25, 14, 62);
  const bodyX = base.x;
  const bodyY = base.y - pandaSize * 0.78;
  const headY = bodyY - pandaSize * 0.56;
  const jerseyColor = entity.team === "A" ? "#ff8b7b" : entity.team === "B" ? "#6fb9ff" : "#9aa4ad";

  pitchCtx.fillStyle = "rgba(20, 29, 19, 0.23)";
  pitchCtx.beginPath();
  pitchCtx.ellipse(base.x, base.y - pandaSize * 0.03, pandaSize * 0.52, pandaSize * 0.22, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#1f2327";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX - pandaSize * 0.16, base.y - pandaSize * 0.22, pandaSize * 0.12, pandaSize * 0.23, 0, 0, Math.PI * 2);
  pitchCtx.ellipse(bodyX + pandaSize * 0.16, base.y - pandaSize * 0.22, pandaSize * 0.12, pandaSize * 0.23, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#f7faf8";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX, bodyY, pandaSize * 0.4, pandaSize * 0.48, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = jerseyColor;
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX, bodyY + pandaSize * 0.04, pandaSize * 0.35, pandaSize * 0.23, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#111417";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - pandaSize * 0.23, headY - pandaSize * 0.25, pandaSize * 0.15, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + pandaSize * 0.23, headY - pandaSize * 0.25, pandaSize * 0.15, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#ffffff";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY, pandaSize * 0.34, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#14181a";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX - pandaSize * 0.16, headY - pandaSize * 0.03, pandaSize * 0.11, pandaSize * 0.14, 0.1, 0, Math.PI * 2);
  pitchCtx.ellipse(bodyX + pandaSize * 0.16, headY - pandaSize * 0.03, pandaSize * 0.11, pandaSize * 0.14, -0.1, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#fefefe";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - pandaSize * 0.15, headY - pandaSize * 0.01, pandaSize * 0.04, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + pandaSize * 0.15, headY - pandaSize * 0.01, pandaSize * 0.04, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#191b1c";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY + pandaSize * 0.07, pandaSize * 0.04, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.strokeStyle = "#191b1c";
  pitchCtx.lineWidth = Math.max(1, pandaSize * 0.04);
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY + pandaSize * 0.1, pandaSize * 0.1, 0.25 * Math.PI, 0.75 * Math.PI);
  pitchCtx.stroke();

  pitchCtx.fillStyle = "rgba(255, 160, 176, 0.72)";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - pandaSize * 0.24, headY + pandaSize * 0.07, pandaSize * 0.05, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + pandaSize * 0.24, headY + pandaSize * 0.07, pandaSize * 0.05, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#f7fbff";
  pitchCtx.font = `${Math.max(10, pandaSize * 0.22)}px "Avenir Next", sans-serif`;
  pitchCtx.textAlign = "center";
  pitchCtx.textBaseline = "middle";
  pitchCtx.fillText(String(entity.id), bodyX, bodyY + pandaSize * 0.02);
}

function drawMonkey(entity, camera, now) {
  const phase = now * 0.007 + Number(entity.id || 0) * 0.61;
  const bob = Math.sin(phase) * 0.09;
  const base = projectWorldPoint(entity.x, 0, entity.y, camera);
  const top = projectWorldPoint(entity.x, 1.72 + bob, entity.y, camera);
  if (!base || !top) {
    return;
  }

  const monkeySize = clamp((base.y - top.y) * 1.22, 14, 62);
  const bodyX = base.x;
  const bodyY = base.y - monkeySize * 0.75;
  const headY = bodyY - monkeySize * 0.54;

  pitchCtx.fillStyle = "rgba(23, 20, 13, 0.24)";
  pitchCtx.beginPath();
  pitchCtx.ellipse(base.x, base.y - monkeySize * 0.03, monkeySize * 0.54, monkeySize * 0.23, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.strokeStyle = "#5e3b21";
  pitchCtx.lineWidth = Math.max(1.2, monkeySize * 0.05);
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX + monkeySize * 0.34, bodyY - monkeySize * 0.12, monkeySize * 0.26, -0.55 * Math.PI, 0.6 * Math.PI);
  pitchCtx.stroke();

  pitchCtx.fillStyle = "#8f5f35";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX, bodyY, monkeySize * 0.4, monkeySize * 0.48, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#6fb9ff";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX, bodyY + monkeySize * 0.06, monkeySize * 0.35, monkeySize * 0.23, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#8f5f35";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - monkeySize * 0.28, headY - monkeySize * 0.08, monkeySize * 0.13, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + monkeySize * 0.28, headY - monkeySize * 0.08, monkeySize * 0.13, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#c99662";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - monkeySize * 0.28, headY - monkeySize * 0.08, monkeySize * 0.075, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + monkeySize * 0.28, headY - monkeySize * 0.08, monkeySize * 0.075, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#9a673d";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY, monkeySize * 0.34, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#d8b188";
  pitchCtx.beginPath();
  pitchCtx.ellipse(bodyX, headY + monkeySize * 0.02, monkeySize * 0.22, monkeySize * 0.17, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#1c1611";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX - monkeySize * 0.11, headY - monkeySize * 0.04, monkeySize * 0.035, 0, Math.PI * 2);
  pitchCtx.arc(bodyX + monkeySize * 0.11, headY - monkeySize * 0.04, monkeySize * 0.035, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#6b3f1f";
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY + monkeySize * 0.04, monkeySize * 0.03, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.strokeStyle = "#6b3f1f";
  pitchCtx.lineWidth = Math.max(1, monkeySize * 0.032);
  pitchCtx.beginPath();
  pitchCtx.arc(bodyX, headY + monkeySize * 0.08, monkeySize * 0.09, 0.2 * Math.PI, 0.8 * Math.PI);
  pitchCtx.stroke();

  pitchCtx.fillStyle = "#f7fbff";
  pitchCtx.font = `${Math.max(10, monkeySize * 0.22)}px "Avenir Next", sans-serif`;
  pitchCtx.textAlign = "center";
  pitchCtx.textBaseline = "middle";
  pitchCtx.fillText(String(entity.id), bodyX, bodyY + monkeySize * 0.03);
}

function drawBall(entity, camera, now) {
  const bounce = 0.16 + Math.abs(Math.sin(now * 0.01 + Number(entity.id || 0))) * 0.11;
  const center = projectWorldPoint(entity.x, 0.25 + bounce, entity.y, camera);
  const base = projectWorldPoint(entity.x, 0, entity.y, camera);
  if (!center || !base) {
    return;
  }

  const radius = clamp((base.y - center.y) * 0.9, 4, 16);

  pitchCtx.fillStyle = "rgba(19, 24, 18, 0.24)";
  pitchCtx.beginPath();
  pitchCtx.ellipse(base.x, base.y - radius * 0.15, radius * 1.05, radius * 0.45, 0, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#ffffff";
  pitchCtx.beginPath();
  pitchCtx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  pitchCtx.fill();

  pitchCtx.fillStyle = "#1f2730";
  pitchCtx.beginPath();
  pitchCtx.arc(center.x, center.y, radius * 0.32, 0, Math.PI * 2);
  pitchCtx.arc(center.x - radius * 0.56, center.y + radius * 0.04, radius * 0.17, 0, Math.PI * 2);
  pitchCtx.arc(center.x + radius * 0.49, center.y + radius * 0.17, radius * 0.14, 0, Math.PI * 2);
  pitchCtx.fill();
}

function drawPitchScene() {
  const { width: canvasW, height: canvasH } = syncPitchCanvas();
  pitchCtx.clearRect(0, 0, canvasW, canvasH);
  pitchCtx.lineJoin = "round";
  pitchCtx.lineCap = "round";

  const camera = buildCameraBasis(canvasW, canvasH);
  drawPitchSurface(camera, canvasW, canvasH);

  const now = performance.now();
  const activeEntities = collectRenderableEntities(now);
  const drawables = [];
  for (const entity of activeEntities) {
    const p = projectWorldPoint(entity.x, 0, entity.y, camera);
    if (!p) {
      continue;
    }
    const kind = entity.type === "ball" ? "ball" : entity.team === "B" ? "monkey" : "panda";
    drawables.push({
      kind,
      entity,
      depth: p.depth,
    });
  }

  drawables.sort((a, b) => b.depth - a.depth);
  for (const item of drawables) {
    if (item.kind === "ball") {
      drawBall(item.entity, camera, now);
    } else if (item.kind === "monkey") {
      drawMonkey(item.entity, camera, now);
    } else {
      drawPanda(item.entity, camera, now);
    }
  }
}

function render(now = performance.now()) {
  const overlayIntervalMs = 33;
  const pitchIntervalMs = isMobileDevice ? 42 : 28;

  if (now - lastOverlayRenderAt >= overlayIntervalMs) {
    drawVideoOverlay();
    lastOverlayRenderAt = now;
  }

  if (now - lastPitchRenderAt >= pitchIntervalMs) {
    drawPitchScene();
    lastPitchRenderAt = now;
  }

  renderGameHud();
  requestAnimationFrame(render);
}

async function bootstrap() {
  bindGameControls();
  renderEventFeed();
  renderGameHud();
  await Promise.all([loadVideoPreview(), fetchGameState()]);
  connect();
  setupPitchCameraControls();
  render();
}

bootstrap();
