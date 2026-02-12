const pitchLength = 105;
const pitchWidth = 68;

const videoEl = document.getElementById("sourceVideo");
const overlayCanvas = document.getElementById("overlay");
const overlayCtx = overlayCanvas.getContext("2d");

const pitchCanvas = document.getElementById("pitch");
const pitchCtx = pitchCanvas.getContext("2d");

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

function wsUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws`;
}

async function loadVideoPreview() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`health status ${res.status}`);
    }
    const health = await res.json();
    const previewUrl = health.video_preview_url || (health.runtime && health.runtime.video_preview_url);
    if (previewUrl) {
      videoEl.src = previewUrl;
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
    const data = JSON.parse(event.data);
    latestSeq = data.seq || 0;
    latestLatency = data.frame_ts_ms ? Date.now() - data.frame_ts_ms : 0;
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
    syncVideoTimeToFrame(latestFrame);

    const seen = new Set();
    const now = performance.now();
    const payloadEntities = Array.isArray(data.entities) ? data.entities : [];
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
          updatedAt: now,
        });
      } else {
        current.type = e.type;
        current.team = e.team;
        current.targetX = e.x;
        current.targetY = e.y;
        current.updatedAt = now;
      }
    }

    for (const id of entities.keys()) {
      if (!seen.has(id)) {
        const item = entities.get(id);
        if (item && now - item.updatedAt > 1000) {
          entities.delete(id);
        }
      }
    }

    metaEl.innerHTML = `
      <span>seq: ${latestSeq}</span>
      <span>entities: ${entities.size}</span>
      <span>tracks: ${latestTracks.length}</span>
      <span>detections: ${latestDetections.length}</span>
      <span>latency: ${latestLatency} ms</span>
      <span>backend: ${latestBackend}</span>
      <span>process: ${latestProcess} ms</span>
      <span>decode: ${latestDecode}</span>
    `;
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
  if (now - lastVideoSyncAt < 180) {
    return;
  }
  lastVideoSyncAt = now;

  const targetSeconds = Math.max(0, frameInfo.source_ts_ms / 1000);
  const drift = Math.abs(videoEl.currentTime - targetSeconds);
  if (drift > 0.25) {
    const safeTarget = Math.min(videoEl.duration - 0.05, targetSeconds);
    if (safeTarget >= 0) {
      videoEl.currentTime = safeTarget;
    }
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

  const frameW = latestFrame.width || videoEl.videoWidth || 1;
  const frameH = latestFrame.height || videoEl.videoHeight || 1;
  const sx = canvasW / frameW;
  const sy = canvasH / frameH;

  overlayCtx.setLineDash([6, 4]);
  overlayCtx.lineWidth = 1.5;
  overlayCtx.strokeStyle = "rgba(255, 224, 87, 0.9)";
  for (const det of latestDetections) {
    const x = det.x * sx;
    const y = det.y * sy;
    const w = det.w * sx;
    const h = det.h * sy;
    overlayCtx.strokeRect(x, y, w, h);
  }

  overlayCtx.setLineDash([]);
  overlayCtx.font = "12px sans-serif";
  overlayCtx.textBaseline = "top";
  for (const track of latestTracks) {
    const x = track.x * sx;
    const y = track.y * sy;
    const w = track.w * sx;
    const h = track.h * sy;
    const color = teamColor(track.team);

    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2.4;
    overlayCtx.strokeRect(x, y, w, h);

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

function pitchToCanvas(x, y) {
  return {
    x: (x / pitchLength) * pitchCanvas.width,
    y: (y / pitchWidth) * pitchCanvas.height,
  };
}

function drawPitch() {
  pitchCtx.fillStyle = "#2b8e46";
  pitchCtx.fillRect(0, 0, pitchCanvas.width, pitchCanvas.height);

  pitchCtx.strokeStyle = "#ecf3d2";
  pitchCtx.lineWidth = 4;
  pitchCtx.strokeRect(4, 4, pitchCanvas.width - 8, pitchCanvas.height - 8);

  pitchCtx.beginPath();
  pitchCtx.moveTo(pitchCanvas.width / 2, 4);
  pitchCtx.lineTo(pitchCanvas.width / 2, pitchCanvas.height - 4);
  pitchCtx.stroke();

  pitchCtx.beginPath();
  pitchCtx.arc(pitchCanvas.width / 2, pitchCanvas.height / 2, 85, 0, Math.PI * 2);
  pitchCtx.stroke();

  pitchCtx.beginPath();
  pitchCtx.rect(4, pitchCanvas.height * 0.2, pitchCanvas.width * 0.16, pitchCanvas.height * 0.6);
  pitchCtx.rect(
    pitchCanvas.width * 0.84,
    pitchCanvas.height * 0.2,
    pitchCanvas.width * 0.16 - 4,
    pitchCanvas.height * 0.6
  );
  pitchCtx.stroke();
}

function drawPitchEntities() {
  const now = performance.now();
  for (const e of entities.values()) {
    e.x += (e.targetX - e.x) * 0.25;
    e.y += (e.targetY - e.y) * 0.25;

    const p = pitchToCanvas(e.x, e.y);
    if (e.type === "ball") {
      pitchCtx.fillStyle = "#ffffff";
      pitchCtx.beginPath();
      pitchCtx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      pitchCtx.fill();
      pitchCtx.strokeStyle = "#111111";
      pitchCtx.lineWidth = 1;
      pitchCtx.stroke();
    } else {
      pitchCtx.fillStyle = e.team === "A" ? "#e04a3f" : e.team === "B" ? "#1f5abf" : "#111111";
      pitchCtx.beginPath();
      pitchCtx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      pitchCtx.fill();

      pitchCtx.fillStyle = "#f8f8f8";
      pitchCtx.font = "11px sans-serif";
      pitchCtx.textAlign = "center";
      pitchCtx.fillText(String(e.id), p.x, p.y - 14);
    }

    if (now - e.updatedAt > 1800) {
      entities.delete(e.id);
    }
  }
}

function render() {
  drawVideoOverlay();
  drawPitch();
  drawPitchEntities();
  requestAnimationFrame(render);
}

loadVideoPreview();
connect();
render();
