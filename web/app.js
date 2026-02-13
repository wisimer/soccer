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

  if (active.length > 0) {
    return active;
  }

  const sample = [];
  for (let i = 0; i < 8; i += 1) {
    const phase = now * 0.001 + i * 0.7;
    sample.push({
      id: 200 + i,
      type: "player",
      team: i % 2 === 0 ? "A" : "B",
      x: pitchLength * 0.22 + (i % 4) * 18 + Math.sin(phase) * 1.7,
      y: pitchWidth * 0.2 + Math.floor(i / 4) * 24 + Math.cos(phase * 0.8) * 1.2,
    });
  }
  sample.push({
    id: 1,
    type: "ball",
    team: "unknown",
    x: pitchLength * 0.5 + Math.sin(now * 0.0012) * 4,
    y: pitchWidth * 0.5 + Math.cos(now * 0.0011) * 2.3,
  });
  return sample;
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
    drawables.push({
      kind: entity.type === "ball" ? "ball" : "panda",
      entity,
      depth: p.depth,
    });
  }

  drawables.sort((a, b) => b.depth - a.depth);
  for (const item of drawables) {
    if (item.kind === "ball") {
      drawBall(item.entity, camera, now);
    } else {
      drawPanda(item.entity, camera, now);
    }
  }
}

function render() {
  drawVideoOverlay();
  drawPitchScene();
  requestAnimationFrame(render);
}

loadVideoPreview();
connect();
setupPitchCameraControls();
render();
