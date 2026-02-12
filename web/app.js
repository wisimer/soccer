const pitchLength = 105;
const pitchWidth = 68;

const canvas = document.getElementById("pitch");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");

const entities = new Map();
let latestSeq = 0;
let latestLatency = 0;
let latestBackend = "-";
let latestProcess = "-";
let latestDecode = "-";

function wsUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws`;
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
    latestSeq = data.seq;
    latestLatency = Date.now() - data.frame_ts_ms;
    latestBackend = data.meta
      ? `${data.meta.detector}/${data.meta.tracker}/${data.meta.projector}`
      : "-";
    latestProcess = data.meta ? data.meta.process_ms : "-";
    latestDecode = data.meta
      ? `${data.meta.decode_backend} drop=${data.meta.decode_dropped} buf=${data.meta.decode_buffered}`
      : "-";

    const seen = new Set();
    for (const e of data.entities) {
      seen.add(e.id);
      const now = performance.now();
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
        if (item && performance.now() - item.updatedAt > 1000) {
          entities.delete(id);
        }
      }
    }

    metaEl.innerHTML = `
      <span>seq: ${latestSeq}</span>
      <span>entities: ${entities.size}</span>
      <span>latency: ${latestLatency} ms</span>
      <span>backend: ${latestBackend}</span>
      <span>process: ${latestProcess} ms</span>
      <span>decode: ${latestDecode}</span>
    `;
  });
}

function pitchToCanvas(x, y) {
  return {
    x: (x / pitchLength) * canvas.width,
    y: (y / pitchWidth) * canvas.height,
  };
}

function drawPitch() {
  ctx.fillStyle = "#2b8e46";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#ecf3d2";
  ctx.lineWidth = 4;
  ctx.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);

  ctx.beginPath();
  ctx.moveTo(canvas.width / 2, 4);
  ctx.lineTo(canvas.width / 2, canvas.height - 4);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(canvas.width / 2, canvas.height / 2, 85, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.rect(4, canvas.height * 0.2, canvas.width * 0.16, canvas.height * 0.6);
  ctx.rect(canvas.width * 0.84, canvas.height * 0.2, canvas.width * 0.16 - 4, canvas.height * 0.6);
  ctx.stroke();
}

function drawEntities() {
  const now = performance.now();

  for (const e of entities.values()) {
    e.x += (e.targetX - e.x) * 0.25;
    e.y += (e.targetY - e.y) * 0.25;

    const p = pitchToCanvas(e.x, e.y);
    if (e.type === "ball") {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.fillStyle = e.team === "A" ? "#e04a3f" : e.team === "B" ? "#1f5abf" : "#111111";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#f8f8f8";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(String(e.id), p.x, p.y - 14);
    }

    if (now - e.updatedAt > 1800) {
      entities.delete(e.id);
    }
  }
}

function render() {
  drawPitch();
  drawEntities();
  requestAnimationFrame(render);
}

connect();
render();
