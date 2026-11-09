import { store } from "./store";

/** Top-down map: only what the seeker camera covered is revealed (perception log-odds map); the rest is black. */
export interface MapOpts {
  reveal: boolean; // true = black outside coverage
  showFire: boolean; // truth fire raster (spectator only)
  hotspots: boolean;
  drones: boolean;
  playerId?: string | null;
  playerPose?: { x: number; y: number; yaw: number } | null;
  anchors?: Map<string, { x: number; y: number; hotspot?: string }>;
  selected?: string | null;
  planLines?: [string, string][];
  footprint?: number; // camera swath (m) to draw around the player
}

const off = document.createElement("canvas");
const offCtx = off.getContext("2d")!;

export function toCanvas(x: number, y: number, size: number, R: number): [number, number] {
  return [((x + R) / (2 * R)) * size, ((R - y) / (2 * R)) * size];
}

export function fromCanvas(px: number, py: number, size: number, R: number): [number, number] {
  return [(px / size) * 2 * R - R, R - (py / size) * 2 * R];
}

function rasterImage(opts: MapOpts): HTMLCanvasElement | null {
  const init = store.init;
  const g = store.groundBytes;
  if (!init || !g) return null;
  const n = init.ground.n;
  if (off.width !== n) {
    off.width = n;
    off.height = n;
  }
  const img = offCtx.createImageData(n, n);
  const cov = store.cov;
  const fire = store.fire;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const src = r * n + c; // row 0 = south
      const dst = ((n - 1 - r) * n + c) * 4; // canvas row 0 = north
      const v = g[src] / 255;
      const seen = !opts.reveal || (cov && store.covN === n && cov[src] > 0);
      let R0 = 0, G0 = 0, B0 = 0;
      if (seen) {
        R0 = 34 + 22 * v;
        G0 = 50 + 26 * v;
        B0 = 42 + 14 * v;
      } else {
        R0 = 4;
        G0 = 6;
        B0 = 10;
      }
      if (opts.showFire && fire && store.fireN === n) {
        const h = fire[src] / 255;
        if (h > 0.02) {
          R0 = Math.min(255, R0 + 255 * h);
          G0 = Math.min(255, G0 + 110 * h * h);
          B0 = B0 * (1 - h);
        }
      }
      img.data[dst] = R0;
      img.data[dst + 1] = G0;
      img.data[dst + 2] = B0;
      img.data[dst + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  return off;
}

export function drawMap(ctx: CanvasRenderingContext2D, size: number, opts: MapOpts): void {
  const init = store.init;
  if (!init) return;
  const R = init.arena_radius;
  ctx.clearRect(0, 0, size, size);
  ctx.save();
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = "#03050a";
  ctx.fillRect(0, 0, size, size);
  const img = rasterImage(opts);
  if (img) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, 0, 0, size, size);
  }
  // grid
  ctx.strokeStyle = "rgba(120,170,255,0.08)";
  ctx.lineWidth = 1;
  for (let m = -150; m <= 150; m += 50) {
    const [a] = toCanvas(m, 0, size, R);
    const [, b] = toCanvas(0, m, size, R);
    ctx.beginPath();
    ctx.moveTo(a, 0);
    ctx.lineTo(a, size);
    ctx.moveTo(0, b);
    ctx.lineTo(size, b);
    ctx.stroke();
  }
  // base
  const [bx, by] = toCanvas(init.base[0], init.base[1] + 8, size, R);
  ctx.fillStyle = "rgba(79,209,255,0.25)";
  ctx.strokeStyle = "#4fd1ff";
  ctx.beginPath();
  ctx.arc(bx, by, Math.max(5, (22 / (2 * R)) * size), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const s = size / (2 * R);
  // plan lines (swarm plan in watch mode)
  if (opts.planLines) {
    const hs = new Map(store.hotspots.map((h) => [h[0], h]));
    ctx.strokeStyle = "rgba(79,209,255,0.7)";
    ctx.setLineDash([4, 4]);
    for (const [, hid] of opts.planLines) {
      const h = hs.get(hid);
      if (!h) continue;
      const [x, y] = toCanvas(h[1], h[2], size, R);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  // anchors (human plan)
  if (opts.anchors) {
    for (const [id, a] of opts.anchors) {
      const [x, y] = toCanvas(a.x, a.y, size, R);
      const sel = opts.selected === id;
      ctx.strokeStyle = sel ? "#ffffff" : "rgba(255,181,71,0.85)";
      ctx.lineWidth = sel ? 2 : 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffb547";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#1a1206";
      ctx.font = "600 9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(id.replace("ext-", ""), x, y + 0.5);
      if (!a.hotspot) {
        ctx.strokeStyle = "rgba(255,181,71,0.25)";
        ctx.beginPath();
        ctx.arc(x, y, 45 * s, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
  // hotspots from perception
  if (opts.hotspots) {
    for (const [, x, y, area, conf, status] of store.hotspots) {
      const [px, py] = toCanvas(x, y, size, R);
      const r = Math.max(4, Math.sqrt(Math.max(area, 0.5) / Math.PI) * s * 2.2 + 3);
      const col = status === 3 ? "#5cff9d" : status === 0 ? "#ffd166" : "#ff5a36";
      ctx.fillStyle = col + "44";
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (status !== 3 && size > 300) {
        ctx.fillStyle = "rgba(255,255,255,0.75)";
        ctx.font = "500 10px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${area.toFixed(1)} m² · ${(conf * 100).toFixed(0)}%`, px + r + 3, py + 3);
      }
    }
  }
  // drones
  if (opts.drones) {
    for (const [id, kind, x, y, , yaw, , , isPlayer] of store.drones) {
      if (isPlayer && opts.playerPose) continue;
      const [px, py] = toCanvas(x, y, size, R);
      ctx.fillStyle = kind === "s" ? "#b49cff" : store.init?.mode === "watch" ? "#4fd1ff" : "#ffb547";
      arrow(ctx, px, py, yaw, kind === "e" ? 5 : 4);
      if (opts.selected === id) {
        ctx.strokeStyle = "#fff";
        ctx.beginPath();
        ctx.arc(px, py, 9, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
  if (opts.playerPose) {
    const p = opts.playerPose;
    const [px, py] = toCanvas(p.x, p.y, size, R);
    if (opts.footprint) {
      ctx.strokeStyle = "rgba(125,255,178,0.6)";
      const w = opts.footprint * s;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate((p.yaw * Math.PI) / 180);
      ctx.strokeRect(-w / 2, -w * 0.4, w, w * 0.8);
      ctx.restore();
    }
    ctx.fillStyle = "#7dffb2";
    arrow(ctx, px, py, p.yaw, 7);
  }
  ctx.restore();
  ctx.strokeStyle = "rgba(79,209,255,0.55)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
  ctx.stroke();
}

function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, yawDeg: number, s: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate((yawDeg * Math.PI) / 180);
  ctx.beginPath();
  ctx.moveTo(0, -s * 1.4);
  ctx.lineTo(s, s);
  ctx.lineTo(0, s * 0.45);
  ctx.lineTo(-s, s);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
