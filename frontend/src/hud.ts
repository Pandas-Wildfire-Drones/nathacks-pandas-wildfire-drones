import { drawMap } from "./mapdraw";
import type { Player } from "./player";
import { store } from "./store";
import type { Phase, ThermalMsg } from "./types";
import { b64bytes } from "./types";

const PHASE_LABEL: Record<Phase, string> = {
  BRIEFING: "Briefing",
  SEEK: "Seek",
  PLAN: "Plan",
  EXTINGUISH: "Extinguish",
  DEMO: "Demo · live",
  SCORE: "Results",
};

export interface FollowInfo {
  id: string | null;
  nextIn: number; // seconds until the camera switches drone (0 = manual / off)
  period: number;
}

// inferno-ish LUT for the thermal picture-in-picture
const LUT = (() => {
  const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const a = Math.floor(t), b = Math.min(stops.length - 1, a + 1), f = t - a;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = stops[a][k] * (1 - f) + stops[b][k] * f;
  }
  return out;
})();

export function fmtTime(s: number): string {
  s = Math.max(0, Math.ceil(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export class Hud {
  el: HTMLElement;
  private mini: HTMLCanvasElement;
  private miniCtx: CanvasRenderingContext2D;
  private th: HTMLCanvasElement;
  private thCtx: CanvasRenderingContext2D;
  private thImg: ImageData | null = null;
  private lastThermalT = -1;
  private $ = (sel: string) => this.el.querySelector(sel) as HTMLElement;
  showThermal = true;

  constructor(ui: HTMLElement, onPause: () => void) {
    this.el = document.createElement("div");
    this.el.className = "hud hidden";
    this.el.innerHTML = `
      <div class="hud-top">
        <button class="icon-btn" id="hud-pause" title="Pause (Esc)"><svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor"/><rect x="8.5" y="1" width="3.5" height="12" rx="1" fill="currentColor"/></svg></button>
        <div class="phase-pill"><span class="phase-dot"></span><span id="hud-phase">Seek</span></div>
        <div class="timer" id="hud-timer">1:15</div>
        <div class="phase-bar"><div id="hud-bar"></div></div>
        <div class="mode-tag" id="hud-mode"></div>
      </div>
      <div class="panel telemetry" id="hud-tele"></div>
      <div class="panel feed" id="hud-feed"></div>
      <div class="panel minimap"><canvas id="hud-mini" width="440" height="440"></canvas><div class="mini-label" id="hud-mini-label"></div></div>
      <div class="panel thermal" id="hud-thermal">
        <div class="thermal-head"><span>LWIR · nadir</span><span id="hud-tmax"></span></div>
        <canvas id="hud-th" width="160" height="128"></canvas>
        <div class="thermal-foot" id="hud-th-foot">FUSED · threshold + CNN (fire_cnn_mix.onnx)</div>
      </div>
      <div class="hint" id="hud-hint"></div>
      <div class="mouse-hint" id="hud-mouse">Click the view to steer with the mouse · or Q / E · drag also turns</div>
      <div class="panel follow-card" id="hud-follow"></div>
      <div class="reticle" id="hud-reticle"><div></div></div>
      <div class="toast" id="hud-toast"></div>`;
    ui.appendChild(this.el);
    this.mini = this.$("#hud-mini") as HTMLCanvasElement;
    this.miniCtx = this.mini.getContext("2d")!;
    this.th = this.$("#hud-th") as HTMLCanvasElement;
    this.thCtx = this.th.getContext("2d")!;
    this.$("#hud-pause").addEventListener("click", onPause);
  }

  show(v: boolean): void {
    this.el.classList.toggle("hidden", !v);
  }

  toast(text: string, ms = 2600): void {
    const t = this.$("#hud-toast");
    t.textContent = text;
    t.classList.add("on");
    clearTimeout((t as unknown as { _h: number })._h);
    (t as unknown as { _h: number })._h = window.setTimeout(() => t.classList.remove("on"), ms);
  }

  update(player: Player, playerActive: boolean, follow: FollowInfo | null = null): void {
    const s = store.last;
    const init = store.init;
    if (!s || !init) return;
    if (init.mode === "demo") {
      this.updateDemo(follow);
      return;
    }
    this.$("#hud-follow").style.display = "none";
    this.$("#hud-mouse").style.display = playerActive && !player.mouseLocked ? "block" : "none";
    const phase = s.phase;
    const total = init.phases[phase] > 0 ? init.phases[phase] : 1;
    this.$("#hud-phase").textContent = PHASE_LABEL[phase];
    this.$("#hud-timer").textContent = fmtTime(s.phase_left);
    (this.$("#hud-bar") as HTMLElement).style.width = `${Math.max(0, Math.min(100, (s.phase_left / total) * 100))}%`;
    this.el.dataset.phase = phase;
    this.$("#hud-mode").textContent = init.mode === "watch"
      ? `AUTONOMOUS SWARM · ${init.swarm_seekers} seeker${init.swarm_seekers > 1 ? "s" : ""} + ${init.n_exts} extinguishers`
      : `YOU + ${init.bots.length} BOTS`;
    this.$("#hud-reticle").style.display = playerActive ? "block" : "none";

    // telemetry
    const me = s.drones.find((d) => d[8] && d[1] === (phase === "EXTINGUISH" ? "e" : "s"));
    const lim = phase === "EXTINGUISH" ? init.limits.ext : init.limits.seeker;
    const rel = (player.z - lim[0]) / (lim[1] - lim[0]);
    let tele = "";
    if (init.mode === "play" && playerActive) {
      const water = me ? me[6] : 0;
      const dropTooHigh = phase === "EXTINGUISH" && player.z > init.limits.drop_max_agl;
      tele += `
        <div class="tele-row"><span>ALT AGL</span><b>${player.z.toFixed(0)} m</b></div>
        <div class="alt-bar"><div style="height:${Math.max(0, Math.min(100, rel * 100))}%"></div><i>${lim[1]}</i><i class="lo">${lim[0]}</i></div>
        <div class="tele-row"><span>SPEED</span><b>${player.speed.toFixed(1)} m/s</b></div>
        <div class="tele-row"><span>HDG</span><b>${player.yaw.toFixed(0).padStart(3, "0")}°</b></div>`;
      if (phase === "SEEK") {
        const swath = 2 * player.z * Math.tan((init.camera.hfov * Math.PI) / 360);
        const gsd = (swath / init.camera.w) * 100;
        tele += `<div class="tele-row"><span>SWATH</span><b>${swath.toFixed(0)} m</b></div>
                 <div class="tele-row"><span>GSD</span><b>${gsd.toFixed(1)} cm/px</b></div>
                 <div class="tele-row"><span>COVERED</span><b>${store.covPct.toFixed(0)}%</b></div>
                 <div class="tele-row"><span>HOTSPOTS</span><b>${store.hotspots.length}</b></div>`;
      } else if (phase === "EXTINGUISH") {
        const pct = (water / init.water_capacity) * 100;
        tele += `<div class="tele-row"><span>WATER</span><b>${water.toFixed(0)} L</b></div>
                 <div class="water-bar"><div style="width:${pct}%"></div></div>
                 <div class="tele-row ${dropTooHigh ? "warn" : ""}"><span>DROP</span><b>${dropTooHigh ? "too high" : player.dropping && water > 0 ? "releasing" : "ready"}</b></div>
                 ${me && me[7] === "REFILL" ? `<div class="tele-row good"><span>PAD</span><b>refilling</b></div>` : ""}`;
      }
    } else {
      tele += `<div class="tele-row"><span>COVERED</span><b>${store.covPct.toFixed(0)}%</b></div>
               <div class="tele-row"><span>HOTSPOTS</span><b>${store.hotspots.length}</b></div>`;
    }
    if (store.baseM2 > 0) {
      const pct = (store.fireM2 / store.baseM2) * 100;
      tele += `<div class="sep"></div>
        <div class="tele-row"><span>FIRE NOW</span><b>${store.fireM2.toFixed(0)} m²</b></div>
        <div class="tele-row"><span>VS NO ACTION</span><b class="${pct < 90 ? "good" : ""}">${pct.toFixed(0)}%</b></div>
        <div class="tele-row"><span>WATER OUT</span><b>${store.water} L</b></div>`;
    }
    this.$("#hud-tele").innerHTML = tele;

    // feed: autonomy alerts / leader log
    const feed = this.$("#hud-feed");
    if (store.log.length && (init.mode === "watch" || phase === "EXTINGUISH")) {
      feed.style.display = "block";
      feed.innerHTML = `<div class="feed-title">autonomy · leader</div>` + store.log.slice(-5).map((l) => `<div class="feed-line">${escapeHtml(l)}</div>`).join("");
    } else feed.style.display = "none";

    // hint
    const hints: Record<string, string> = {
      SEEK: "W A S D move · Q E / mouse turn · R F altitude · Shift boost · click to capture mouse · Esc pause",
      EXTINGUISH: "Fly over fire, get below 30 m, hold Space to drop · land on a base pad to refill · Esc pause",
      BRIEFING: "Press Enter to start now",
    };
    this.$("#hud-hint").textContent = init.mode === "play" ? hints[phase] ?? "" : "Drag to orbit · scroll to zoom · Tab cycle drones · Esc pause";

    // minimap
    const size = this.mini.width;
    drawMap(this.miniCtx, size, {
      reveal: init.mode === "play" && phase !== "SCORE",
      showFire: init.mode === "watch" || phase === "SCORE",
      hotspots: true,
      drones: true,
      playerPose: init.mode === "play" && playerActive ? { x: player.x, y: player.y, yaw: player.yaw } : null,
      footprint: phase === "SEEK" ? 2 * player.z * Math.tan((init.camera.hfov * Math.PI) / 360) : undefined,
    });
    this.$("#hud-mini-label").textContent = init.mode === "play" ? "known map (perception)" : "swarm map + true fire";

    // thermal
    const showTh = this.showThermal && store.thermal && (phase === "SEEK" || (init.mode === "watch" && phase !== "SCORE"));
    this.$("#hud-thermal").style.display = showTh ? "block" : "none";
    if (showTh && store.thermal && store.thermal.t !== this.lastThermalT) this.drawThermal(store.thermal);
  }

  private updateDemo(follow: FollowInfo | null): void {
    const s = store.last!;
    const init = store.init!;
    const d = store.demo;
    this.$("#hud-phase").textContent = PHASE_LABEL[s.phase];
    this.$("#hud-timer").textContent = fmtTime(s.phase_left);
    const total = init.phases.DEMO > 0 ? init.phases.DEMO : 1;
    (this.$("#hud-bar") as HTMLElement).style.width = `${Math.max(0, Math.min(100, (s.phase_left / total) * 100))}%`;
    this.el.dataset.phase = "EXTINGUISH";
    const wet = init.lakes.some((l) => l[1]);
    this.$("#hud-mode").textContent = `DEMO · ${init.n_seekers} SEEKERS + ${init.n_exts} ATTACKERS · ${wet ? `${init.lakes.length} LAKES` : "NO LAKES (BASE ONLY)"}`;
    this.$("#hud-reticle").style.display = "none";
    this.$("#hud-mouse").style.display = "none";

    let tele = `<div class="fought"><div class="eyebrow">Fire fought</div><div class="big">${d ? d.fire_fought_pct.toFixed(0) : 0}<span>% vs no drones</span></div></div>`;
    if (d) {
      tele += `
        <div class="tele-row"><span>COVERED</span><b>${store.covPct.toFixed(0)}%</b></div>
        <div class="tele-row"><span>CONFIRMED FIRES</span><b>${d.confirmed}</b></div>
        <div class="tele-row"><span>IN QUEUE</span><b>${d.queue}</b></div>
        <div class="tele-row"><span>DROPS</span><b>${d.drops}</b></div>
        <div class="tele-row"><span>REFILLS · LAKE</span><b class="${d.fills_lake ? "good" : ""}">${d.fills_lake}</b></div>
        <div class="tele-row"><span>REFILLS · BASE</span><b>${d.fills_base}</b></div>
        <div class="tele-row"><span>VERIFY: OUT / HOT</span><b>${d.suppressed} / ${d.verify_hot}</b></div>
        <div class="sep"></div>
        <div class="tele-row"><span>CLOSE-RANGE LOOKS</span><b>${d.investigated}</b></div>
        <div class="tele-row"><span>· CONFIRMED / REJECTED</span><b>${d.confirmed_investigate} / ${d.rejected_investigate}</b></div>
        <div class="tele-row"><span>· GROUND REVIEW</span><b>${d.reviews}</b></div>
        <div class="tele-row"><span>· OPERATOR FIRE / NOT</span><b>${d.ground_fire} / ${d.ground_not_fire}</b></div>
        <div class="tele-row"><span>· NO ANSWER → AUTO</span><b>${d.review_timeouts}</b></div>
        <div class="sep"></div>
        <div class="tele-row"><span>FIRE NOW</span><b>${d.fire_m2.toFixed(0)} m²</b></div>
        <div class="tele-row"><span>NO-DRONE BASELINE</span><b>${d.baseline_m2.toFixed(0)} m²</b></div>
        <div class="tele-row"><span>FOAM OUT</span><b>${d.water_dropped_l} L</b></div>`;
    }
    this.$("#hud-tele").innerHTML = tele;

    const feed = this.$("#hud-feed");
    feed.style.display = "block";
    const rows = store.queue
      .map(([id, p, l]) => `<div class="queue-row"><span>${escapeHtml(id.replace("seeker-", "s"))}</span><i style="width:${Math.max(4, p * 100)}%"></i><span>${l.toFixed(0)} L</span></div>`)
      .join("");
    feed.innerHTML = `<div class="feed-title">priority queue · max-heap (size × growth × heat)</div>${rows || `<div class="feed-line">waiting for seekers to confirm fires…</div>`}` +
      (store.log.length ? `<div class="feed-title" style="margin-top:10px">leader</div>` + store.log.slice(-3).map((l) => `<div class="feed-line">${escapeHtml(l)}</div>`).join("") : "");

    const card = this.$("#hud-follow");
    const fid = follow?.id ?? null;
    if (fid) {
      const row = store.drones.find((r) => r[0] === fid);
      const seeker = row?.[1] === "s";
      const color = seeker ? "#b49cff" : "#4fd1ff";
      const act = store.acts[fid] ?? (row ? init.status_text[row[7]] ?? row[7] : "");
      const next = follow && follow.nextIn > 0
        ? `<div class="next">next drone in ${Math.ceil(follow.nextIn)} s<i><b style="width:${(follow.nextIn / follow.period) * 100}%"></b></i></div>`
        : `<div class="next">manual · Tab next · A auto</div>`;
      card.innerHTML = `<div class="swatch" style="color:${color};background:${color}"></div>
        <div><div class="who">${escapeHtml(fid)}<small>${seeker ? "seeker · thermal search" : "attacker · field refill"}</small></div>
        <div class="what">${escapeHtml(act)}</div></div>${next}`;
      card.style.display = "flex";
    } else card.style.display = "none";

    this.$("#hud-hint").textContent = "Camera hops drone every 10 s · Tab next · A auto on/off · F free camera · drag orbit · scroll zoom · Esc pause";

    const size = this.mini.width;
    drawMap(this.miniCtx, size, { reveal: false, showFire: true, hotspots: true, drones: true, zones: true, links: true, highlight: fid });
    this.$("#hud-mini-label").textContent = "swarm map · true fire · lakes";

    const showTh = this.showThermal && store.thermal && s.phase !== "SCORE";
    this.$("#hud-thermal").style.display = showTh ? "block" : "none";
    if (showTh && store.thermal && store.thermal.t !== this.lastThermalT) this.drawThermal(store.thermal);
  }

  private drawThermal(th: ThermalMsg): void {
    this.lastThermalT = th.t;
    const bytes = b64bytes(th.data);
    if (!this.thImg || this.thImg.width !== th.w) {
      this.th.width = th.w;
      this.th.height = th.h;
      this.thImg = this.thCtx.createImageData(th.w, th.h);
    }
    const d = this.thImg.data;
    for (let i = 0; i < bytes.length; i++) {
      const v = bytes[i];
      d[i * 4] = LUT[v * 3];
      d[i * 4 + 1] = LUT[v * 3 + 1];
      d[i * 4 + 2] = LUT[v * 3 + 2];
      d[i * 4 + 3] = 255;
    }
    this.thCtx.putImageData(this.thImg, 0, 0);
    this.thCtx.lineWidth = 1;
    for (const [u, v, conf] of th.boxes) {
      this.thCtx.strokeStyle = "#7dffb2";
      this.thCtx.strokeRect(u - 6, v - 6, 12, 12);
      this.thCtx.fillStyle = "#7dffb2";
      this.thCtx.font = "8px JetBrains Mono, monospace";
      this.thCtx.fillText(`${Math.round(conf * 100)}`, u + 7, v - 4);
    }
    this.$("#hud-tmax").textContent = `${th.id ? `${th.id} · ` : ""}${th.lo.toFixed(0)}–${th.hi.toFixed(0)} °C · max ${th.tmax.toFixed(0)} °C`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
