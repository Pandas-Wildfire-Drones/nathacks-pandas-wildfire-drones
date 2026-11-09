import { drawMap, fromCanvas, toCanvas } from "./mapdraw";
import { fmtTime } from "./hud";
import { store } from "./store";

export type Plan = Map<string, { x: number; y: number; hotspot?: string }>;

/** Planning phase: the revealed map (black where the seeker never looked) + drag-free click assignment of bots. */
export class PlanScreen {
  el: HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private size = 640;
  plan: Plan = new Map();
  selected: string | null = null;
  readonly = false;
  onChange: (p: Plan) => void = () => {};
  onLaunch: () => void = () => {};
  private $ = (s: string) => this.el.querySelector(s) as HTMLElement;

  constructor(ui: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "plan hidden";
    this.el.innerHTML = `
      <div class="plan-side panel">
        <div class="eyebrow" id="plan-eyebrow">Planning · 30 s</div>
        <h2 id="plan-title">Send your extinguishers</h2>
        <p class="muted" id="plan-help">Pick a bot, then click a detected fire (direct task) or any spot (it will
          work the nearest known fires within 45 m). Unassigned bots stay at base. You fly one extinguisher yourself.</p>
        <div class="bots" id="plan-bots"></div>
        <div class="plan-actions">
          <button class="btn ghost" id="plan-auto">Spread over fires</button>
          <button class="btn ghost" id="plan-clear">Clear</button>
        </div>
        <div class="legend">
          <span><i style="background:#ff5a36"></i>confirmed hotspot</span>
          <span><i style="background:#ffd166"></i>detected (unconfirmed)</span>
          <span><i style="background:#03050a;border:1px solid #2a3550"></i>never seen</span>
        </div>
        <button class="btn primary wide" id="plan-launch">Launch now ▸</button>
      </div>
      <div class="plan-map">
        <div class="plan-timer" id="plan-timer">0:30</div>
        <canvas id="plan-canvas"></canvas>
        <div class="plan-stats" id="plan-stats"></div>
      </div>`;
    ui.appendChild(this.el);
    this.canvas = this.$("#plan-canvas") as HTMLCanvasElement;
    this.ctx = this.canvas.getContext("2d")!;
    this.canvas.addEventListener("click", (e) => this.click(e));
    this.$("#plan-launch").addEventListener("click", () => this.onLaunch());
    this.$("#plan-clear").addEventListener("click", () => {
      this.plan.clear();
      this.changed();
    });
    this.$("#plan-auto").addEventListener("click", () => this.autoSpread());
    window.addEventListener("resize", () => this.layout());
  }

  open(readonly: boolean): void {
    this.readonly = readonly;
    this.plan.clear();
    const bots = store.init?.bots ?? [];
    this.selected = readonly ? null : bots[0] ?? null;
    this.el.classList.toggle("readonly", readonly);
    this.$("#plan-title").textContent = readonly ? "Swarm brain is allocating" : "Send your extinguishers";
    this.$("#plan-help").textContent = readonly
      ? "Swarm brain (core/allocation.py): confirmed fires go on a max-heap by size × growth × heat; the highest goes to the extinguisher with the lowest refill-aware ETA, then the next."
      : "Pick a bot, then click a detected fire (direct task) or any spot (it works the nearest known fires within 45 m of it). Unassigned bots stay at base. You fly one extinguisher yourself.";
    this.el.classList.remove("hidden");
    this.layout();
    this.renderBots();
  }

  close(): void {
    this.el.classList.add("hidden");
  }

  private layout(): void {
    const s = Math.floor(Math.min(window.innerHeight * 0.82, window.innerWidth - 420, 760));
    this.size = Math.max(320, s);
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = this.size * dpr;
    this.canvas.height = this.size * dpr;
    this.canvas.style.width = `${this.size}px`;
    this.canvas.style.height = `${this.size}px`;
  }

  private renderBots(): void {
    const bots = store.init?.bots ?? [];
    const box = this.$("#plan-bots");
    if (this.readonly) {
      box.innerHTML = `<div class="muted small">${store.plan.length} tasks planned for ${bots.length} drones</div>`;
      return;
    }
    box.innerHTML = bots
      .map((b) => {
        const a = this.plan.get(b);
        const what = a ? (a.hotspot ? "→ fire" : "→ area") : "at base";
        return `<button class="bot ${this.selected === b ? "sel" : ""} ${a ? "set" : ""}" data-id="${b}"><b>${b.replace("ext-", "#")}</b><span>${what}</span></button>`;
      })
      .join("");
    box.querySelectorAll<HTMLButtonElement>(".bot").forEach((btn) =>
      btn.addEventListener("click", () => {
        this.selected = btn.dataset.id!;
        this.renderBots();
      }),
    );
  }

  private click(e: MouseEvent): void {
    if (this.readonly || !this.selected || !store.init) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * this.size;
    const py = ((e.clientY - rect.top) / rect.height) * this.size;
    const R = store.init.arena_radius;
    const [x, y] = fromCanvas(px, py, this.size, R);
    if (Math.hypot(x, y) > R) return;
    // snap to a hotspot if close
    let best: { id: string; x: number; y: number } | null = null;
    let bd = 14 * ((2 * R) / this.size);
    for (const h of store.hotspots) {
      if (h[5] === 3) continue;
      const d = Math.hypot(h[1] - x, h[2] - y);
      if (d < bd) {
        bd = d;
        best = { id: h[0], x: h[1], y: h[2] };
      }
    }
    this.plan.set(this.selected, best ? { x: best.x, y: best.y, hotspot: best.id } : { x, y });
    const bots = store.init.bots;
    const next = bots.find((b) => !this.plan.has(b));
    this.selected = next ?? this.selected;
    this.changed();
  }

  private autoSpread(): void {
    // convenience for the human: one bot per known hotspot, biggest first (still a human choice to press it)
    const bots = store.init?.bots ?? [];
    const hs = [...store.hotspots].filter((h) => h[5] === 1 || h[5] === 2).sort((a, b) => b[3] - a[3]);
    this.plan.clear();
    bots.forEach((b, i) => {
      const h = hs[i % Math.max(hs.length, 1)];
      if (h) this.plan.set(b, { x: h[1], y: h[2], hotspot: h[0] });
    });
    this.changed();
  }

  private changed(): void {
    this.renderBots();
    this.onChange(this.plan);
  }

  draw(): void {
    if (this.el.classList.contains("hidden") || !store.last || !store.init) return;
    const dpr = this.canvas.width / this.size;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const watch = store.init.mode === "watch";
    drawMap(this.ctx, this.size, {
      reveal: !watch,
      showFire: false,
      hotspots: true,
      drones: watch,
      anchors: watch ? undefined : this.plan,
      selected: this.selected,
      planLines: watch ? store.plan : undefined,
    });
    if (!watch && this.selected) {
      const [bx, by] = toCanvas(store.init.base[0], store.init.base[1] + 8, this.size, store.init.arena_radius);
      this.ctx.fillStyle = "rgba(255,255,255,0.85)";
      this.ctx.font = "600 12px Inter, sans-serif";
      this.ctx.textAlign = "center";
      this.ctx.fillText(`placing ${this.selected.replace("ext-", "bot #")}`, bx, by - 18);
    }
    this.$("#plan-timer").textContent = fmtTime(store.last.phase_left);
    const conf = store.hotspots.filter((h) => h[5] === 1 || h[5] === 2).length;
    this.$("#plan-stats").innerHTML = `<span><b>${store.covPct.toFixed(0)}%</b> of the map seen</span><span><b>${conf}</b> confirmed · <b>${store.hotspots.length - conf}</b> other hotspots</span>`;
  }
}
