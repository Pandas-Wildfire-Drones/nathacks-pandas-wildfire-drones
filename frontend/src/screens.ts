import { store } from "./store";
import type { Mode, ScoreSide } from "./types";

export interface StartOpts {
  mode: Mode;
  map: string;
  seed: number;
  seekers: number;
}

export interface MapInfo {
  id: string;
  title: string;
  available: boolean;
  wet?: boolean;
}

function el(html: string): HTMLElement {
  const d = document.createElement("div");
  d.innerHTML = html.trim();
  return d.firstElementChild as HTMLElement;
}

export class Screens {
  menu: HTMLElement;
  brief: HTMLElement;
  pause: HTMLElement;
  score: HTMLElement;
  loading: HTMLElement;
  opts: StartOpts = { mode: "play", map: "p1", seed: 1, seekers: 1 };
  onStart: (o: StartOpts) => void = () => {};
  onResume: () => void = () => {};
  onRestart: (newSeed: boolean) => void = () => {};
  onMenu: () => void = () => {};
  onWatch: () => void = () => {};
  onSkip: () => void = () => {};
  onDemoMap: (mapId: string) => void = () => {};
  private maps: MapInfo[] = [];
  private demoMaps: MapInfo[] = [];
  private mapByMode: Record<string, string> = { play: "p1", watch: "p1", demo: "demo-lakes-1" };

  constructor(ui: HTMLElement) {
    this.menu = el(`
      <div class="screen menu">
        <div class="menu-card panel">
          <div class="eyebrow">natHacks 2026 · Team Pandas</div>
          <h1>Night Initial Attack</h1>
          <p class="lede">A night of spot fires. Find them with a thermal seeker, plan, then knock them down with a
            fleet of heavy extinguisher drones. Then see if our autonomous swarm did better on the exact same map.
            Or run <b>DEMO</b>: a whole cluster, seekers and field-refill attackers at once, lakes as refill points.</p>
          <div class="field"><label>Mode</label>
            <div class="seg" data-k="mode">
              <button data-v="play" class="on">Play <small>you + 18 bots</small></button>
              <button data-v="watch">Watch swarm <small>fully autonomous</small></button>
              <button data-v="demo">DEMO <small>large map · 48 drones</small></button>
            </div></div>
          <div class="field"><label>Map</label><div class="seg" data-k="map" id="menu-maps"></div></div>
          <div class="row2">
            <div class="field"><label>Seed</label>
              <div class="seed"><input id="menu-seed" type="number" min="0" value="1" /><button class="btn ghost" id="menu-dice" title="Random seed">⚄</button></div></div>
            <div class="field" id="menu-seekers-field"><label>Swarm seekers</label>
              <div class="seg small" data-k="seekers">
                <button data-v="1" class="on">1</button><button data-v="2">2</button><button data-v="3">3</button><button data-v="4">4</button>
              </div></div>
          </div>
          <button class="btn primary wide" id="menu-go">Start ▸</button>
          <div class="status" id="menu-status">connecting to sim server…</div>
        </div>
        <div class="menu-info panel">
          <h3>What's real here</h3>
          <ul>
            <li><b>Thermal world</b> built from FLAME 3 NADIR radiometric frames (Hanna Hammock): real ground, real char, real fire pixels.</li>
            <li><b>Perception</b>: every seeker frame (640×512, rendered from that data) runs our threshold proposals + CNN verifier (ONNX), georeferencing, tracker and log-odds map. The map you plan on is what it actually saw.</li>
            <li><b>Autonomy</b>: extinguishers execute through our mission planner, CBS paths, path claims, drop safety gate and refill flow. Swarm seekers fly our lawnmower coverage.</li>
            <li><b>Swarm brain</b> (<code>core/</code>): fires go on a max-heap by size × growth × heat; each goes to the extinguisher with the lowest refill-aware ETA (drone → best refill point → fire); failed VERIFY puts the fire back on the heap; seekers get equal-time zones.</li>
            <li><b>DEMO</b>: small attackers mix foam into water from any lake or the base and refill after every drop. Same map with dry basins for comparison.</li>
          </ul>
          <h3>Scoring</h3>
          <p class="muted">% fire left vs. the same map with no drones at all (100%). Lower is better.</p>
        </div>
      </div>`);
    this.brief = el(`
      <div class="screen brief hidden">
        <div class="panel brief-card">
          <div class="eyebrow">Briefing</div>
          <h2 id="brief-title">Seek · 75 s</h2>
          <div class="brief-grid">
            <div><b>1 · Seek</b><span>Fly the only seeker. Wherever its camera looks, the map fills in and our detector finds hotspots. Higher = wider view, coarser pixels.</span></div>
            <div><b>2 · Plan</b><span>30 s on the map you revealed. Send your bots to fires or areas.</span></div>
            <div><b>3 · Extinguish</b><span>Fly one extinguisher yourself. Get low, hold Space over fire. Refill on the base pads.</span></div>
          </div>
          <div class="keys">
            <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
            <span><kbd>Q</kbd><kbd>E</kbd> / mouse turn</span>
            <span><kbd>R</kbd><kbd>F</kbd> altitude</span>
            <span><kbd>Shift</kbd> boost</span>
            <span><kbd>Space</kbd> drop water</span>
            <span><kbd>T</kbd> thermal view</span>
            <span><kbd>Esc</kbd> pause</span>
          </div>
          <button class="btn primary" id="brief-go">Start now (Enter)</button>
        </div>
      </div>`);
    this.pause = el(`
      <div class="screen pause hidden">
        <div class="panel pause-card">
          <h2>Paused</h2>
          <button class="btn primary wide" id="pause-resume">Resume</button>
          <button class="btn wide" id="pause-restart">Restart (same map &amp; seed)</button>
          <button class="btn wide" id="pause-new">New seed</button>
          <button class="btn ghost wide" id="pause-menu">Main menu</button>
        </div>
      </div>`);
    this.score = el(`<div class="screen score hidden"></div>`);
    this.loading = el(`<div class="screen loading hidden"><div class="spinner"></div><div id="loading-text">Building the thermal world…</div></div>`);
    for (const e of [this.menu, this.brief, this.pause, this.score, this.loading]) ui.appendChild(e);
    this.wireMenu();
    this.pause.querySelector("#pause-resume")!.addEventListener("click", () => this.onResume());
    this.pause.querySelector("#pause-restart")!.addEventListener("click", () => this.onRestart(false));
    this.pause.querySelector("#pause-new")!.addEventListener("click", () => this.onRestart(true));
    this.pause.querySelector("#pause-menu")!.addEventListener("click", () => this.onMenu());
    this.brief.querySelector("#brief-go")!.addEventListener("click", () => this.onSkip());
  }

  private wireMenu(): void {
    this.menu.querySelectorAll<HTMLElement>(".seg").forEach((seg) => {
      seg.addEventListener("click", (e) => {
        const b = (e.target as HTMLElement).closest("button") as HTMLButtonElement | null;
        if (!b || b.disabled) return;
        seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        const k = seg.dataset.k as keyof StartOpts;
        const v = b.dataset.v!;
        (this.opts as unknown as Record<string, unknown>)[k] = k === "seekers" ? Number(v) : v;
        if (k === "map") this.mapByMode[this.opts.mode] = v;
        if (k === "mode") {
          this.opts.map = this.mapByMode[v];
          this.renderMaps();
        }
      });
    });
    const seed = this.menu.querySelector("#menu-seed") as HTMLInputElement;
    seed.addEventListener("change", () => (this.opts.seed = Math.max(0, Number(seed.value) || 0)));
    this.menu.querySelector("#menu-dice")!.addEventListener("click", () => {
      this.opts.seed = Math.floor(Math.random() * 9999);
      seed.value = String(this.opts.seed);
    });
    this.menu.querySelector("#menu-go")!.addEventListener("click", () => this.onStart({ ...this.opts }));
  }

  setMaps(maps: MapInfo[], demoMaps: MapInfo[] = []): void {
    this.maps = maps;
    this.demoMaps = demoMaps.length ? demoMaps : [
      { id: "demo-lakes-1", title: "DEMO · Lakes · Hanna Hammock 1", available: true, wet: true },
      { id: "demo-dry-1", title: "DEMO · Dry basins · Hanna Hammock 1", available: true, wet: false },
    ];
    this.renderMaps();
  }

  private renderMaps(): void {
    const box = this.menu.querySelector("#menu-maps")!;
    const demo = this.opts.mode === "demo";
    const list = demo ? this.demoMaps : this.maps;
    box.innerHTML = list
      .map((m) => `<button data-v="${m.id}" class="${m.id === this.opts.map ? "on" : ""}" ${m.available ? "" : "disabled title='dataset not on this machine'"}>${m.title.replace("DEMO · ", "")}</button>`)
      .join("");
    (this.menu.querySelector("#menu-seekers-field") as HTMLElement).style.visibility = demo ? "hidden" : "visible";
  }

  /** The other DEMO variant of the same layout (lakes <-> dry basins). */
  static otherDemoMap(id: string): string {
    return id.includes("lakes") ? id.replace("lakes", "dry") : id.replace("dry", "lakes");
  }

  setStatus(text: string, ok: boolean): void {
    const s = this.menu.querySelector("#menu-status") as HTMLElement;
    s.textContent = text;
    s.classList.toggle("ok", ok);
  }

  only(which: HTMLElement | null): void {
    for (const e of [this.menu, this.brief, this.pause, this.score, this.loading]) e.classList.toggle("hidden", e !== which);
  }

  showBriefing(seekS: number): void {
    (this.brief.querySelector("#brief-title") as HTMLElement).textContent = `Seek · ${seekS} s`;
    this.only(this.brief);
  }

  showScore(): void {
    const sc = store.score;
    const init = store.init;
    if (!sc || !init) return;
    if (sc.demo) {
      this.showDemoScore();
      return;
    }
    const side = (label: string, cls: string, s?: ScoreSide) =>
      s
        ? `<div class="score-side ${cls}">
            <div class="eyebrow">${label}</div>
            <div class="big">${s.fire_left_pct.toFixed(0)}<span>% fire left</span></div>
            <div class="bar"><div style="width:${Math.min(100, s.fire_left_pct)}%"></div></div>
            <table>
              <tr><td>Map covered by camera</td><td>${s.coverage_pct.toFixed(0)}%</td></tr>
              <tr><td>Fires detected</td><td>${s.fires_detected} / ${sc.fires_total}</td></tr>
              <tr><td>Decoys reported as fire</td><td>${s.decoys_reported} / ${sc.decoys_total}</td></tr>
              <tr><td>Water dropped</td><td>${s.water_dropped_l} L</td></tr>
              <tr><td>…landing on fire</td><td>${s.water_on_fire_l} L</td></tr>
              <tr><td>Autonomy tasks run</td><td>${s.tasks}</td></tr>
            </table>
          </div>`
        : "";
    let verdict = "";
    if (sc.human && sc.swarm) {
      const d = sc.human.fire_left_pct - sc.swarm.fire_left_pct;
      verdict = d > 1 ? `The swarm left <b>${d.toFixed(0)} points</b> less fire than you.` : d < -1 ? `You beat the swarm by <b>${(-d).toFixed(0)} points</b>.` : "Dead heat.";
    }
    this.score.innerHTML = `
      <div class="panel score-card">
        <div class="eyebrow">Results · ${init.map_title} · seed ${init.seed}</div>
        <h2>${verdict || "Autonomous swarm"}</h2>
        <div class="score-grid">${side(`You + ${init.bots.length} bots`, "human", sc.human)}${side(`Autonomous swarm · ${init.swarm_seekers} seeker${init.swarm_seekers > 1 ? "s" : ""}`, "swarm", sc.swarm)}</div>
        <p class="muted small">No-action baseline on the same map: ${sc.baseline_m2.toFixed(0)} m² burning (= 100%). Same seed, same fire physics, same perception and autonomy code on both sides.</p>
        <div class="score-actions">
          <button class="btn primary" id="sc-again">Play again</button>
          <button class="btn" id="sc-watch">Watch the swarm on this map</button>
          <button class="btn" id="sc-new">New seed</button>
          <button class="btn ghost" id="sc-menu">Menu</button>
        </div>
      </div>`;
    this.score.querySelector("#sc-again")!.addEventListener("click", () => this.onRestart(false));
    this.score.querySelector("#sc-new")!.addEventListener("click", () => this.onRestart(true));
    this.score.querySelector("#sc-watch")!.addEventListener("click", () => this.onWatch());
    this.score.querySelector("#sc-menu")!.addEventListener("click", () => this.onMenu());
    this.only(this.score);
  }

  private showDemoScore(): void {
    const init = store.init!;
    const d = store.score!.demo!;
    const other = Screens.otherDemoMap(init.map);
    const wet = !!d.wet;
    const fills = d.fills_lake + d.fills_base;
    this.score.innerHTML = `
      <div class="panel score-card demo-score">
        <div class="eyebrow">DEMO results · ${init.map_title} · seed ${init.seed}</div>
        <div class="big">${d.fire_fought_pct.toFixed(0)}<span>% fire fought vs. the same map with no drones</span></div>
        <div class="bar" style="margin:14px 0 6px"><div style="width:${Math.min(100, d.fire_fought_pct)}%"></div></div>
        <table>
          <tr><td>Drones</td><td>${d.seekers} seekers + ${d.attackers} attackers</td></tr>
          <tr><td>Map covered by seeker cameras</td><td>${(d.coverage_pct ?? 0).toFixed(0)}%</td></tr>
          <tr><td>Fires detected</td><td>${d.fires_detected ?? 0} / ${d.fires_total ?? 0}</td></tr>
          <tr><td>Decoys reported as fire</td><td>${d.decoys_reported ?? 0}</td></tr>
          <tr><td>Drops (≈15 L foam each)</td><td>${d.drops}</td></tr>
          <tr><td>Refills at lakes / at base</td><td>${d.fills_lake} / ${d.fills_base}${fills ? ` (${((100 * d.fills_lake) / fills).toFixed(0)}% lakes)` : ""}</td></tr>
          <tr><td>VERIFY: suppressed / still hot → back on the heap</td><td>${d.suppressed} / ${d.verify_hot}</td></tr>
          <tr><td>Close-range confirmations (seeker INVESTIGATE)</td><td>${d.investigated}: ${d.confirmed_investigate} fire, ${d.rejected_investigate} rejected</td></tr>
          <tr><td>Sent to ground review → operator fire / not fire / no answer</td><td>${d.reviews} → ${d.ground_fire} / ${d.ground_not_fire} / ${d.review_timeouts}</td></tr>
          <tr><td>Foam dropped / landing on fire</td><td>${d.water_dropped_l} L / ${d.water_on_fire_l} L</td></tr>
        </table>
        <p class="muted small">${wet ? "Lakes on: attackers refill at whichever point (lake or base) gives the shortest drone → refill → fire trip." : "No lakes (dry basins): every refill is a trip to the base, so the allocator degrades to highest-priority-first."} Same fires, seeds and perception on both variants — run the other one to compare.</p>
        <div class="score-actions">
          <button class="btn primary" id="sc-other">Run ${wet ? "without lakes" : "with lakes"} (same fires)</button>
          <button class="btn" id="sc-again">Replay</button>
          <button class="btn" id="sc-new">New seed</button>
          <button class="btn ghost" id="sc-menu">Menu</button>
        </div>
      </div>`;
    this.score.querySelector("#sc-other")!.addEventListener("click", () => this.onDemoMap(other));
    this.score.querySelector("#sc-again")!.addEventListener("click", () => this.onRestart(false));
    this.score.querySelector("#sc-new")!.addEventListener("click", () => this.onRestart(true));
    this.score.querySelector("#sc-menu")!.addEventListener("click", () => this.onMenu());
    this.only(this.score);
  }
}
