import "./style.css";
import { Audio } from "./audio";
import { type FollowInfo, Hud } from "./hud";
import { Net } from "./net";
import { PlanScreen } from "./plan";
import { Player } from "./player";
import { Screens, type StartOpts } from "./screens";
import { absorb, resetStore, store } from "./store";
import type { InitMsg, Mode, Phase, ServerMsg, StateMsg } from "./types";
import { World3D } from "./world3d";

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const ui = document.getElementById("ui") as HTMLElement;

const world = new World3D(canvas);
const player = new Player(canvas);
const net = new Net();
const audio = new Audio();
const screens = new Screens(ui);
const hud = new Hud(ui, () => pause());
const plan = new PlanScreen(ui);
const qp = new URLSearchParams(location.search);

let phase: Phase | null = null;
let mode: Mode = "play";
let inGame = false;
let paused = false;
let lastOpts: StartOpts = screens.opts;
let followIdx = 0;

// DEMO camera: hop to another drone every FOLLOW_PERIOD seconds (seekers every 3rd hop, otherwise a busy attacker)
const FOLLOW_PERIOD = 10;
let autoFollow = true;
let followLeft = FOLLOW_PERIOD;
let hop = 0;
let lastExt = "";
let lastSeeker = "";

// ---------------------------------------------------------------------------------------------------------------
net.onStatus = (ok) => screens.setStatus(ok ? "sim server connected" : "connecting to sim server… (python -m sim.server)", ok);
net.onMessage = (m: ServerMsg) => {
  if (m.type === "loading") {
    screens.only(screens.loading);
  } else if (m.type === "init") {
    onInit(m);
  } else if (m.type === "state") {
    onState(m);
  } else if (m.type === "error") {
    hud.toast(`Sim error: ${m.message}`, 8000);
    console.error(m.message);
  }
};
net.connect();
fetch("/api/maps")
  .then((r) => r.json())
  .then((d) => screens.setMaps(d.maps, d.demo_maps ?? []))
  .catch(() => screens.setMaps([{ id: "p1", title: "Hanna Hammock · test plot", available: true }, { id: "synth", title: "Synthetic", available: true }]));

screens.onStart = (o) => {
  audio.start();
  lastOpts = o;
  start(o);
};
screens.onResume = () => resume();
screens.onRestart = (newSeed) => {
  const seed = newSeed ? Math.floor(Math.random() * 9999) : lastOpts.seed;
  lastOpts = { ...lastOpts, seed };
  start(lastOpts);
};
screens.onMenu = () => {
  net.send({ type: "stop" });
  inGame = false;
  paused = false;
  phase = null;
  player.enabled = false;
  hud.show(false);
  plan.close();
  world.follow(null);
  document.exitPointerLock?.();
  screens.only(screens.menu);
};
screens.onWatch = () => {
  lastOpts = { ...lastOpts, mode: "watch" };
  start(lastOpts);
};
screens.onDemoMap = (map) => {
  lastOpts = { ...lastOpts, mode: "demo", map };
  start(lastOpts);
};
screens.onSkip = () => net.send({ type: "skip" });
plan.onChange = (p) => {
  const out: Record<string, object> = {};
  for (const [id, a] of p) out[id] = a.hotspot ? { hotspot: a.hotspot } : { x: a.x, y: a.y };
  net.send({ type: "plan", plan: out });
};
plan.onLaunch = () => net.send({ type: "skip" });

function start(o: StartOpts): void {
  inGame = false;
  paused = false;
  phase = null;
  player.enabled = false;
  plan.close();
  hud.show(false);
  world.follow(null);
  screens.only(screens.loading);
  net.send({ type: "start", mode: o.mode, map: o.map, seed: o.seed, seekers: o.seekers, fast: qp.get("fast") === "1" });
}

/** Orbit camera looking at world point (x, y) from `back` metres south and `up` metres above. */
function lookAt(x: number, y: number, up: number, back: number): void {
  world.orbit.target.set(x, 0, -y);
  world.camera.position.set(x, up, -y + back);
  world.orbit.update();
}

function onInit(m: InitMsg): void {
  resetStore(m);
  mode = m.mode;
  world.build(m);
  world.useOrbit = true;
  world.orbit.enabled = true;
  world.follow(null);
  const R = m.arena_radius;
  if (mode === "demo") lookAt(m.base[0], m.base[1] + 60, 110, 120);
  else if (mode === "watch") lookAt(m.base[0], m.base[1] + 30, 70, 95);
  else lookAt(0, 0, R * 0.9, R * 1.05);
  inGame = true;
  phase = null;
  autoFollow = true;
  followLeft = 3; // first hop shortly after the start, once the drones are moving
  hop = 0;
}

function onState(s: StateMsg): void {
  if (!inGame || !store.init) return;
  const { fireChanged, burnt } = absorb(s);
  world.setDrones(s.drones, mode, s.t);
  if (fireChanged && store.fire) world.setFire(store.fire, store.fireN);
  if (burnt) world.setBurnt(burnt);
  if (s.hotspots) world.setHotspots(s.hotspots, s.t);
  if (s.links) world.setLinks(s.links);
  if (s.drops) {
    world.addDrops(s.drops, (x, y) => {
      let best = 15, bd = Infinity;
      for (const d of s.drones) {
        const dd = Math.hypot(d[2] - x, d[3] - y);
        if (d[1] === "e" && dd < bd) {
          bd = dd;
          best = d[4];
        }
      }
      return best;
    });
  }
  if (s.phase !== phase) enterPhase(s.phase, s);
}

// ---- DEMO follow camera ------------------------------------------------------------------------------------------
function setFollow(id: string | null): void {
  const row = store.drones.find((r) => r[0] === id);
  world.follow(id, row?.[1] === "s" ? 120 : 55);
  store.followId = id;
  net.send({ type: "follow", id });
  followLeft = FOLLOW_PERIOD;
}

function nextAfter(ids: string[], last: string): string | undefined {
  if (!ids.length) return undefined;
  const sorted = [...ids].sort();
  return sorted.find((x) => x > last) ?? sorted[0];
}

function nextFollow(): void {
  const rows = store.drones;
  if (!rows.length) return;
  hop++;
  const seekers = rows.filter((r) => r[1] === "s").map((r) => r[0]);
  const exts = rows.filter((r) => r[1] === "e");
  const busy = exts.filter((r) => r[7] !== "IDLE").map((r) => r[0]);
  let id: string | undefined;
  if (hop % 3 === 1 || !exts.length) {
    id = nextAfter(seekers, lastSeeker);
    if (id) lastSeeker = id;
  } else {
    id = nextAfter(busy.length ? busy : exts.map((r) => r[0]), lastExt);
    if (id) lastExt = id;
  }
  setFollow(id ?? null);
}

function followInfo(): FollowInfo {
  return { id: store.followId, nextIn: autoFollow ? followLeft : 0, period: FOLLOW_PERIOD };
}

// ---------------------------------------------------------------------------------------------------------------
function enterPhase(p: Phase, s: StateMsg): void {
  const prev = phase;
  phase = p;
  const init = store.init!;
  if (prev !== null) audio.chime(p === "SCORE");
  if (p === "DEMO") {
    screens.only(null);
    hud.show(true);
    hud.toast(`DEMO · ${init.n_seekers} seekers search while ${init.n_exts} attackers fight — camera hops drone every ${FOLLOW_PERIOD} s`, 4200);
    return;
  }
  if (p === "BRIEFING") {
    hud.show(false);
    if (mode === "watch") {
      net.send({ type: "skip" });
      return;
    }
    screens.showBriefing(init.phases.SEEK);
    return;
  }
  if (p === "SEEK") {
    screens.only(null);
    hud.show(true);
    if (mode === "play") {
      const me = s.drones.find((d) => d[8] && d[1] === "s");
      const [cruise, max, accel] = init.speeds.seeker;
      player.spawn(me ? me[2] : init.base[0], me ? me[3] : init.base[1], Math.max(40, me ? me[4] : 40), 0,
        { minZ: init.limits.seeker[0], maxZ: init.limits.seeker[1], cruise, max, accel, vz: 6, radius: init.arena_radius }, -32);
      enterFpv();
      hud.toast("Seek: fly the seeker — the map fills in where the camera looks · click the view to steer with the mouse");
    } else {
      world.follow("seeker-1", 120);
      hud.toast("Swarm seekers: lawnmower coverage over equal-time zones (core/zones.py)");
    }
    return;
  }
  if (p === "PLAN") {
    player.enabled = false;
    exitFpv();
    world.follow(null);
    hud.show(false);
    plan.open(mode === "watch");
    return;
  }
  if (p === "EXTINGUISH") {
    plan.close();
    hud.show(true);
    if (mode === "play") {
      const me = s.drones.find((d) => d[8] && d[1] === "e");
      const [cruise, max, accel] = init.speeds.ext;
      player.spawn(me ? me[2] : init.base[0], me ? me[3] : init.base[1], 20, 0,
        { minZ: 3, maxZ: init.limits.ext[1], cruise, max, accel, vz: 5, radius: init.arena_radius }, -24);
      enterFpv();
      hud.toast("Extinguish: get below 30 m over fire and hold Space");
    } else {
      world.follow(null);
      const R = init.arena_radius;
      lookAt(0, -R * 0.15, R * 0.55, R * 0.7);
      hud.toast("Swarm brain: priority queue → nearest refill-aware extinguisher · autonomy plans, flies, drops, verifies");
    }
    return;
  }
  if (p === "SCORE") {
    player.enabled = false;
    exitFpv();
    hud.show(false);
    world.follow(null);
    const R = init.arena_radius;
    lookAt(0, 0, R * 1.0, R * 1.0);
    screens.showScore();
  }
}

function enterFpv(): void {
  player.enabled = true;
  world.useOrbit = false;
  world.orbit.enabled = false;
  world.hidePlayerMesh = true;
  player.lock(); // may be refused (not a user gesture): click the view, or drag, to steer with the mouse
}

function exitFpv(): void {
  world.useOrbit = true;
  world.orbit.enabled = true;
  world.hidePlayerMesh = false;
  if (document.pointerLockElement) document.exitPointerLock();
  // orbit from where the player was
  world.orbit.target.set(player.x, 0, -player.y);
  world.camera.position.set(player.x, Math.max(player.z, 60) + 80, -player.y + 120);
}

let pausedAt = 0;
function pause(): void {
  if (!inGame || paused || phase === "SCORE" || phase === null) return;
  paused = true;
  pausedAt = performance.now();
  net.send({ type: "pause" });
  player.enabled = false;
  if (document.pointerLockElement) document.exitPointerLock(); // the menu needs a visible cursor
  screens.only(screens.pause);
}

function resume(): void {
  if (!paused) return;
  paused = false;
  net.send({ type: "resume" });
  if (phase === "BRIEFING" && mode === "play") screens.showBriefing(store.init!.phases.SEEK);
  else screens.only(null);
  if (mode === "play" && (phase === "SEEK" || phase === "EXTINGUISH")) {
    player.enabled = true;
    player.lock();
  }
}

// ---------------------------------------------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (e.code === "Escape" || e.code === "KeyP") {
    // the same Esc may also have released pointer lock (which already paused): don't immediately un-pause
    if (paused && performance.now() - pausedAt > 400) resume();
    else if (!paused) pause();
  } else if (e.code === "Enter" && (phase === "BRIEFING" || phase === "PLAN") && !paused) {
    net.send({ type: "skip" });
  } else if (e.code === "KeyT") {
    hud.showThermal = !hud.showThermal;
  } else if (e.code === "KeyM") {
    audio.setMuted(!audio.muted);
    hud.toast(audio.muted ? "sound off" : "sound on", 1200);
  } else if (e.code === "Tab" && mode === "demo" && inGame) {
    e.preventDefault();
    nextFollow();
  } else if (e.code === "KeyA" && mode === "demo" && inGame) {
    autoFollow = !autoFollow;
    followLeft = FOLLOW_PERIOD;
    if (autoFollow && !store.followId) nextFollow();
    hud.toast(autoFollow ? `auto camera: new drone every ${FOLLOW_PERIOD} s` : "auto camera off (Tab = next drone)", 1500);
  } else if (e.code === "KeyF" && mode === "demo" && inGame) {
    autoFollow = false;
    setFollow(null);
    hud.toast("free camera · A = auto follow", 1500);
  } else if (e.code === "Tab" && mode === "watch" && inGame) {
    e.preventDefault();
    const ids = world.droneIds();
    followIdx = (followIdx + 1) % (ids.length + 1);
    world.follow(followIdx === ids.length ? null : ids[followIdx], null);
    hud.toast(world.followId ? `following ${world.followId}` : "free camera", 1200);
  }
});
document.addEventListener("pointerlockchange", () => {
  // Esc releases pointer lock before keydown reaches us: treat an unexpected unlock during flight as pause
  if (!document.pointerLockElement && player.enabled && inGame && !paused) pause();
});
canvas.addEventListener("click", () => {
  if (player.enabled) player.lock();
});

setInterval(() => {
  if (inGame && player.enabled && !paused) net.send(player.pose());
}, 50);

// ---------------------------------------------------------------------------------------------------------------
let last = performance.now();
let hudT = 0;
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const t = now / 1000;
  if (inGame) {
    if (mode === "demo" && phase === "DEMO" && autoFollow && !paused) {
      followLeft -= dt;
      if (followLeft <= 0) nextFollow();
    }
    player.update(dt);
    if (!world.useOrbit) player.applyCamera(world.camera);
    world.update(dt, t);
    plan.draw();
    hudT += dt;
    if (hudT > 0.066) {
      hudT = 0;
      hud.update(player, mode === "play" && player.enabled, mode === "demo" ? followInfo() : null);
    }
    const flying = mode === "play" && player.enabled;
    audio.rotor(flying, player.speed, phase === "EXTINGUISH");
    const me = store.drones.find((d) => d[8] && d[1] === "e");
    audio.waterOn(flying && phase === "EXTINGUISH" && player.dropping && !!me && me[6] > 0);
  } else {
    // idle menu: slow orbit
    world.camera.position.x = Math.cos(t * 0.05) * 320;
    world.camera.position.z = Math.sin(t * 0.05) * 320;
    world.camera.position.y = 180;
    world.camera.lookAt(0, 0, 0);
  }
  world.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
screens.only(screens.menu);

// ?auto=play|watch|demo&map=p1|demo-lakes-1&seed=1&seekers=1 starts straight away (demo kiosk / screenshots)
const auto = qp.get("auto");
if (auto === "play" || auto === "watch" || auto === "demo") {
  const map = qp.get("map") ?? (auto === "demo" ? "demo-lakes-1" : "p1");
  lastOpts = { mode: auto, map, seed: Number(qp.get("seed") ?? 1), seekers: Number(qp.get("seekers") ?? 1) };
  start(lastOpts);
}
