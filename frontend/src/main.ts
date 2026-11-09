import "./style.css";
import { Audio } from "./audio";
import { Hud } from "./hud";
import { Net } from "./net";
import { PlanScreen } from "./plan";
import { Player } from "./player";
import { Screens, type StartOpts } from "./screens";
import { absorb, resetStore, store } from "./store";
import type { InitMsg, Phase, ServerMsg, StateMsg } from "./types";
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

let phase: Phase | null = null;
let mode: "play" | "watch" = "play";
let inGame = false;
let paused = false;
let lastOpts: StartOpts = screens.opts;
let followIdx = 0;

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
  .then((d) => screens.setMaps(d.maps))
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
  document.exitPointerLock?.();
  screens.only(screens.menu);
};
screens.onWatch = () => {
  lastOpts = { ...lastOpts, mode: "watch" };
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
  screens.only(screens.loading);
  net.send({ type: "start", mode: o.mode, map: o.map, seed: o.seed, seekers: o.seekers, fast: qp.get("fast") === "1" });
}

function onInit(m: InitMsg): void {
  resetStore(m);
  mode = m.mode;
  world.build(m);
  world.useOrbit = true;
  world.followId = null;
  world.camera.position.set(0, 230, 260);
  world.orbit.target.set(0, 0, 0);
  inGame = true;
  phase = null;
}

function onState(s: StateMsg): void {
  if (!inGame || !store.init) return;
  const { fireChanged, burnt } = absorb(s);
  world.setDrones(s.drones, mode, s.t);
  if (fireChanged && store.fire) world.setFire(store.fire, store.fireN);
  if (burnt) world.setBurnt(burnt);
  if (s.hotspots) world.setHotspots(s.hotspots, s.t);
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

function enterPhase(p: Phase, s: StateMsg): void {
  const prev = phase;
  phase = p;
  const init = store.init!;
  if (prev !== null) audio.chime(p === "SCORE");
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
      hud.toast("Seek: fly the seeker — the map fills in where the camera looks");
    } else {
      world.followId = "seeker-1";
      world.camera.position.set(init.base[0] + 60, 160, -init.base[1] + 140);
      hud.toast("Swarm seekers: lawnmower coverage over stub zones");
    }
    return;
  }
  if (p === "PLAN") {
    player.enabled = false;
    exitFpv();
    world.followId = null;
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
      world.followId = null;
      world.camera.position.set(0, 210, 230);
      world.orbit.target.set(0, 0, 0);
      hud.toast("Stub leader assigns tasks · autonomy plans, flies, drops and verifies");
    }
    return;
  }
  if (p === "SCORE") {
    player.enabled = false;
    exitFpv();
    hud.show(false);
    world.camera.position.set(0, 260, 260);
    world.orbit.target.set(0, 0, 0);
    screens.showScore();
  }
}

function enterFpv(): void {
  player.enabled = true;
  world.useOrbit = false;
  world.orbit.enabled = false;
  world.hidePlayerMesh = true;
  player.lock();
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
  if (!inGame || paused || phase === "SCORE") return;
  paused = true;
  pausedAt = performance.now();
  net.send({ type: "pause" });
  player.enabled = false;
  screens.only(screens.pause);
}

function resume(): void {
  paused = false;
  net.send({ type: "resume" });
  screens.only(null);
  if (mode === "play" && (phase === "SEEK" || phase === "EXTINGUISH")) {
    player.enabled = true;
    player.lock();
  }
}

// ---------------------------------------------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (e.code === "Escape") {
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
  } else if (e.code === "Tab" && mode === "watch" && inGame) {
    e.preventDefault();
    const ids = world.droneIds();
    followIdx = (followIdx + 1) % (ids.length + 1);
    world.followId = followIdx === ids.length ? null : ids[followIdx];
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
    player.update(dt);
    if (!world.useOrbit) player.applyCamera(world.camera);
    world.update(dt, t);
    plan.draw();
    hudT += dt;
    if (hudT > 0.066) {
      hudT = 0;
      hud.update(player, mode === "play" && player.enabled);
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

// ?auto=play|watch&map=p1&seed=1&seekers=1 starts straight away (demo kiosk / screenshots)
const qp = new URLSearchParams(location.search);
const auto = qp.get("auto");
if (auto === "play" || auto === "watch") {
  lastOpts = { mode: auto, map: qp.get("map") ?? "p1", seed: Number(qp.get("seed") ?? 1), seekers: Number(qp.get("seekers") ?? 1) };
  start(lastOpts);
}
