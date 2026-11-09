import type { DroneRow, HotspotRow, InitMsg, ScoreMsg, StateMsg, ThermalMsg } from "./types";
import { b64bytes } from "./types";

/** Latest decoded server state, shared by HUD, plan screen and score screen. */
export const store = {
  init: null as InitMsg | null,
  last: null as StateMsg | null,
  drones: [] as DroneRow[],
  hotspots: [] as HotspotRow[],
  cov: null as Uint8Array | null,
  covN: 0,
  covPct: 0,
  fire: null as Uint8Array | null,
  fireN: 0,
  thermal: null as ThermalMsg | null,
  log: [] as string[],
  fireM2: 0,
  baseM2: 0,
  water: 0,
  plan: [] as [string, string][],
  score: null as ScoreMsg | null,
  groundBytes: null as Uint8Array | null,
};

export function absorb(s: StateMsg): { fireChanged: boolean; burnt: Uint8Array | null } {
  store.last = s;
  store.drones = s.drones;
  let fireChanged = false;
  let burnt: Uint8Array | null = null;
  if (s.hotspots) store.hotspots = s.hotspots;
  if (s.cov && s.cov_n) {
    store.cov = b64bytes(s.cov);
    store.covN = s.cov_n;
  }
  if (s.cov_pct !== undefined) store.covPct = s.cov_pct;
  if (s.fire && s.fire_n) {
    store.fire = b64bytes(s.fire);
    store.fireN = s.fire_n;
    fireChanged = true;
  }
  if (s.burnt) burnt = b64bytes(s.burnt);
  if (s.thermal) store.thermal = s.thermal;
  if (s.log) store.log = s.log;
  if (s.fire_m2 !== undefined) store.fireM2 = s.fire_m2;
  if (s.baseline_m2 !== undefined) store.baseM2 = s.baseline_m2;
  if (s.water_dropped !== undefined) store.water = s.water_dropped;
  if (s.plan) store.plan = s.plan;
  if (s.score) store.score = s.score;
  return { fireChanged, burnt };
}

export function resetStore(init: InitMsg): void {
  store.init = init;
  store.last = null;
  store.drones = [];
  store.hotspots = [];
  store.cov = null;
  store.covN = 0;
  store.covPct = 0;
  store.fire = null;
  store.fireN = 0;
  store.thermal = null;
  store.log = [];
  store.plan = [];
  store.score = null;
  store.groundBytes = b64bytes(init.ground.data);
}
