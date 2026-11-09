// Messages from the Python sim server (sim/game.py). World coords: ENU metres (x east, y north, z up).

export type Phase = "BRIEFING" | "SEEK" | "PLAN" | "EXTINGUISH" | "SCORE";
export type Mode = "play" | "watch";

export interface InitMsg {
  type: "init";
  mode: Mode;
  map: string;
  map_title: string;
  seed: number;
  real_data: boolean;
  swarm_seekers: number;
  arena_radius: number;
  base: [number, number];
  station: [number, number];
  pads: [number, number][];
  trees: [number, number, number, number, number][]; // x, y, h, r, kind
  decoys: [string, number, number, number, number][]; // kind, x, y, yaw, area
  ground: { n: number; cell: number; data: string };
  raster_cell: number;
  phases: Record<Phase, number>;
  limits: { seeker: [number, number]; ext: [number, number]; drop_max_agl: number };
  speeds: { seeker: [number, number, number]; ext: [number, number, number] };
  camera: { hfov: number; w: number; h: number };
  water_capacity: number;
  bots: string[];
  player_ext: string | null;
}

// [id, kind ("s"|"e"), x, y, z, yaw, water, status, isPlayer]
export type DroneRow = [string, "s" | "e", number, number, number, number, number, string, boolean];
// [id, x, y, area_m2, confidence, status(0 det,1 conf,2 assigned,3 suppressed,4 fp), litres still needed]
export type HotspotRow = [string, number, number, number, number, number, number];

export interface ThermalMsg {
  w: number;
  h: number;
  data: string;
  lo: number;
  hi: number;
  t: number;
  boxes: [number, number, number][];
  tmax: number;
}

export interface ScoreSide {
  fire_left_pct: number;
  fire_m2: number;
  coverage_pct: number;
  fires_detected: number;
  decoys_reported: number;
  water_dropped_l: number;
  water_on_fire_l: number;
  tasks: number;
}

export interface ScoreMsg {
  baseline_m2: number;
  fires_total: number;
  decoys_total: number;
  human?: ScoreSide;
  swarm?: ScoreSide;
}

export interface StateMsg {
  type: "state";
  t: number;
  phase: Phase;
  phase_left: number;
  paused: boolean;
  drones: DroneRow[];
  fire?: string;
  fire_n?: number;
  burnt?: string;
  cov?: string;
  cov_n?: number;
  cov_pct?: number;
  hotspots?: HotspotRow[];
  thermal?: ThermalMsg;
  drops?: [number, number, number, number, number][]; // seq, x, y, radius, litres
  log?: string[];
  fire_m2?: number;
  baseline_m2?: number;
  water_dropped?: number;
  plan?: [string, string][];
  score?: ScoreMsg;
}

export type ServerMsg = InitMsg | StateMsg | { type: "loading"; map: string } | { type: "error"; message: string };

export function b64bytes(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
