// Messages from the Python sim server (sim/game.py). World coords: ENU metres (x east, y north, z up).

export type Phase = "BRIEFING" | "SEEK" | "PLAN" | "EXTINGUISH" | "DEMO" | "SCORE";
export type Mode = "play" | "watch" | "demo";

// [id, wet, nominal radius m, outline [[x, y], ...]]
export type LakeRow = [string, boolean, number, [number, number][]];

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
  lakes: LakeRow[];
  n_seekers: number;
  n_exts: number;
  zones: Record<string, [number, number][]>;
  status_text: Record<string, string>;
}

// [id, kind ("s"|"e"), x, y, z, yaw, water, status, isPlayer]
export type DroneRow = [string, "s" | "e", number, number, number, number, number, string, boolean];
// [id, x, y, area_m2, confidence, status, litres still needed, priority 0..1]
// status: 0 detected, 1 confirmed, 2 assigned, 3 suppressed, 4 false positive, 5 seeker investigating, 6 ground review
export type HotspotRow = [string, number, number, number, number, number, number, number];

export interface ThermalMsg {
  w: number;
  h: number;
  data: string;
  lo: number;
  hi: number;
  t: number;
  boxes: [number, number, number][];
  tmax: number;
  id?: string;
}

export interface DemoStats {
  fire_fought_pct: number;
  fire_left_pct: number;
  fire_m2: number;
  baseline_m2: number;
  drops: number;
  fills_lake: number;
  fills_base: number;
  suppressed: number;
  verify_hot: number;
  reopened: number;
  confirmed: number;
  queue: number;
  tasks: number;
  water_dropped_l: number;
  water_on_fire_l: number;
  seekers: number;
  attackers: number;
  lakes: number;
  investigated: number;
  confirmed_investigate: number;
  rejected_investigate: number;
  reviews: number;
  ground_fire: number;
  ground_not_fire: number;
  review_timeouts: number;
  false_positives: number;
  coverage_pct?: number;
  fires_detected?: number;
  decoys_reported?: number;
  fires_total?: number;
  wet?: boolean;
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
  demo?: DemoStats;
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
  acts?: Record<string, string>;
  links?: [string, string][]; // [drone id, hotspot id] active assignments
  demo?: DemoStats;
  queue?: [string, number, number][]; // top of the priority queue: [hotspot id, priority, litres unassigned]
}

export type ServerMsg = InitMsg | StateMsg | { type: "loading"; map: string } | { type: "error"; message: string };

export function b64bytes(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
