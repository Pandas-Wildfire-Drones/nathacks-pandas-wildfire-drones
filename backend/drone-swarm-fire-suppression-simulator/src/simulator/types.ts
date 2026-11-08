export interface Vector2D {
  x: number;
  y: number;
}

export type DroneState = 
  | 'SEARCHING' 
  | 'SUPPRESSING' 
  | 'RETURNING' 
  | 'REFILLING' 
  | 'HOVERING' 
  | 'COLLIDED';

export interface Drone {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  heading: number; // in radians
  speed: number;
  maxSpeed: number;
  maxForce: number;
  radius: number; // collision radius (units)
  sensorRadius: number; // vision/avoidance scan radius
  commRadius: number; // radio mesh link radius
  water: number; // liters (0 to maxWater)
  maxWater: number;
  dischargeRate: number; // liters/sec
  battery: number; // percentage (0 to 100)
  batteryDrainRate: number; // %/sec
  state: DroneState;
  targetFireId: string | null;
  trail: Vector2D[];
  isManualRTB: boolean;
  nearMissCount: number;
  waterDroppedTotal: number;
  color?: string;
}

export interface FireSpot {
  id: string;
  x: number;
  y: number;
  intensity: number; // 0 to 100
  initialIntensity: number;
  radius: number;
  maxRadius: number;
  spreadRate: number;
  isActive: boolean;
  isExtinguished: boolean;
  smokeParticles?: Vector2D[];
}

export type ObstacleType = 'tower' | 'ridge' | 'structure' | 'nofly';

export interface Obstacle {
  id: string;
  type: ObstacleType;
  x: number;
  y: number;
  radius: number;
  label?: string;
  color?: string;
  height?: number;
}

export interface BaseStation {
  x: number;
  y: number;
  radius: number; // docking / refill radius
  refillRate: number; // water liters/sec
  rechargeRate: number; // battery %/sec
}

export interface Wind {
  speed: number; // 0 - 15 m/s
  direction: number; // 0 - 360 degrees
}

export interface DroneAction {
  id: string;
  vx?: number;
  vy?: number;
  targetHeading?: number;
  suppress?: boolean;
  rtb?: boolean;
  state?: DroneState;
}

export interface SimulationMetrics {
  elapsedTime: number; // seconds
  firesRemaining: number;
  initialFireCount: number;
  fireContainedPct: number;
  totalWaterDropped: number;
  waterEffectivePct: number;
  droneCollisions: number;
  nearMisses: number;
  swarmAvgDistance: number;
  batteryAvg: number;
  waterAvg: number;
  dronesActive: number;
  dronesRTB: number;
  dronesRefilling: number;
  algorithmName: string;
  algorithmFps: number;
  executionEngine: 'pyodide' | 'native';
  lastError: string | null;
}

export interface AlgorithmPreset {
  id: string;
  name: string;
  description: string;
  pythonCode: string;
}

export interface BenchmarkResult {
  algorithmName: string;
  scenarioName: string;
  durationSec: number;
  containmentPct: number;
  firesSuppressed: number;
  totalFires: number;
  collisionCount: number;
  nearMissCount: number;
  waterEfficiencyPct: number;
  avgBatteryRemaining: number;
  grade: 'S' | 'A' | 'B' | 'C' | 'FAIL';
  summary: string;
  timestamp: string;
}
