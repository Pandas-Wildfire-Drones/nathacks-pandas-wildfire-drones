import { Drone, FireSpot, Obstacle, BaseStation, Wind } from './types';

export interface Scenario {
  id: string;
  name: string;
  badge: string;
  description: string;
  worldWidth: number;
  worldHeight: number;
  baseStation: BaseStation;
  drones: Drone[];
  fires: FireSpot[];
  obstacles: Obstacle[];
  wind: Wind;
  recommendedDuration: number;
}

export function createDrone(
  id: string,
  x: number,
  y: number,
  index: number = 0
): Drone {
  const colors = [
    '#38bdf8', '#34d399', '#f472b6', '#a78bfa',
    '#fbbf24', '#f87171', '#2dd4bf', '#818cf8',
    '#fb923c', '#4ade80', '#e879f9', '#22d3ee',
  ];

  return {
    id,
    x,
    y,
    vx: 0,
    vy: 0,
    heading: 0,
    speed: 0,
    maxSpeed: 68,
    maxForce: 12,
    radius: 7,
    sensorRadius: 75,
    commRadius: 130,
    water: 100,
    maxWater: 100,
    dischargeRate: 25,
    battery: 100,
    batteryDrainRate: 0.6,
    state: 'SEARCHING',
    targetFireId: null,
    trail: [],
    isManualRTB: false,
    nearMissCount: 0,
    waterDroppedTotal: 0,
    color: colors[index % colors.length],
  };
}

export function getScenarios(): Scenario[] {
  return [
    {
      id: 'canyon_gorge',
      name: 'Canyon Gorge Firestorm',
      badge: 'CHALLENGING',
      description: 'Narrow canyon corridor between mountain ridges with 4 fire clusters and gusting crosswinds.',
      worldWidth: 960,
      worldHeight: 640,
      recommendedDuration: 45,
      baseStation: {
        x: 90,
        y: 320,
        radius: 52,
        refillRate: 35,
        rechargeRate: 25,
      },
      wind: {
        speed: 4.8,
        direction: 35, // blows towards northeast
      },
      drones: [
        createDrone('DRN-01', 90, 290, 0),
        createDrone('DRN-02', 70, 320, 1),
        createDrone('DRN-03', 90, 350, 2),
        createDrone('DRN-04', 110, 320, 3),
        createDrone('DRN-05', 70, 290, 4),
        createDrone('DRN-06', 110, 350, 5),
      ],
      obstacles: [
        // Top canyon wall
        { id: 'obs-wall-t1', type: 'ridge', x: 300, y: 110, radius: 48, label: 'North Ridge A' },
        { id: 'obs-wall-t2', type: 'ridge', x: 420, y: 130, radius: 52, label: 'North Ridge B' },
        { id: 'obs-wall-t3', type: 'ridge', x: 560, y: 100, radius: 58, label: 'North Ridge C' },
        { id: 'obs-wall-t4', type: 'ridge', x: 700, y: 140, radius: 46, label: 'North Ridge D' },
        // Bottom canyon wall
        { id: 'obs-wall-b1', type: 'ridge', x: 320, y: 530, radius: 50, label: 'South Ridge A' },
        { id: 'obs-wall-b2', type: 'ridge', x: 460, y: 510, radius: 56, label: 'South Ridge B' },
        { id: 'obs-wall-b3', type: 'ridge', x: 600, y: 540, radius: 52, label: 'South Ridge C' },
        { id: 'obs-wall-b4', type: 'ridge', x: 740, y: 500, radius: 45, label: 'South Ridge D' },
        // Center choke pillar
        { id: 'obs-pillar-1', type: 'tower', x: 480, y: 310, radius: 36, label: 'Gorge Monolith' },
      ],
      fires: [
        { id: 'fire-1', x: 380, y: 240, intensity: 75, initialIntensity: 75, radius: 24, maxRadius: 42, spreadRate: 2.2, isActive: true, isExtinguished: false },
        { id: 'fire-2', x: 620, y: 260, intensity: 85, initialIntensity: 85, radius: 28, maxRadius: 48, spreadRate: 2.5, isActive: true, isExtinguished: false },
        { id: 'fire-3', x: 400, y: 400, intensity: 65, initialIntensity: 65, radius: 22, maxRadius: 38, spreadRate: 1.8, isActive: true, isExtinguished: false },
        { id: 'fire-4', x: 760, y: 340, intensity: 90, initialIntensity: 90, radius: 30, maxRadius: 52, spreadRate: 2.8, isActive: true, isExtinguished: false },
      ],
    },
    {
      id: 'tower_facility',
      name: 'Communications Tower Facility',
      badge: 'HIGH OBSTACLE DENSITY',
      description: 'Multiple broadcast antenna towers with restricted no-fly radiuses and scattered fire breakout spots.',
      worldWidth: 960,
      worldHeight: 640,
      recommendedDuration: 50,
      baseStation: {
        x: 80,
        y: 80,
        radius: 50,
        refillRate: 35,
        rechargeRate: 25,
      },
      wind: {
        speed: 2.5,
        direction: 180, // blows westward
      },
      drones: [
        createDrone('DRN-01', 70, 60, 0),
        createDrone('DRN-02', 90, 60, 1),
        createDrone('DRN-03', 70, 90, 2),
        createDrone('DRN-04', 90, 90, 3),
        createDrone('DRN-05', 110, 80, 4),
        createDrone('DRN-06', 80, 110, 5),
        createDrone('DRN-07', 55, 80, 6),
      ],
      obstacles: [
        { id: 'tower-1', type: 'tower', x: 260, y: 160, radius: 28, label: 'Mast Alpha' },
        { id: 'tower-2', type: 'tower', x: 500, y: 140, radius: 30, label: 'Mast Bravo' },
        { id: 'tower-3', type: 'tower', x: 740, y: 180, radius: 28, label: 'Mast Charlie' },
        { id: 'tower-4', type: 'tower', x: 340, y: 380, radius: 32, label: 'Radar Array' },
        { id: 'tower-5', type: 'tower', x: 620, y: 390, radius: 30, label: 'Mast Delta' },
        { id: 'tower-6', type: 'tower', x: 220, y: 520, radius: 26, label: 'Repeater 1' },
        { id: 'tower-7', type: 'tower', x: 800, y: 490, radius: 34, label: 'Main Transmitter' },
        { id: 'bldg-1', type: 'structure', x: 480, y: 520, radius: 42, label: 'Substation Core' },
      ],
      fires: [
        { id: 'fire-1', x: 380, y: 220, intensity: 70, initialIntensity: 70, radius: 22, maxRadius: 40, spreadRate: 2.0, isActive: true, isExtinguished: false },
        { id: 'fire-2', x: 660, y: 240, intensity: 80, initialIntensity: 80, radius: 26, maxRadius: 44, spreadRate: 2.4, isActive: true, isExtinguished: false },
        { id: 'fire-3', x: 440, y: 340, intensity: 85, initialIntensity: 85, radius: 28, maxRadius: 46, spreadRate: 2.6, isActive: true, isExtinguished: false },
        { id: 'fire-4', x: 250, y: 390, intensity: 65, initialIntensity: 65, radius: 20, maxRadius: 36, spreadRate: 1.8, isActive: true, isExtinguished: false },
        { id: 'fire-5', x: 700, y: 530, intensity: 90, initialIntensity: 90, radius: 32, maxRadius: 50, spreadRate: 3.0, isActive: true, isExtinguished: false },
      ],
    },
    {
      id: 'swarm_stress',
      name: 'Multi-Swarm High Density Benchmark',
      badge: '12 DRONES / STRESS TEST',
      description: 'Dense 12-drone swarm operating simultaneously across 7 wildfire outbreaks with tight obstacle clearances.',
      worldWidth: 960,
      worldHeight: 640,
      recommendedDuration: 60,
      baseStation: {
        x: 480,
        y: 320,
        radius: 65,
        refillRate: 45,
        rechargeRate: 35,
      },
      wind: {
        speed: 3.2,
        direction: 120, // southeast
      },
      drones: [
        createDrone('DRN-01', 450, 290, 0),
        createDrone('DRN-02', 480, 280, 1),
        createDrone('DRN-03', 510, 290, 2),
        createDrone('DRN-04', 530, 320, 3),
        createDrone('DRN-05', 510, 350, 4),
        createDrone('DRN-06', 480, 360, 5),
        createDrone('DRN-07', 450, 350, 6),
        createDrone('DRN-08', 430, 320, 7),
        createDrone('DRN-09', 460, 310, 8),
        createDrone('DRN-10', 500, 310, 9),
        createDrone('DRN-11', 460, 335, 10),
        createDrone('DRN-12', 500, 335, 11),
      ],
      obstacles: [
        { id: 'obs-1', type: 'structure', x: 250, y: 180, radius: 35, label: 'Bunker North' },
        { id: 'obs-2', type: 'structure', x: 710, y: 180, radius: 35, label: 'Bunker East' },
        { id: 'obs-3', type: 'structure', x: 250, y: 470, radius: 35, label: 'Bunker West' },
        { id: 'obs-4', type: 'structure', x: 710, y: 470, radius: 35, label: 'Bunker South' },
        { id: 'obs-t1', type: 'tower', x: 480, y: 120, radius: 24, label: 'North Beacon' },
        { id: 'obs-t2', type: 'tower', x: 480, y: 520, radius: 24, label: 'South Beacon' },
      ],
      fires: [
        { id: 'fire-1', x: 160, y: 120, intensity: 80, initialIntensity: 80, radius: 25, maxRadius: 45, spreadRate: 2.2, isActive: true, isExtinguished: false },
        { id: 'fire-2', x: 800, y: 120, intensity: 85, initialIntensity: 85, radius: 28, maxRadius: 48, spreadRate: 2.5, isActive: true, isExtinguished: false },
        { id: 'fire-3', x: 140, y: 340, intensity: 75, initialIntensity: 75, radius: 24, maxRadius: 42, spreadRate: 2.0, isActive: true, isExtinguished: false },
        { id: 'fire-4', x: 820, y: 340, intensity: 90, initialIntensity: 90, radius: 30, maxRadius: 50, spreadRate: 2.8, isActive: true, isExtinguished: false },
        { id: 'fire-5', x: 170, y: 540, intensity: 70, initialIntensity: 70, radius: 22, maxRadius: 40, spreadRate: 1.9, isActive: true, isExtinguished: false },
        { id: 'fire-6', x: 790, y: 530, intensity: 85, initialIntensity: 85, radius: 26, maxRadius: 46, spreadRate: 2.4, isActive: true, isExtinguished: false },
        { id: 'fire-7', x: 380, y: 220, intensity: 65, initialIntensity: 65, radius: 20, maxRadius: 36, spreadRate: 1.7, isActive: true, isExtinguished: false },
      ],
    },
    {
      id: 'open_sandbox',
      name: 'Open Proving Ground (Sandbox)',
      badge: 'CUSTOMIZABLE',
      description: 'Clear baseline proving ground. Click anywhere to ignite fires, spawn obstacles, and command drone flights.',
      worldWidth: 960,
      worldHeight: 640,
      recommendedDuration: 60,
      baseStation: {
        x: 100,
        y: 100,
        radius: 54,
        refillRate: 35,
        rechargeRate: 25,
      },
      wind: {
        speed: 1.5,
        direction: 45,
      },
      drones: [
        createDrone('DRN-01', 90, 80, 0),
        createDrone('DRN-02', 110, 80, 1),
        createDrone('DRN-03', 90, 110, 2),
        createDrone('DRN-04', 110, 110, 3),
        createDrone('DRN-05', 130, 95, 4),
      ],
      obstacles: [
        { id: 'obs-1', type: 'tower', x: 450, y: 280, radius: 32, label: 'Central Pylon' },
        { id: 'obs-2', type: 'ridge', x: 680, y: 440, radius: 40, label: 'Boulder Cluster' },
      ],
      fires: [
        { id: 'fire-1', x: 550, y: 220, intensity: 80, initialIntensity: 80, radius: 26, maxRadius: 45, spreadRate: 2.2, isActive: true, isExtinguished: false },
        { id: 'fire-2', x: 720, y: 320, intensity: 75, initialIntensity: 75, radius: 24, maxRadius: 42, spreadRate: 2.0, isActive: true, isExtinguished: false },
      ],
    },
  ];
}
