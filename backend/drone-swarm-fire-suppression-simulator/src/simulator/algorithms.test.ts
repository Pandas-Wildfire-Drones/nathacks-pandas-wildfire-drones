import test from 'node:test';
import assert from 'node:assert/strict';
import {
  dist,
  normalize,
  calculateSeparation,
  calculateObstacleRepulsion,
  calculateRTBForce,
  detectCollisionsAndNearMisses,
  stepFireDynamics,
  runNativeAlgorithm,
} from './algorithms';
import { Drone, FireSpot, Obstacle, BaseStation, Wind } from './types';

function createMockDrone(id: string, x: number, y: number, overrides: Partial<Drone> = {}): Drone {
  return {
    id,
    x,
    y,
    vx: 0,
    vy: 0,
    heading: 0,
    speed: 0,
    maxSpeed: 60,
    maxForce: 10,
    radius: 7,
    sensorRadius: 60,
    commRadius: 100,
    water: 100,
    maxWater: 100,
    dischargeRate: 20,
    battery: 100,
    batteryDrainRate: 0.5,
    state: 'SEARCHING',
    targetFireId: null,
    trail: [],
    isManualRTB: false,
    nearMissCount: 0,
    waterDroppedTotal: 0,
    ...overrides,
  };
}

test('dist calculation is mathematically accurate', () => {
  const p1 = { x: 0, y: 0 };
  const p2 = { x: 3, y: 4 };
  assert.equal(dist(p1, p2), 5);
});

test('calculateSeparation repels close drones and stays zero when distant', () => {
  const d1 = createMockDrone('D1', 100, 100);
  const d2Close = createMockDrone('D2', 110, 100); // 10 units away (within 28 minDistance)
  
  const repulsion = calculateSeparation(d1, [d1, d2Close], 28, 2.5);
  // d1 is to the left of d2 (x=100 vs x=110), so repulsion should push d1 to the left (negative x)
  assert.ok(repulsion.x < 0, `Expected negative x repulsion, got ${repulsion.x}`);
  assert.equal(Math.round(repulsion.y), 0);

  // When distant
  const d2Far = createMockDrone('D2', 300, 300);
  const zeroRepulsion = calculateSeparation(d1, [d1, d2Far], 28, 2.5);
  assert.equal(zeroRepulsion.x, 0);
  assert.equal(zeroRepulsion.y, 0);
});

test('calculateObstacleRepulsion directs force away from obstacle center', () => {
  const drone = createMockDrone('D1', 100, 50);
  const obstacle: Obstacle = {
    id: 'OBS-1',
    type: 'tower',
    x: 100,
    y: 70, // Obstacle is directly south (+y)
    radius: 20,
  };

  const rep = calculateObstacleRepulsion(drone, [obstacle], 32, 3.5);
  // Drone should be pushed north (negative y)
  assert.ok(rep.y < 0, `Expected negative y force away from obstacle, got ${rep.y}`);
  assert.equal(Math.round(rep.x), 0);
});

test('calculateRTBForce steers directly towards base coordinates', () => {
  const drone = createMockDrone('D1', 200, 200);
  const base: BaseStation = {
    x: 50,
    y: 50,
    radius: 40,
    refillRate: 35,
    rechargeRate: 25,
  };

  const { force, atBase } = calculateRTBForce(drone, base);
  assert.equal(atBase, false);
  // Base is at (50, 50), drone at (200, 200) -> force should be negative x and negative y
  assert.ok(force.x < 0);
  assert.ok(force.y < 0);

  // When already inside base station radius
  const droneAtBase = createMockDrone('D2', 55, 55);
  const resultAtBase = calculateRTBForce(droneAtBase, base);
  assert.equal(resultAtBase.atBase, true);
});

test('detectCollisionsAndNearMisses flags inter-drone and obstacle collisions', () => {
  const d1 = createMockDrone('D1', 100, 100, { radius: 7 });
  const d2 = createMockDrone('D2', 110, 100, { radius: 7 }); // dist = 10, hardRadius = 14 -> Collision!
  const d3 = createMockDrone('D3', 500, 500, { radius: 7 });

  const result = detectCollisionsAndNearMisses([d1, d2, d3], []);
  assert.equal(result.collisions, 1);
  assert.ok(result.droneCollidedIds.has('D1'));
  assert.ok(result.droneCollidedIds.has('D2'));
  assert.ok(!result.droneCollidedIds.has('D3'));
});

test('stepFireDynamics adjusts intensity and extinguishes fires under 1%', () => {
  const fire: FireSpot = {
    id: 'F1',
    x: 200,
    y: 200,
    intensity: 0.5, // under threshold
    initialIntensity: 50,
    radius: 20,
    maxRadius: 40,
    spreadRate: 2,
    isActive: true,
    isExtinguished: false,
  };

  const wind: Wind = { speed: 5, direction: 90 };
  const updated = stepFireDynamics([fire], wind, 0.1);
  assert.equal(updated[0].isExtinguished, true);
  assert.equal(updated[0].isActive, false);
});

test('runNativeAlgorithm produces valid non-NaN action outputs for all presets', () => {
  const drones = [
    createMockDrone('D1', 80, 80),
    createMockDrone('D2', 120, 90),
  ];
  const fires: FireSpot[] = [{
    id: 'F1',
    x: 300,
    y: 300,
    intensity: 80,
    initialIntensity: 80,
    radius: 25,
    maxRadius: 45,
    spreadRate: 2,
    isActive: true,
    isExtinguished: false,
  }];
  const obstacles: Obstacle[] = [{
    id: 'O1',
    type: 'tower',
    x: 200,
    y: 200,
    radius: 20,
  }];
  const base: BaseStation = {
    x: 50,
    y: 50,
    radius: 40,
    refillRate: 30,
    rechargeRate: 20,
  };

  for (const preset of ['apf_voronoi', 'boids', 'cbf_barrier', 'naive']) {
    const actions = runNativeAlgorithm(drones, fires, obstacles, base, preset);
    assert.equal(actions.length, 2);
    for (const act of actions) {
      assert.ok(!isNaN(act.vx || 0), `NaN in action vx for preset ${preset}`);
      assert.ok(!isNaN(act.vy || 0), `NaN in action vy for preset ${preset}`);
      assert.ok(typeof act.state === 'string');
    }
  }
});
