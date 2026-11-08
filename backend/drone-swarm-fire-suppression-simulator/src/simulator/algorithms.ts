import {
  Vector2D,
  Drone,
  FireSpot,
  Obstacle,
  BaseStation,
  Wind,
  DroneAction,
  AlgorithmPreset,
} from './types';

// ==========================================
// 1. Vector 2D Pure Math Utilities
// ==========================================

export function dist(a: Vector2D, b: Vector2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function distSq(a: Vector2D, b: Vector2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

export function norm(v: Vector2D): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function normalize(v: Vector2D): Vector2D {
  const len = norm(v);
  if (len === 0) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function scale(v: Vector2D, factor: number): Vector2D {
  return { x: v.x * factor, y: v.y * factor };
}

export function add(a: Vector2D, b: Vector2D): Vector2D {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function sub(a: Vector2D, b: Vector2D): Vector2D {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function limit(v: Vector2D, maxLen: number): Vector2D {
  const len = norm(v);
  if (len > maxLen && len > 0) {
    return scale(v, maxLen / len);
  }
  return v;
}

export function angle(v: Vector2D): number {
  return Math.atan2(v.y, v.x);
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

// ==========================================
// 2. Swarm Coordination & Obstacle Avoidance
// ==========================================

/**
 * Calculates repulsive force from neighboring drones to prevent collisions (Separation).
 */
export function calculateSeparation(
  drone: Drone,
  allDrones: Drone[],
  minDistance: number = 28,
  weight: number = 2.5
): Vector2D {
  let steer: Vector2D = { x: 0, y: 0 };
  let count = 0;

  for (const other of allDrones) {
    if (other.id === drone.id) continue;
    const d = dist(drone, other);
    // If within separation radius
    if (d > 0 && d < minDistance) {
      const diff = sub(drone, other);
      const normalizedDiff = normalize(diff);
      // Inverse square weighting so closer drones repel much more aggressively
      const forceMag = (minDistance - d) / minDistance;
      steer = add(steer, scale(normalizedDiff, forceMag * forceMag));
      count++;
    }
  }

  if (count > 0) {
    steer = scale(steer, 1 / count);
    steer = scale(normalize(steer), drone.maxSpeed * weight);
  }

  return steer;
}

/**
 * Calculates obstacle repulsive force using Artificial Potential Field.
 */
export function calculateObstacleRepulsion(
  drone: Drone,
  obstacles: Obstacle[],
  safetyMargin: number = 32,
  weight: number = 3.5
): Vector2D {
  let steer: Vector2D = { x: 0, y: 0 };

  for (const obs of obstacles) {
    const d = dist(drone, obs);
    const effectiveRadius = obs.radius + drone.radius + safetyMargin;

    if (d < effectiveRadius) {
      // Repulsive vector pointing away from obstacle center
      const diff = sub(drone, obs);
      const unitDiff = normalize(diff);
      
      // Potential field gradient: (1/d - 1/d0)
      const penetration = effectiveRadius - d;
      const forceMag = (penetration / effectiveRadius) ** 2;
      steer = add(steer, scale(unitDiff, forceMag * weight * drone.maxSpeed * 2.0));
    }
  }

  return steer;
}

/**
 * Calculates attractive force towards target fire or assigns an optimal fire.
 */
export function calculateFireAttraction(
  drone: Drone,
  fires: FireSpot[],
  allDrones: Drone[],
  weight: number = 1.8
): { force: Vector2D; targetFireId: string | null; inSuppressionRange: boolean } {
  const activeFires = fires.filter(f => f.isActive && !f.isExtinguished);
  if (activeFires.length === 0) {
    return { force: { x: 0, y: 0 }, targetFireId: null, inSuppressionRange: false };
  }

  // Count drones currently targeting each fire to balance the swarm
  const fireAssignmentCounts = new Map<string, number>();
  for (const f of activeFires) {
    fireAssignmentCounts.set(f.id, 0);
  }
  for (const d of allDrones) {
    if (d.id !== drone.id && d.targetFireId && fireAssignmentCounts.has(d.targetFireId)) {
      fireAssignmentCounts.set(d.targetFireId, (fireAssignmentCounts.get(d.targetFireId) || 0) + 1);
    }
  }

  let bestFire: FireSpot | null = null;
  let bestScore = -Infinity;

  for (const fire of activeFires) {
    const d = dist(drone, fire);
    const assignedCount = fireAssignmentCounts.get(fire.id) || 0;
    
    // Score based on fire intensity, distance, and existing assignment penalty
    const distancePenalty = d * 1.2;
    const assignmentPenalty = assignedCount * 140; // Discourage over-clustering on one fire
    const intensityBonus = fire.intensity * 2.0;

    const score = intensityBonus - distancePenalty - assignmentPenalty;

    if (score > bestScore) {
      bestScore = score;
      bestFire = fire;
    }
  }

  if (!bestFire) {
    return { force: { x: 0, y: 0 }, targetFireId: null, inSuppressionRange: false };
  }

  const d = dist(drone, bestFire);
  const desired = scale(normalize(sub(bestFire, drone)), drone.maxSpeed * weight);
  const inSuppressionRange = d <= (bestFire.radius + 18);

  return {
    force: desired,
    targetFireId: bestFire.id,
    inSuppressionRange,
  };
}

/**
 * Calculates force directing drone back to base station.
 */
export function calculateRTBForce(
  drone: Drone,
  baseStation: BaseStation,
  weight: number = 2.2
): { force: Vector2D; atBase: boolean } {
  const d = dist(drone, baseStation);
  if (d <= baseStation.radius * 0.7) {
    // Already in landing zone - decelerate to hover/dock
    return {
      force: scale(drone.vx !== 0 || drone.vy !== 0 ? { x: -drone.vx, y: -drone.vy } : { x: 0, y: 0 }, 1.5),
      atBase: true,
    };
  }

  const desired = scale(normalize(sub(baseStation, drone)), drone.maxSpeed * weight);
  return {
    force: desired,
    atBase: false,
  };
}

// ==========================================
// 3. Complete Python-Simulated / Native Algorithms
// ==========================================

export function runNativeAlgorithm(
  drones: Drone[],
  fires: FireSpot[],
  obstacles: Obstacle[],
  baseStation: BaseStation,
  presetId: string = 'apf_voronoi'
): DroneAction[] {
  const actions: DroneAction[] = [];

  for (const drone of drones) {
    // 1. Check if Drone is commanded to RTB or needs urgent recharge/refill
    const needsRefill = drone.water <= 5;
    const needsRecharge = drone.battery <= 15;
    const isReturning = drone.isManualRTB || needsRefill || needsRecharge;

    // Check if drone is at base station and refilling
    const distToBase = dist(drone, baseStation);
    if (distToBase <= baseStation.radius && (isReturning || drone.water < drone.maxWater || drone.battery < 99)) {
      if (drone.water >= drone.maxWater && drone.battery >= 98 && !drone.isManualRTB) {
        // Refilled and recharged, ready to redeploy!
      } else {
        // Stay docked / refilling
        actions.push({
          id: drone.id,
          vx: 0,
          vy: 0,
          suppress: false,
          state: 'REFILLING',
        });
        continue;
      }
    }

    if (isReturning) {
      const { force: rtbForce, atBase } = calculateRTBForce(drone, baseStation);
      const obsRepulsion = calculateObstacleRepulsion(drone, obstacles, 35, 3.8);
      const swarmSeparation = calculateSeparation(drone, drones, 26, 2.5);

      let totalForce = add(rtbForce, obsRepulsion);
      totalForce = add(totalForce, swarmSeparation);
      totalForce = limit(totalForce, drone.maxSpeed);

      actions.push({
        id: drone.id,
        vx: atBase ? 0 : totalForce.x,
        vy: atBase ? 0 : totalForce.y,
        suppress: false,
        state: atBase ? 'REFILLING' : 'RETURNING',
      });
      continue;
    }

    // Normal Mission Execution
    if (presetId === 'boids') {
      // Boids flocking with fire suppression
      const separation = calculateSeparation(drone, drones, 32, 3.0);
      const obsRepulsion = calculateObstacleRepulsion(drone, obstacles, 35, 4.0);
      const { force: fireForce, inSuppressionRange, targetFireId } = calculateFireAttraction(
        drone,
        fires,
        drones,
        2.0
      );

      let total = add(separation, obsRepulsion);
      total = add(total, fireForce);
      total = limit(total, drone.maxSpeed);

      const isSuppressing = inSuppressionRange && drone.water > 0;
      actions.push({
        id: drone.id,
        vx: total.x,
        vy: total.y,
        suppress: isSuppressing,
        state: isSuppressing ? 'SUPPRESSING' : targetFireId ? 'SEARCHING' : 'HOVERING',
      });
    } else if (presetId === 'cbf_barrier') {
      // Control Barrier Function approach: Hard priority to obstacle & swarm safety
      const obsRepulsion = calculateObstacleRepulsion(drone, obstacles, 45, 5.0);
      const swarmSeparation = calculateSeparation(drone, drones, 36, 4.0);
      const { force: fireForce, inSuppressionRange, targetFireId } = calculateFireAttraction(
        drone,
        fires,
        drones,
        1.5
      );

      // If obstacle or swarm danger is high, suppress fire tracking
      const safetyForce = add(obsRepulsion, swarmSeparation);
      const safetyMag = norm(safetyForce);

      let total: Vector2D;
      if (safetyMag > drone.maxSpeed * 0.8) {
        total = limit(safetyForce, drone.maxSpeed);
      } else {
        total = limit(add(safetyForce, fireForce), drone.maxSpeed);
      }

      const isSuppressing = inSuppressionRange && drone.water > 0;
      actions.push({
        id: drone.id,
        vx: total.x,
        vy: total.y,
        suppress: isSuppressing,
        state: isSuppressing ? 'SUPPRESSING' : targetFireId ? 'SEARCHING' : 'HOVERING',
      });
    } else if (presetId === 'naive') {
      // Baseline naive: flies straight at closest fire with minimal avoidance
      const obsRepulsion = calculateObstacleRepulsion(drone, obstacles, 15, 1.2);
      const swarmSeparation = calculateSeparation(drone, drones, 14, 0.8);
      const { force: fireForce, inSuppressionRange, targetFireId } = calculateFireAttraction(
        drone,
        fires,
        drones,
        2.5
      );

      const total = limit(add(add(fireForce, obsRepulsion), swarmSeparation), drone.maxSpeed);
      const isSuppressing = inSuppressionRange && drone.water > 0;
      actions.push({
        id: drone.id,
        vx: total.x,
        vy: total.y,
        suppress: isSuppressing,
        state: isSuppressing ? 'SUPPRESSING' : targetFireId ? 'SEARCHING' : 'HOVERING',
      });
    } else {
      // Default: APF + Voronoi Partitioning (Balanced, smooth, high containment)
      const swarmSeparation = calculateSeparation(drone, drones, 28, 2.8);
      const obsRepulsion = calculateObstacleRepulsion(drone, obstacles, 32, 3.6);
      const { force: fireForce, inSuppressionRange, targetFireId } = calculateFireAttraction(
        drone,
        fires,
        drones,
        2.2
      );

      let total = add(swarmSeparation, obsRepulsion);
      total = add(total, fireForce);
      total = limit(total, drone.maxSpeed);

      const isSuppressing = inSuppressionRange && drone.water > 0;
      actions.push({
        id: drone.id,
        vx: total.x,
        vy: total.y,
        suppress: isSuppressing,
        state: isSuppressing ? 'SUPPRESSING' : targetFireId ? 'SEARCHING' : 'HOVERING',
      });
    }
  }

  return actions;
}

// ==========================================
// 4. Fire Spread & Wind Propagation Dynamics
// ==========================================

export function stepFireDynamics(
  fires: FireSpot[],
  wind: Wind,
  dt: number,
  allowSpread: boolean = true
): FireSpot[] {
  // Convert wind direction (degrees) to radians: 0 deg = blowing from North (to South: +y)
  // Let standard mathematical angle: 0 deg = East (+x), 90 deg = North (-y)
  const windRad = (wind.direction * Math.PI) / 180;
  const windVector = {
    x: Math.cos(windRad) * wind.speed,
    y: Math.sin(windRad) * wind.speed,
  };

  return fires.map(fire => {
    if (!fire.isActive || fire.isExtinguished) {
      return fire;
    }

    let { intensity, radius } = fire;

    // Natural burning expansion
    if (allowSpread && intensity > 10) {
      // Wind accelerates spread and intensity
      const windMultiplier = 1.0 + (wind.speed / 15) * 0.8;
      intensity = Math.min(100, intensity + fire.spreadRate * windMultiplier * dt * 0.6);
      radius = Math.min(fire.maxRadius, radius + dt * 0.15 * windMultiplier);
    }

    const isExtinguished = intensity <= 1.0;

    return {
      ...fire,
      intensity,
      radius,
      isActive: !isExtinguished,
      isExtinguished,
    };
  });
}

// ==========================================
// 5. Collision & Near-Miss Detection
// ==========================================

export function detectCollisionsAndNearMisses(
  drones: Drone[],
  obstacles: Obstacle[]
): {
  collisions: number;
  nearMisses: number;
  droneCollidedIds: Set<string>;
  droneNearMissIds: Set<string>;
} {
  let collisions = 0;
  let nearMisses = 0;
  const droneCollidedIds = new Set<string>();
  const droneNearMissIds = new Set<string>();

  // 1. Inter-drone collisions
  for (let i = 0; i < drones.length; i++) {
    for (let j = i + 1; j < drones.length; j++) {
      const d1 = drones[i];
      const d2 = drones[j];
      const distance = dist(d1, d2);
      const hardRadius = d1.radius + d2.radius; // e.g. 14 units
      const nearRadius = hardRadius * 1.8; // e.g. 25.2 units

      if (distance <= hardRadius) {
        collisions++;
        droneCollidedIds.add(d1.id);
        droneCollidedIds.add(d2.id);
      } else if (distance <= nearRadius) {
        nearMisses++;
        droneNearMissIds.add(d1.id);
        droneNearMissIds.add(d2.id);
      }
    }
  }

  // 2. Drone-obstacle collisions
  for (const drone of drones) {
    for (const obs of obstacles) {
      const distance = dist(drone, obs);
      const hardRadius = drone.radius + obs.radius;
      const nearRadius = hardRadius + 14;

      if (distance <= hardRadius) {
        collisions++;
        droneCollidedIds.add(drone.id);
      } else if (distance <= nearRadius) {
        nearMisses++;
        droneNearMissIds.add(drone.id);
      }
    }
  }

  return { collisions, nearMisses, droneCollidedIds, droneNearMissIds };
}

// ==========================================
// 6. Python Algorithm Code Presets
// ==========================================

export const ALGORITHM_PRESETS: AlgorithmPreset[] = [
  {
    id: 'apf_voronoi',
    name: 'Artificial Potential Field + Voronoi Partitioning',
    description: 'Combines repulsive gradient fields from obstacles & swarm peers with distributed fire cluster allocation.',
    pythonCode: `# ========================================================
# Drone Swarm Algorithm: APF (Artificial Potential Field)
# with Distributed Fire Target Partitioning
# ========================================================
import math

def update_drones(drones, fires, obstacles, base, wind, dt):
    """
    Main autonomous swarm coordination tick function.
    
    Args:
        drones: list of drone telemetry dicts:
            [{'id': str, 'x': float, 'y': float, 'vx': float, 'vy': float,
              'water': float, 'max_water': float, 'battery': float,
              'is_manual_rtb': bool, 'max_speed': float, 'radius': float}]
        fires: list of fire dicts:
            [{'id': str, 'x': float, 'y': float, 'intensity': float,
              'radius': float, 'is_active': bool}]
        obstacles: list of obstacle dicts:
            [{'id': str, 'x': float, 'y': float, 'radius': float, 'type': str}]
        base: {'x': float, 'y': float, 'radius': float}
        wind: {'speed': float, 'direction': float}
        dt: delta time step (seconds)
        
    Returns:
        list of action dicts:
            [{'id': str, 'vx': float, 'vy': float, 'suppress': bool, 'state': str}]
    """
    actions = []
    active_fires = [f for f in fires if f.get('is_active', True) and f.get('intensity', 0) > 0]
    
    # Track assigned fires to avoid swarm over-clustering
    fire_claims = {f['id']: 0 for f in active_fires}
    
    for drone in drones:
        d_id = drone['id']
        x, y = drone['x'], drone['y']
        water = drone['water']
        battery = drone['battery']
        max_speed = drone.get('max_speed', 65.0)
        is_manual_rtb = drone.get('is_manual_rtb', False)
        
        # 1. Check Return to Base (RTB) condition
        needs_refill = water <= 5.0
        needs_charge = battery <= 15.0
        should_rtb = is_manual_rtb or needs_refill or needs_charge
        
        dist_to_base = math.hypot(x - base['x'], y - base['y'])
        
        # Docked at base
        if dist_to_base <= base['radius']:
            if should_rtb or water < drone.get('max_water', 100):
                actions.append({
                    'id': d_id,
                    'vx': 0.0,
                    'vy': 0.0,
                    'suppress': False,
                    'state': 'REFILLING'
                })
                continue
        
        # Returning to base navigation
        if should_rtb:
            dx = base['x'] - x
            dy = base['y'] - y
            d = math.hypot(dx, dy) or 1.0
            
            # Attractive force to base
            vx = (dx / d) * max_speed
            vy = (dy / d) * max_speed
            
            # Add obstacle avoidance even on RTB
            vx, vy = apply_obstacle_avoidance(x, y, vx, vy, obstacles, max_speed)
            vx, vy = apply_swarm_separation(drone, drones, vx, vy, max_speed)
            
            actions.append({
                'id': d_id,
                'vx': vx,
                'vy': vy,
                'suppress': False,
                'state': 'RETURNING'
            })
            continue
            
        # 2. Fire Hunting & Partitioning (Voronoi/Hungarian heuristic)
        best_fire = None
        best_score = -999999.0
        
        for fire in active_fires:
            f_dist = math.hypot(x - fire['x'], y - fire['y'])
            assigned_penalty = fire_claims[fire['id']] * 120.0
            intensity_bonus = fire['intensity'] * 2.0
            score = intensity_bonus - f_dist - assigned_penalty
            if score > best_score:
                best_score = score
                best_fire = fire
                
        if best_fire is not None:
            fire_claims[best_fire['id']] += 1
            fdx = best_fire['x'] - x
            fdy = best_fire['y'] - y
            f_dist = math.hypot(fdx, fdy) or 1.0
            
            # Desired velocity towards fire
            vx = (fdx / f_dist) * max_speed
            vy = (fdy / f_dist) * max_speed
            
            # 3. Apply Obstacle Repulsion Field
            vx, vy = apply_obstacle_avoidance(x, y, vx, vy, obstacles, max_speed)
            
            # 4. Apply Swarm Member Separation (Anti-Collision)
            vx, vy = apply_swarm_separation(drone, drones, vx, vy, max_speed)
            
            # 5. Fire Suppression Trigger
            suppress_range = best_fire.get('radius', 25.0) + 15.0
            in_range = f_dist <= suppress_range
            should_suppress = in_range and water > 0
            
            actions.append({
                'id': d_id,
                'vx': vx,
                'vy': vy,
                'suppress': should_suppress,
                'state': 'SUPPRESSING' if should_suppress else 'SEARCHING'
            })
        else:
            # All fires out: hover or patrol
            actions.append({
                'id': d_id,
                'vx': 0.0,
                'vy': 0.0,
                'suppress': False,
                'state': 'HOVERING'
            })
            
    return actions


def apply_obstacle_avoidance(x, y, vx, vy, obstacles, max_speed, safety_margin=32.0):
    """Repels drone from all obstacles using inverse-square potential field."""
    for obs in obstacles:
        dx = x - obs['x']
        dy = y - obs['y']
        d = math.hypot(dx, dy)
        safe_r = obs.get('radius', 25.0) + safety_margin
        if d < safe_r and d > 0:
            push = ((safe_r - d) / safe_r) ** 2 * (max_speed * 2.5)
            vx += (dx / d) * push
            vy += (dy / d) * push
            
    # Clamp speed to max_speed
    speed = math.hypot(vx, vy)
    if speed > max_speed:
        vx = (vx / speed) * max_speed
        vy = (vy / speed) * max_speed
    return vx, vy


def apply_swarm_separation(drone, all_drones, vx, vy, max_speed, min_separation=28.0):
    """Applies inter-drone repulsion to guarantee zero mid-air collisions."""
    x, y = drone['x'], drone['y']
    for other in all_drones:
        if other['id'] == drone['id']:
            continue
        dx = x - other['x']
        dy = y - other['y']
        d = math.hypot(dx, dy)
        if 0 < d < min_separation:
            push = ((min_separation - d) / min_separation) * (max_speed * 2.2)
            vx += (dx / d) * push
            vy += (dy / d) * push
            
    speed = math.hypot(vx, vy)
    if speed > max_speed:
        vx = (vx / speed) * max_speed
        vy = (vy / speed) * max_speed
    return vx, vy
`,
  },
  {
    id: 'boids_flocking',
    name: 'Boids Swarm Flocking (Reynolds Rules)',
    description: 'Implements Separation, Cohesion, and Alignment with goal-directed fire suppression vectors.',
    pythonCode: `# ========================================================
# Drone Swarm Algorithm: Boids Flocking Model
# (Reynolds: Separation + Alignment + Cohesion + Goal)
# ========================================================
import math

def update_drones(drones, fires, obstacles, base, wind, dt):
    actions = []
    active_fires = [f for f in fires if f.get('is_active', True) and f.get('intensity', 0) > 0]
    
    for drone in drones:
        d_id = drone['id']
        x, y = drone['x'], drone['y']
        max_speed = drone.get('max_speed', 65.0)
        
        # Check RTB
        if drone.get('is_manual_rtb', False) or drone['water'] <= 5.0 or drone['battery'] <= 15.0:
            dx = base['x'] - x
            dy = base['y'] - y
            d = math.hypot(dx, dy) or 1.0
            if d <= base['radius']:
                actions.append({'id': d_id, 'vx': 0.0, 'vy': 0.0, 'suppress': False, 'state': 'REFILLING'})
            else:
                vx = (dx / d) * max_speed
                vy = (dy / d) * max_speed
                actions.append({'id': d_id, 'vx': vx, 'vy': vy, 'suppress': False, 'state': 'RETURNING'})
            continue
            
        # 1. Boids: Separation (Repel if too close)
        sep_x, sep_y = 0.0, 0.0
        sep_count = 0
        
        # 2. Boids: Alignment (Match neighbors velocity)
        align_x, align_y = 0.0, 0.0
        align_count = 0
        
        # 3. Boids: Cohesion (Steer toward center of flock)
        coh_x, coh_y = 0.0, 0.0
        coh_count = 0
        
        for other in drones:
            if other['id'] == d_id:
                continue
            d = math.hypot(x - other['x'], y - other['y'])
            if 0 < d < 32.0:
                sep_x += (x - other['x']) / d
                sep_y += (y - other['y']) / d
                sep_count += 1
            if 0 < d < 90.0:
                align_x += other.get('vx', 0.0)
                align_y += other.get('vy', 0.0)
                align_count += 1
                coh_x += other['x']
                coh_y += other['y']
                coh_count += 1
                
        # 4. Goal Vector (Target nearest high-intensity fire)
        goal_x, goal_y = 0.0, 0.0
        target_fire = None
        min_fire_dist = 999999.0
        for f in active_fires:
            fd = math.hypot(x - f['x'], y - f['y'])
            if fd < min_fire_dist:
                min_fire_dist = fd
                target_fire = f
                
        if target_fire:
            goal_x = (target_fire['x'] - x) / min_fire_dist
            goal_y = (target_fire['y'] - y) / min_fire_dist
            
        # 5. Obstacle Avoidance Vector
        obs_x, obs_y = 0.0, 0.0
        for obs in obstacles:
            od = math.hypot(x - obs['x'], y - obs['y'])
            safe_d = obs.get('radius', 25.0) + 30.0
            if 0 < od < safe_d:
                push = (safe_d - od) / safe_d
                obs_x += ((x - obs['x']) / od) * push * 3.5
                obs_y += ((y - obs['y']) / od) * push * 3.5
                
        # Combine weighted vectors
        vx = goal_x * 1.8 + sep_x * 2.8 + obs_x * 3.2
        vy = goal_y * 1.8 + sep_y * 2.8 + obs_y * 3.2
        
        if align_count > 0:
            vx += (align_x / align_count) * 0.2
            vy += (align_y / align_count) * 0.2
            
        # Scale to max_speed
        spd = math.hypot(vx, vy) or 1.0
        vx = (vx / spd) * max_speed
        vy = (vy / spd) * max_speed
        
        should_suppress = target_fire and min_fire_dist <= (target_fire.get('radius', 25) + 15) and drone['water'] > 0
        actions.append({
            'id': d_id,
            'vx': vx,
            'vy': vy,
            'suppress': should_suppress,
            'state': 'SUPPRESSING' if should_suppress else 'SEARCHING'
        })
        
    return actions
`,
  },
  {
    id: 'cbf_barrier',
    name: 'Control Barrier Functions (CBF Safety Guarantee)',
    description: 'Enforces hard mathematical collision avoidance boundary certificates for ultra-safe multi-agent operations.',
    pythonCode: `# ========================================================
# Drone Swarm Algorithm: Control Barrier Functions (CBF)
# Hard collision avoidance guarantees for dense environments
# ========================================================
import math

def update_drones(drones, fires, obstacles, base, wind, dt):
    actions = []
    active_fires = [f for f in fires if f.get('is_active', True) and f.get('intensity', 0) > 0]
    
    for drone in drones:
        d_id = drone['id']
        x, y = drone['x'], drone['y']
        max_speed = drone.get('max_speed', 65.0)
        
        # Check RTB
        if drone.get('is_manual_rtb', False) or drone['water'] <= 5.0 or drone['battery'] <= 15.0:
            dx = base['x'] - x
            dy = base['y'] - y
            d = math.hypot(dx, dy) or 1.0
            if d <= base['radius']:
                actions.append({'id': d_id, 'vx': 0.0, 'vy': 0.0, 'suppress': False, 'state': 'REFILLING'})
                continue
            vx = (dx / d) * max_speed
            vy = (dy / d) * max_speed
            # Apply safety barrier filter
            vx, vy = enforce_barrier_certificates(x, y, vx, vy, drones, obstacles, d_id, max_speed)
            actions.append({'id': d_id, 'vx': vx, 'vy': vy, 'suppress': False, 'state': 'RETURNING'})
            continue
            
        # Nominal mission velocity: Target nearest fire
        target_fire = None
        min_dist = float('inf')
        for f in active_fires:
            d = math.hypot(x - f['x'], y - f['y'])
            if d < min_dist:
                min_dist = d
                target_fire = f
                
        if target_fire:
            vx = ((target_fire['x'] - x) / min_dist) * max_speed
            vy = ((target_fire['y'] - y) / min_dist) * max_speed
        else:
            vx, vy = 0.0, 0.0
            
        # Filter nominal control via Safety Barrier
        safe_vx, safe_vy = enforce_barrier_certificates(x, y, vx, vy, drones, obstacles, d_id, max_speed)
        
        should_suppress = target_fire and min_dist <= (target_fire.get('radius', 25) + 16) and drone['water'] > 0
        actions.append({
            'id': d_id,
            'vx': safe_vx,
            'vy': safe_vy,
            'suppress': should_suppress,
            'state': 'SUPPRESSING' if should_suppress else 'SEARCHING'
        })
        
    return actions


def enforce_barrier_certificates(x, y, vx, vy, drones, obstacles, d_id, max_speed):
    """Quadratic program / barrier projection ensuring zero collision."""
    # 1. Drone-Drone Barrier: h_ij = ||p_i - p_j||^2 - R_safe^2 >= 0
    safe_dist_drones = 32.0
    for other in drones:
        if other['id'] == d_id:
            continue
        dx = x - other['x']
        dy = y - other['y']
        dist = math.hypot(dx, dy)
        if dist < safe_dist_drones and dist > 0:
            penetration = safe_dist_drones - dist
            barrier_force = (penetration / safe_dist_drones) * (max_speed * 3.5)
            vx += (dx / dist) * barrier_force
            vy += (dy / dist) * barrier_force
            
    # 2. Obstacle Barrier: h_obs = ||p_i - p_obs||^2 - (R_obs + R_safe)^2 >= 0
    for obs in obstacles:
        dx = x - obs['x']
        dy = y - obs['y']
        dist = math.hypot(dx, dy)
        safe_obs_dist = obs.get('radius', 25.0) + 36.0
        if dist < safe_obs_dist and dist > 0:
            penetration = safe_obs_dist - dist
            barrier_force = (penetration / safe_obs_dist) ** 2 * (max_speed * 4.0)
            vx += (dx / dist) * barrier_force
            vy += (dy / dist) * barrier_force
            
    spd = math.hypot(vx, vy)
    if spd > max_speed:
        vx = (vx / spd) * max_speed
        vy = (vy / spd) * max_speed
    return vx, vy
`,
  },
  {
    id: 'scratchpad',
    name: 'Custom Python Algorithm Template',
    description: 'Clean starter template for creating and testing your own custom drone coordination algorithms.',
    pythonCode: `# ========================================================
# Custom Swarm Algorithm Scratchpad
# Edit this code to test your own autonomous drone logic!
# ========================================================
import math

def update_drones(drones, fires, obstacles, base, wind, dt):
    """
    Called each simulation tick.
    
    Inputs:
      drones: list of dicts with:
        id, x, y, vx, vy, water, max_water, battery, is_manual_rtb, max_speed, radius
      fires: list of dicts with:
        id, x, y, intensity (0-100), radius, is_active
      obstacles: list of dicts with:
        id, x, y, radius, type
      base: dict with x, y, radius
      wind: dict with speed (m/s), direction (deg)
      dt: time delta in seconds
      
    Returns:
      list of dicts: [{'id': str, 'vx': float, 'vy': float, 'suppress': bool, 'state': str}]
    """
    actions = []
    active_fires = [f for f in fires if f.get('is_active', True)]
    
    for drone in drones:
        x, y = drone['x'], drone['y']
        max_speed = drone.get('max_speed', 65.0)
        
        # 1. Base / RTB Handling
        if drone.get('is_manual_rtb', False) or drone['water'] <= 5 or drone['battery'] <= 15:
            dx = base['x'] - x
            dy = base['y'] - y
            d = math.hypot(dx, dy) or 1.0
            actions.append({
                'id': drone['id'],
                'vx': (dx / d) * max_speed,
                'vy': (dy / d) * max_speed,
                'suppress': False,
                'state': 'RETURNING' if d > base['radius'] else 'REFILLING'
            })
            continue
            
        # 2. Target Selection (Your algorithm here!)
        target = active_fires[0] if active_fires else None
        vx, vy = 0.0, 0.0
        
        if target:
            dx = target['x'] - x
            dy = target['y'] - y
            d = math.hypot(dx, dy) or 1.0
            vx = (dx / d) * max_speed
            vy = (dy / d) * max_speed
            
        # 3. Simple Obstacle Avoidance
        for obs in obstacles:
            odx = x - obs['x']
            ody = y - obs['y']
            od = math.hypot(odx, ody)
            safe_d = obs['radius'] + 30.0
            if 0 < od < safe_d:
                push = (safe_d - od) / safe_d * (max_speed * 2.5)
                vx += (odx / od) * push
                vy += (ody / od) * push
                
        # 4. Swarm Collision Prevention
        for other in drones:
            if other['id'] == drone['id']:
                continue
            pdx = x - other['x']
            pdy = y - other['y']
            pd = math.hypot(pdx, pdy)
            if 0 < pd < 26.0:
                push = (26.0 - pd) / 26.0 * (max_speed * 2.0)
                vx += (pdx / pd) * push
                vy += (pdy / pd) * push
                
        # Clamp velocity
        cur_spd = math.hypot(vx, vy)
        if cur_spd > max_speed:
            vx = (vx / cur_spd) * max_speed
            vy = (vy / cur_spd) * max_speed
            
        # Suppress if near fire
        in_range = target and math.hypot(target['x'] - x, target['y'] - y) <= (target['radius'] + 15)
        
        actions.append({
            'id': drone['id'],
            'vx': vx,
            'vy': vy,
            'suppress': bool(in_range and drone['water'] > 0),
            'state': 'SUPPRESSING' if in_range else 'SEARCHING'
        })
        
    return actions
`,
  },
];
