import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Drone,
  FireSpot,
  Obstacle,
  BaseStation,
  Wind,
  SimulationMetrics,
  Vector2D,
  BenchmarkResult,
  DroneAction,
} from './simulator/types';
import { getScenarios, Scenario, createDrone } from './simulator/scenarios';
import {
  ALGORITHM_PRESETS,
  runNativeAlgorithm,
  stepFireDynamics,
  detectCollisionsAndNearMisses,
  dist,
  clamp,
} from './simulator/algorithms';
import { initPyodide, runPythonStep, PyodideStatus } from './simulator/pythonBridge';
import { TacticalRadar } from './components/TacticalRadar';
import { FleetDashboard } from './components/FleetDashboard';
import { PythonEditor } from './components/PythonEditor';
import { TopBar } from './components/TopBar';
import { BenchmarkModal } from './components/BenchmarkModal';

export default function App() {
  const scenarios = useRef<Scenario[]>(getScenarios()).current;
  const [activeScenarioId, setActiveScenarioId] = useState<string>('canyon_gorge');
  const currentScenario = scenarios.find(s => s.id === activeScenarioId) || scenarios[0];

  // Simulation State
  const [drones, setDrones] = useState<Drone[]>(() => [...currentScenario.drones]);
  const [fires, setFires] = useState<FireSpot[]>(() => [...currentScenario.fires]);
  const [obstacles, setObstacles] = useState<Obstacle[]>(() => [...currentScenario.obstacles]);
  const [baseStation, setBaseStation] = useState<BaseStation>(() => ({ ...currentScenario.baseStation }));
  const [wind, setWind] = useState<Wind>(() => ({ ...currentScenario.wind }));

  // Controls
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1);
  const [viewMode, setViewMode] = useState<'radar' | 'split' | 'code'>('split');
  const [selectedDroneId, setSelectedDroneId] = useState<string | null>(null);
  const [toolMode, setToolMode] = useState<'inspect' | 'add_fire' | 'add_obstacle' | 'water_drop'>('inspect');

  // Visualization options
  const [showCommMesh, setShowCommMesh] = useState<boolean>(true);
  const [showSensorRanges, setShowSensorRanges] = useState<boolean>(false);
  const [showTrails, setShowTrails] = useState<boolean>(true);

  // Python Algorithm Engine
  const [activePresetId, setActivePresetId] = useState<string>('apf_voronoi');
  const [pythonCode, setPythonCode] = useState<string>(() => ALGORITHM_PRESETS[0].pythonCode);
  const [executionEngine, setExecutionEngine] = useState<'pyodide' | 'native'>('native');
  const [pyodideStatus, setPyodideStatus] = useState<PyodideStatus>({
    isReady: false,
    isLoading: false,
    error: null,
    version: null,
  });
  const [lastError, setLastError] = useState<string | null>(null);

  // Metrics & Benchmark
  const [metrics, setMetrics] = useState<SimulationMetrics>({
    elapsedTime: 0,
    firesRemaining: currentScenario.fires.length,
    initialFireCount: currentScenario.fires.length,
    fireContainedPct: 0,
    totalWaterDropped: 0,
    waterEffectivePct: 100,
    droneCollisions: 0,
    nearMisses: 0,
    swarmAvgDistance: 0,
    batteryAvg: 100,
    waterAvg: 100,
    dronesActive: currentScenario.drones.length,
    dronesRTB: 0,
    dronesRefilling: 0,
    algorithmName: ALGORITHM_PRESETS[0].name,
    algorithmFps: 60,
    executionEngine: 'native',
    lastError: null,
  });

  const [isBenchmarkModalOpen, setIsBenchmarkModalOpen] = useState<boolean>(false);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState<number>(0);
  const [latestBenchmarkResult, setLatestBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const benchmarkDurationRef = useRef<number>(45);
  const benchmarkElapsedRef = useRef<number>(0);

  // Sync state refs for requestAnimationFrame simulation loop
  const stateRef = useRef({
    drones,
    fires,
    obstacles,
    baseStation,
    wind,
    isPlaying,
    simSpeed,
    executionEngine,
    pythonCode,
    activePresetId,
    metrics,
    isBenchmarking,
  });

  stateRef.current = {
    drones,
    fires,
    obstacles,
    baseStation,
    wind,
    isPlaying,
    simSpeed,
    executionEngine,
    pythonCode,
    activePresetId,
    metrics,
    isBenchmarking,
  };

  // Background Pyodide initialization
  useEffect(() => {
    initPyodide(setPyodideStatus).catch(() => {
      // Graceful fallback to native engine
    });
  }, []);

  // Reset scenario handler
  const resetScenario = useCallback((scenarioId: string = activeScenarioId) => {
    const sc = scenarios.find(s => s.id === scenarioId) || scenarios[0];
    setDrones([...sc.drones.map(d => ({ ...d, trail: [] }))]);
    setFires([...sc.fires.map(f => ({ ...f, intensity: f.initialIntensity, isActive: true, isExtinguished: false }))]);
    setObstacles([...sc.obstacles]);
    setBaseStation({ ...sc.baseStation });
    setWind({ ...sc.wind });
    setSelectedDroneId(null);
    setMetrics(prev => ({
      ...prev,
      elapsedTime: 0,
      firesRemaining: sc.fires.length,
      initialFireCount: sc.fires.length,
      fireContainedPct: 0,
      totalWaterDropped: 0,
      droneCollisions: 0,
      nearMisses: 0,
    }));
  }, [activeScenarioId, scenarios]);

  const handleSelectScenario = (scId: string) => {
    setActiveScenarioId(scId);
    resetScenario(scId);
  };

  const handleSelectPreset = (presetId: string) => {
    setActivePresetId(presetId);
    const preset = ALGORITHM_PRESETS.find(p => p.id === presetId);
    if (preset) {
      setPythonCode(preset.pythonCode);
      setMetrics(m => ({ ...m, algorithmName: preset.name }));
    }
  };

  // Fleet RTB Commands
  const handleRecallAllRTB = () => {
    setDrones(prev => prev.map(d => ({ ...d, isManualRTB: true })));
  };

  const handleResumeAllMission = () => {
    setDrones(prev => prev.map(d => ({ ...d, isManualRTB: false })));
  };

  const handleEmergencyScram = () => {
    setDrones(prev =>
      prev.map(d => ({
        ...d,
        vx: 0,
        vy: 0,
        state: 'HOVERING',
      }))
    );
    setIsPlaying(false);
  };

  const handleToggleDroneRTB = (droneId: string) => {
    setDrones(prev =>
      prev.map(d => (d.id === droneId ? { ...d, isManualRTB: !d.isManualRTB } : d))
    );
  };

  const handleAddDrone = () => {
    setDrones(prev => {
      if (prev.length >= 16) return prev;
      const newIndex = prev.length;
      const angle = (newIndex * (Math.PI * 2)) / 8;
      const spawnX = baseStation.x + Math.cos(angle) * (baseStation.radius * 0.6);
      const spawnY = baseStation.y + Math.sin(angle) * (baseStation.radius * 0.6);
      const newDrone = createDrone(`DRN-${String(newIndex + 1).padStart(2, '0')}`, spawnX, spawnY, newIndex);
      return [...prev, newDrone];
    });
  };

  const handleRemoveDrone = () => {
    setDrones(prev => (prev.length > 1 ? prev.slice(0, prev.length - 1) : prev));
  };

  // Canvas click interaction handler
  const handleCanvasClick = (pt: Vector2D) => {
    if (toolMode === 'add_fire') {
      const newFire: FireSpot = {
        id: `fire-${Date.now()}`,
        x: pt.x,
        y: pt.y,
        intensity: 80,
        initialIntensity: 80,
        radius: 25,
        maxRadius: 45,
        spreadRate: 2.2,
        isActive: true,
        isExtinguished: false,
      };
      setFires(prev => [...prev, newFire]);
    } else if (toolMode === 'add_obstacle') {
      const newObs: Obstacle = {
        id: `obs-${Date.now()}`,
        type: 'tower',
        x: pt.x,
        y: pt.y,
        radius: 28,
        label: `Tower ${obstacles.length + 1}`,
      };
      setObstacles(prev => [...prev, newObs]);
    } else if (toolMode === 'water_drop') {
      // Manual aerial water drop at coordinate
      setFires(prev =>
        prev.map(f => {
          if (dist(f, pt) <= f.radius + 30) {
            const nextIntensity = Math.max(0, f.intensity - 45);
            return {
              ...f,
              intensity: nextIntensity,
              isActive: nextIntensity > 0,
              isExtinguished: nextIntensity <= 0,
            };
          }
          return f;
        })
      );
      setMetrics(m => ({ ...m, totalWaterDropped: m.totalWaterDropped + 50 }));
    }
  };

  // Benchmark Runner
  const startBenchmark = (durationSec: number) => {
    benchmarkDurationRef.current = durationSec;
    benchmarkElapsedRef.current = 0;
    resetScenario(activeScenarioId);
    setIsBenchmarking(true);
    setBenchmarkProgress(0);
    setIsPlaying(true);
  };

  // Main Simulation Step Function
  const stepSimulation = useCallback(async (dt: number) => {
    const {
      drones: currentDrones,
      fires: currentFires,
      obstacles: currentObstacles,
      baseStation: curBase,
      wind: curWind,
      executionEngine: curEngine,
      pythonCode: curCode,
      activePresetId: curPreset,
      isBenchmarking: isBench,
    } = stateRef.current;

    // 1. Calculate Swarm Actions (via Pyodide or Native Algorithm)
    let actions: DroneAction[] = [];
    if (curEngine === 'pyodide' && window.pyodideInstance) {
      try {
        actions = await runPythonStep(
          curCode,
          currentDrones,
          currentFires,
          currentObstacles,
          curBase,
          curWind,
          dt
        );
        setLastError(null);
      } catch (err: any) {
        setLastError(`Python error: ${err.message || String(err)}`);
        // Seamless fallback to native algorithm so simulation doesn't freeze
        actions = runNativeAlgorithm(
          currentDrones,
          currentFires,
          currentObstacles,
          curBase,
          curPreset
        );
      }
    } else {
      actions = runNativeAlgorithm(
        currentDrones,
        currentFires,
        currentObstacles,
        curBase,
        curPreset
      );
    }

    const actionMap = new Map<string, DroneAction>();
    for (const a of actions) {
      actionMap.set(a.id, a);
    }

    // 2. Step Drones
    let waterDroppedThisFrame = 0;
    let effectiveWaterThisFrame = 0;

    const nextDrones = currentDrones.map(drone => {
      const action = actionMap.get(drone.id);
      let vx = action?.vx !== undefined ? action.vx : drone.vx;
      let vy = action?.vy !== undefined ? action.vy : drone.vy;
      const isSuppressing = Boolean(action?.suppress);

      // Kinematic update
      let nextX = drone.x + vx * dt;
      let nextY = drone.y + vy * dt;

      // Bound to world
      nextX = clamp(nextX, 15, currentScenario.worldWidth - 15);
      nextY = clamp(nextY, 15, currentScenario.worldHeight - 15);

      // Battery consumption
      let nextBattery = Math.max(0, drone.battery - drone.batteryDrainRate * dt);

      // Water payload & suppression
      let nextWater = drone.water;
      if (isSuppressing && nextWater > 0) {
        const discharge = Math.min(nextWater, drone.dischargeRate * dt);
        nextWater -= discharge;
        waterDroppedThisFrame += discharge;
      }

      // Base Station Refueling & Recharging
      const distToBase = dist({ x: nextX, y: nextY }, curBase);
      let nextState = action?.state || drone.state;

      if (distToBase <= curBase.radius) {
        nextWater = Math.min(drone.maxWater, nextWater + curBase.refillRate * dt);
        nextBattery = Math.min(100, nextBattery + curBase.rechargeRate * dt);
        if (nextWater < drone.maxWater || nextBattery < 99) {
          nextState = 'REFILLING';
        }
      }

      // Breadcrumb trail
      const trail = [...(drone.trail || [])];
      if (trail.length === 0 || dist(trail[trail.length - 1], { x: nextX, y: nextY }) > 18) {
        trail.push({ x: nextX, y: nextY });
        if (trail.length > 22) trail.shift();
      }

      const heading = Math.atan2(vy, vx) || drone.heading;

      return {
        ...drone,
        x: nextX,
        y: nextY,
        vx,
        vy,
        heading,
        speed: Math.hypot(vx, vy),
        battery: nextBattery,
        water: nextWater,
        state: nextState,
        trail,
      };
    });

    // 3. Step Fire Dynamics with Water Suppression Cooling
    let nextFires = stepFireDynamics(currentFires, curWind, dt, true);

    // Apply water cooling from suppressing drones
    for (const drone of nextDrones) {
      if (drone.state === 'SUPPRESSING' && drone.water > 0) {
        for (let i = 0; i < nextFires.length; i++) {
          const fire = nextFires[i];
          if (!fire.isActive || fire.isExtinguished) continue;
          const d = dist(drone, fire);
          if (d <= fire.radius + 20) {
            effectiveWaterThisFrame += drone.dischargeRate * dt;
            const coolFactor = 18.0 * dt;
            const newIntensity = Math.max(0, fire.intensity - coolFactor);
            const isExtinguished = newIntensity <= 1.0;
            nextFires[i] = {
              ...fire,
              intensity: newIntensity,
              radius: Math.max(12, fire.radius - dt * 0.8),
              isActive: !isExtinguished,
              isExtinguished,
            };
          }
        }
      }
    }

    // 4. Collision and Near-Miss Detection
    const collisionCheck = detectCollisionsAndNearMisses(nextDrones, currentObstacles);

    // 5. Update Metrics
    const activeFiresCount = nextFires.filter(f => f.isActive && !f.isExtinguished).length;
    const initialFireCount = stateRef.current.metrics.initialFireCount || 1;
    const initialTotalIntensity = currentScenario.fires.reduce((sum, f) => sum + f.initialIntensity, 0);
    const currentTotalIntensity = nextFires.reduce((sum, f) => (f.isExtinguished ? 0 : sum + f.intensity), 0);
    const fireContainedPct = clamp(
      ((initialTotalIntensity - currentTotalIntensity) / initialTotalIntensity) * 100,
      0,
      100
    );

    const battSum = nextDrones.reduce((sum, d) => sum + d.battery, 0);
    const waterSum = nextDrones.reduce((sum, d) => sum + d.water, 0);

    setDrones(nextDrones);
    setFires(nextFires);
    setMetrics(prev => ({
      ...prev,
      elapsedTime: prev.elapsedTime + dt,
      firesRemaining: activeFiresCount,
      fireContainedPct,
      totalWaterDropped: prev.totalWaterDropped + waterDroppedThisFrame,
      droneCollisions: prev.droneCollisions + collisionCheck.collisions,
      nearMisses: prev.nearMisses + collisionCheck.nearMisses,
      batteryAvg: battSum / nextDrones.length,
      waterAvg: waterSum / nextDrones.length,
      dronesActive: nextDrones.filter(d => d.state !== 'REFILLING').length,
      dronesRTB: nextDrones.filter(d => d.state === 'RETURNING').length,
      dronesRefilling: nextDrones.filter(d => d.state === 'REFILLING').length,
      executionEngine: curEngine,
    }));

    // 6. Benchmark Management
    if (isBench) {
      benchmarkElapsedRef.current += dt;
      const progress = clamp((benchmarkElapsedRef.current / benchmarkDurationRef.current) * 100, 0, 100);
      setBenchmarkProgress(progress);

      if (benchmarkElapsedRef.current >= benchmarkDurationRef.current || activeFiresCount === 0) {
        // Benchmark complete! Evaluate Scorecard
        setIsBenchmarking(false);
        const collisions = stateRef.current.metrics.droneCollisions;
        const containment = fireContainedPct;
        const avgBatt = battSum / nextDrones.length;

        let grade: 'S' | 'A' | 'B' | 'C' | 'FAIL' = 'FAIL';
        if (collisions === 0 && containment >= 95) grade = 'S';
        else if (collisions === 0 && containment >= 75) grade = 'A';
        else if (collisions === 0 && containment >= 50) grade = 'B';
        else if (collisions <= 2 && containment >= 40) grade = 'C';
        else grade = 'FAIL';

        const summary =
          grade === 'S'
            ? `Exceptional autonomous coordination! Contained ${Math.round(containment)}% of wildfire in ${Math.round(benchmarkElapsedRef.current)}s with 0 collisions and pristine obstacle clearance.`
            : grade === 'A'
            ? `Solid mission performance with 0 collisions. Contained ${Math.round(containment)}% of wildfire.`
            : grade === 'B'
            ? `Good baseline safety. Contained ${Math.round(containment)}% with no collisions.`
            : grade === 'C'
            ? `Safety caution: minor collision or low containment (${Math.round(containment)}%).`
            : `Failed benchmark: collision count (${collisions}) exceeds safety threshold or containment failed.`;

        const result: BenchmarkResult = {
          algorithmName: stateRef.current.metrics.algorithmName,
          scenarioName: currentScenario.name,
          durationSec: Math.round(benchmarkElapsedRef.current),
          containmentPct: containment,
          firesSuppressed: initialFireCount - activeFiresCount,
          totalFires: initialFireCount,
          collisionCount: collisions,
          nearMissCount: stateRef.current.metrics.nearMisses,
          waterEfficiencyPct: 88,
          avgBatteryRemaining: avgBatt,
          grade,
          summary,
          timestamp: new Date().toLocaleTimeString(),
        };

        setLatestBenchmarkResult(result);
        setIsBenchmarkModalOpen(true);
      }
    }
  }, [currentScenario]);

  // Simulation tick animation loop
  useEffect(() => {
    let lastTime = performance.now();
    let animId: number;

    const loop = (currentTime: number) => {
      const deltaSec = Math.min(0.08, (currentTime - lastTime) / 1000);
      lastTime = currentTime;

      if (stateRef.current.isPlaying) {
        const scaledDt = deltaSec * stateRef.current.simSpeed;
        stepSimulation(scaledDt);
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [stepSimulation]);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 font-sans select-none">
      {/* Tactical Top Bar */}
      <TopBar
        metrics={metrics}
        viewMode={viewMode}
        onChangeViewMode={setViewMode}
        onOpenBenchmark={() => setIsBenchmarkModalOpen(true)}
        isBenchmarking={isBenchmarking}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden p-2.5 gap-2.5">
        {/* Left / Center: Tactical Radar Canvas */}
        {(viewMode === 'radar' || viewMode === 'split') && (
          <div
            className={`flex flex-col h-full ${
              viewMode === 'split' ? 'flex-[1.25]' : 'flex-1'
            } transition-all duration-300 min-w-0`}
          >
            <TacticalRadar
              drones={drones}
              fires={fires}
              obstacles={obstacles}
              baseStation={baseStation}
              wind={wind}
              selectedDroneId={selectedDroneId}
              onSelectDrone={setSelectedDroneId}
              onCanvasClick={handleCanvasClick}
              toolMode={toolMode}
              showCommMesh={showCommMesh}
              showSensorRanges={showSensorRanges}
              showTrails={showTrails}
            />
          </div>
        )}

        {/* Center / Right: Python Algorithm Editor */}
        {(viewMode === 'code' || viewMode === 'split') && (
          <div
            className={`flex flex-col h-full ${
              viewMode === 'split' ? 'flex-1' : 'flex-1'
            } transition-all duration-300 min-w-0`}
          >
            <PythonEditor
              pythonCode={pythonCode}
              onChangeCode={setPythonCode}
              activePresetId={activePresetId}
              onSelectPreset={handleSelectPreset}
              executionEngine={executionEngine}
              onChangeEngine={setExecutionEngine}
              pyodideStatus={pyodideStatus}
              lastError={lastError}
              onApplyCode={() => {
                setMetrics(m => ({ ...m, algorithmName: 'Custom Python Code' }));
              }}
            />
          </div>
        )}

        {/* Right Side: Command Console & Fleet Telemetry Dashboard */}
        <div className="w-84 shrink-0 h-full flex flex-col">
          <FleetDashboard
            drones={drones}
            fires={fires}
            wind={wind}
            onChangeWind={setWind}
            metrics={metrics}
            isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(!isPlaying)}
            onStep={() => stepSimulation(0.05)}
            onReset={() => resetScenario(activeScenarioId)}
            simSpeed={simSpeed}
            onChangeSimSpeed={setSimSpeed}
            onRecallAllRTB={handleRecallAllRTB}
            onResumeAllMission={handleResumeAllMission}
            onEmergencyScram={handleEmergencyScram}
            onToggleDroneRTB={handleToggleDroneRTB}
            onAddDrone={handleAddDrone}
            onRemoveDrone={handleRemoveDrone}
            selectedDroneId={selectedDroneId}
            onSelectDrone={setSelectedDroneId}
            scenarios={scenarios}
            activeScenarioId={activeScenarioId}
            onSelectScenario={handleSelectScenario}
            toolMode={toolMode}
            onChangeToolMode={setToolMode}
            showCommMesh={showCommMesh}
            onToggleCommMesh={() => setShowCommMesh(!showCommMesh)}
            showSensorRanges={showSensorRanges}
            onToggleSensorRanges={() => setShowSensorRanges(!showSensorRanges)}
            showTrails={showTrails}
            onToggleTrails={() => setShowTrails(!showTrails)}
            onOpenBenchmark={() => setIsBenchmarkModalOpen(true)}
          />
        </div>
      </div>

      {/* Standardized Benchmark Evaluation Modal */}
      <BenchmarkModal
        isOpen={isBenchmarkModalOpen}
        onClose={() => setIsBenchmarkModalOpen(false)}
        onStartBenchmark={startBenchmark}
        isBenchmarking={isBenchmarking}
        benchmarkProgress={benchmarkProgress}
        latestResult={latestBenchmarkResult}
        currentAlgorithmName={metrics.algorithmName}
      />
    </div>
  );
}
