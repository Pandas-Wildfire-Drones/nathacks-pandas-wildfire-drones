import React from 'react';
import {
  Home,
  Shield,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Flame,
  Plus,
  Minus,
  Navigation,
  Droplets,
  Battery,
  Wind as WindIcon,
  Crosshair,
  Layers,
  Award,
} from 'lucide-react';
import { Drone, FireSpot, Wind, SimulationMetrics } from '../simulator/types';
import { Scenario } from '../simulator/scenarios';

interface FleetDashboardProps {
  drones: Drone[];
  fires: FireSpot[];
  wind: Wind;
  onChangeWind: (wind: Wind) => void;
  metrics: SimulationMetrics;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStep: () => void;
  onReset: () => void;
  simSpeed: number;
  onChangeSimSpeed: (speed: number) => void;
  onRecallAllRTB: () => void;
  onResumeAllMission: () => void;
  onEmergencyScram: () => void;
  onToggleDroneRTB: (droneId: string) => void;
  onAddDrone: () => void;
  onRemoveDrone: () => void;
  selectedDroneId: string | null;
  onSelectDrone: (droneId: string | null) => void;
  scenarios: Scenario[];
  activeScenarioId: string;
  onSelectScenario: (scenarioId: string) => void;
  toolMode: 'inspect' | 'add_fire' | 'add_obstacle' | 'water_drop';
  onChangeToolMode: (mode: 'inspect' | 'add_fire' | 'add_obstacle' | 'water_drop') => void;
  showCommMesh: boolean;
  onToggleCommMesh: () => void;
  showSensorRanges: boolean;
  onToggleSensorRanges: () => void;
  showTrails: boolean;
  onToggleTrails: () => void;
  onOpenBenchmark: () => void;
}

export const FleetDashboard: React.FC<FleetDashboardProps> = ({
  drones,
  fires,
  wind,
  onChangeWind,
  metrics,
  isPlaying,
  onTogglePlay,
  onStep,
  onReset,
  simSpeed,
  onChangeSimSpeed,
  onRecallAllRTB,
  onResumeAllMission,
  onEmergencyScram,
  onToggleDroneRTB,
  onAddDrone,
  onRemoveDrone,
  selectedDroneId,
  onSelectDrone,
  scenarios,
  activeScenarioId,
  onSelectScenario,
  toolMode,
  onChangeToolMode,
  showCommMesh,
  onToggleCommMesh,
  showSensorRanges,
  onToggleSensorRanges,
  showTrails,
  onToggleTrails,
  onOpenBenchmark,
}) => {
  const allInRTB = drones.length > 0 && drones.every(d => d.isManualRTB);

  return (
    <div className="flex flex-col h-full bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-xl text-slate-100 text-xs font-mono">
      {/* 1. Header & Primary RTB Command Deck */}
      <div className="p-3.5 bg-slate-950 border-b border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-ping" />
            <span className="font-bold text-sky-400 tracking-wide">FLEET COMMAND CONSOLE</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-slate-400">FLEET SIZE:</span>
            <span className="font-bold text-white bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
              {drones.length} DRONES
            </span>
            <button
              onClick={onRemoveDrone}
              disabled={drones.length <= 1}
              className="p-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded border border-slate-700"
              title="Remove drone"
              aria-label="Remove Drone"
            >
              <Minus className="w-3 h-3" />
            </button>
            <button
              onClick={onAddDrone}
              disabled={drones.length >= 16}
              className="p-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded border border-slate-700"
              title="Add drone to swarm"
              aria-label="Add Drone"
            >
              <Plus className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Big RTB Buttons (Crucial requested feature!) */}
        <div className="grid grid-cols-2 gap-2 pt-1">
          {allInRTB ? (
            <button
              onClick={onResumeAllMission}
              className="col-span-1 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-bold flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition-all border border-emerald-500/50"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>RESUME AUTONOMY</span>
            </button>
          ) : (
            <button
              onClick={onRecallAllRTB}
              className="col-span-1 px-3 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-bold flex items-center justify-center gap-2 shadow-lg shadow-amber-600/20 transition-all border border-amber-500/50"
            >
              <Home className="w-4 h-4" />
              <span>RECALL ALL TO BASE</span>
            </button>
          )}

          <button
            onClick={onEmergencyScram}
            className="col-span-1 px-3 py-2 bg-slate-800 hover:bg-rose-900/60 hover:text-rose-200 text-slate-300 rounded-lg font-bold flex items-center justify-center gap-2 transition-all border border-slate-700"
            title="Immediate hover and collision lock"
          >
            <Shield className="w-4 h-4 text-rose-400" />
            <span>EMERGENCY SCRAM</span>
          </button>
        </div>
      </div>

      {/* 2. Simulation Playback & Speed */}
      <div className="px-3.5 py-2.5 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <button
            onClick={onTogglePlay}
            className={`px-3 py-1.5 rounded-md font-bold flex items-center gap-1.5 transition-colors ${
              isPlaying
                ? 'bg-amber-600 text-white hover:bg-amber-500'
                : 'bg-emerald-600 text-white hover:bg-emerald-500'
            }`}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>{isPlaying ? 'PAUSE' : 'SIMULATE'}</span>
          </button>

          <button
            onClick={onStep}
            disabled={isPlaying}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded-md border border-slate-700"
            title="Single simulation tick"
            aria-label="Step simulation forward"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={onReset}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md border border-slate-700"
            title="Reset scenario to initial state"
            aria-label="Reset scenario"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Speed multiplier */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-500">SPEED:</span>
          {[0.5, 1, 2, 4].map(s => (
            <button
              key={s}
              onClick={() => onChangeSimSpeed(s)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                simSpeed === s
                  ? 'bg-sky-600 text-white font-bold'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* 3. Scenario & Interactive Tool Toggles */}
      <div className="p-3.5 border-b border-slate-800 space-y-2.5">
        <div className="flex items-center justify-between">
          <label className="text-[10px] text-slate-400 flex items-center gap-1">
            <Layers className="w-3 h-3 text-sky-400" />
            <span>TACTICAL SCENARIO</span>
          </label>
          <button
            onClick={onOpenBenchmark}
            className="text-[10px] px-2 py-0.5 bg-sky-950 hover:bg-sky-900 text-sky-300 border border-sky-800 rounded flex items-center gap-1 transition-colors"
          >
            <Award className="w-3 h-3 text-sky-400" />
            <span>RUN TEST BENCH</span>
          </button>
        </div>

        <select
          value={activeScenarioId}
          onChange={e => onSelectScenario(e.target.value)}
          className="w-full bg-slate-950 text-slate-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 focus:outline-none"
        >
          {scenarios.map(s => (
            <option key={s.id} value={s.id}>
              {s.name} ({s.badge})
            </option>
          ))}
        </select>

        {/* Interactive Click Tools */}
        <div className="pt-1 space-y-1.5">
          <span className="text-[10px] text-slate-400 flex items-center gap-1">
            <Crosshair className="w-3 h-3 text-emerald-400" />
            <span>RADAR CLICK ACTION</span>
          </span>
          <div className="grid grid-cols-4 gap-1 text-[10px]">
            <button
              onClick={() => onChangeToolMode('inspect')}
              className={`py-1 rounded text-center border transition-colors ${
                toolMode === 'inspect'
                  ? 'bg-sky-950 border-sky-500 text-sky-300 font-bold'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              Select
            </button>
            <button
              onClick={() => onChangeToolMode('add_fire')}
              className={`py-1 rounded text-center border transition-colors ${
                toolMode === 'add_fire'
                  ? 'bg-orange-950 border-orange-500 text-orange-300 font-bold'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              + Fire
            </button>
            <button
              onClick={() => onChangeToolMode('add_obstacle')}
              className={`py-1 rounded text-center border transition-colors ${
                toolMode === 'add_obstacle'
                  ? 'bg-rose-950 border-rose-500 text-rose-300 font-bold'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              + Tower
            </button>
            <button
              onClick={() => onChangeToolMode('water_drop')}
              className={`py-1 rounded text-center border transition-colors ${
                toolMode === 'water_drop'
                  ? 'bg-blue-950 border-blue-500 text-blue-300 font-bold'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              Drop
            </button>
          </div>
        </div>

        {/* Wind Controller */}
        <div className="pt-1 space-y-1.5">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <WindIcon className="w-3 h-3 text-sky-400" />
              <span>WIND VECTOR</span>
            </span>
            <span className="text-white font-mono">
              {wind.speed.toFixed(1)} m/s @ {Math.round(wind.direction)}°
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-[9px] text-slate-500">Speed</span>
              <input
                type="range"
                min="0"
                max="12"
                step="0.5"
                value={wind.speed}
                onChange={e => onChangeWind({ ...wind, speed: parseFloat(e.target.value) })}
                className="w-full accent-sky-500 h-1 bg-slate-800 rounded"
              />
            </div>
            <div>
              <span className="text-[9px] text-slate-500">Direction</span>
              <input
                type="range"
                min="0"
                max="360"
                step="5"
                value={wind.direction}
                onChange={e => onChangeWind({ ...wind, direction: parseFloat(e.target.value) })}
                className="w-full accent-sky-500 h-1 bg-slate-800 rounded"
              />
            </div>
          </div>
        </div>

        {/* Visualization layer checkboxes */}
        <div className="pt-1 flex items-center justify-between text-[10px] text-slate-400">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showCommMesh}
              onChange={onToggleCommMesh}
              className="accent-sky-500"
            />
            <span>Mesh Links</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showSensorRanges}
              onChange={onToggleSensorRanges}
              className="accent-sky-500"
            />
            <span>Sensors</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={onToggleTrails}
              className="accent-sky-500"
            />
            <span>Trails</span>
          </label>
        </div>
      </div>

      {/* 4. Fleet Drones Telemetry Table (Granular Per-Drone RTB Controls) */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="px-3.5 py-2 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-semibold text-slate-300">
          <span>INDIVIDUAL DRONE TELEMETRY</span>
          <span className="text-[10px] text-slate-500">CLICK TO COMMAND RTB</span>
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60 p-1">
          {drones.map(drone => {
            const isSelected = drone.id === selectedDroneId;
            return (
              <div
                key={drone.id}
                onClick={() => onSelectDrone(drone.id)}
                className={`p-2.5 rounded-lg cursor-pointer transition-colors ${
                  isSelected
                    ? 'bg-sky-950/70 border border-sky-700/80'
                    : 'hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: drone.color || '#38bdf8' }}
                    />
                    <span className="font-bold text-white">{drone.id}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-semibold ${
                        drone.state === 'SUPPRESSING'
                          ? 'bg-blue-900/60 text-blue-300 border border-blue-700'
                          : drone.state === 'RETURNING'
                          ? 'bg-amber-900/60 text-amber-300 border border-amber-700'
                          : drone.state === 'REFILLING'
                          ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {drone.state}
                    </span>
                  </div>

                  {/* Individual RTB Toggle Button */}
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      onToggleDroneRTB(drone.id);
                    }}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
                      drone.isManualRTB
                        ? 'bg-amber-600 text-white border-amber-500 font-bold'
                        : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                    }`}
                    title={drone.isManualRTB ? 'Cancel RTB order' : 'Order this drone to RTB'}
                  >
                    {drone.isManualRTB ? 'CANCEL RTB' : 'CALL RTB'}
                  </button>
                </div>

                {/* Battery and Water gauges */}
                <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400">
                  <div className="space-y-0.5">
                    <div className="flex justify-between">
                      <span className="flex items-center gap-1">
                        <Droplets className="w-2.5 h-2.5 text-sky-400" />
                        <span>Water</span>
                      </span>
                      <span className="text-white font-mono">{Math.round(drone.water)}L</span>
                    </div>
                    <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sky-500"
                        style={{ width: `${(drone.water / drone.maxWater) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="space-y-0.5">
                    <div className="flex justify-between">
                      <span className="flex items-center gap-1">
                        <Battery className="w-2.5 h-2.5 text-emerald-400" />
                        <span>Battery</span>
                      </span>
                      <span className="text-white font-mono">{Math.round(drone.battery)}%</span>
                    </div>
                    <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          drone.battery > 40 ? 'bg-emerald-500' : 'bg-amber-500'
                        }`}
                        style={{ width: `${drone.battery}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
