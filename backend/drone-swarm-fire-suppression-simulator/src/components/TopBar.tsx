import React from 'react';
import { Shield, ShieldAlert, Flame, Droplets, Terminal, Radio, Award } from 'lucide-react';
import { SimulationMetrics } from '../simulator/types';

interface TopBarProps {
  metrics: SimulationMetrics;
  viewMode: 'radar' | 'split' | 'code';
  onChangeViewMode: (mode: 'radar' | 'split' | 'code') => void;
  onOpenBenchmark: () => void;
  isBenchmarking: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  metrics,
  viewMode,
  onChangeViewMode,
  onOpenBenchmark,
  isBenchmarking,
}) => {
  return (
    <header className="h-14 bg-slate-950 border-b border-slate-800 px-4 flex items-center justify-between select-none z-20">
      {/* Brand & Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-sky-950 border border-sky-800 flex items-center justify-center text-sky-400">
            <Radio className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold text-white tracking-wider font-mono">
                AERO-SWARM SIMULATOR
              </h1>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono border border-slate-700">
                PYTHON ENGINE
              </span>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
              <span className="flex items-center gap-1 text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ONLINE
              </span>
              <span>•</span>
              <span>ALGO: {metrics.algorithmName}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Center Live Mission Telemetry Counters */}
      <div className="hidden md:flex items-center gap-6 font-mono text-xs">
        {/* Fire Containment Gauge */}
        <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-lg border border-slate-800">
          <Flame className="w-4 h-4 text-orange-500" />
          <div>
            <div className="flex items-center justify-between gap-3 text-[10px] text-slate-400">
              <span>CONTAINED:</span>
              <span className="font-bold text-white">{Math.round(metrics.fireContainedPct)}%</span>
            </div>
            <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-0.5">
              <div
                className="h-full bg-gradient-to-r from-orange-500 to-emerald-500 transition-all duration-300"
                style={{ width: `${metrics.fireContainedPct}%` }}
              />
            </div>
          </div>
        </div>

        {/* Safety & Collision Alert Counter */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono ${
            metrics.droneCollisions === 0
              ? 'bg-slate-900/90 border-slate-800 text-slate-300'
              : 'bg-rose-950/60 border-rose-800 text-rose-300 animate-pulse'
          }`}
        >
          {metrics.droneCollisions === 0 ? (
            <Shield className="w-4 h-4 text-emerald-400" />
          ) : (
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          )}
          <div>
            <div className="text-[10px] text-slate-400">COLLISIONS:</div>
            <div className={`font-bold ${metrics.droneCollisions === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {metrics.droneCollisions} <span className="text-[10px] text-slate-500 font-normal">({metrics.nearMisses} near)</span>
            </div>
          </div>
        </div>

        {/* Water Payload Remaining */}
        <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-lg border border-slate-800">
          <Droplets className="w-4 h-4 text-sky-400" />
          <div>
            <div className="text-[10px] text-slate-400">WATER DROPPED:</div>
            <div className="font-bold text-sky-300">
              {Math.round(metrics.totalWaterDropped)} L
            </div>
          </div>
        </div>
      </div>

      {/* Right Action Deck & View Mode Switcher */}
      <div className="flex items-center gap-3">
        {/* Test Bench Button */}
        <button
          onClick={onOpenBenchmark}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-all ${
            isBenchmarking
              ? 'bg-amber-600 text-white animate-pulse'
              : 'bg-slate-900 hover:bg-slate-800 text-sky-400 border border-sky-800/80 shadow-md shadow-sky-950'
          }`}
          aria-label="Open Test Bench Benchmark"
        >
          <Award className="w-3.5 h-3.5" />
          <span>{isBenchmarking ? 'Testing...' : 'Test Bench'}</span>
        </button>

        {/* View Mode Buttons */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
          <button
            onClick={() => onChangeViewMode('radar')}
            className={`px-2.5 py-1 rounded transition-colors ${
              viewMode === 'radar'
                ? 'bg-sky-600 text-white font-medium shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            aria-label="Switch to Radar View"
          >
            Radar
          </button>
          <button
            onClick={() => onChangeViewMode('split')}
            className={`px-2.5 py-1 rounded transition-colors ${
              viewMode === 'split'
                ? 'bg-sky-600 text-white font-medium shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            aria-label="Switch to Split View"
          >
            Split
          </button>
          <button
            onClick={() => onChangeViewMode('code')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1 ${
              viewMode === 'code'
                ? 'bg-sky-600 text-white font-medium shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            aria-label="Switch to Code View"
          >
            <Terminal className="w-3 h-3" />
            <span>Python</span>
          </button>
        </div>
      </div>
    </header>
  );
};
