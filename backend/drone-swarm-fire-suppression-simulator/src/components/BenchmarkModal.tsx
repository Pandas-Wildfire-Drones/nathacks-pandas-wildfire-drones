import React, { useState } from 'react';
import { Award, ShieldAlert, CheckCircle, X, Download, Play, Flame, Droplets, BatteryCharging } from 'lucide-react';
import { BenchmarkResult } from '../simulator/types';

interface BenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartBenchmark: (durationSec: number) => void;
  isBenchmarking: boolean;
  benchmarkProgress: number; // 0 to 100
  latestResult: BenchmarkResult | null;
  currentAlgorithmName: string;
}

export const BenchmarkModal: React.FC<BenchmarkModalProps> = ({
  isOpen,
  onClose,
  onStartBenchmark,
  isBenchmarking,
  benchmarkProgress,
  latestResult,
  currentAlgorithmName,
}) => {
  const [selectedDuration, setSelectedDuration] = useState<number>(45);

  if (!isOpen) return null;

  const handleExport = () => {
    if (!latestResult) return;
    const blob = new Blob([JSON.stringify(latestResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_${latestResult.algorithmName.replace(/\s+/g, '_').toLowerCase()}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-sky-950 border border-sky-800 text-sky-400">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 font-mono">SWARM ALGORITHM TEST BENCH</h2>
              <p className="text-xs text-slate-400">Standardized verification & pass/fail scorecard</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
            aria-label="Close Benchmark Modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Active Configuration */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-mono">TARGET ALGORITHM</span>
              <span className="text-sky-400 font-bold font-mono">{currentAlgorithmName}</span>
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-mono">BENCHMARK DURATION</span>
              <div className="flex items-center gap-1.5">
                {[10, 20, 30, 45, 60].map(sec => (
                  <button
                    key={sec}
                    onClick={() => setSelectedDuration(sec)}
                    disabled={isBenchmarking}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                      selectedDuration === sec
                        ? 'bg-sky-600 text-white font-bold'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    {sec}s
                  </button>
                ))}
              </div>
            </div>

            {/* Benchmark Progress Bar */}
            {isBenchmarking && (
              <div className="space-y-1.5 pt-2 border-t border-slate-800">
                <div className="flex justify-between text-xs font-mono text-slate-400">
                  <span>Executing live automated trial...</span>
                  <span className="text-sky-400 font-bold">{Math.round(benchmarkProgress)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-sky-500 to-emerald-500 transition-all duration-200"
                    style={{ width: `${benchmarkProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Latest Result Card */}
          {latestResult ? (
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono tracking-wider text-slate-400">TRIAL SCORECARD</span>
                  <h3 className="text-lg font-bold text-white font-mono">{latestResult.scenarioName}</h3>
                </div>

                <div
                  className={`flex items-center justify-center w-14 h-14 rounded-xl text-2xl font-black font-mono shadow-lg ${
                    latestResult.grade === 'S'
                      ? 'bg-emerald-950 border border-emerald-500 text-emerald-400 shadow-emerald-900/30'
                      : latestResult.grade === 'A'
                      ? 'bg-sky-950 border border-sky-500 text-sky-400 shadow-sky-900/30'
                      : latestResult.grade === 'B'
                      ? 'bg-amber-950 border border-amber-500 text-amber-400 shadow-amber-900/30'
                      : 'bg-rose-950 border border-rose-500 text-rose-400 shadow-rose-900/30'
                  }`}
                >
                  {latestResult.grade}
                </div>
              </div>

              {/* KPI Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <Flame className="w-3.5 h-3.5 text-orange-400" />
                    <span>CONTAINED</span>
                  </div>
                  <div className="text-base font-bold text-emerald-400">
                    {Math.round(latestResult.containmentPct)}%
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {latestResult.firesSuppressed}/{latestResult.totalFires} fires
                  </div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                    <span>COLLISIONS</span>
                  </div>
                  <div
                    className={`text-base font-bold ${
                      latestResult.collisionCount === 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {latestResult.collisionCount}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {latestResult.nearMissCount} near-misses
                  </div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <Droplets className="w-3.5 h-3.5 text-sky-400" />
                    <span>WATER EFF</span>
                  </div>
                  <div className="text-base font-bold text-sky-400">
                    {Math.round(latestResult.waterEfficiencyPct)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Target hit accuracy</div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <BatteryCharging className="w-3.5 h-3.5 text-amber-400" />
                    <span>RESERVE BATT</span>
                  </div>
                  <div className="text-base font-bold text-amber-400">
                    {Math.round(latestResult.avgBatteryRemaining)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Fleet average</div>
                </div>
              </div>

              {/* Summary text */}
              <p className="text-xs text-slate-300 bg-slate-900/50 p-3 rounded-lg border border-slate-800/80 leading-relaxed">
                {latestResult.summary}
              </p>
            </div>
          ) : (
            <div className="text-center py-8 px-4 border border-dashed border-slate-800 rounded-xl">
              <ShieldAlert className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-300">No Benchmark Trial Run Yet</p>
              <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                Launch a standardized test trial to evaluate obstacle avoidance, fire containment, and zero-collision safety guarantees.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950 border-t border-slate-800">
          {latestResult ? (
            <button
              onClick={handleExport}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono border border-slate-700 flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export JSON Report</span>
            </button>
          ) : <div />}

          <button
            onClick={() => onStartBenchmark(selectedDuration)}
            disabled={isBenchmarking}
            className="px-5 py-2 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-xs font-mono font-semibold flex items-center gap-2 shadow-lg shadow-sky-600/20 transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{isBenchmarking ? 'Running Benchmark...' : 'Run Automated Benchmark'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
