import React, { useState } from 'react';
import { Play, CheckCircle2, AlertTriangle, Copy, Download, RotateCcw, BookOpen, Terminal, Cpu } from 'lucide-react';
import { ALGORITHM_PRESETS } from '../simulator/algorithms';
import { testPythonCode, PyodideStatus } from '../simulator/pythonBridge';

interface PythonEditorProps {
  pythonCode: string;
  onChangeCode: (code: string) => void;
  activePresetId: string;
  onSelectPreset: (presetId: string) => void;
  executionEngine: 'pyodide' | 'native';
  onChangeEngine: (engine: 'pyodide' | 'native') => void;
  pyodideStatus: PyodideStatus;
  lastError: string | null;
  onApplyCode: () => void;
}

export const PythonEditor: React.FC<PythonEditorProps> = ({
  pythonCode,
  onChangeCode,
  activePresetId,
  onSelectPreset,
  executionEngine,
  onChangeEngine,
  pyodideStatus,
  lastError,
  onApplyCode,
}) => {
  const [testResult, setTestResult] = useState<{
    running: boolean;
    success?: boolean;
    message?: string;
  } | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleTestCode = async () => {
    setTestResult({ running: true });
    try {
      const result = await testPythonCode(pythonCode, 4);
      setTestResult({
        running: false,
        success: result.success,
        message: result.success ? result.message : `${result.message}\n${result.traceback || ''}`,
      });
      if (result.success) {
        onApplyCode();
      }
    } catch (err: any) {
      setTestResult({
        running: false,
        success: false,
        message: err.message || 'Error executing Python test',
      });
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(pythonCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([pythonCode], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'swarm_algorithm.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const newCode = pythonCode.substring(0, start) + '    ' + pythonCode.substring(end);
      onChangeCode(newCode);
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      }, 0);
    }
  };

  // Line count for gutter
  const lineCount = pythonCode.split('\n').length;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

  return (
    <div className="flex flex-col h-full bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
      {/* Editor Top Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-950 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-sky-400 font-semibold font-mono">
            <Terminal className="w-4 h-4" />
            <span>ALGORITHM PYTHON ENGINE</span>
          </div>

          {/* Preset Selector */}
          <select
            value={activePresetId}
            onChange={e => onSelectPreset(e.target.value)}
            className="bg-slate-900 text-slate-200 border border-slate-700 rounded px-2.5 py-1 text-xs focus:ring-1 focus:ring-sky-500 focus:outline-none"
            aria-label="Select Algorithm Preset"
          >
            {ALGORITHM_PRESETS.map(preset => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
        </div>

        {/* Engine Switcher & Status */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-900 border border-slate-700 rounded p-0.5">
            <button
              onClick={() => onChangeEngine('native')}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                executionEngine === 'native'
                  ? 'bg-sky-600 text-white font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Compiled high-FPS simulation mode"
            >
              Native Engine
            </button>
            <button
              onClick={() => onChangeEngine('pyodide')}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors flex items-center gap-1 ${
                executionEngine === 'pyodide'
                  ? 'bg-amber-600 text-white font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Runs actual Python in WebAssembly"
            >
              <Cpu className="w-3 h-3" />
              Pyodide Python
            </button>
          </div>

          <button
            onClick={() => setShowDocs(!showDocs)}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 flex items-center gap-1 transition-colors"
            title="API Documentation"
            aria-label="Toggle Python API Reference"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>API Docs</span>
          </button>

          <button
            onClick={handleCopy}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition-colors"
            title="Copy Code"
            aria-label="Copy Python Code"
          >
            <Copy className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={handleDownload}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition-colors"
            title="Download .py file"
            aria-label="Download Python Code"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* API Documentation Drawer */}
      {showDocs && (
        <div className="bg-slate-950/95 border-b border-sky-900/50 p-4 text-xs font-mono text-slate-300 max-h-56 overflow-y-auto">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
            <span className="text-sky-400 font-bold">Python Function Signature Reference</span>
            <button
              onClick={() => setShowDocs(false)}
              className="text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>
          <p className="text-slate-400 mb-2 font-sans">
            Your algorithm must define the <code className="text-amber-300">update_drones(drones, fires, obstacles, base, wind, dt)</code> function:
          </p>
          <pre className="text-[11px] text-slate-300 bg-slate-900 p-2.5 rounded border border-slate-800 overflow-x-auto">
{`def update_drones(drones, fires, obstacles, base, wind, dt):
    # drones: list of dicts: [{'id': 'DRN-01', 'x': float, 'y': float, 'vx': float, 'vy': float,
    #                          'water': float, 'battery': float, 'is_manual_rtb': bool, 'max_speed': float}]
    # fires: list of dicts: [{'id': 'fire-1', 'x': float, 'y': float, 'intensity': float, 'radius': float}]
    # obstacles: list of dicts: [{'id': 'obs-1', 'x': float, 'y': float, 'radius': float, 'type': str}]
    # base: {'x': float, 'y': float, 'radius': float}
    # wind: {'speed': float, 'direction': float (deg)}
    # dt: float (seconds per tick)
    #
    # RETURN: list of action dicts:
    # [{'id': 'DRN-01', 'vx': float, 'vy': float, 'suppress': bool, 'state': 'SEARCHING'|'SUPPRESSING'|'RETURNING'}]`}
          </pre>
        </div>
      )}

      {/* Code Editor Body */}
      <div className="relative flex-1 flex overflow-hidden font-mono text-xs">
        {/* Line Numbers Gutter */}
        <div className="w-10 bg-slate-950 py-3 text-right pr-2 text-slate-600 select-none border-r border-slate-800/80 overflow-hidden">
          {lineNumbers.map(n => (
            <div key={n} className="leading-5 h-5">
              {n}
            </div>
          ))}
        </div>

        {/* Text Area */}
        <textarea
          value={pythonCode}
          onChange={e => onChangeCode(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          className="flex-1 w-full bg-slate-950/80 text-emerald-300 p-3 leading-5 resize-none focus:outline-none selection:bg-sky-800/50 overflow-auto whitespace-pre font-mono"
          aria-label="Python Code Editor"
        />
      </div>

      {/* Live Error / Test Result Bar */}
      {(testResult || lastError) && (
        <div
          className={`px-4 py-2 border-t text-xs font-mono flex items-start gap-2 ${
            testResult?.success
              ? 'bg-emerald-950/40 border-emerald-800 text-emerald-300'
              : 'bg-rose-950/40 border-rose-800 text-rose-300'
          }`}
        >
          {testResult?.success ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          )}
          <div className="flex-1 overflow-x-auto whitespace-pre-wrap max-h-24">
            {testResult?.running
              ? 'Validating Python code contract...'
              : testResult?.message || lastError}
          </div>
        </div>
      )}

      {/* Editor Bottom Actions */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-950 border-t border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>
            Engine: {executionEngine === 'pyodide' ? 'Pyodide Wasm Python' : 'Simulated Native Python'}
          </span>
          {copied && <span className="text-sky-400">Code copied to clipboard!</span>}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onSelectPreset(activePresetId)}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs font-mono border border-slate-700 flex items-center gap-1.5 transition-colors"
            title="Reset code to current preset template"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          <button
            onClick={handleTestCode}
            disabled={testResult?.running}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-sky-300 rounded text-xs font-mono border border-sky-800/80 flex items-center gap-1.5 transition-colors"
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>{testResult?.running ? 'Testing...' : 'Test Syntax'}</span>
          </button>

          <button
            onClick={onApplyCode}
            className="px-4 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded text-xs font-mono font-semibold flex items-center gap-1.5 shadow-lg shadow-sky-600/20 transition-all"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Apply to Swarm</span>
          </button>
        </div>
      </div>
    </div>
  );
};
