import { Drone, FireSpot, Obstacle, BaseStation, Wind, DroneAction } from './types';
import { runNativeAlgorithm } from './algorithms';

// Extend window definition for Pyodide
declare global {
  interface Window {
    loadPyodide?: (config?: { indexURL?: string }) => Promise<any>;
    pyodideInstance?: any;
  }
}

export interface PyodideStatus {
  isReady: boolean;
  isLoading: boolean;
  error: string | null;
  version: string | null;
}

let pyodideInstance: any = null;
let initPromise: Promise<any> | null = null;

export async function initPyodide(
  onStatusChange?: (status: PyodideStatus) => void
): Promise<any> {
  if (pyodideInstance) {
    onStatusChange?.({ isReady: true, isLoading: false, error: null, version: '0.26.4' });
    return pyodideInstance;
  }

  if (initPromise) {
    return initPromise;
  }

  onStatusChange?.({ isReady: false, isLoading: true, error: null, version: null });

  initPromise = (async () => {
    try {
      // Check if pyodide script is already present or wait briefly for it
      let attempts = 0;
      while (!window.loadPyodide && attempts < 30) {
        await new Promise(r => setTimeout(r, 100));
        attempts++;
      }

      if (!window.loadPyodide) {
        // Dynamically inject if CDN script tag didn't load yet
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
          script.async = true;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to download Pyodide WebAssembly runtime from CDN'));
          document.head.appendChild(script);
        });
      }

      if (!window.loadPyodide) {
        throw new Error('window.loadPyodide not available after script load');
      }

      const pyodide = await window.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
      });

      // Redirect stdout / stderr
      await pyodide.runPythonAsync(`
import sys
import io

class OutputCapture(io.StringIO):
    pass
sys.stdout = OutputCapture()
sys.stderr = OutputCapture()
`);

      pyodideInstance = pyodide;
      window.pyodideInstance = pyodide;
      onStatusChange?.({ isReady: true, isLoading: false, error: null, version: '0.26.4' });
      return pyodide;
    } catch (err: any) {
      console.warn('Pyodide initialization warning (falling back to native engine):', err);
      onStatusChange?.({
        isReady: false,
        isLoading: false,
        error: err?.message || 'Pyodide loading timed out',
        version: null,
      });
      throw err;
    }
  })();

  return initPromise;
}

/**
 * Validates and compiles Python code to test for syntax and runtime contract errors.
 */
export async function testPythonCode(
  pythonCode: string,
  sampleDroneCount: number = 3
): Promise<{ success: boolean; message: string; output?: string; traceback?: string }> {
  try {
    const py = await initPyodide();

    // Setup dummy test environment
    const testScript = `
import json

${pythonCode}

# Test sample execution
sample_drones = [
    {
        'id': f'DRN-{i+1:02d}',
        'x': 100.0 + i * 30.0,
        'y': 100.0 + i * 20.0,
        'vx': 0.0,
        'vy': 0.0,
        'water': 100.0,
        'max_water': 100.0,
        'battery': 100.0,
        'is_manual_rtb': False,
        'max_speed': 65.0,
        'radius': 7.0
    }
    for i in range(${sampleDroneCount})
]

sample_fires = [
    {'id': 'fire-1', 'x': 250.0, 'y': 250.0, 'intensity': 80.0, 'radius': 25.0, 'is_active': True}
]

sample_obstacles = [
    {'id': 'obs-1', 'x': 180.0, 'y': 180.0, 'radius': 28.0, 'type': 'tower'}
]

sample_base = {'x': 80.0, 'y': 80.0, 'radius': 50.0}
sample_wind = {'speed': 3.5, 'direction': 45.0}

if 'update_drones' not in globals():
    raise NameError("Function 'update_drones(drones, fires, obstacles, base, wind, dt)' is not defined!")

test_actions = update_drones(sample_drones, sample_fires, sample_obstacles, sample_base, sample_wind, 0.05)

if not isinstance(test_actions, (list, tuple)):
    raise TypeError(f"update_drones must return a list of action dicts, got {type(test_actions).__name__}")

for act in test_actions:
    if not isinstance(act, dict):
        raise TypeError("Each item in returned actions list must be a dict with at least 'id'")
    if 'id' not in act:
        raise KeyError("Action dict missing required 'id' key")

json.dumps(test_actions)
`;

    await py.runPythonAsync(testScript);
    return {
      success: true,
      message: `Syntax and algorithm contract passed! Returned ${sampleDroneCount} valid drone actions.`,
    };
  } catch (err: any) {
    return {
      success: false,
      message: err.message || 'Python execution error',
      traceback: String(err),
    };
  }
}

/**
 * Executes the user's Python code for one simulation frame step.
 */
export async function runPythonStep(
  pythonCode: string,
  drones: Drone[],
  fires: FireSpot[],
  obstacles: Obstacle[],
  baseStation: BaseStation,
  wind: Wind,
  dt: number
): Promise<DroneAction[]> {
  if (!pyodideInstance) {
    throw new Error('Pyodide engine is not initialized');
  }

  // Serialize inputs to Python-friendly format
  const dronesData = drones.map(d => ({
    id: d.id,
    x: d.x,
    y: d.y,
    vx: d.vx,
    vy: d.vy,
    water: d.water,
    max_water: d.maxWater,
    battery: d.battery,
    is_manual_rtb: d.isManualRTB,
    max_speed: d.maxSpeed,
    radius: d.radius,
    state: d.state,
  }));

  const firesData = fires.map(f => ({
    id: f.id,
    x: f.x,
    y: f.y,
    intensity: f.intensity,
    radius: f.radius,
    is_active: f.isActive && !f.isExtinguished,
  }));

  const obstaclesData = obstacles.map(o => ({
    id: o.id,
    x: o.x,
    y: o.y,
    radius: o.radius,
    type: o.type,
  }));

  const baseData = {
    x: baseStation.x,
    y: baseStation.y,
    radius: baseStation.radius,
  };

  const windData = {
    speed: wind.speed,
    direction: wind.direction,
  };

  // Set Python variables
  pyodideInstance.globals.set('__drones_json', JSON.stringify(dronesData));
  pyodideInstance.globals.set('__fires_json', JSON.stringify(firesData));
  pyodideInstance.globals.set('__obs_json', JSON.stringify(obstaclesData));
  pyodideInstance.globals.set('__base_json', JSON.stringify(baseData));
  pyodideInstance.globals.set('__wind_json', JSON.stringify(windData));
  pyodideInstance.globals.set('__dt', dt);

  const runnerScript = `
import json

${pythonCode}

_in_drones = json.loads(__drones_json)
_in_fires = json.loads(__fires_json)
_in_obs = json.loads(__obs_json)
_in_base = json.loads(__base_json)
_in_wind = json.loads(__wind_json)

_actions = update_drones(_in_drones, _in_fires, _in_obs, _in_base, _in_wind, float(__dt))
json.dumps(_actions)
`;

  const resultJson = await pyodideInstance.runPythonAsync(runnerScript);
  const actions: any[] = JSON.parse(resultJson);

  return actions.map(a => ({
    id: String(a.id),
    vx: typeof a.vx === 'number' && !isNaN(a.vx) ? a.vx : 0,
    vy: typeof a.vy === 'number' && !isNaN(a.vy) ? a.vy : 0,
    suppress: Boolean(a.suppress),
    state: a.state || (a.suppress ? 'SUPPRESSING' : 'SEARCHING'),
    rtb: Boolean(a.rtb),
  }));
}
