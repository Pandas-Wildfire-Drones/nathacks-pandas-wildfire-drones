# Decision 0002: Perception sensor wiring (pose sync, terrain/AGL, RTK loss, camera health, lens distortion)

**Status:** ACCEPTED, implemented in `perception/` (2026-11-07) · **Owner:** C · **No contract change**

> **Judge question:** "A thermal pixel is just a temperature. How do you turn it into a GPS point a 800 kg drone can
> fly to, and what happens when a sensor misbehaves?"
>
> **Our answer:** "Every frame is matched to the drone's pose at the exact moment of exposure, projected onto the
> terrain model, and corrected for the lens. When RTK or the camera degrade, perception widens its uncertainty or
> drops the frame, and it never records 'searched, nothing here' from a bad frame."

## What was missing and what we did

| Gap | Risk if ignored | Implementation | Default `[ASSUMPTION]` |
|---|---|---|---|
| **Pose ↔ frame time sync** | Using "latest pose" instead of "pose at exposure": 15 m/s × 50 ms = **0.75 m** error, worse while turning | `PoseBuffer` interpolates position and attitude (shortest-arc yaw) at `ThermalFrame.timestamp` (= exposure time); frames with no pose within 0.25 s are dropped | max gap 0.25 s, extrapolation ≤ 50 ms |
| **Terrain + height above ground** | Flat-ground georef on a slope: a 5 m height error moves an edge hotspot ~1.4 m (seeker), more for the extinguisher's 50° lens; wrong pixel size → wrong area and wrong patch scale | `DemSampler` (bilinear, grid-aligned) → `TerrainGeoreferencer`; AGL from the altimeter when within its rated range, else pose z − DEM (decision 0001) | altimeter max 120 m |
| **GNSS-RTK lost** (`sensors_ok["gnss_rtk"]=False`) | Positions jump metres; the tracker spawns duplicate hotspots; map evidence lands on the wrong cells | Tracker gate +3 m, map hit radius +3 m, position weight × 0.25 while RTK is down | non-RTK GNSS ~1–3 m `[VERIFY]` |
| **Thermal health / flat-field correction (FFC)** | Boson-class cores pause to recalibrate (shutter); frames during FFC are frozen. A faulty camera reports "nothing hot" | Frame skipped **and no "searched, empty" evidence written** to the log-odds map | — |
| **Gain-mode saturation** | Bypass trigger tied to the wrong range (decision 0003) | `FrameContext.saturation_c` from the camera's current gain mode | config 550 °C, but Boson+ low gain ≈ 350 °C [S23]: E to set |
| **Friendly drones / industrial heat** | An extinguisher's engine or a gas flare passes the hot-blob bypass | `FrameContext.friendly_positions` / `known_hot_sources` → image-space exclusion zones that remove the bypass (not the detection) | 15 m radius |
| **Lens distortion** | See below | `LensDistortion` (Brown–Conrady k1, k2, p1, p2), undistort before ray casting; `FakeDetector` can distort for the sim | none (pinhole) |

All of this goes through a perception-local `FrameContext` passed next to the frame:
`SeekerPerception.process(frame, ctx)`. It doesn't change `ThermalFrame`, which is a contract.

## Lens distortion: is it worth it?

Edge-of-image georeferencing error if a real lens's distortion is ignored. **The k1 values are illustrative
assumptions** (typical mild barrel distortion), not calibrated values:

| Camera | AGL | k1 | Error at image edge | In pixels |
|---|---|---|---|---|
| Seeker 640×512, 32° | 80 m | −0.05 | 0.18 m | 2.4 px |
| Seeker 640×512, 32° | 80 m | −0.10 | 0.35 m | 4.9 px |
| Extinguisher 320×256, 50° | 120 m (transit) | −0.10 | **2.3 m** | 6.5 px |
| Extinguisher 320×256, 50° | 120 m (transit) | −0.20 | **4.5 m** | 12.9 px |
| Extinguisher 320×256, 50° | 15 m (drop height) | −0.20 | 0.57 m | 12.9 px |

**Decision:**
- **Simulator / demo:** pinhole by default (the synthetic frames are rendered pinhole, so there's nothing to
  correct). The code path exists: pass the same `LensDistortion` to `FakeDetector` and to the georeferencer to
  show "uncalibrated vs calibrated" in the demo.
- **Seeker:** negligible next to the tracker's 5 m gate.
- **Extinguisher:** matters at transit altitude (metres) if it re-acquires a hotspot near the image edge. Mitigation
  without calibration: centre the hotspot before trusting its position (the error shrinks to ~0 at the image
  centre).
- **Real hardware:** calibrate each thermal camera (heated or cooled checkerboard) and then propose adding
  distortion coefficients to `contracts.CameraModel`. That's a contract change, so it needs the integration owner's OK
  (context §0.3).

## Deferred contract changes (for the integration owner, when real hardware arrives)
- `CameraModel`: distortion coefficients and intrinsics (fx, fy, cx, cy) from calibration.
- `ThermalFrame`: optional metadata (gain mode, FFC flag, exposure timestamp source), or keep `FrameContext` local.
- `config.py`: rangefinder/altimeter range (decision 0001), thermal saturation per gain mode (context §12 #11).
