# Decision 0003: Hot-blob bypass (the ML can't veto an obvious fire)

**Status:** ACCEPTED, implemented in `perception/detectors/fused.py` (2026-11-07) · **Owner:** C
**Replaces:** the first draft's fixed "peak ≥ 300 °C skips the CNN" rule

> **Judge question:** "What if your neural network says 'not a fire' when it obviously is?"
>
> **Our answer:** "It can't overrule physics. A blob that saturates the camera, or a hot fire too big for the network's
> view, is reported regardless. The bypass never skips confirmation: a seeker still checks before any water is dropped."

## Why the fixed 300 °C rule was replaced

| Problem | Detail |
|---|---|
| No source | 300 °C was a guess |
| Only catches easy fires | Smouldering peaks are 450–700 °C internally [S24], but the camera sees a cooler ash surface plus sub-pixel mixing (a fire filling 19% of a pixel at 500 °C reads ~168 °C). Small embers, the hard cases, never reach 300 °C; big hot fires, which the CNN already gets, do |
| Conflicts with the camera | Boson+ low gain tops out ~350 °C, high gain ~150 °C [S23]. In high gain a 300 °C rule can never fire; in low gain it fires only just below saturation |
| Real non-fire heat ≥ 300 °C at night | Gas flares (common in Alberta; FIRMS also detects them), engine exhausts `[VERIFY]`, **our own extinguisher's engine** seen from above |
| No size minimum | One dead or noisy pixel could bypass the CNN |

## The rule now

A blob skips the CNN veto if **either**:
1. **"saturated":** ≥ **3 pixels** at ≥ **0.9 × the camera's saturation for its current gain mode**. This is tied to
   the sensor, so it adapts automatically (high gain 150 °C → trigger 135 °C; low gain 350 °C → 315 °C). The pixel
   minimum stops single-pixel spikes.
2. **"oversize":** the blob spans ≥ **75% of the CNN's patch window** (2.3 m of ground) **and** peaks ≥ **150 °C**.
   The CNN compares the centre of its patch with the border, so a fire bigger than the patch has a hot border and the
   CNN is blind. The temperature floor exists because size alone isn't evidence: our own tests caught a 3 m²,
   35 °C sun-warmed rock triggering a size-only rule.

**Exclusion zones remove the bypass, never the detection.** Within 15 m of a friendly drone (from heartbeats) or a
known industrial heat source, the blob goes to the CNN like any other. An extinguisher hovering over the fire it's
fighting must not hide that fire.

**Cost of a wrong bypass is small by design:** a bypassed blob becomes a DETECTED hotspot, which gets an INVESTIGATE
pass (decision 0004) before any extinguisher is sent.

## How we keep it honest (data checks, automatic)
- `training/train_cnn.py` reports the hottest non-fire training patch and **how many non-fire patches would have
  bypassed the CNN** (should be 0). See RESULTS.md, CNN section.
- `perception/evaluate.py` reports, on the test split, how often each bypass fired and how often on non-fire blobs.

## Assumptions to confirm
`[ASSUMPTION]` 0.9 × saturation, 3 px, 75% window, 150 °C floor, 15 m exclusion radius. Camera saturation per gain
mode is E's (context §12 #11).
