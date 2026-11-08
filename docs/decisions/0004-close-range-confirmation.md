# Decision 0004: Close-range confirmation (INVESTIGATE) and post-drop verification (VERIFY)

**Status:** ACCEPTED, implemented in `perception/confirm.py` (2026-11-07) · **Owner:** C

> **Judge question:** "How do you avoid sending an 800 kg water bomber to a warm rock?"
>
> **Our answer:** "Search is tuned never to miss. Then a light seeker drops to ~30 m and watches the candidate for a
> few seconds. Only if the same network, now held to a much stricter bar, agrees across many frames do we call it
> CONFIRMED and send an extinguisher. Humans still declare fires out."

## Question
Does close-range confirmation need its own model and threshold, or does the search pipeline cover it?
(An idea we considered: "sensors normalize for distance." They don't. Radiometric calibration corrects the
camera's response, not geometry. At 30 m a fire covers ~7× more pixels than at 80 m, and with less sub-pixel
dilution it reads hotter. A fixed 32-pixel patch would show the CNN a different-looking object at each altitude.)

## Decision

| Aspect | Search (SCAN_ZONE) | Confirm (INVESTIGATE) | Verify (after a drop) |
|---|---|---|---|
| Who | seeker at 80 m | seeker hovering at 25–40 m | seeker or extinguisher |
| Question | "could this be a fire?" | "is it really a fire?" | "is anything still hot?" |
| Cost of a wrong "yes" | one INVESTIGATE pass | a wasted extinguisher sortie | a fire left burning |
| Method | threshold proposals → CNN | **same CNN**, many frames, mean probability | physics: hottest pixel within 2 m vs frame background |
| Threshold | **recall-first** (recall ≥ 0.95) | **three bands:** CONFIRMED above a precision-first bar (precision ≥ 0.98, never below the search threshold); FALSE_POSITIVE only below a bar that keeps recall ≥ 0.99; **inconclusive in between → escalate to a human** | still hot if ≥ 15 °C above background |
| Output | DETECTED hotspot | CONFIRMED / FALSE_POSITIVE / inconclusive | SUPPRESSED (only humans set VERIFIED_OUT, hard rule 10) |

1. **One model, trained across altitudes.** Training data should cover search and confirm altitudes (synthetic:
   25–150 m). Scale normalization (a fixed 2.3 m ground window, resampled in radiance, PatchSpec v2) is available
   and automatic when the manifest has `gsd_m`; see the A/B below for why it is **not yet proven necessary**.
2. **Separate operating points, same model.** The training script writes search, confirm and reject thresholds
   into the model sidecar.
3. **Multi-frame evidence.** Mean probability over hover frames (not "any frame says yes", which turns noise into
   confirmations). Inconclusive after the frame budget → keep it DETECTED and escalate to a human.
4. **Separate confirm model only if real data says so.** Revisit when close-range real imagery exists; the A/B below
   is the check.

## Evidence (synthetic, see RESULTS.md `confirm_synth`)

Two bugs found and fixed while building this, both worth mentioning in Q&A:
- **Confirm bar below search bar.** "Max recall at precision ≥ 0.98" picked 0.21 on easy data, *laxer* than the
  search threshold (0.96). Confirm is now never below search.
- **Rejecting real fires.** With only confirm/reject, the model wrongly rejected **9 real fires** up close. With the
  three-band rule it rejects **0**; those 9 become "inconclusive → human" (21 targets total, 12 of them decoys).

**Confirm stage, shipped model** (300 targets at 25–40 m AGL, 9 frames each; half fires 0.01–4 m², half decoys):

| | Confirm precision | Fires confirmed | Fires wrongly rejected | Decoys confirmed | Inconclusive (→ human) |
|---|---|---|---|---|---|
| Full-range model, scale-normalized (shipped) | 1.000 | 93.7% | **0** | 0 | 21 |

**A/B: does scale normalization matter?**

| Training altitudes | Patches | Fires confirmed | Fires wrongly rejected | Decoys confirmed | Inconclusive |
|---|---|---|---|---|---|
| 25–150 m | scale-normalized | 93.7% | 0 | 0 | 21 |
| 25–150 m | plain crops | 99.3% | 1 | 0 | 0 |
| **70–90 m only** (like a single-altitude real dataset) | scale-normalized | 98.6% | 1 | 1 | 15 |
| **70–90 m only** | plain crops | 99.3% | 1 | 0 | 1 |

**Honest reading:** on synthetic data, plain crops did **as well or slightly better**, even when trained only at
70–90 m and tested at 25–40 m. The likely reason: our synthetic decoys (≤ 90 °C) and fires (≥ 150 °C) separate on
absolute temperature, which doesn't depend on scale, so the CNN never needed shape. Real decoys (hot exhausts,
flares, sun-heated rock at dusk) overlap fires in temperature, and that's where scale should matter.
**So:** the claim "sensors normalize for distance" is wrong physically (pixel count and apparent temperature both
change with altitude), but on synthetic data the CNN coped either way. **Re-run this A/B on real data**
(`train_cnn --no-scale-normalize` vs default) once FLAME 3 altitude metadata is known, and keep whichever wins.

**Update 2026-11-07: A/B re-run on FLAME 3 (issue #4).** Altitude is known now: the raw RGB twin of each Sycan frame
carries DJI XMP (height, gimbal pitch, laser range), and every frame is **oblique** (pitch median −15°), so GSD is
computed per image row (0.14–1 m/px; `training/FLAME3.md`). Data: synthetic fires + decoys composited onto the 116 real
No-Fire frames (`training/composite.py`, foreshortened, ground-only), 10 train / 4 test ground-cell groups, same split
and seed for both arms (`bash training/retrain_flame3.sh`).

| Patches | CV F1 (5 folds) | CV AP | Test patch F1 / AP | Test object F1 [95% CI] | Test frame F1 |
|---|---|---|---|---|---|
| scale-normalized (per-row GSD) | 0.969 ± 0.019 | 0.989 | 0.958 / 0.986 | 0.940 [0.90, 0.97] | 1.000 |
| plain crops | 0.973 ± 0.026 | 0.995 | 0.948 / 0.975 | 0.934 [0.90, 0.96] | 0.994 |

**Reading:** still a tie within noise (CV favours plain by 0.004 F1, the held-out test favours scale-normalized by
0.006–0.010). Real backgrounds didn't make scale normalization *necessary*, and it didn't hurt. **Kept: PatchSpec v2
(scale-normalized) stays the default**, because it is the physically right input when one model serves 80 m search
and 30 m confirm. Cross-domain check on the same split: the shipped synthetic-only model scores object F1
**0.698** (recall 0.54) on these real backgrounds, while the composite model scores **0.950** on the synthetic night
test set. Real backgrounds matter far more than the patch-scaling choice.
Limits: fires are still synthetic; Sycan frames are daytime and oblique (the seeker is nadir at night); 4 test groups.
The NADIR plot set (Hanna Hammock, 122 m nadir) is the better place for a decisive scale test, and hand-labelled real
blobs (`training/label_tool.py`) are the only real-fire check.

For context, single-frame search from 80 m on the same targets: 143/143 fires detected, 0/157 decoys passed.

## Assumptions
`[ASSUMPTION]` 25–40 m confirm altitude, 9 frames (1 s at 9 Hz), 3 m search radius around the expected position,
precision ≥ 0.98 confirm target, 15 °C "still hot" margin, 2 m verify radius. Synthetic decoys are cooler than
fires by construction, so real-data numbers will be worse.
