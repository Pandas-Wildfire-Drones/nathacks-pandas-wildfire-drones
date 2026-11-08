# FLAME 3: citation and labeling notes

## Citation

The dataset page asks users to cite the original paper. Cite it in anything that uses FLAME 3: dev reports, slides, `perception/RESULTS.md`, and the pitch.

**APA**

> Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F. (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management. arXiv preprint arXiv:2412.02831.

**BibTeX**

```bibtex
@article{hopkins2024flame3,
  title   = {{FLAME 3} Dataset: Unleashing the Power of Radiometric Thermal {UAV} Imagery for Wildfire Management},
  author  = {Hopkins, Bryce and O'Neill, Leo and Marinaccio, Michael and Rowell, Eric and Parsons, Russell and
             Flanary, Sarah and Nazim, Irtija and Seielstad, Carl and Afghah, Fatemeh},
  journal = {arXiv preprint arXiv:2412.02831},
  year    = {2024}
}
```

Paper: https://arxiv.org/abs/2412.02831 (we read v2, 28 Jun 2026; local copy `exclude/flame3_paper.pdf`, gitignored). The APA text above is exactly what the dataset page asks for. The v2 author list also includes Mobin Habibpour.

## How FLAME 3 labels are made (from the paper)

| What | What the paper says | Where |
|---|---|---|
| Thermal TIFFs | 2-D float raster, **°C per pixel**, from the sensor's calibrated radiometric response (DJI: annual factory recalibration; Autel: manual blackbody calibration) | §III.A.1 p.5, §III.B p.7 |
| Image-level Fire / No-Fire | **Heuristics on the TIFF's max pixel temperature**: max < ~80 °C → almost always no fire; max > ~200 °C → typically active flaming. Images between the two thresholds are **reviewed by hand or excluded** | §III.A.1 p.5 |
| Tooling | The "image labeling tool" does fast **manual** Fire/No-Fire labeling, sorts images into output folders, and has a temperature-thresholding feature | §III.C.1 p.9 |
| Pixel masks | **The paper does not describe shipped segmentation masks.** It only says TIFFs *enable* pixel labels by direct thresholding, Otsu or hysteresis thresholding | §III.A.1 p.5 |
| Prescribed-burn temperatures | typically 55–650 °C; prescribed fires burn cooler than wildfires | §III.B p.7 |
| Their benchmark task | binary Fire / No-Fire **image classification**, Willamette + Sycan held out as the test set | §V p.10 |

### Cameras, gain and GSD (paper Table III p.7, §III p.4, §IV.A)

| Burn | Camera | Thermal | Saturation, low gain / high gain (paper) |
|---|---|---|---|
| Hanna Hammock (nadir modeling subset), Hundred, Wild Bill | Autel EVO II Dual 640T | 640×512 | 0–550 °C / −20–150 °C |
| Sycan (Kaggle CV subset) | DJI M30T | 640×512 | 0–500 °C / −20–150 °C |
| Shoetank, Willamette | DJI M2EA | 640×512 | −20–450 °C / −20–150 °C |

- **GSD in general:** `GSD = AGL × pixel_pitch / focal_length`, or `2 × AGL × tan(HFOV/2) / 640` for a nadir view. The paper gives ~17.6 cm/px for the M30T at 122 m nadir.
- *Nadir subset (Autel 640T, 122 m):* compute the GSD from the 640T's thermal focal length and pixel pitch `[VERIFY]`, or measure it from the ground control plates (`GSD = metres between two plates / pixels between them`).

### What the Sycan CV subset actually contains (measured on the download, 2026-11-07)

`python -m training.build_manifest_flame3` reads all of this and prints a summary. Per-frame values are in `exclude/data/flame3/frames_meta.csv`.

| Fact | Value | Consequence |
|---|---|---|
| Files | 622 Fire + 116 No Fire quartets (Celsius TIFF, palette JPG, raw RGB 4000×3000, corrected-FOV RGB 640×512) | only the TIFF is used for training |
| File names | stems **restart per class folder** (`Fire/00001` ≠ `No Fire/00001`) and are **shuffled** (Spearman(stem, time) ≈ 0) | pair by folder + stem; order by EXIF time. The old "sort by name, blocks of 50" grouping leaked neighbouring frames across the split |
| Capture | 25–27 Oct 2022, **15:00–19:23 local**: daytime to dusk, a few frames after sunset | warmer, sun-heated backgrounds than our night scenes |
| Metadata | thermal JPG/TIFF carry no XMP; the **raw RGB** has DJI XMP: `RelativeAltitude` (above takeoff), `GimbalPitch/Yaw`, `LRFTarget*` (laser rangefinder slant distance and the ground point at the image centre) | the builder reads geometry from the RGB twin |
| View | **every frame is oblique**: gimbal pitch −46° to +12° (median −15°); 16 Fire frames point above the horizon. No-Fire frames −41° to −10° | GSD changes row by row; the top rows of a typical frame reach the horizon |
| Height | 55–124 m above takeoff. The ground under the image centre is median ~10 m above takeoff (LRF; it agrees with pitch × height to 0.01°) | `agl_m = RelativeAltitude − LRFTargetAlt` (day median when the LRF has no return or returns nonsense) |
| Focal length | **≈ 696 px (HFOV 49.4°)**. DJI's 61° DFOV gives 696 px. Matching the corrected-FOV RGB inside the wide RGB (0.636 of its width, 12 frames, sd 0.004) gives ~698 px. EXIF 9.1 mm / 12 µm (758 px) disagrees with both | `--focal-px 696` |
| GSD | centre row: median **0.31–0.32 m/px** (5–95 %: 0.16–1.0). Bottom row of a −15° frame ≈ 0.14 m/px; the middle ≈ 0.33 m/px; the top runs to the horizon. Vertical foreshortening ≈ 1/sin(15°) ≈ 3.9× | **4–14× coarser than our seeker** (7.2 cm/px at 80 m nadir) |
| Gain | two modes, recognised by the values pixels pile up at: **high = floor −40.6 / clip 187.4 °C**, **low = floor −22.8 / clip 608.7 °C** (not the paper's 150 / 500). Frames touching neither take the gain of their session (frames ≤ 120 s apart) | No-Fire backgrounds: 108 high / 8 low. Fire frames: 84 high / 538 low |
| Hard band | **all 84 Fire frames whose max T is 80–200 °C are high-gain frames**: 67 sit exactly at the 187.4 °C clip, 15 more at 182–187 °C, and only 2 (108 and 165 °C) look genuinely in band | at frame level the subset has essentially **no** real 80–200 °C fire frames, only clipping. The hard band has to come from blobs (label tool) and composites |
| Groups | one day × one 250 m cell of the image-centre ground point → 37 groups (No Fire: 14) | frames of the same ground on the same day stay together |

### What this means for us

1. **Fire / No-Fire folder labels come from a threshold rule plus human review.** Confident frames (max < 80 °C or > 200 °C) are effectively labeled by `max(T)`. On Sycan the 80–200 °C frames are high-gain clipping (table above).
   - A detector that just checks `max(T) > ~150 °C` will look near-perfect on these labels. That shows it agrees with the labeling rule; it doesn't show it detects fires.
   - So **report image-level F1 by `max(T)` band** (< 80, 80–200, > 200 °C), and don't treat the overall FLAME 3 number as evidence for the > 90 % F1 criterion.
2. **Any pixel mask is threshold-derived unless shown otherwise.** The Sycan subset ships no masks. If a later release does, run `python -m training.check_masks` before scoring against it.
3. **The thresholds are tuned to prescribed daytime burns** (the paper calls them "practical heuristics rather than fixed rules"). Don't copy 80 °C / 200 °C (or 187.4 / 608.7 °C) into `perception/` as constants.
4. **Use FLAME 3 mainly as real backgrounds.** `python -m training.composite` inserts synthetic fires (and, with `--decoys`, rocks, vehicles and animals) into the No-Fire frames using the `perception.synth` renderer. Mixing happens in radiance (sub-pixel dilution preserved), masks are exact, groups are kept. On oblique frames each object is sized with its row's GSD, foreshortened by sin(depression), and only placed on ground (≥ 10° below the horizon, not sky). Composites clip at **our camera's** saturation (`--clip-mode camera`, default 550 °C `[VERIFY]`): clipping is a camera setting, not part of the scene, and 108/116 backgrounds were shot in high gain. `--clip-mode frame` mimics FLAME 3's own gain modes. Recipe:
   ```
   python -m training.build_manifest_flame3 --root exclude/data/flame3_sycan_marsh_raw --out exclude/data/flame3/manifest.csv
   python -m training.composite --manifest exclude/data/flame3/manifest.csv --out exclude/data/flame3_composite --per-background 3 --decoys
   python -m training.make_split --manifest exclude/data/flame3_composite/manifest.csv --test-frac 0.25 --seed 0
   bash training/retrain_flame3.sh     # A/B (scale-normalized vs plain) + cross-domain evaluations
   ```
   With the defaults, of 551 inserted fires 156 read < 80 °C (sub-pixel at these GSDs), 174 read 80–200 °C and 212 read > 200 °C (9 fell off the frame). Shift the mix with `--core-range` / `--area-range`.
   - *Domain gap (decide what the composites are for):* daytime vs night, oblique vs nadir, 0.15–1 m/px vs 7 cm/px. Composites give **real clutter and real sensor noise**; they don't give our viewing geometry. Keep the night synthetic set as the second test set and report both.
   - *Why not HIT-UAV / BIRDSAI ([S21]) for negatives?* Both release **8-bit** frames: HIT-UAV states "bit depth is 8", and BIRDSAI's real frames are `.jpg`. Neither carries °C, and our CNN's first channel is absolute temperature.
5. **Real-fire test set for the hard band (optional):** `python -m training.label_tool` (candidates → label → agree). A person labels 80–200 °C blobs from the paired RGB photo: active / smouldering / not fire / unsure. Sycan has 12,762 unclipped in-band blobs in Fire frames (23 groups); the default draw is ~300 (≤ 15 per group). Blobs at the gain clip are skipped (`--include-clipped` keeps them). The RGB panel crops the 4000×3000 raw photo through `--rgb-box 0.1638 0.1511 0.8004 0.8289` (where the thermal FOV sits in the wide camera). About 200–300 blobs take 20–30 min. These labels are a **test set**, not training data. Scored in `training/train_verifier.ipynb` §7.
6. **Group by burn.** The paper holds out whole burns (Willamette, Sycan) for testing. The Kaggle CV subset is a single burn, so the builder groups by day × 250 m ground cell. With only 14 No-Fire groups, CV and test estimates on composites stay noisy.
