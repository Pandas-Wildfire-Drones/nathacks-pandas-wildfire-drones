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

| Burn | Camera | Thermal | Saturation, low gain / high gain |
|---|---|---|---|
| Hanna Hammock (nadir modeling subset), Hundred, Wild Bill | Autel EVO II Dual 640T | 640×512 | 0–550 °C / −20–150 °C |
| Sycan (Kaggle CV subset) | DJI M30T | 640×512 | 0–500 °C / −20–150 °C |
| Shoetank, Willamette | DJI M2EA | 640×512 | −20–450 °C / −20–150 °C |

- **High gain clips at 150 °C, inside our 80–200 °C hard band.** Check which mode each burn used: if fire frames show many pixels piled up at exactly ~150 °C, it was high gain. Hot spots there are clipped, so set `--saturation-c 150` and treat > 150 °C as unknown.
- **Flight geometry:** oblique image pairs at 61–122 m AGL (mapping flights use a 50° oblique angle); nadir plots at 122 m AGL, straight down. The **No-Fire frames are mostly pre-burn oblique imagery** "collected to serve as negative samples", which makes them clean composite backgrounds.
- **GSD (ground sample distance, metres of ground per pixel):** the paper gives **~17.6 cm/px thermal at 122 m nadir for the M30T**. In general, `GSD = AGL × pixel_pitch / focal_length`, or `2 × AGL × tan(HFOV/2) / 640`. GSD scales linearly with altitude: the same camera at 61 m gives ~8.8 cm/px.
  - *Nadir subset (Autel 640T, 122 m):* compute it from the 640T's thermal focal length and pixel pitch `[VERIFY]`, or measure it: the four aluminium plates are ground control points with RTK positions, so `GSD = (metres between two plates) / (pixels between them in the frame)`. That measurement is the most trustworthy number.
  - *Oblique frames:* GSD varies across the image (coarser toward the horizon) and with altitude. If the TIFFs kept their XMP metadata (`exiftool -RelativeAltitude -GimbalPitchDegree -FocalLength <file>`), compute it per frame. Otherwise use a single middle value (≈0.13 m/px for the M30T at ~90 m). A ±50% error only shifts the fire-size distribution of the composites, whose areas are log-uniform anyway.

### What this means for us

1. **Fire / No-Fire folder labels come from a threshold rule plus human review.** Confident frames (max < 80 °C or > 200 °C) are effectively labeled by `max(T)`. Only the 80–200 °C band is checked by hand, and those frames may be missing entirely.
   - A detector that just checks `max(T) > ~150 °C` will look near-perfect on these labels. That shows it agrees with the labeling rule; it doesn't show it detects fires.
   - The hard cases (warm decoys, sub-pixel smouldering spots between 80 and 200 °C) are the ones most likely to be missing. Our night spot-fire targets are often in this band.
   - So **report image-level F1 by `max(T)` band** (< 80, 80–200, > 200 °C), and don't treat the overall FLAME 3 number as evidence for the > 90 % F1 criterion.
2. **Any pixel mask is threshold-derived unless shown otherwise.** If a mask ships with the data (Kaggle subset or a later release), assume it was made with a fixed threshold, Otsu or hysteresis. The paper names these methods, and arXiv 2609.01390 reports thresholding. Run `python -m training.check_masks` before using masks to score the threshold + CNN pipeline: it detects both fixed and per-frame (Otsu-style) thresholds.
3. **The thresholds are tuned to prescribed daytime burns** (the paper calls them "practical heuristics rather than fixed rules"). Our camera, emissivity and night background differ, so don't copy 80 °C / 200 °C into `perception/` as constants.
4. **Use FLAME 3 mainly as real backgrounds.** `python -m training.composite` takes the No-Fire frames and inserts synthetic fires (and, with `--decoys`, rocks, vehicles and animals) using the `perception.synth` renderer. Mixing happens in radiance, so sub-pixel dilution is preserved, and the masks are exact. Groups are kept, so splits don't leak. `objects.csv` records each fire's observed peak and band for per-band F1. Recipe:
   ```
   python -m training.build_manifest_flame3 --root exclude/data/flame3_raw --out exclude/data/flame3/manifest.csv --group-by sequence
   python -m training.composite --manifest exclude/data/flame3/manifest.csv --out exclude/data/flame3_composite --gsd-m <m/px> --decoys
   python -m training.make_split --manifest exclude/data/flame3_composite/manifest.csv   # then train as usual
   ```
   `--gsd-m` is required unless the manifest has `gsd_m` (see "Cameras, gain and GSD" below). Set `--saturation-c` to the background camera's gain range. On a synthetic stand-in run, about 17% of inserted fires landed in the 80–200 °C band; shift that with `--core-range` / `--area-range`.
   - *Why not HIT-UAV / BIRDSAI ([S21]) for negatives?* Both release **8-bit** frames: HIT-UAV states "bit depth is 8", and BIRDSAI's real frames are `.jpg`. Neither carries °C, and our CNN's first channel is absolute temperature. At most they could inform decoy shapes and sizes.
5. **Real-fire test set for the hard band (optional):** `python -m training.label_tool` (candidates → label → agree). A person labels 80–200 °C blobs from the paired RGB photo: active / smouldering / not fire / unsure. About 200–300 blobs take 20–30 min. Scored in `training/train_verifier.ipynb` §7.
6. **Group by burn.** The paper holds out whole burns (Willamette, Sycan) for testing. We do the same where we have more than one burn; the Kaggle CV subset is a single burn, so we group by time blocks (`build_manifest_flame3.py --group-by sequence`).
