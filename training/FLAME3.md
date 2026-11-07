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

### What this means for us

1. **Fire / No-Fire folder labels come from a threshold rule plus human review.** Confident frames (max < 80 °C or > 200 °C) are effectively labeled by `max(T)`. Only the 80–200 °C band is checked by hand, and those frames may be missing entirely.
   - A detector that just checks `max(T) > ~150 °C` will look near-perfect on these labels. That shows it agrees with the labeling rule; it doesn't show it detects fires.
   - The hard cases (warm decoys, sub-pixel smouldering spots between 80 and 200 °C) are the ones most likely to be missing. Our night spot-fire targets are often in this band.
   - So **report image-level F1 by `max(T)` band** (< 80, 80–200, > 200 °C), and don't treat the overall FLAME 3 number as evidence for the > 90 % F1 criterion.
2. **Any pixel mask is threshold-derived unless shown otherwise.** If a mask ships with the data (Kaggle subset or a later release), assume it was made with a fixed threshold, Otsu or hysteresis. The paper names these methods, and arXiv 2609.01390 reports thresholding. Run `python -m training.check_masks` before using masks to score the threshold + CNN pipeline: it detects both fixed and per-frame (Otsu-style) thresholds.
3. **The thresholds are tuned to prescribed daytime burns** (the paper calls them "practical heuristics rather than fixed rules"). Our camera, emissivity and night background differ, so don't copy 80 °C / 200 °C into `perception/` as constants.
4. **Group by burn.** The paper holds out whole burns (Willamette, Sycan) for testing. We do the same where we have more than one burn; the Kaggle CV subset is a single burn, so we group by time blocks (`build_manifest_flame3.py --group-by sequence`).
