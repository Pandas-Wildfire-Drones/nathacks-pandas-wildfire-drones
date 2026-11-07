# Perception results

Numbers for the dev reports, the pitch and the demo (workstream C). The sections between `<!-- BEGIN:key -->`
markers are rewritten by the scripts; machine-readable copies are in `perception/results/<key>.json` (the
frontend should read those). Edit only the text outside the markers.

| Section key | Written by | Data |
|---|---|---|
| `cnn_cv_<tag>` | `python -m training.train_cnn` | training split, group k-fold CV (patch level) |
| `gbm_cv_<tag>` | `python -m training.train_gbm` **[STRETCH]** | same patches and folds as the CNN |
| `pipeline_test_<tag>` | `python -m perception.evaluate` | held-out TEST split, full mixed pipeline (object/frame/patch level) |

**How to read them.** We pick thresholds recall-first: recall ≥ 0.95 on cross-validation, then the best
precision. A missed spot fire costs more than a false alarm, because a seeker re-checks every alarm. The design
criterion is F1 > 0.90 at the object level on real data. Accuracy is listed but misleading, because most
patches and frames contain no fire.

**`synth` sections are a wiring check only.** The synthetic task is easier than real data: decoys are ≤ 90 °C
and fires are ≥ 150 °C. Those numbers show that the code works and how sub-pixel spots and decoys behave. They
are not evidence of real-world detection performance. Report real numbers only from FLAME 3 (or other real
data) test splits.

<!-- BEGIN:cnn_cv_synth -->
### CNN verifier (PRIMARY): 5-fold group CV · synth

Patch-level, on classical proposals + mask-centred positives. 2923 patches (1857 fire / 1066 not, 586 hard negatives), 12 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 15, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=508, pos=328, thr=0.859) | 0.997 | 0.994 | 0.995 | 0.995 | 0.994 | 0.998 | 0.996 |
| fold 1 (n=504, pos=327, thr=0.840) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| fold 2 (n=510, pos=331, thr=0.840) | 0.997 | 0.979 | 0.988 | 0.982 | 0.984 | 0.994 | 0.984 |
| fold 3 (n=693, pos=423, thr=0.840) | 1.000 | 0.967 | 0.983 | 0.973 | 0.980 | 0.998 | 0.997 |
| fold 4 (n=708, pos=448, thr=0.788) | 0.998 | 0.993 | 0.996 | 0.994 | 0.994 | 1.000 | 1.000 |
| **mean ± std over folds** | 0.998 ± 0.001 | 0.987 ± 0.012 | 0.992 ± 0.006 | 0.989 ± 0.010 | 0.991 ± 0.007 | 0.998 ± 0.002 | 0.995 ± 0.006 |
| pooled OOF @ final thr 0.840 | 0.999 | 0.987 | 0.993 | 0.989 | 0.991 | 0.998 | 0.995 |
| pooled OOF @ 0.5 (reference) | 0.997 | 0.990 | 0.994 | 0.991 | 0.992 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
<!-- END:cnn_cv_synth -->

<!-- BEGIN:gbm_cv_synth -->
### [STRETCH] GBM verifier: 5-fold group CV · synth

Stretch-goal comparison against the primary CNN: same patches, folds and threshold rule.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (thr=0.985) | 1.000 | 0.951 | 0.975 | 0.961 | 0.969 | 0.999 | 0.999 |
| fold 1 (thr=0.985) | 0.991 | 0.997 | 0.994 | 0.996 | 0.992 | 0.999 | 0.999 |
| fold 2 (thr=0.976) | 0.988 | 0.955 | 0.971 | 0.961 | 0.963 | 0.996 | 0.992 |
| fold 3 (thr=0.990) | 0.995 | 0.915 | 0.953 | 0.930 | 0.945 | 0.993 | 0.986 |
| fold 4 (thr=0.985) | 0.995 | 0.964 | 0.980 | 0.970 | 0.975 | 0.999 | 0.999 |
| **mean ± std over folds** | 0.994 ± 0.004 | 0.956 ± 0.026 | 0.975 ± 0.013 | 0.964 ± 0.021 | 0.969 ± 0.015 | 0.998 ± 0.002 | 0.995 ± 0.005 |
| pooled OOF @ final thr 0.985 | 0.995 | 0.959 | 0.977 | 0.966 | 0.971 | 0.997 | 0.993 |
<!-- END:gbm_cv_synth -->

<!-- BEGIN:pipeline_test_synth -->
### Pipeline evaluation: synthetic test flights (4 held-out flights)

160 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Object level** (detections matched to mask components):

| | precision | recall | f1 | f2 |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.856 | 0.984 | 0.915 | 0.955 |
| all classical proposals (recall ceiling) | 0.500 | 0.984 | 0.663 | 0.824 |
| FUSED + CNN (PRIMARY) | 0.989 | 0.984 | 0.987 | 0.985 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 0.989 | 0.984 | 0.987 | 0.985 |
| FUSED + GBM [STRETCH] | 0.984 | 0.984 | 0.984 | 0.984 |
| FUSED + GBM, no hot-bypass [STRETCH] | 0.984 | 0.984 | 0.984 | 0.984 |

95% CI on F1 (bootstrap over groups): THRESHOLD only (peak >= 60 C, classical) [0.880, 0.944], FUSED + CNN (PRIMARY) [0.975, 0.997], FUSED + CNN, no hot-bypass (PRIMARY) [0.975, 0.997], FUSED + GBM [STRETCH] [0.976, 0.995], FUSED + GBM, no hot-bypass [STRETCH] [0.976, 0.995]

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.919 | 1.000 | 0.958 | 0.944 |
| all classical proposals (recall ceiling) | 0.734 | 1.000 | 0.846 | 0.769 |
| FUSED + CNN (PRIMARY) | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + GBM [STRETCH] | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + GBM, no hot-bypass [STRETCH] | 1.000 | 1.000 | 1.000 | 1.000 |

**Patch level** (verifier on 368 proposals, 188 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.874 | 1.000 | 0.933 | 0.999 | 0.999 | 60.000 |
| CNN (PRIMARY) verifier | 0.995 | 0.984 | 0.989 | 0.998 | 0.996 | 0.840 |
| GBM [STRETCH] verifier | 0.995 | 0.989 | 0.992 | 0.999 | 0.999 | 0.985 |

CPU latency per frame (ms, this machine): threshold 1.8, fused:cnn 2.5, fused:gbm 2.0
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_synth -->
