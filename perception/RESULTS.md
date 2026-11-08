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

Patch-level, on classical proposals + mask-centred positives. 2937 patches (1853 fire / 1084 not, 604 hard negatives), 12 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 15, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=506, pos=327, thr=0.955) | 1.000 | 0.979 | 0.989 | 0.983 | 0.986 | 0.999 | 0.999 |
| fold 1 (n=691, pos=426, thr=0.955) | 1.000 | 0.967 | 0.983 | 0.974 | 0.980 | 1.000 | 1.000 |
| fold 2 (n=493, pos=319, thr=0.955) | 1.000 | 0.975 | 0.987 | 0.980 | 0.984 | 1.000 | 1.000 |
| fold 3 (n=710, pos=449, thr=0.932) | 0.995 | 0.978 | 0.987 | 0.981 | 0.983 | 0.999 | 0.998 |
| fold 4 (n=537, pos=332, thr=0.955) | 0.994 | 0.949 | 0.971 | 0.957 | 0.965 | 0.992 | 0.983 |
| **mean ± std over folds** | 0.998 ± 0.003 | 0.969 ± 0.011 | 0.983 ± 0.007 | 0.975 ± 0.009 | 0.979 ± 0.008 | 0.998 ± 0.003 | 0.996 ± 0.007 |
| pooled OOF @ final thr 0.955 | 0.998 | 0.969 | 0.984 | 0.975 | 0.980 | 0.998 | 0.996 |
| pooled OOF @ 0.5 (reference) | 0.995 | 0.988 | 0.991 | 0.989 | 0.989 |  |  |
| CONFIRM op. point @ 0.955 (precision >= 0.98) | 0.998 | 0.969 | 0.984 | 0.975 | 0.980 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: True.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 284.6 °C (p99.9 89.7 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 1084. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_synth -->

<!-- BEGIN:gbm_cv_synth -->
### [STRETCH] GBM verifier: 5-fold group CV · synth

Stretch-goal comparison against the primary CNN: same patches, folds and threshold rule.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (thr=0.978) | 1.000 | 0.976 | 0.988 | 0.980 | 0.984 | 1.000 | 0.999 |
| fold 1 (thr=0.994) | 1.000 | 0.840 | 0.913 | 0.868 | 0.902 | 0.998 | 0.996 |
| fold 2 (thr=0.968) | 0.994 | 0.966 | 0.979 | 0.971 | 0.974 | 0.999 | 0.999 |
| fold 3 (thr=0.976) | 0.989 | 0.982 | 0.985 | 0.983 | 0.982 | 0.998 | 0.997 |
| fold 4 (thr=0.985) | 0.988 | 0.952 | 0.969 | 0.959 | 0.963 | 0.995 | 0.990 |
| **mean ± std over folds** | 0.994 ± 0.005 | 0.943 ± 0.052 | 0.967 ± 0.028 | 0.952 ± 0.043 | 0.961 ± 0.031 | 0.998 ± 0.002 | 0.996 ± 0.003 |
| pooled OOF @ final thr 0.985 | 0.995 | 0.950 | 0.972 | 0.959 | 0.966 | 0.998 | 0.995 |
<!-- END:gbm_cv_synth -->

<!-- BEGIN:pipeline_test_synth -->
### Pipeline evaluation: synthetic test flights (4 held-out flights, 25-150 m AGL)

160 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Object level** (detections matched to mask components):

| | precision | recall | f1 | f2 |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.858 | 0.978 | 0.915 | 0.952 |
| all classical proposals (recall ceiling) | 0.493 | 0.978 | 0.656 | 0.818 |
| FUSED + CNN (PRIMARY) | 0.989 | 0.978 | 0.984 | 0.981 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 0.973 | 0.986 | 0.978 |
| FUSED + GBM [STRETCH] | 0.973 | 0.978 | 0.976 | 0.977 |
| FUSED + GBM, no hot-bypass [STRETCH] | 0.978 | 0.978 | 0.978 | 0.978 |

95% CI on F1 (bootstrap over groups): THRESHOLD only (peak >= 60 C, classical) [0.880, 0.940], FUSED + CNN (PRIMARY) [0.974, 0.995], FUSED + CNN, no hot-bypass (PRIMARY) [0.975, 0.997], FUSED + GBM [STRETCH] [0.964, 0.989], FUSED + GBM, no hot-bypass [STRETCH] [0.967, 0.989]

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.919 | 1.000 | 0.958 | 0.944 |
| all classical proposals (recall ceiling) | 0.734 | 1.000 | 0.846 | 0.769 |
| FUSED + CNN (PRIMARY) | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 0.990 | 0.995 | 0.994 |
| FUSED + GBM [STRETCH] | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + GBM, no hot-bypass [STRETCH] | 1.000 | 0.990 | 0.995 | 0.994 |

**Patch level** (verifier on 369 proposals, 189 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.877 | 0.984 | 0.928 | 0.995 | 0.993 | 60.000 |
| CNN (PRIMARY) verifier | 1.000 | 0.958 | 0.978 | 0.997 | 0.995 | 0.955 |
| GBM [STRETCH] verifier | 0.995 | 0.979 | 0.987 | 0.998 | 0.998 | 0.985 |

**Hot-blob bypass** (trigger ≥ 3 px at ≥ 495 °C = 0.9 × saturation 550 °C, or blob larger than the patch): CNN saturated: fired 37x, 0 on non-fire; CNN oversize: fired 29x, 0 on non-fire; GBM saturated: fired 37x, 0 on non-fire; GBM oversize: fired 29x, 0 on non-fire.

CPU latency per frame (ms, this machine): threshold 1.7, fused:cnn 2.6, fused:gbm 2.1
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_synth -->

<!-- BEGIN:cnn_cv_synth_plain -->
### CNN verifier (PRIMARY): 5-fold group CV · synth_plain

Patch-level, on classical proposals + mask-centred positives. 2937 patches (1853 fire / 1084 not, 604 hard negatives), 12 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 15, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=506, pos=327, thr=0.815) | 1.000 | 0.994 | 0.997 | 0.995 | 0.996 | 1.000 | 1.000 |
| fold 1 (n=691, pos=426, thr=0.815) | 1.000 | 0.993 | 0.996 | 0.994 | 0.996 | 1.000 | 1.000 |
| fold 2 (n=493, pos=319, thr=0.851) | 1.000 | 0.987 | 0.994 | 0.990 | 0.992 | 1.000 | 1.000 |
| fold 3 (n=710, pos=449, thr=0.815) | 1.000 | 0.989 | 0.994 | 0.991 | 0.993 | 0.999 | 0.999 |
| fold 4 (n=537, pos=332, thr=0.989) | 0.997 | 0.934 | 0.964 | 0.946 | 0.957 | 0.988 | 0.972 |
| **mean ± std over folds** | 0.999 ± 0.001 | 0.979 ± 0.023 | 0.989 ± 0.013 | 0.983 ± 0.019 | 0.987 ± 0.015 | 0.997 ± 0.005 | 0.994 ± 0.011 |
| pooled OOF @ final thr 0.815 | 0.999 | 0.987 | 0.993 | 0.990 | 0.991 | 0.997 | 0.994 |
| pooled OOF @ 0.5 (reference) | 0.998 | 0.990 | 0.994 | 0.991 | 0.993 |  |  |
| CONFIRM op. point @ 0.815 (precision >= 0.98) | 0.999 | 0.987 | 0.993 | 0.990 | 0.991 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: False.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 89.7 °C (p99.9 89.3 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 1084. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_synth_plain -->

<!-- BEGIN:confirm_synth -->
### Close-range confirmation (INVESTIGATE), synthetic · 300 targets, 9 frames each, 25-40 m AGL

Synthetic behaviour check (decoys are cooler than fires by construction; background regenerated per frame). Not a real-world performance claim.

| model | confirm precision | fires confirmed | fires wrongly rejected | decoys confirmed | decoys rejected | inconclusive | thresholds (confirm / reject) |
|---|---|---|---|---|---|---|---|
| scale (scale-normalized: True) | 1.000 | 93.7% | 0 | 0 | 145 | 21 | 0.955 / 0.091 |
| plain (scale-normalized: False) | 1.000 | 99.3% | 1 | 0 | 157 | 0 | 0.815 / 0.453 |

**By target type** (close-range outcome · single 80 m search frame):

| model | target | CONFIRMED | FALSE_POSITIVE | INCONCLUSIVE | search @80 m detected / missed |
|---|---|---|---|---|---|
| scale | decoy | 0 | 145 | 12 | 0 / 157 |
| scale | fire 0.1-1 m² | 52 | 0 | 6 | 58 / 0 |
| scale | fire <0.1 m² | 42 | 0 | 3 | 45 / 0 |
| scale | fire >1 m² | 40 | 0 | 0 | 40 / 0 |
| plain | decoy | 0 | 157 | 0 | 0 / 157 |
| plain | fire 0.1-1 m² | 58 | 0 | 0 | 58 / 0 |
| plain | fire <0.1 m² | 44 | 1 | 0 | 45 / 0 |
| plain | fire >1 m² | 40 | 0 | 0 | 40 / 0 |
<!-- END:confirm_synth -->

<!-- BEGIN:cnn_cv_synth_narrow -->
### CNN verifier (PRIMARY): 5-fold group CV · synth_narrow

Patch-level, on classical proposals + mask-centred positives. 2826 patches (1784 fire / 1042 not, 562 hard negatives), 12 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 15, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=479, pos=316, thr=0.680) | 1.000 | 0.987 | 0.994 | 0.990 | 0.992 | 1.000 | 1.000 |
| fold 1 (n=672, pos=430, thr=0.680) | 1.000 | 0.991 | 0.995 | 0.993 | 0.994 | 1.000 | 1.000 |
| fold 2 (n=685, pos=418, thr=0.680) | 1.000 | 0.969 | 0.984 | 0.975 | 0.981 | 0.998 | 0.995 |
| fold 3 (n=510, pos=318, thr=0.987) | 0.997 | 0.912 | 0.952 | 0.928 | 0.943 | 0.997 | 0.997 |
| fold 4 (n=480, pos=302, thr=0.680) | 1.000 | 0.983 | 0.992 | 0.987 | 0.990 | 0.999 | 0.997 |
| **mean ± std over folds** | 0.999 ± 0.001 | 0.968 ± 0.029 | 0.983 ± 0.016 | 0.974 ± 0.024 | 0.980 ± 0.019 | 0.999 ± 0.001 | 0.998 ± 0.002 |
| pooled OOF @ final thr 0.680 | 0.999 | 0.985 | 0.992 | 0.988 | 0.990 | 0.998 | 0.997 |
| pooled OOF @ 0.5 (reference) | 0.998 | 0.987 | 0.992 | 0.989 | 0.990 |  |  |
| CONFIRM op. point @ 0.680 (precision >= 0.98) | 0.999 | 0.985 | 0.992 | 0.988 | 0.990 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: True.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 90.0 °C (p99.9 89.5 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 1042. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_synth_narrow -->

<!-- BEGIN:cnn_cv_synth_narrow_plain -->
### CNN verifier (PRIMARY): 5-fold group CV · synth_narrow_plain

Patch-level, on classical proposals + mask-centred positives. 2826 patches (1784 fire / 1042 not, 562 hard negatives), 12 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 15, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=479, pos=316, thr=0.740) | 1.000 | 0.991 | 0.995 | 0.992 | 0.994 | 1.000 | 1.000 |
| fold 1 (n=672, pos=430, thr=0.679) | 1.000 | 0.991 | 0.995 | 0.993 | 0.994 | 1.000 | 0.999 |
| fold 2 (n=685, pos=418, thr=0.679) | 1.000 | 0.964 | 0.982 | 0.971 | 0.978 | 0.997 | 0.994 |
| fold 3 (n=510, pos=318, thr=0.985) | 0.997 | 0.903 | 0.947 | 0.920 | 0.937 | 0.997 | 0.997 |
| fold 4 (n=480, pos=302, thr=0.632) | 0.997 | 0.983 | 0.990 | 0.986 | 0.988 | 0.999 | 0.998 |
| **mean ± std over folds** | 0.999 ± 0.002 | 0.966 ± 0.033 | 0.982 ± 0.018 | 0.972 ± 0.027 | 0.978 ± 0.021 | 0.998 ± 0.001 | 0.997 ± 0.002 |
| pooled OOF @ final thr 0.679 | 0.999 | 0.984 | 0.992 | 0.987 | 0.989 | 0.998 | 0.997 |
| pooled OOF @ 0.5 (reference) | 0.998 | 0.985 | 0.992 | 0.988 | 0.990 |  |  |
| CONFIRM op. point @ 0.679 (precision >= 0.98) | 0.999 | 0.984 | 0.992 | 0.987 | 0.989 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: False.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 90.1 °C (p99.9 89.6 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 1042. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_synth_narrow_plain -->

<!-- BEGIN:confirm_synth_narrow_training -->
### Close-range confirmation (INVESTIGATE), synthetic · 300 targets, 9 frames each, 25-40 m AGL

Synthetic behaviour check (decoys are cooler than fires by construction; background regenerated per frame). Not a real-world performance claim.

| model | confirm precision | fires confirmed | fires wrongly rejected | decoys confirmed | decoys rejected | inconclusive | thresholds (confirm / reject) |
|---|---|---|---|---|---|---|---|
| scale_trained_70-90m (scale-normalized: True) | 0.993 | 98.6% | 1 | 1 | 142 | 15 | 0.680 / 0.106 |
| plain_trained_70-90m (scale-normalized: False) | 1.000 | 99.3% | 1 | 0 | 156 | 1 | 0.679 / 0.088 |

**By target type** (close-range outcome · single 80 m search frame):

| model | target | CONFIRMED | FALSE_POSITIVE | INCONCLUSIVE | search @80 m detected / missed |
|---|---|---|---|---|---|
| scale_trained_70-90m | decoy | 1 | 142 | 14 | 0 / 157 |
| scale_trained_70-90m | fire 0.1-1 m² | 57 | 0 | 1 | 58 / 0 |
| scale_trained_70-90m | fire <0.1 m² | 44 | 1 | 0 | 45 / 0 |
| scale_trained_70-90m | fire >1 m² | 40 | 0 | 0 | 40 / 0 |
| plain_trained_70-90m | decoy | 0 | 156 | 1 | 0 / 157 |
| plain_trained_70-90m | fire 0.1-1 m² | 58 | 0 | 0 | 58 / 0 |
| plain_trained_70-90m | fire <0.1 m² | 44 | 1 | 0 | 45 / 0 |
| plain_trained_70-90m | fire >1 m² | 40 | 0 | 0 | 40 / 0 |
<!-- END:confirm_synth_narrow_training -->

<!-- BEGIN:cnn_cv_flame3_composite -->
### CNN verifier (PRIMARY): 5-fold group CV · flame3_composite

Patch-level, on classical proposals + mask-centred positives. 8042 patches (1108 fire / 6934 not, 6610 hard negatives), 10 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 20, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=5866, pos=443, thr=0.882) | 0.954 | 0.928 | 0.941 | 0.933 | 0.991 | 0.957 | 0.991 |
| fold 1 (n=661, pos=129, thr=0.834) | 0.992 | 0.915 | 0.952 | 0.929 | 0.982 | 0.996 | 0.999 |
| fold 2 (n=449, pos=124, thr=0.747) | 1.000 | 0.968 | 0.984 | 0.974 | 0.991 | 0.998 | 0.999 |
| fold 3 (n=500, pos=188, thr=0.747) | 0.989 | 0.984 | 0.987 | 0.985 | 0.990 | 0.999 | 0.999 |
| fold 4 (n=566, pos=224, thr=0.747) | 0.982 | 0.982 | 0.982 | 0.982 | 0.986 | 0.997 | 0.998 |
| **mean ± std over folds** | 0.983 ± 0.016 | 0.955 ± 0.029 | 0.969 ± 0.019 | 0.961 ± 0.025 | 0.988 ± 0.004 | 0.989 ± 0.016 | 0.997 ± 0.003 |
| pooled OOF @ final thr 0.747 | 0.969 | 0.960 | 0.965 | 0.962 | 0.990 | 0.985 | 0.996 |
| pooled OOF @ 0.5 (reference) | 0.934 | 0.976 | 0.955 | 0.967 | 0.987 |  |  |
| CONFIRM op. point @ 0.904 (precision >= 0.98) | 0.981 | 0.941 | 0.961 | 0.949 | 0.989 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: True.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 88.5 °C (p99.9 85.2 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 6934. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_flame3_composite -->

<!-- BEGIN:cnn_cv_flame3_composite_plain -->
### CNN verifier (PRIMARY): 5-fold group CV · flame3_composite_plain

Patch-level, on classical proposals + mask-centred positives. 8042 patches (1108 fire / 6934 not, 6610 hard negatives), 10 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 20, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=5866, pos=443, thr=0.970) | 0.973 | 0.878 | 0.923 | 0.895 | 0.989 | 0.979 | 0.998 |
| fold 1 (n=661, pos=129, thr=0.777) | 0.970 | 1.000 | 0.985 | 0.994 | 0.994 | 0.997 | 0.999 |
| fold 2 (n=449, pos=124, thr=0.905) | 0.992 | 0.984 | 0.988 | 0.985 | 0.993 | 1.000 | 1.000 |
| fold 3 (n=500, pos=188, thr=0.777) | 0.995 | 1.000 | 0.997 | 0.999 | 0.998 | 1.000 | 1.000 |
| fold 4 (n=566, pos=224, thr=0.931) | 0.986 | 0.955 | 0.971 | 0.961 | 0.977 | 0.998 | 0.999 |
| **mean ± std over folds** | 0.983 ± 0.010 | 0.963 ± 0.046 | 0.973 ± 0.026 | 0.967 ± 0.038 | 0.990 ± 0.007 | 0.995 ± 0.008 | 0.999 ± 0.001 |
| pooled OOF @ final thr 0.939 | 0.983 | 0.950 | 0.966 | 0.957 | 0.991 | 0.991 | 0.999 |
| pooled OOF @ 0.5 (reference) | 0.971 | 0.980 | 0.975 | 0.978 | 0.993 |  |  |
| CONFIRM op. point @ 0.939 (precision >= 0.98) | 0.983 | 0.950 | 0.966 | 0.957 | 0.991 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: False.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 458.8 °C (p99.9 86.6 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 6934. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_flame3_composite_plain -->

<!-- BEGIN:pipeline_test_cnn_flame3 -->
### Pipeline evaluation: FLAME 3 Sycan composites, 4 held-out ground cells (cnn_flame3)

140 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.976 | 0.920 | 0.947 | 0.936 |
| all classical proposals (recall ceiling) | 0.674 | 1.000 | 0.806 | 0.700 |
| FUSED + CNN (PRIMARY) | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 1.000 | 1.000 | 1.000 |

**Patch level** (verifier on 2893 proposals, 162 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.893 | 0.821 | 0.855 | 0.930 | 0.985 | 60.000 |
| CNN (PRIMARY) verifier | 0.940 | 0.975 | 0.958 | 0.986 | 0.997 | 0.747 |

**Hot-blob bypass** (trigger ≥ 3 px at ≥ 495 °C = 0.9 × saturation 550 °C, or blob larger than the patch): CNN saturated: fired 6x, 0 on non-fire; CNN oversize: fired 25x, 0 on non-fire.

CPU latency per frame (ms, this machine): threshold 8.4, fused:cnn 18.1
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_cnn_flame3 -->

<!-- BEGIN:pipeline_test_cnn_flame3_plain -->
### Pipeline evaluation: FLAME 3 Sycan composites, 4 held-out ground cells (cnn_flame3_plain)

140 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.976 | 0.920 | 0.947 | 0.936 |
| all classical proposals (recall ceiling) | 0.674 | 1.000 | 0.806 | 0.700 |
| FUSED + CNN (PRIMARY) | 1.000 | 0.989 | 0.994 | 0.993 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 0.989 | 0.994 | 0.993 |

**Patch level** (verifier on 2893 proposals, 162 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.893 | 0.821 | 0.855 | 0.930 | 0.985 | 60.000 |
| CNN (PRIMARY) verifier | 0.934 | 0.963 | 0.948 | 0.975 | 0.997 | 0.939 |

**Hot-blob bypass** (trigger ≥ 3 px at ≥ 495 °C = 0.9 × saturation 550 °C, or blob larger than the patch): CNN saturated: fired 6x, 0 on non-fire.

CPU latency per frame (ms, this machine): threshold 8.4, fused:cnn 17.3
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_cnn_flame3_plain -->

<!-- BEGIN:pipeline_test_flame3_by_synth_model -->
### Pipeline evaluation: FLAME 3 Sycan composites, shipped synthetic model

140 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.976 | 0.920 | 0.947 | 0.936 |
| all classical proposals (recall ceiling) | 0.674 | 1.000 | 0.806 | 0.700 |
| FUSED + CNN (PRIMARY) | 1.000 | 0.713 | 0.832 | 0.821 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 1.000 | 0.713 | 0.832 | 0.821 |

**Patch level** (verifier on 2893 proposals, 162 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.893 | 0.821 | 0.855 | 0.930 | 0.985 | 60.000 |
| CNN (PRIMARY) verifier | 1.000 | 0.549 | 0.709 | 0.843 | 0.891 | 0.955 |

**Hot-blob bypass** (trigger ≥ 3 px at ≥ 495 °C = 0.9 × saturation 550 °C, or blob larger than the patch): CNN saturated: fired 6x, 0 on non-fire; CNN oversize: fired 25x, 0 on non-fire.

CPU latency per frame (ms, this machine): threshold 8.5, fused:cnn 18.1
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_flame3_by_synth_model -->

<!-- BEGIN:pipeline_test_synth_by_flame3_model -->
### Pipeline evaluation: synthetic night test flights, FLAME 3 composite model

160 frames, 4 groups (flights/burns). Thresholds were chosen recall-first on training CV and never tuned on this split.

**Object level** (detections matched to mask components):

| | precision | recall | f1 | f2 |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.858 | 0.978 | 0.915 | 0.952 |
| all classical proposals (recall ceiling) | 0.493 | 0.978 | 0.656 | 0.818 |
| FUSED + CNN (PRIMARY) | 0.924 | 0.978 | 0.950 | 0.967 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 0.929 | 0.978 | 0.953 | 0.968 |

95% CI on F1 (bootstrap over groups): THRESHOLD only (peak >= 60 C, classical) [0.880, 0.940], FUSED + CNN (PRIMARY) [0.893, 0.989], FUSED + CNN, no hot-bypass (PRIMARY) [0.900, 0.989]

**Frame level** (does the frame contain fire):

| | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| THRESHOLD only (peak >= 60 C, classical) | 0.919 | 1.000 | 0.958 | 0.944 |
| all classical proposals (recall ceiling) | 0.734 | 1.000 | 0.846 | 0.769 |
| FUSED + CNN (PRIMARY) | 0.981 | 1.000 | 0.990 | 0.988 |
| FUSED + CNN, no hot-bypass (PRIMARY) | 0.981 | 0.990 | 0.985 | 0.981 |

**Patch level** (verifier on 369 proposals, 189 fire):

| | precision | recall | f1 | ap | roc_auc | threshold |
|---|---|---|---|---|---|---|
| peak temperature verifier | 0.877 | 0.984 | 0.928 | 0.995 | 0.993 | 60.000 |
| CNN (PRIMARY) verifier | 0.949 | 0.984 | 0.966 | 0.969 | 0.989 | 0.747 |

**Hot-blob bypass** (trigger ≥ 3 px at ≥ 495 °C = 0.9 × saturation 550 °C, or blob larger than the patch): CNN saturated: fired 37x, 0 on non-fire; CNN oversize: fired 29x, 0 on non-fire.

CPU latency per frame (ms, this machine): threshold 1.7, fused:cnn 2.6
Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5).
<!-- END:pipeline_test_synth_by_flame3_model -->
