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

<!-- BEGIN:compare_handlabels_DoqCA -->
### Approach comparison: FLAME 3 Sycan hand labels (DoqCA, 80-200 °C, active+smoulder = fire) · composite CNN

98 candidates (78 fire), 18 groups. Hot-blob bypass fired on 31.

| approach | ROC AUC [95% CI, groups] | AP | search thr | precision | recall |
|---|---|---|---|---|---|
| algorithmic (peak °C) | 0.552 [0.406, 0.670] | 0.857 | 60.000 | 0.796 | 1.000 |
| CNN alone (no bypass) | 0.437 [0.327, 0.565] | 0.801 | 0.747 | 0.786 | 0.846 |
| full pipeline (CNN + bypass) | 0.440 [0.329, 0.574] | 0.802 | 0.747 | 0.798 | 0.910 |

Triage (NOT_FIRE < low ≤ REVIEW < high ≤ FIRE):

| approach | low | high | fires missed (NOT_FIRE) | review rate | FIRE precision | non-fires auto-FIRE |
|---|---|---|---|---|---|---|
| algorithmic (peak °C) | 60.000 | 150.000 | 0/78 (0.000) | 0.643 | 0.800 | 7/20 |
| CNN alone (no bypass) | 0.215 | 0.904 | 5/78 (0.064) | 0.153 | 0.766 | 18/20 |
| full pipeline (CNN + bypass) | 0.215 | 0.904 | 3/78 (0.038) | 0.173 | 0.766 | 18/20 |
<!-- END:compare_handlabels_DoqCA -->

**How to read the hand-label comparison (2026-11-08):**
* 297 labelled, **199 unsure** (195 of them low-gain frames), so 98 are usable: 78 fire, only 20 not-fire from 9 groups. The CIs are wide.
* **Nothing separates fire from not-fire in this band** (all AUC CIs include 0.5). Medians: not-fire 122 °C / 52 px vs fire 119–135 °C / 41–56 px.
* The 20 negatives are physically suspect. A compact 100–180 °C spot can't be sun-heated grass, and several sit in smoke. "No fire visible in a daytime RGB photo" ≠ "no heat source". Treat this set mainly as a **recall** check until a second labeller confirms the negatives.
* The recall check is a real finding: at the search threshold the full pipeline drops **7 of 78** (9 %) human-confirmed fires that peak-temperature-alone keeps (CNN without the bypass: 12 of 78).
* Why: the composite training set has almost no hot non-fire examples (composite test split, 80–200 °C: 58 fire vs **3** non-fire), so the CNN learned "hot ⇒ fire" there. It still clearly beats the classical baseline **below 80 °C** (diluted small fires), where the threshold misses 29 of 37.
* Re-run: `python -m training.compare handlabels ... --key compare_handlabels_<labeller>`; plots and sliders: `training/compare_approaches.ipynb`.

<!-- BEGIN:compare_flame3_composite_test -->
### Approach comparison: FLAME 3 composite test split · composite CNN

2893 candidates (162 fire), 4 groups. Hot-blob bypass fired on 31.
Proposer missed 7 of 168 fire objects entirely (no approach can recover those).

| approach | ROC AUC [95% CI, groups] | AP | search thr | precision | recall |
|---|---|---|---|---|---|
| algorithmic (peak °C) | 0.985 [0.968, 0.996] | 0.930 | 60.000 | 0.893 | 0.821 |
| CNN alone (no bypass) | 0.997 [0.993, 1.000] | 0.986 | 0.747 | 0.940 | 0.975 |
| full pipeline (CNN + bypass) | 0.997 [0.993, 1.000] | 0.986 | 0.747 | 0.940 | 0.975 |

Triage (NOT_FIRE < low ≤ REVIEW < high ≤ FIRE):

| approach | low | high | fires missed (NOT_FIRE) | review rate | FIRE precision | non-fires auto-FIRE |
|---|---|---|---|---|---|---|
| algorithmic (peak °C) | 60.000 | 150.000 | 29/162 (0.179) | 0.020 | 1.000 | 0/2731 |
| CNN alone (no bypass) | 0.215 | 0.904 | 1/162 (0.006) | 0.007 | 0.963 | 6/2731 |
| full pipeline (CNN + bypass) | 0.215 | 0.904 | 1/162 (0.006) | 0.007 | 0.963 | 6/2731 |
<!-- END:compare_flame3_composite_test -->

<!-- BEGIN:cnn_cv_mix_v2 -->
### CNN verifier (PRIMARY): 5-fold group CV · mix_v2

Patch-level, on classical proposals + mask-centred positives. 24220 patches (13496 fire / 10724 not, 6945 hard negatives), 40 groups. Threshold rule: recall >= 0.95 then max precision; each fold's threshold is chosen on the other folds. Epochs 20, width 16, augment True.

| | precision | recall | f1 | f2 | accuracy | ap | roc_auc |
|---|---|---|---|---|---|---|---|
| fold 0 (n=9004, pos=2411, thr=0.987) | 0.999 | 0.773 | 0.872 | 0.809 | 0.939 | 0.996 | 0.998 |
| fold 1 (n=3389, pos=3098, thr=0.899) | 1.000 | 0.989 | 0.995 | 0.991 | 0.990 | 1.000 | 1.000 |
| fold 2 (n=3798, pos=2612, thr=0.920) | 0.997 | 0.969 | 0.983 | 0.975 | 0.977 | 0.999 | 0.998 |
| fold 3 (n=3913, pos=2637, thr=0.948) | 0.999 | 0.947 | 0.972 | 0.957 | 0.963 | 0.999 | 0.998 |
| fold 4 (n=4116, pos=2738, thr=0.914) | 0.994 | 0.980 | 0.987 | 0.983 | 0.983 | 0.999 | 0.999 |
| **mean ± std over folds** | 0.998 ± 0.002 | 0.932 ± 0.081 | 0.962 ± 0.046 | 0.943 ± 0.068 | 0.971 ± 0.018 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| pooled OOF @ final thr 0.935 | 0.998 | 0.953 | 0.975 | 0.962 | 0.973 | 0.999 | 0.999 |
| pooled OOF @ 0.5 (reference) | 0.994 | 0.981 | 0.987 | 0.984 | 0.986 |  |  |
| CONFIRM op. point @ 0.935 (precision >= 0.98) | 0.998 | 0.953 | 0.975 | 0.962 | 0.973 |  |  |

Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).
Patches scale-normalized by GSD: True.

**Hot-blob bypass data check** (trigger 495 °C = 0.9 × saturation 550 °C): hottest non-fire patch 109.7 °C (p99.9 87.6 °C, patch centre); non-fire patches that could bypass the CNN (upper bound): **0** of 10724. The blob-based count on the test split is in the pipeline section.
<!-- END:cnn_cv_mix_v2 -->

<!-- BEGIN:matrix_mix_v2 -->
### Models × test views (three approaches)

| view | approach | composite (shipped)<br>roc_auc | composite (shipped)<br>precision | composite (shipped)<br>recall | composite (shipped)<br>f1 | composite (shipped)<br>missed | composite (shipped)<br>review | mix_v2<br>roc_auc | mix_v2<br>precision | mix_v2<br>recall | mix_v2<br>f1 | mix_v2<br>missed | mix_v2<br>review |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nadir (n=3407, fire=1980) | algorithmic (peak °C) | 1.000 | 0.990 | 1.000 | 0.995 | 0.000 | 0.176 | 1.000 | 0.990 | 1.000 | 0.995 | 0.000 | 0.176 |
|  | CNN alone | 0.956 | 0.968 | 0.726 | 0.830 | 0.113 | 0.178 | 1.000 | 0.999 | 0.997 | 0.998 | 0.000 | 0.026 |
|  | full pipeline | 0.960 | 0.970 | 0.755 | 0.849 | 0.099 | 0.186 | 1.000 | 0.999 | 0.997 | 0.998 | 0.000 | 0.026 |
| handlabels (n=98, fire=78) | algorithmic (peak °C) | 0.552 | 0.796 | 1.000 | 0.886 | 0.000 | 0.643 | 0.552 | 0.796 | 1.000 | 0.886 | 0.000 | 0.643 |
|  | CNN alone | 0.437 | 0.786 | 0.846 | 0.815 | 0.064 | 0.153 | 0.424 | 0.789 | 0.910 | 0.845 | 0.000 | 0.082 |
|  | full pipeline | 0.440 | 0.798 | 0.910 | 0.850 | 0.038 | 0.173 | 0.424 | 0.793 | 0.936 | 0.859 | 0.000 | 0.082 |
| composite (n=2893, fire=162) | algorithmic (peak °C) | 0.985 | 0.893 | 0.821 | 0.855 | 0.179 | 0.020 | 0.985 | 0.893 | 0.821 | 0.855 | 0.179 | 0.020 |
|  | CNN alone | 0.997 | 0.940 | 0.975 | 0.958 | 0.006 | 0.007 | 0.995 | 0.987 | 0.932 | 0.959 | 0.006 | 0.008 |
|  | full pipeline | 0.997 | 0.940 | 0.975 | 0.958 | 0.006 | 0.007 | 0.995 | 0.987 | 0.932 | 0.959 | 0.006 | 0.008 |
| cvsubset (n=4670, fire=3776) | algorithmic (peak °C) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.002 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.002 |
|  | CNN alone | 0.997 | 1.000 | 0.872 | 0.932 | 0.051 | 0.113 | 1.000 | 1.000 | 0.974 | 0.987 | 0.005 | 0.017 |
|  | full pipeline | 1.000 | 1.000 | 0.998 | 0.999 | 0.002 | 0.152 | 1.000 | 1.000 | 0.998 | 0.999 | 0.000 | 0.021 |
| all_with_hand (n=10915, fire=5984) | algorithmic (peak °C) | 0.998 | 0.991 | 0.995 | 0.993 | 0.005 | 0.066 | 0.998 | 0.991 | 0.995 | 0.993 | 0.005 | 0.066 |
|  | CNN alone | 0.986 | 0.986 | 0.826 | 0.899 | 0.071 | 0.104 | 0.998 | 0.996 | 0.980 | 0.988 | 0.003 | 0.018 |
|  | full pipeline | 0.990 | 0.987 | 0.916 | 0.950 | 0.035 | 0.124 | 0.998 | 0.996 | 0.995 | 0.996 | 0.000 | 0.019 |
| all_no_hand (n=10817, fire=5906) | algorithmic (peak °C) | 0.999 | 0.994 | 0.995 | 0.995 | 0.005 | 0.061 | 0.999 | 0.994 | 0.995 | 0.995 | 0.005 | 0.061 |
|  | CNN alone | 0.988 | 0.989 | 0.826 | 0.900 | 0.071 | 0.104 | 1.000 | 0.999 | 0.981 | 0.990 | 0.003 | 0.017 |
|  | full pipeline | 0.992 | 0.990 | 0.916 | 0.952 | 0.035 | 0.123 | 1.000 | 0.999 | 0.996 | 0.998 | 0.000 | 0.019 |
| nadir_proposals (n=2570, fire=2544) | algorithmic (peak °C) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.572 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.572 |
|  | CNN alone | 0.834 | 0.999 | 0.623 | 0.768 | 0.208 | 0.244 | 0.990 | 1.000 | 0.941 | 0.970 | 0.008 | 0.051 |
|  | full pipeline | 0.862 | 0.999 | 0.717 | 0.835 | 0.168 | 0.283 | 0.992 | 1.000 | 0.961 | 0.980 | 0.001 | 0.058 |
<!-- END:matrix_mix_v2 -->

**How to read the matrix (2026-11-08):**
* `mix_v2` = NADIR plot 2 (labels from each pixel's history) + composites + Sycan weak labels, plus hard examples drawn from the masks (`training/patch_data.py`). Thresholds come from pooled out-of-fold scores only; no test view was used for tuning.
* **Real fire it used to drop, it now keeps.** NADIR plot 1 (a plot it never saw): CNN recall 0.73 → 0.997 at the search threshold. Sycan CV subset: 0.87 → 0.97. Pooled pipeline: fires sent to NOT_FIRE 3.5 % → 0 %, REVIEW load 12 % → 2 %.
* **Composite / night synthetic:** discrimination is unchanged (AUC 0.995–0.997, night 0.989 → 0.992). Recall at the search threshold dips (composite 0.975 → 0.932) because the new threshold is stricter (0.935 vs 0.747). Those fires land in REVIEW, not NOT_FIRE (composite fires missed: 1 for both models). Use `reject_threshold` (0.177) as the keep cut once the search REVIEW band exists (#13).
* `nadir` and `cvsubset` separate on temperature **by construction**: their labels only cover clear cases, and the hard ones are ignored. So algorithmic = 1.0 there, and those views test whether the CNN drops real fire or calls warm char/rocks fire. Don't cite them as evidence of discrimination in the hard band.
* `handlabels` is unchanged (AUC ~0.42–0.55 for every approach), and that's expected: nothing in any training set looks like the 20 "not fire" blobs at 80–200 °C, which are probably real heat (#19). Pipeline recall on your labelled fires: 0.91 → 0.94, with 0 sent to NOT_FIRE.
* Re-run: `python -m training.eval_sets --model "name=path.onnx" ... --key matrix_<tag>`; plots and sliders: `training/compare_approaches.ipynb`.
