# FLAME 3 Sycan composites: issue #4 A/B (scale-normalized vs plain crops) + cross-domain checks.
# Run from the repo root:  bash training/retrain_flame3.sh      (PY=... to pick the interpreter)
# Prereqs (training/FLAME3.md "Recipe"):
#   python -m training.build_manifest_flame3 --root exclude/data/flame3_sycan_marsh_raw --out exclude/data/flame3/manifest.csv
#   python -m training.composite --manifest exclude/data/flame3/manifest.csv --out exclude/data/flame3_composite --per-background 3 --decoys
#   python -m training.make_split --manifest exclude/data/flame3_composite/manifest.csv --test-frac 0.25 --seed 0
set -e
PY=${PY:-.venv/bin/python}
M=exclude/data/flame3_composite/manifest_split.csv
echo "=== CNN, scale-normalized (per-row GSD) ==="
$PY -m training.train_cnn --manifest $M --out-dir exclude/runs/cnn_flame3 --tag flame3_composite \
  --epochs 20 --device cpu 2>&1 | grep -v "epoch [0-9]*/"
echo "=== CNN, plain crops (A/B) ==="
$PY -m training.train_cnn --manifest $M --out-dir exclude/runs/cnn_flame3_plain --tag flame3_composite_plain \
  --epochs 20 --device cpu --no-scale-normalize 2>&1 | grep -v "epoch [0-9]*/"
echo "=== test split: FLAME 3 composites ==="
for m in cnn_flame3 cnn_flame3_plain; do
  $PY -m perception.evaluate --manifest $M --split test --cnn exclude/runs/$m/fire_cnn.onnx \
    --label "FLAME 3 Sycan composites, 4 held-out ground cells ($m)" --key pipeline_test_$m \
    --json-out exclude/runs/$m/test.json | tail -25
done
echo "=== cross-domain: shipped synth (nadir, night) model on FLAME 3 composites ==="
$PY -m perception.evaluate --manifest $M --split test --cnn perception/models/fire_cnn_synth.onnx \
  --label "FLAME 3 Sycan composites, shipped synthetic model" --key pipeline_test_flame3_by_synth_model \
  --json-out exclude/runs/cnn_flame3/test_by_synth_model.json | tail -25
echo "=== cross-domain: FLAME 3 model on the synthetic night test set ==="
$PY -m perception.evaluate --manifest exclude/data/synth/manifest_split.csv --split test \
  --cnn exclude/runs/cnn_flame3/fire_cnn.onnx --label "synthetic night test flights, FLAME 3 composite model" \
  --key pipeline_test_synth_by_flame3_model --json-out exclude/runs/cnn_flame3/test_synth.json | tail -25
