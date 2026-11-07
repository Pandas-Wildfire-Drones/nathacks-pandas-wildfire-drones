"""PRIMARY: train the CNN patch verifier with group k-fold CV, pick a recall-first threshold, export ONNX.

    python -m training.train_cnn --manifest exclude/data/flame3/manifest_split.csv \
        --out-dir exclude/runs/cnn_flame3 --export-to perception/models/fire_cnn.onnx

Reads ONLY rows with split == "train" (make_split.py). The test split is for perception/evaluate.py, once.

Outputs (out-dir):  patches.npz (cache, reused by train_gbm.py) · cv.json · fire_cnn.onnx + fire_cnn.json
Also: --export-to copies the ONNX + sidecar (commit those) · RESULTS.md section "cnn_cv_<tag>" + results/*.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from perception.datasets import load_manifest
from perception.patches import PATCH_SPEC, normalize_patches
from perception.results_doc import fmt_metric_table, update_section

from .cv import evaluate_oof, make_folds
from .model import TinyFireNet
from .patch_data import PatchSet, augment_batch, build_patch_set

REPO = Path(__file__).resolve().parents[1]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(x_c: np.ndarray, y: np.ndarray, args, device, seed: int, log_prefix: str = "") -> TinyFireNet:
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = TinyFireNet(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(y) / args.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    pos = max(int(y.sum()), 1)
    pos_weight = torch.tensor([min((len(y) - pos) / pos, args.max_pos_weight)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = normalize_patches(augment_batch(x_c[idx], rng) if args.augment else x_c[idx], PATCH_SPEC)
            xb_t = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(y[idx].astype(np.float32)).to(device)[:, None]
            loss = loss_fn(model(xb_t), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"{log_prefix}epoch {epoch + 1}/{args.epochs} loss {total / len(y):.4f}")
    return model


@torch.no_grad()
def predict(model: TinyFireNet, x_c: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x_c), batch_size):
        xb = torch.from_numpy(normalize_patches(x_c[i:i + batch_size], PATCH_SPEC)).to(device)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy().ravel())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def export_onnx(model: TinyFireNet, path: Path) -> None:
    model = model.eval().cpu()
    dummy = torch.zeros(2, PATCH_SPEC.channels, PATCH_SPEC.size_px, PATCH_SPEC.size_px)
    kw = dict(input_names=["patches"], output_names=["logits"], opset_version=17)
    try:  # classic TorchScript exporter: no extra deps, stable dynamic batch axis
        torch.onnx.export(model, (dummy,), str(path), dynamo=False,
                          dynamic_axes={"patches": {0: "N"}, "logits": {0: "N"}}, **kw)
    except Exception as e:  # newer torch may only offer the dynamo exporter (needs onnxscript)
        print(f"[export] legacy exporter unavailable ({e.__class__.__name__}); using dynamo exporter")
        batch = torch.export.Dim("N", min=1)
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, dynamic_shapes={"x": {0: batch}}, **kw)


def check_onnx_parity(model: TinyFireNet, path: Path, x_c: np.ndarray) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xb = normalize_patches(x_c[:64], PATCH_SPEC)
    onnx_logits = sess.run(None, {sess.get_inputs()[0].name: xb})[0].ravel()
    with torch.no_grad():
        torch_logits = model.eval().cpu()(torch.from_numpy(xb)).numpy().ravel()
    return float(np.max(np.abs(onnx_logits - torch_logits)))


def cv_markdown(tag: str, stats: dict, cv: dict, args) -> str:
    s, fin = cv["summary"], cv["final_threshold"]
    rows = [(f"fold {m['fold']} (n={m['n']}, pos={m['positives']}, thr={m['threshold']:.3f})", m) for m in cv["per_fold"]]
    rows.append(("**mean ± std over folds**", s))
    rows.append((f"pooled OOF @ final thr {fin['threshold']:.3f}", cv["pooled_oof_at_final_threshold"]))
    rows.append(("pooled OOF @ 0.5 (reference)", cv["pooled_oof_at_0.5"]))
    warn = "" if fin["met_target"] else f"\n\n**WARNING: recall target {fin['min_recall']} not reached on OOF.**"
    return "\n".join([
        f"### CNN verifier (PRIMARY): {len(cv['per_fold'])}-fold group CV · {tag}", "",
        f"Patch-level, on classical proposals + mask-centred positives. {stats['patches']} patches "
        f"({stats['positives']} fire / {stats['negatives']} not, {stats['proposal_negatives_hard']} hard negatives), "
        f"{stats['groups']} groups. Threshold rule: recall >= {args.min_recall} then max precision; each fold's "
        f"threshold is chosen on the other folds. Epochs {args.epochs}, width {args.width}, augment {args.augment}.",
        "", fmt_metric_table(rows, ("precision", "recall", "f1", "f2", "accuracy", "ap", "roc_auc")) + warn,
        "", "Accuracy is shown for completeness only (class imbalance makes it misleading; see perception/evaluation.py).",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=str(REPO / "exclude" / "runs" / f"cnn_{dt.datetime.now():%Y%m%d_%H%M}"))
    ap.add_argument("--export-to", help="copy final .onnx + .json here (e.g. perception/models/fire_cnn.onnx)")
    ap.add_argument("--tag", default="", help="label for RESULTS.md section (default: manifest folder name)")
    ap.add_argument("--results-doc", default=str(REPO / "perception" / "RESULTS.md"))
    ap.add_argument("--patch-cache", help="reuse a patches.npz instead of rebuilding")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--max-pos-weight", type=float, default=20.0)
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--weak-labels", action="store_true", help="label proposals in mask-less fire frames positive")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.manifest).parent.name
    device = pick_device(args.device)
    print(f"[train_cnn] device={device} out={out}")

    if args.patch_cache:
        ps = PatchSet.load(args.patch_cache)
        stats = {"patches": len(ps.y), "positives": int(ps.y.sum()), "negatives": int((ps.y == 0).sum()),
                 "groups": len(set(ps.groups)), "proposal_negatives_hard": int(((ps.origin == "proposal") & (ps.y == 0)).sum())}
    else:
        recs = load_manifest(args.manifest, split="train")
        if not recs:
            raise SystemExit("no split == 'train' rows: run `python -m training.make_split` first")
        ps = build_patch_set(recs, weak_labels=args.weak_labels, seed=args.seed)
        stats = ps.stats
        ps.save(out / "patches.npz")
    print(f"[train_cnn] patches: {json.dumps(stats)}")

    folds = make_folds(ps.y, ps.groups, args.folds, args.seed)
    oof = np.full(len(ps.y), np.nan, dtype=np.float32)
    for k, (tr, va) in enumerate(folds):
        model = train_model(ps.patches_c[tr], ps.y[tr], args, device, seed=args.seed + k, log_prefix=f"[fold {k}] ")
        oof[va] = predict(model, ps.patches_c[va], device)
    cv = evaluate_oof(ps.y, oof, folds, args.min_recall)
    print(json.dumps(cv["summary"], indent=1))

    final = train_model(ps.patches_c, ps.y, args, device, seed=args.seed + 100, log_prefix="[final] ")
    onnx_path = out / "fire_cnn.onnx"
    export_onnx(final, onnx_path)
    parity = check_onnx_parity(final, onnx_path, ps.patches_c)
    print(f"[export] {onnx_path} max |torch - onnx| logit diff = {parity:.2e}")
    if parity > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch; do not ship this model")

    meta = {
        "model_kind": "cnn", "architecture": "TinyFireNet", "width": args.width, "patch_spec": PATCH_SPEC.to_dict(),
        "input_name": "patches", "output": "logits (apply sigmoid)",
        "threshold": cv["final_threshold"]["threshold"], "threshold_rule": f"recall>={args.min_recall} then max precision (pooled OOF)",
        "cv_summary": cv["summary"], "patch_stats": stats, "manifest": str(args.manifest), "tag": tag,
        "epochs": args.epochs, "seed": args.seed, "onnx_parity_max_abs": parity,
        "created": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
    }
    (out / "fire_cnn.json").write_text(json.dumps(meta, indent=2, default=float))
    (out / "cv.json").write_text(json.dumps({"cv": cv, "args": vars(args)}, indent=2, default=float))
    if args.export_to:
        dst = Path(args.export_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_path, dst)
        shutil.copy(out / "fire_cnn.json", dst.with_suffix(".json"))
        print(f"[export] copied to {dst}")
    update_section(args.results_doc, f"cnn_cv_{tag}", cv_markdown(tag, stats, cv, args),
                   {"cv": cv, "meta": meta})
    print(f"[results] updated {args.results_doc}")


if __name__ == "__main__":
    main()
