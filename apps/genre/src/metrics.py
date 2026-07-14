"""Full metrics recompute over a FROZEN bundle — the metrics wall (no retrain).

Model 1 is frozen, so every metric the dashboard shows is *derived from the
delivered weights*, not produced by a new training run. This script loads the
bundle, runs it once over each split to harvest (y_true, y_pred, probs), and
writes one fat ``metrics_full.json`` next to the bundle. Bit-identical to the
delivered model by construction; cheap (inference over a few hundred clips).

    python projects/genre/src/metrics.py \
        --config projects/genre/configs/beardown.yaml \
        --bundle projects/genre/models/beardown

Output  ->  projects/genre/models/beardown/metrics_full.json
            { schema_version, generated, genres, cv, acceptance,
              splits: { val: {...}, test: {...} } }

Each split block carries:
  scalars      accuracy, balanced_accuracy, macro/weighted/micro-F1,
               macro precision/recall, cohen_kappa, top2_accuracy,
               log_loss, brier (multiclass)
  per_genre    precision / recall / f1 / support  (per class)
  confusion    raw [10x10] + row-normalized (recall view)
  roc          per-genre OvR AUC + macro + micro, PR-AUC (avg precision) per genre
  calibration  ECE, MCE, Brier; reliability bins [{conf, acc, count}]

The viewer (Training panel toggle) reads this; the legacy metrics.json contract
is left untouched, so nothing downstream breaks.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

# repo root on path so `_shared` and `projects.genre...` resolve from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

C, A, R = "\033[36m", "\033[33m", "\033[0m"


def _bar(done, total, w=28):
    f = int(w * done / max(total, 1))
    return "█" * f + "·" * (w - f)


# --------------------------------------------------------------- metric kernels
def _confusion(y_true, y_pred, k):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row == 0, 1, row))     # recall-normalized
    return cm.astype(int).tolist(), np.round(cm_norm, 6).tolist()


def _roc(y_true, probs, k):
    """Per-genre one-vs-rest AUC + macro/micro, and PR-AUC (average precision)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    Y = np.eye(k)[y_true]                                   # one-hot [N,k]
    per_auc, per_ap = {}, {}
    for i in range(k):
        yi = Y[:, i]
        if yi.sum() == 0 or yi.sum() == len(yi):            # degenerate class in this split
            per_auc[i], per_ap[i] = None, None
            continue
        per_auc[i] = float(roc_auc_score(yi, probs[:, i]))
        per_ap[i] = float(average_precision_score(yi, probs[:, i]))
    valid = [v for v in per_auc.values() if v is not None]
    macro = float(np.mean(valid)) if valid else None
    try:
        micro = float(roc_auc_score(Y, probs, average="micro", multi_class="ovr"))
    except Exception:
        micro = None
    return per_auc, per_ap, macro, micro


def _calibration(y_true, y_pred, probs, bins=15):
    """ECE / MCE + reliability bins on the max-probability (confidence) axis."""
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out, ece, mce, N = [], 0.0, 0.0, len(y_true)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            out.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                        "conf": None, "acc": None, "count": 0}); continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(a - c)
        ece += (n / N) * gap
        mce = max(mce, gap)
        out.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                    "conf": round(c, 6), "acc": round(a, 6), "count": n})
    return {"ece": round(ece, 6), "mce": round(mce, 6),
            "n_bins": bins, "bins": out}


def split_metrics(y_true, y_pred, probs, genres):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_recall_fscore_support, precision_score,
                                 recall_score, cohen_kappa_score, log_loss)
    k = len(genres)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); probs = np.asarray(probs)
    Y = np.eye(k)[y_true]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), average=None, zero_division=0)
    per_genre = {genres[i]: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(s[i])} for i in range(k)}

    # top-2 accuracy
    top2 = np.argsort(-probs, axis=1)[:, :2]
    top2_acc = float(np.mean([yt in row for yt, row in zip(y_true, top2)]))
    # multiclass Brier = mean squared error vs one-hot, summed over classes
    brier = float(np.mean(np.sum((probs - Y) ** 2, axis=1)))

    per_auc, per_ap, macro_auc, micro_auc = _roc(y_true, probs, k)
    cm_raw, cm_norm = _confusion(y_true, y_pred, k)

    return {
        "scalars": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "top2_accuracy": top2_acc,
            "log_loss": float(log_loss(y_true, probs, labels=list(range(k)))),
            "brier": brier,
            "roc_auc_macro": macro_auc,
            "roc_auc_micro": micro_auc,
            "n": int(len(y_true)),
        },
        "per_genre": per_genre,
        "roc": {"per_genre_auc": {genres[i]: per_auc[i] for i in range(k)},
                "per_genre_ap":  {genres[i]: per_ap[i]  for i in range(k)},
                "macro_auc": macro_auc, "micro_auc": micro_auc},
        "confusion": {"labels": genres, "raw": cm_raw, "row_normalized": cm_norm},
        "calibration": _calibration(y_true, y_pred, probs),
    }


# ----------------------------------------------------------------- harvest path
def _evaluate(model, loader, device):
    import torch
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            img, tab, y = batch
            logits = model(img.to(device), tab.to(device))
            ps.append(torch.softmax(logits, 1).cpu().numpy())
            ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys); probs = np.concatenate(ps)
    return y_true, probs.argmax(1), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", required=True, help="models/<name> dir of the frozen bundle")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out", default=None, help="default: <bundle>/metrics_full.json")
    args = ap.parse_args()

    import torch, yaml
    from projects.genre.src import dataio
    from projects.genre.src.models import beardown as _beardown   # noqa: F401 (registers)
    from projects.genre.src.bundle import load_bundle

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"{C}🐻 metrics wall · frozen recompute · {args.bundle}{R}")
    print(f"{C}   device={device}  splits={splits}{R}")

    print(f"{C}  [1/3] loading data …{R}", flush=True)
    loaded = dataio.load(
        representation=cfg.get("representation", "fused"),
        split_strategy=cfg["split"]["mode"],
        data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
        seed=cfg["split"].get("seed", 0), standardize=True,
        drop_length=cfg.get("features", {}).get("drop_length", False),
        image_size=cfg.get("features", {}).get("image_size", 128),
        image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    print(f"        {loaded.summary()}")

    print(f"{C}  [2/3] loading frozen bundle …{R}", flush=True)
    b = load_bundle(args.bundle, device=device)
    genres = b.genres
    legacy = b.metrics or {}

    print(f"{C}  [3/3] scoring splits …{R}")
    blocks = {}
    for i, sp in enumerate(splits):
        print(f"\r{C}        [{_bar(i, len(splits))}] {sp} …{R}", end="", flush=True)
        loader = dataio.to_torch_loader(loaded, sp, batch=256, shuffle=False)
        yt, yp, pr = _evaluate(b.model, loader, device)
        blocks[sp] = split_metrics(yt, yp, pr, genres)
        acc = blocks[sp]["scalars"]["accuracy"]
        print(f"\r{C}        [{_bar(i + 1, len(splits))}] {sp}: acc={acc:.4f}  "
              f"macroF1={blocks[sp]['scalars']['macro_f1']:.4f}  "
              f"AUC={blocks[sp]['scalars']['roc_auc_macro']}{R}")

    out = {
        "schema_version": "metrics_full/1.0",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bundle": os.path.basename(os.path.normpath(args.bundle)),
        "genres": genres,
        "cv": legacy.get("cv"),
        "acceptance": legacy.get("acceptance"),
        "splits": blocks,
    }
    out_path = args.out or os.path.join(args.bundle, "metrics_full.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{C}  metrics_full.json -> {out_path}   {A}🐻 [{len(splits)} split(s) recorded]{R}")


if __name__ == "__main__":
    main()
