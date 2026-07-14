"""mk2.5 metric-validation study — refit ALL sweep configs on full-train, record test.

    python projects/genre/src/batch_refit.py --config projects/genre/configs/beardown_rrm.yaml

=== FIREWALL (read this) =====================================================
Model SELECTION for deployment was already done on CROSS-VALIDATION columns only
(variance-weighted RRM_w -> cfg 17; see reselect.py / reselect.json). The test
scores produced here DID NOT and DO NOT influence which model is deployed.

Their ONLY purpose is to evaluate the selection metric's predictive validity:
"does a CV-side metric (RRM_w, RRM, accuracy, ...) rank configs the way held-out
TEST accuracy does?" That is a hypothesis about the METRIC, not a selector over
models. Recording test for all configs to study the metric is legitimate; letting
a test score decide the deployed model would be leakage. We do the former, not the
latter — and this file emits that statement into batch_test.json so the artifact is
self-documenting.
==============================================================================

For each config it reuses promote_cfg's refit path (so numbers match the deployed
model's computation exactly), records val+test, then reports rank-agreement
(Spearman) between each CV metric and test accuracy. Bundles are written to a throwaway
dir by default (NOT the deployed slot) and can be discarded after the table is read.

Cost: one full-train refit per config (~90s CPU each). Use --only to refit a subset.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, shutil

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import dataio                          # noqa: E402
from projects.genre.src.models import REGISTRY                 # noqa: E402
from projects.genre.src.models import beardown as _beardown    # noqa: F401,E402
from projects.genre.src import rrm as R                        # noqa: E402
from projects.genre.src.sweep import _resolve_cfg              # noqa: E402
from projects.genre.src.train import train_model, _split_metrics  # noqa: E402
import yaml                                                    # noqa: E402

C, A_, Rz = "\033[36m", "\033[33m", "\033[0m"


def _find_sweep(run_name):
    cands = glob.glob(f"projects/genre/runs/{run_name}/*/sweep.json")
    if not cands:
        sys.exit(f"no sweep.json under projects/genre/runs/{run_name}/")
    return max(cands, key=os.path.getmtime)


def _spearman(x, y):
    """Spearman rank correlation (ties averaged) — pure numpy."""
    def ranks(v):
        v = np.asarray(v, float); order = v.argsort()
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        for u in np.unique(v):
            m = v == u; r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--only", default=None, help="comma list of 1-indexed cfgs (default: all)")
    ap.add_argument("--keep-bundles", action="store_true", help="keep per-config bundles (default: discard)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run_name = cfg.get("run_name", "beardown_rrm")
    rep = cfg.get("representation", "fused")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sweep_path = args.sweep or _find_sweep(run_name)
    sweep = json.load(open(sweep_path, encoding="utf-8"))
    rows = sorted(sweep["configs"], key=lambda r: r["cfg_index"])
    if args.only:
        want = {int(x) - 1 for x in args.only.split(",")}
        rows = [r for r in rows if r["cfg_index"] in want]

    print(f"{C}🐻 mk2.5 metric-validation · refit {len(rows)} configs · FIREWALLED from selection{Rz}")
    print(f"{C}   selection already done on CV (variance-weighted RRM_w). Test here = metric study only.{Rz}\n")

    loaded = dataio.load(representation=rep, split_strategy=cfg["split"]["mode"],
                         data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
                         seed=cfg["split"].get("seed", 0), standardize=True,
                         drop_length=cfg.get("features", {}).get("drop_length", False),
                         image_size=cfg.get("features", {}).get("image_size", 128),
                         image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    print(f"  {loaded.summary()}\n")
    dims = {"tab_in": len(loaded.feature_cols), "img_ch": 1}
    scratch = os.path.join("projects/genre/models", "_batch_scratch")

    out_rows, t0 = [], time.time()
    for r in rows:
        i = r["cfg_index"]
        wcfg = _resolve_cfg(cfg, r["draw"]); tr = wcfg["train"]
        model = REGISTRY[wcfg["model"]](wcfg, dims).to(device)
        val_loader = dataio.to_torch_loader(loaded, "val", batch=256, shuffle=False)
        train_model(model, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"], shuffle=True),
                    epochs=tr["epochs"], lr=tr["lr"], wd=tr.get("weight_decay", 0.0), rep=rep,
                    device=device, tag=f"cfg{i+1}", optimizer=tr.get("optimizer", "adam"),
                    val_loader=val_loader, early_stop=tr.get("early_stopping"),
                    reduce_lr=tr.get("reduce_lr_on_plateau"))
        val_m = _split_metrics(model, loaded, "val", rep, device)
        test_m = _split_metrics(model, loaded, "test", rep, device)
        out_rows.append({
            "cfg": i + 1, "A": r["A"], "s": r["s"], "U": r["U"], "off_diag": r["off_diag"],
            "RRM": r["RRM"], "macro_f1_cv": r["macro_f1"],
            "val_acc": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
            "test_acc": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
            "test_per_genre_f1": test_m.get("per_genre_f1", {}),
        })
        el = time.time() - t0
        print(f"{C}  cfg{i+1:>2}  CV-A={r['A']:.4f} RRM={r['RRM']:+.4f}  ->  "
              f"VAL={val_m['accuracy']:.4f}  TEST={test_m['accuracy']:.4f}  "
              f"[{len(out_rows)}/{len(rows)} {el:4.0f}s]{Rz}")

    # ---- variance-weighted RRM_w over the cohort (the deployed selector) ----
    A = np.array([o["A"] for o in out_rows]); s = np.array([o["s"] for o in out_rows])
    U = np.array([o["U"] for o in out_rows]); test = np.array([o["test_acc"] for o in out_rows])
    rrm_locked = np.array([o["RRM"] for o in out_rows])
    vw = R.variance_weights(A, s, U)
    rrm_w = R.rrm_weighted_cohort(A, s, U, vw)["rrm"]
    cfgs = [o["cfg"] for o in out_rows]

    # ---- the hypothesis: which CV metric best predicts TEST rank? -----------
    agree = {
        "RRM_w (variance, deployed selector)": _spearman(rrm_w, test),
        "RRM (equal-weight, locked)":          _spearman(rrm_locked, test),
        "CV accuracy (A) alone":               _spearman(A, test),
        "fold-stability (-s)":                 _spearman(-s, test),
        "epistemic (-U)":                      _spearman(-U, test),
    }
    print(f"\n{C}══ metric validation — Spearman rank-agreement with TEST accuracy ══{Rz}")
    for k, v in sorted(agree.items(), key=lambda kv: -kv[1]):
        print(f"   {v:+.3f}   {k}")
    best_metric = max(agree, key=agree.get)
    deployed = cfgs[int(np.argmax(rrm_w))]
    print(f"\n{C}   deployed (variance-weighted RRM_w winner) = cfg{deployed}   "
          f"test_acc={test[int(np.argmax(rrm_w))]:.4f}{Rz}")
    print(f"{C}   best test-rank predictor among CV metrics: {best_metric} ({agree[best_metric]:+.3f}){Rz}")

    doc = {
        "FIREWALL": ("Model selection for deployment used CROSS-VALIDATION metrics only "
                     "(variance-weighted RRM_w). Test scores below were recorded POST-HOC "
                     "solely to evaluate the selection metric's predictive validity and did "
                     "NOT influence which model is deployed."),
        "sweep_path": sweep_path,
        "deployed_cfg": int(deployed),
        "variance_weights": vw.tolist(),
        "metric_vs_test_spearman": agree,
        "configs": out_rows,
    }
    out_path = os.path.join(os.path.dirname(sweep_path), "batch_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"{C}   batch_test.json -> {out_path}{Rz}")
    if not args.keep_bundles and os.path.isdir(scratch):
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
