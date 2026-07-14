"""Refit ANY recorded sweep config on full-train and report it once on val+test.

    python projects/genre/src/promote_cfg.py --config projects/genre/configs/beardown_rrm.yaml --cfg 17

Pulls the config's EXACT sampled draw from sweep.json, rebuilds the resolved cfg,
and runs the SAME refit→bundle→val/test path sweep.py used for its winner — so the
numbers are computed identically (no methodology drift between configs).

Methodology note (leakage): this is a legitimate report-once pass. Selection already
happened on CV-RRM; reading a config's test score AFTER selection does not influence
which model was picked. Report it alongside the recorded winner's score for honesty.

By default writes the bundle to models/<run_name>_cfg<N>/ so it does NOT clobber the
recorded winner bundle. Pass --out-name beardown_rrm to overwrite the deployed slot
once you've decided.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.profiler import profile_run                       # noqa: E402
from _shared.schema import RunRecord                           # noqa: E402
from projects.genre.src import dataio                          # noqa: E402
from projects.genre.src.models import REGISTRY                 # noqa: E402
from projects.genre.src.models import beardown as _beardown    # noqa: F401,E402
from projects.genre.src.bundle import write_bundle, load_bundle  # noqa: E402
from projects.genre.src import registry as reg                 # noqa: E402
from projects.genre.src.sweep import _resolve_cfg              # noqa: E402  (reuse the EXACT resolver)
from projects.genre.src.train import (train_model, _split_metrics,  # noqa: E402
                                      _git_sha)
import yaml                                                    # noqa: E402

C, A_, Rz, RED = "\033[36m", "\033[33m", "\033[0m", "\033[31m"


def _find_sweep(run_name):
    cands = glob.glob(f"projects/genre/runs/{run_name}/*/sweep.json")
    if not cands:
        sys.exit(f"no sweep.json under projects/genre/runs/{run_name}/")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cfg", type=int, required=True, help="1-indexed config number to refit")
    ap.add_argument("--sweep", default=None, help="sweep.json (default: newest for this run)")
    ap.add_argument("--out-name", default=None, help="bundle dir name (default: <run>_cfg<N>)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run_name = cfg.get("run_name", "beardown_rrm")
    rep = cfg.get("representation", "fused")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_name = args.out_name or f"{run_name}_cfg{args.cfg}"

    sweep_path = args.sweep or _find_sweep(run_name)
    sweep = json.load(open(sweep_path, encoding="utf-8"))
    idx = args.cfg - 1
    row = next((r for r in sweep["configs"] if r["cfg_index"] == idx), None)
    if row is None:
        sys.exit(f"cfg {args.cfg} (index {idx}) not in {sweep_path}")
    draw = row["draw"]

    print(f"{C}🐻 promote · {A_}cfg {args.cfg}{C} from {sweep_path}{Rz}")
    print(f"{C}   recorded CV: A={row['A']:.4f} s={row['s']:.4f} U={row['U']:.4f} "
          f"off={row['off_diag']:.4f} RRM={row['RRM']:+.4f}{Rz}")
    print(f"{C}   draw: {draw}{Rz}")
    print(f"{C}   bundle -> models/{out_name}   (clobbers winner slot only if --out-name {run_name}){Rz}")

    wcfg = _resolve_cfg(cfg, draw)
    tr = wcfg["train"]
    loaded = dataio.load(representation=rep, split_strategy=cfg["split"]["mode"],
                         data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
                         seed=cfg["split"].get("seed", 0), standardize=True,
                         drop_length=cfg.get("features", {}).get("drop_length", False),
                         image_size=cfg.get("features", {}).get("image_size", 128),
                         image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    print(f"  {loaded.summary()}")
    dims = {"tab_in": len(loaded.feature_cols), "img_ch": 1}

    with profile_run(device=device, label=out_name) as prof:
        model = REGISTRY[wcfg["model"]](wcfg, dims).to(device)
        s_img, s_tab, _ = next(iter(dataio.to_torch_loader(loaded, "val", batch=8, shuffle=False)))
        prof.set_model(model, sample=(s_img.to(device), s_tab.to(device)))
        val_loader = dataio.to_torch_loader(loaded, "val", batch=256, shuffle=False)
        refit_hist = train_model(
            model, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"], shuffle=True),
            epochs=tr["epochs"], lr=tr["lr"], wd=tr.get("weight_decay", 0.0),
            rep=rep, device=device, prof=prof, tag=f"cfg{args.cfg}",
            optimizer=tr.get("optimizer", "adam"), val_loader=val_loader,
            early_stop=tr.get("early_stopping"), reduce_lr=tr.get("reduce_lr_on_plateau"),
            show_curve=True)
        with prof.block("predict", n=s_img.size(0)):
            with torch.no_grad():
                model.eval(); model(s_img.to(device), s_tab.to(device))

    val_m = _split_metrics(model, loaded, "val", rep, device)
    test_m = _split_metrics(model, loaded, "test", rep, device)
    target = cfg.get("paper_target", {})
    accept_min = target.get("accept_min", 0.75)
    passed = val_m["accuracy"] >= accept_min
    metrics = {
        "cv": {"folds": sweep["sweep"]["folds"], "fold_accuracies": row["fold_acc"],
               "mean": row["A"], "std": row["s"]},
        "val": val_m, "test": test_m,
        "acceptance": {"target_val_accuracy": target.get("val_accuracy"),
                       "accept_min": accept_min, "passed": passed},
        "selection": {"by": "promote", "from_cfg": args.cfg, "rrm": row["RRM"],
                      "U": row["U"], "off_diag": row["off_diag"]},
    }

    print(f"\n{C}══ cfg {args.cfg} refit results{Rz}")
    print(f"{C}  VAL  acc={val_m['accuracy']:.4f}  macroF1={val_m['macro_f1']:.4f}{Rz}")
    print(f"{C}  TEST acc={test_m['accuracy']:.4f}  macroF1={test_m['macro_f1']:.4f}{Rz}")
    mark = f"{A_}PASS{Rz}" if passed else f"{RED}BELOW TARGET{Rz}"
    print(f"{C}  acceptance (val ≥ {accept_min}): {mark}{Rz}")

    # head-to-head vs the recorded sweep winner (if different)
    wsel = sweep.get("selection", {})
    win_idx = wsel.get("winner_cfg_index")
    if win_idx is not None and win_idx != idx:
        wm = "projects/genre/models/" + run_name + "/metrics.json"
        if os.path.exists(wm):
            wj = json.load(open(wm, encoding="utf-8"))
            wv = wj.get("val", {}); wt = wj.get("test", {})
            print(f"\n{C}── head-to-head ──{Rz}")
            print(f"{C}  recorded winner cfg{win_idx+1}:  VAL {wv.get('accuracy',0):.4f}  TEST {wt.get('accuracy',0):.4f}{Rz}")
            print(f"{C}  this    config  cfg{args.cfg}:  VAL {val_m['accuracy']:.4f}  TEST {test_m['accuracy']:.4f}{Rz}")

    models_dir = os.path.join("projects/genre/models", out_name)
    phi_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False)
    binfo = write_bundle(models_dir, model, loaded, metrics, phi_loader, device=device)
    print(f"{C}  bundle -> {binfo['dir']}  φ{binfo['phi_shape']}{Rz}")

    rb = load_bundle(models_dir, device=device)
    img, tab, _ = next(iter(dataio.to_torch_loader(loaded, "val", batch=64, shuffle=False)))
    model.eval()
    with torch.no_grad():
        a = model(img.to(device), tab.to(device)); bb = rb.model(img.to(device), tab.to(device))
    assert torch.allclose(a, bb, atol=1e-5), "bundle round-trip mismatch!"
    print(f"{C}  round-trip OK · max|Δ|={(a-bb).abs().max().item():.2e}{Rz}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("projects/genre/runs", out_name, ts)
    rec = RunRecord(
        project="genre", model=out_name, git_sha=_git_sha(),
        config=wcfg, split_mode=cfg["split"]["mode"], epochs=refit_hist,
        final_metrics={"val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
                       "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
                       "cv_mean_acc": row["A"], "cv_std_acc": row["s"], "rrm": row["RRM"],
                       "promoted_from_cfg": args.cfg},
        paper_target=target)
    rec.compute = prof.record()
    rec.write(run_dir)
    print(f"{C}  run.json -> {run_dir}{Rz}")
    try:
        reg.register(models_dir, models_dir, git_sha=_git_sha(), run_id=ts, select="never")
        print(f"{C}  registered {models_dir}/registry.json{Rz}")
    except Exception as e:
        print(f"{A_}  registry skip: {e}{Rz}")


if __name__ == "__main__":
    main()
