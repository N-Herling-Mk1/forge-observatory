"""beardown_rrm (Model 2) — broad random sweep, RRM-selected.

    python projects/genre/src/sweep.py --config projects/genre/configs/beardown_rrm.yaml

Two passes, because RRM needs the cohort's s_max / U_max:

  Pass 1 (per config, n≈20):  3-fold GroupKFold-by-track. From the SAME three fold
    models collect  A (mean fold acc), s (std fold acc), off_diag rate, and U (mean
    LLLA predictive σ on held-out folds at a shared-α τ). One coherent CV substrate
    → all four RRM inputs.  Also records the per-config empirical-Bayes τ (B-column).
  Pass 2 (cohort):  s_max,U_max → RRM per config; RAM, Pareto, Pearson, dCor.
  Promote:  argmax-RRM winner → single refit on full train → 8-file bundle at
    models/beardown_rrm/ → run.json (Model-panel dot flips) + sweep.json (catalogue).

The head is `det`; the Bayesian layer (llla.LastLayerLaplace) is the post-hoc
evaluation lens applied identically to every config — the SAME object the FORGE tab
serves, so the U that selects the winner is the U FORGE displays (single source of
truth). Reuses train.py (train_model/evaluate/_split_metrics) verbatim.

Sandbox note: imports torch transitively (via train.py) — runs on the GPU box, not
the numpy-only sandbox. The metric math is in rrm.py and is unit-tested there.
"""
from __future__ import annotations
import argparse, copy, os, sys, time, json

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.profiler import profile_run                       # noqa: E402
from _shared.schema import RunRecord                           # noqa: E402
from projects.genre.src import dataio                          # noqa: E402
from projects.genre.src.models import REGISTRY                 # noqa: E402
from projects.genre.src.models import beardown as _beardown    # noqa: F401,E402
from projects.genre.src.bundle import (write_bundle, load_bundle,  # noqa: E402
                                       compute_phi, ggn_eigenbasis)
from projects.genre.src.bayes.llla import LastLayerLaplace      # noqa: E402
from projects.genre.src import rrm as R                         # noqa: E402
from projects.genre.src import registry as reg                  # noqa: E402
from projects.genre.src.train import (train_model, evaluate,    # noqa: E402
                                      _split_metrics, _git_sha)

C, A_, Rz = "\033[36m", "\033[33m", "\033[0m"
RED = "\033[31m"


# --------------------------------------------------------------- config sampling
def _sample_space(space: dict, rng: np.random.Generator) -> dict:
    """One random draw from the search space (seeded)."""
    out = {}
    for name, spec in space.items():
        d = spec["dist"]
        if d == "uniform":
            out[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif d == "loguniform":
            lo, hi = np.log10(spec["low"]), np.log10(spec["high"])
            out[name] = float(10.0 ** rng.uniform(lo, hi))
        elif d == "choice":
            vals = spec["values"]
            out[name] = vals[int(rng.integers(len(vals)))]
        else:
            raise ValueError(f"unknown dist {d!r} for {name}")
    return out


def _resolve_cfg(base: dict, draw: dict) -> dict:
    """Apply one sampled draw onto a deep copy of the base cfg. head stays det;
    penultimate tracks dense_units_img (paper: fusion_units)."""
    cfg = copy.deepcopy(base)
    a, tr = cfg["arch"], cfg["train"]
    a["spec_cnn"]["embed"] = int(draw["dense_units_img"])
    a["spec_cnn"]["dropout"] = float(draw["dropout_backbone"])
    a["tab_mlp"]["hidden"] = [int(draw["dense_units_tab"])]
    a["tab_mlp"]["embed"] = int(draw["dense_units_tab"])
    a["tab_mlp"]["dropout"] = float(draw["dropout_tab"])
    a["fusion"] = str(draw["fusion"])
    a["head"]["penultimate"] = int(draw["dense_units_img"])
    tr["lr"] = float(draw["lr"])
    tr["epochs"] = int(draw["epochs"])
    tr["optimizer"] = str(draw["optimizer"])
    return cfg


# ----------------------------------------------------------------- per-fold U
def _fold_U(model, loaded, a_idx, b_idx, device, *, tau_alpha, method,
            n_samples, record_mackay) -> dict:
    """Post-hoc LLLA on the frozen fold model. φ on fold-train builds Λ,U + the
    output factor; score fold-val at the shared-α τ. Returns U (selection) and,
    optionally, U at the per-config MacKay τ (recorded B-column)."""
    n_classes = model.classifier.out_features
    phi_tr_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False, indices=a_idx)
    phi_tr, _ = compute_phi(model, phi_tr_loader, device=device, return_y=True)
    lam, Ueig = ggn_eigenbasis(phi_tr)
    W = model.classifier.weight.detach().cpu().numpy()
    b = model.classifier.bias.detach().cpu().numpy()
    lap = LastLayerLaplace(lam, Ueig, W, b, phi_train=phi_tr)

    phi_val_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False, indices=b_idx)
    phi_val, _ = compute_phi(model, phi_val_loader, device=device, return_y=True)

    tau_A = tau_alpha * float(lam.max())
    sig_A = [lap.predict_posterior(phi_val[i], tau_A, method=method,
                                   n_samples=n_samples)["sigma"].mean()
             for i in range(phi_val.shape[0])]
    out = {"U": float(np.mean(sig_A)) if sig_A else 0.0,
           "tau_A": float(tau_A), "n_starved": int((lam < tau_A).sum()), "d": int(lam.size)}

    if record_mackay:
        tau_B = R.optimal_tau_mackay(lam, w_energy=float((W ** 2).sum()), n_classes=n_classes)
        sig_B = [lap.predict_posterior(phi_val[i], tau_B, method=method,
                                       n_samples=n_samples)["sigma"].mean()
                 for i in range(phi_val.shape[0])]
        out["U_mackay"] = float(np.mean(sig_B)) if sig_B else 0.0
        out["tau_B"] = float(tau_B)
    return out


# ------------------------------------------------------------------- one config
def _run_config(cfg, loaded, dims, device, sw, tag) -> dict:
    """3-fold CV for one resolved config → all RRM inputs + recorded columns."""
    tr = cfg["train"]; rep = cfg.get("representation", "fused")
    folds = sw["folds"]
    gkf = GroupKFold(n_splits=folds)
    groups = loaded.track_ids["train"]
    tr_idx = np.arange(loaded.n["train"])

    fold_acc, fold_U, fold_U_b, off_rows = [], [], [], []
    yt_all, yp_all = [], []
    n_classes = len(loaded.label_map)
    for k, (a_idx, b_idx) in enumerate(gkf.split(tr_idx, groups=groups)):
        m = REGISTRY[cfg["model"]](cfg, dims).to(device)
        vl = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False, indices=b_idx)
        train_model(m, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"],
                                              shuffle=True, indices=a_idx),
                    epochs=tr["epochs"], lr=tr["lr"], wd=tr.get("weight_decay", 0.0),
                    rep=rep, device=device, tag=f"{tag} cv{k+1}",
                    optimizer=tr.get("optimizer", "adam"), val_loader=vl,
                    early_stop=tr.get("early_stopping"), reduce_lr=tr.get("reduce_lr_on_plateau"))
        acc, yt, yp, _ = evaluate(m, vl, rep, device)
        fold_acc.append(acc)
        yt_all.append(yt); yp_all.append(yp)
        off_rows.append(R.off_diagonal_rate(yt, yp, n_classes))
        u = _fold_U(m, loaded, a_idx, b_idx, device,
                    tau_alpha=sw["tau_alpha"], method=sw["llla"]["method"],
                    n_samples=sw["llla"].get("n_samples", 2000),
                    record_mackay=sw.get("record_mackay_tau", True))
        fold_U.append(u["U"]); fold_U_b.append(u.get("U_mackay", float("nan")))

    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)
    pooled_off = R.pool_off_diagonal(off_rows)
    Aval = float(np.mean(fold_acc))
    return {
        "A": Aval, "s": float(np.std(fold_acc)),
        "U": float(np.mean(fold_U)), "U_mackay": float(np.nanmean(fold_U_b)),
        "macro_f1": float(f1_score(yt_all, yp_all, average="macro", zero_division=0)),
        "micro_f1": Aval,                              # micro-F1 == accuracy (single-label)
        "off_diag": pooled_off["rate"], "off_top_pair": pooled_off["top_pair"],
        "fold_acc": [float(x) for x in fold_acc],
    }


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=None, help="override sweep.n (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sw = cfg["sweep"]
    run_name = cfg.get("run_name", "beardown_rrm")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rep = cfg.get("representation", "fused")
    n = args.n or sw["n"]

    print(f"{C}🐻 ═══════════ {A_}BEAR DOWN · RRM SWEEP{C} ═══════════ 🐻{Rz}")
    print(f"{C}   {A_}{run_name}{C}  n={n}  folds={sw['folds']}  select={sw['select']}  dev={device}{Rz}")
    print(f"{C}   τ(sel)=α·λ_max  α={sw['tau_alpha']}   llla={sw['llla']['method']}{Rz}")
    print(f"{C}🐻 ════════════════════════════════════════════════ 🐻{Rz}")

    loaded = dataio.load(representation=rep, split_strategy=cfg["split"]["mode"],
                         data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
                         seed=cfg["split"].get("seed", 0), standardize=True,
                         drop_length=cfg.get("features", {}).get("drop_length", False),
                         image_size=cfg.get("features", {}).get("image_size", 128),
                         image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    print(f"  {loaded.summary()}")
    dims = {"tab_in": len(loaded.feature_cols), "img_ch": 1}

    rng = np.random.default_rng(sw["seed"])
    draws = [_sample_space(sw["space"], rng) for _ in range(n)]

    # ---- Pass 1: per config ------------------------------------------------
    rows = []
    t0 = time.time()
    for i, draw in enumerate(draws):
        ccfg = _resolve_cfg(cfg, draw)
        print(f"{C}── cfg {i+1:>2}/{n}  "
              f"img{draw['dense_units_img']} tab{draw['dense_units_tab']} "
              f"{draw['fusion']} {draw['optimizer']} lr{draw['lr']:.1e} "
              f"do({draw['dropout_backbone']:.2f},{draw['dropout_tab']:.2f}) ep{draw['epochs']}{Rz}")
        r = _run_config(ccfg, loaded, dims, device, sw, tag=f"c{i+1:02d}")
        r["cfg_index"] = i; r["draw"] = draw
        rows.append(r)
        el = time.time() - t0
        print(f"{C}   A={r['A']:.4f}  s={r['s']:.4f}  U={r['U']:.4f}  "
              f"off={r['off_diag']:.4f}  macroF1={r['macro_f1']:.4f}   "
              f"[{i+1}/{n}  {el:5.0f}s]{Rz}")

    # ---- Pass 2: cohort ----------------------------------------------------
    Aarr = np.array([r["A"] for r in rows])
    sarr = np.array([r["s"] for r in rows])
    Uarr = np.array([r["U"] for r in rows])
    offarr = np.array([r["off_diag"] for r in rows])
    macro = np.array([r["macro_f1"] for r in rows])
    micro = np.array([r["micro_f1"] for r in rows])

    coh = R.rrm_cohort(Aarr, sarr, Uarr)
    rrm_vals = coh["rrm"]
    ram = R.rank_aggregation(Aarr, rrm_vals, macro, micro)
    tier = R.pareto_front(Aarr, rrm_vals)
    for i, r in enumerate(rows):
        r["RRM"] = float(rrm_vals[i]); r["RAM"] = float(ram["ram"][i])
        r["pareto_tier"] = int(tier[i])

    pearson = R.pearson_matrix({"A": Aarr, "s": sarr, "U": Uarr,
                                "off_diag": offarr, "RRM": rrm_vals})
    dcor = {
        "A__RRM": R.distance_correlation(Aarr, rrm_vals),
        "off_diag__RRM": R.distance_correlation(offarr, rrm_vals),
    }

    # ---- select winner (argmax RRM; tiebreak lower off_diag) ---------------
    best_rrm = rrm_vals.max()
    cands = [i for i in range(n) if abs(rrm_vals[i] - best_rrm) < 1e-9]
    win = min(cands, key=lambda i: offarr[i])
    wrow = rows[win]
    print(f"\n{C}══ winner: cfg {win+1}  RRM={wrow['RRM']:.4f}  A={wrow['A']:.4f}  "
          f"s={wrow['s']:.4f}  U={wrow['U']:.4f}  off={wrow['off_diag']:.4f}{Rz}")
    print(f"{C}   Pearson r(A,RRM)={pearson['matrix'][0,4]:.3f}  "
          f"r(off,RRM)={pearson['matrix'][3,4]:.3f}   "
          f"dCor(off,RRM)={dcor['off_diag__RRM']:.3f}{Rz}")

    # ---- refit winner on full train (profiled) -> the delivered model ------
    wcfg = _resolve_cfg(cfg, wrow["draw"])
    tr = wcfg["train"]
    with profile_run(device=device, label=run_name) as prof:
        model = REGISTRY[wcfg["model"]](wcfg, dims).to(device)
        s_img, s_tab, _ = next(iter(dataio.to_torch_loader(loaded, "val", batch=8, shuffle=False)))
        prof.set_model(model, sample=(s_img.to(device), s_tab.to(device)))
        val_loader = dataio.to_torch_loader(loaded, "val", batch=256, shuffle=False)
        refit_hist = train_model(
            model, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"], shuffle=True),
            epochs=tr["epochs"], lr=tr["lr"], wd=tr.get("weight_decay", 0.0),
            rep=rep, device=device, prof=prof, tag="refit",
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
        "cv": {"folds": sw["folds"], "fold_accuracies": wrow["fold_acc"],
               "mean": wrow["A"], "std": wrow["s"]},
        "val": val_m, "test": test_m,
        "acceptance": {"target_val_accuracy": target.get("val_accuracy"),
                       "accept_min": accept_min, "passed": passed},
        "selection": {"by": "rrm", "rrm": wrow["RRM"], "U": wrow["U"],
                      "off_diag": wrow["off_diag"], "cfg_index": win},
    }
    print(f"{C}  VAL  acc={val_m['accuracy']:.4f}  macroF1={val_m['macro_f1']:.4f}{Rz}")
    print(f"{C}  TEST acc={test_m['accuracy']:.4f}  macroF1={test_m['macro_f1']:.4f}{Rz}")
    mark = f"{A_}PASS{Rz}" if passed else f"{RED}BELOW TARGET{Rz}"
    print(f"{C}  acceptance (val ≥ {accept_min}): {mark}{Rz}")

    # ---- bundle (same 8-file contract as mk1) ------------------------------
    models_dir = os.path.join("projects/genre/models", run_name)
    phi_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False)
    binfo = write_bundle(models_dir, model, loaded, metrics, phi_loader, device=device)
    print(f"{C}  bundle -> {binfo['dir']}  φ{binfo['phi_shape']}{Rz}")

    rb = load_bundle(models_dir, device=device)
    vb = next(iter(dataio.to_torch_loader(loaded, "val", batch=64, shuffle=False)))
    img, tab, _ = vb
    model.eval()
    with torch.no_grad():
        a = model(img.to(device), tab.to(device)); bb = rb.model(img.to(device), tab.to(device))
    assert torch.allclose(a, bb, atol=1e-5), "bundle round-trip mismatch!"
    print(f"{C}  round-trip OK · max|Δ|={(a-bb).abs().max().item():.2e}{Rz}")

    # ---- sweep.json (the full-monty catalogue) -----------------------------
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("projects/genre/runs", run_name, ts)
    os.makedirs(run_dir, exist_ok=True)
    sweep_doc = {
        "run_name": run_name, "generated": ts, "device": device,
        "sweep": {k: sw[k] for k in ("n", "seed", "folds", "select", "tau_alpha", "llla")},
        "space": sw["space"],
        "selection": {"by": "rrm", "winner_cfg_index": win, "winner_rrm": wrow["RRM"],
                      "s_max": coh["s_max"], "U_max": coh["U_max"]},
        "correlations": {
            "pearson": {"labels": pearson["labels"], "matrix": pearson["matrix"].tolist()},
            "dcor": dcor,
        },
        "configs": [
            {"cfg_index": r["cfg_index"], "draw": r["draw"],
             "A": r["A"], "s": r["s"], "U": r["U"], "U_mackay": r["U_mackay"],
             "off_diag": r["off_diag"], "off_top_pair": r["off_top_pair"],
             "macro_f1": r["macro_f1"], "micro_f1": r["micro_f1"],
             "RRM": r["RRM"], "RAM": r["RAM"], "pareto_tier": r["pareto_tier"],
             "fold_acc": r["fold_acc"]}
            for r in rows
        ],
    }
    with open(os.path.join(run_dir, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep_doc, f, indent=2)
    print(f"{C}  sweep.json -> {run_dir}/sweep.json{Rz}")

    # ---- run record (Model panel reads this) -------------------------------
    rec = RunRecord(
        project="genre", model=run_name, git_sha=_git_sha(),
        config=wcfg, split_mode=cfg["split"]["mode"], epochs=refit_hist,
        final_metrics={
            "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
            "cv_mean_acc": wrow["A"], "cv_std_acc": wrow["s"],
            "rrm": wrow["RRM"], "selection_U": wrow["U"], "off_diag": wrow["off_diag"],
        },
        paper_target=target,
    )
    rec.compute = prof.record()
    rec.write(run_dir)
    print(f"{C}  run.json -> {run_dir}   {A_}🐻 [{run_name} delivered]{Rz}")

    # ---- register the bundle ----------------------------------------------
    try:
        reg.register(models_dir, models_dir, git_sha=_git_sha(), run_id=ts, select="never")
        print(f"{C}  registered in {models_dir}/registry.json (select=never){Rz}")
    except Exception as e:                              # registry is best-effort
        print(f"{A_}  registry skip: {e}{Rz}")


if __name__ == "__main__":
    main()
