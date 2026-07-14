"""Entrypoint: config -> train (profiled) -> bundle + run.json/compute.json.

    python projects/genre/src/train.py --config projects/genre/configs/beardown.yaml

Model 1 (faithful repro, NO sweep): train the fixed cfg_14 config.
  • 3-fold GroupKFold-by-track on the TRAIN split  -> CV mean/σ (RRM stability axis)
  • refit on full train                            -> the delivered weights
  • headline accuracy on VAL  (the ≥0.75 acceptance gate)
  • report-once metric on TEST (never used for selection)
  • write the bundle, then RELOAD it and assert predictions match (save/load proven)
When run.json lands, the Model-panel dot for the run flips to delivered.
"""
from __future__ import annotations
import argparse, os, sys, time, json, subprocess

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# repo root on path so `_shared` and `projects.genre...` resolve from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.profiler import profile_run                      # noqa: E402
from _shared.schema import RunRecord                          # noqa: E402
from projects.genre.src import dataio                         # noqa: E402
from projects.genre.src.models import REGISTRY                # noqa: E402
from projects.genre.src.models import beardown as _beardown   # noqa: F401,E402  (registers)
from projects.genre.src.bundle import write_bundle, load_bundle  # noqa: E402

C, A, R = "\033[36m", "\033[33m", "\033[0m"


def _unpack(batch, rep, device):
    if rep in ("fused", "fused3"):
        img, tab, y = batch
        return (img.to(device), tab.to(device)), y.to(device)
    x, y = batch
    return (x.to(device),), y.to(device)


def _make_optimizer(name, params, lr, wd):
    name = (name or "adam").lower()
    if name == "adamw":   return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def _val_metrics(model, loader, rep, device, lossf):
    """val (accuracy, mean_loss) in one pass."""
    model.eval()
    ls = correct = n = 0
    with torch.no_grad():
        for batch in loader:
            inp, y = _unpack(batch, rep, device)
            logits = model(*inp)
            ls += lossf(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return correct / max(n, 1), ls / max(n, 1)


def train_model(model, loader, *, epochs, lr, wd, rep, device, prof=None, tag="",
                optimizer="adam", val_loader=None, early_stop=None, reduce_lr=None,
                show_curve=False):
    """Train. If val_loader is given, evaluate val each epoch and apply (config-driven)
    early stopping with best-weight restore + ReduceLROnPlateau — matching BEARDOWN's
    callbacks. monitor = 'val_accuracy' (maximize) or 'val_loss' (minimize, smoother on
    small val sets). show_curve prints the per-epoch val trajectory at the end."""
    opt = _make_optimizer(optimizer, model.parameters(), lr, wd)
    lossf = nn.CrossEntropyLoss()
    es = early_stop or {}; rl = reduce_lr or {}
    monitor = es.get("monitor", "val_accuracy"); mode_min = (monitor == "val_loss")
    es_pat = int(es.get("patience", 0)); es_restore = bool(es.get("restore_best", True))
    rl_factor = float(rl.get("factor", 0.5)); rl_pat = int(rl.get("patience", 0)); rl_min = float(rl.get("min_lr", 0.0))
    best_metric = float("inf") if mode_min else -1.0
    best_acc, best_state, best_ep = -1.0, None, -1
    es_wait = rl_wait = 0
    hist = []
    for ep in range(epochs):
        model.train()
        lsum = correct = n = 0
        for batch in loader:
            inp, y = _unpack(batch, rep, device)
            opt.zero_grad()
            logits = model(*inp)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            if prof is not None:
                prof.tick(y.size(0))
            bs = y.size(0)
            lsum += loss.item() * bs; n += bs
            correct += (logits.argmax(1) == y).sum().item()
        rec = {"epoch": ep, "loss": lsum / max(n, 1), "train_acc": correct / max(n, 1)}
        va = None
        if val_loader is not None:
            va, vloss = _val_metrics(model, val_loader, rep, device, lossf)
            rec["val_acc"] = va; rec["val_loss"] = vloss
            cur = vloss if mode_min else va
            improved = (cur < best_metric - 1e-5) if mode_min else (cur > best_metric + 1e-5)
            if improved:
                best_metric, best_ep, best_acc = cur, ep, va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                es_wait = rl_wait = 0
            else:
                es_wait += 1; rl_wait += 1
                if rl_pat and rl_wait >= rl_pat:        # ReduceLROnPlateau
                    for g in opt.param_groups:
                        g["lr"] = max(g["lr"] * rl_factor, rl_min)
                    rl_wait = 0
        hist.append(rec)
        bar = "█" * int(28 * (ep + 1) / epochs)
        vtxt = f"  val {va:.3f}" if va is not None else ""
        print(f"\r{C}  {tag:<10} [{bar:<28}] ep {ep+1}/{epochs}  "
              f"loss {rec['loss']:.3f}  acc {rec['train_acc']:.3f}{vtxt}{R}",
              end="", flush=True)
        if es_pat and es_wait >= es_pat:                # early stopping
            print(f"\n{A}  {tag}: early stop @ ep {ep+1}  (best val {best_acc:.4f} @ ep {best_ep+1}){R}", end="")
            break
    print()
    if val_loader is not None and es_restore and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if show_curve and val_loader is not None and hist:
        print(f"{C}  val curve (best {monitor} @ ep {best_ep+1}, val_acc={best_acc:.4f}):{R}")
        for h in hist:
            mark = f"{A}  ◀ best{R}" if h["epoch"] == best_ep else ""
            print(f"    ep {h['epoch']+1:>2}  val_acc {h.get('val_acc', 0):.4f}  "
                  f"val_loss {h.get('val_loss', 0):.4f}{mark}")
    return hist


@torch.no_grad()
def evaluate(model, loader, rep, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        inp, y = _unpack(batch, rep, device)
        ps.append(torch.softmax(model(*inp), 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys); probs = np.concatenate(ps); y_pred = probs.argmax(1)
    return float((y_pred == y_true).mean()), y_true, y_pred, probs


def _per_genre_f1(y_true, y_pred, label_map):
    inv = {v: k for k, v in label_map.items()}
    f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(inv))), zero_division=0)
    return {inv[i]: float(f1[i]) for i in range(len(inv))}


def _split_metrics(model, loaded, split, rep, device, batch=256):
    loader = dataio.to_torch_loader(loaded, split, batch=batch, shuffle=False)
    acc, yt, yp, _ = evaluate(model, loader, rep, device)
    return {
        "accuracy": acc,
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "per_genre_f1": _per_genre_f1(yt, yp, loaded.label_map),
    }


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="override cfg data_root (point at existing GTZAN)")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg train.epochs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable early stopping (run full epochs) + print the val curve — diagnostic")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rep = cfg.get("representation", "fused")
    run_name = cfg.get("run_name", cfg["model"])
    tr = cfg["train"]
    epochs = args.epochs or tr["epochs"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    folds = cfg.get("cv", {}).get("folds", 3)

    build = REGISTRY[cfg["model"]]

    print(f"{C}🐻 ═══════════════ {A}BEAR DOWN{C} ═══════════════ 🐻{R}")
    print(f"{C}   train · {A}{run_name}{C}   rep={rep}  dev={device}  epochs={epochs}{R}")
    print(f"{C}🐻 ══════════════════════════════════════════ 🐻{R}")

    # ---- data (split already train/val/test, track-level, scaled) -----------
    loaded = dataio.load(representation=rep,
                         split_strategy=cfg["split"]["mode"],
                         data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
                         seed=cfg["split"].get("seed", 0),
                         standardize=True,
                         drop_length=cfg.get("features", {}).get("drop_length", False),
                         image_size=cfg.get("features", {}).get("image_size", 128),
                         image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    print(f"  {loaded.summary()}")
    dims = {"tab_in": len(loaded.feature_cols), "img_ch": 1}

    # ---- 3-fold GroupKFold-by-track on TRAIN (CV stability) -----------------
    tr_idx = np.arange(loaded.n["train"])
    groups = loaded.track_ids["train"]
    gkf = GroupKFold(n_splits=folds)
    fold_acc = []
    for k, (a_idx, b_idx) in enumerate(gkf.split(tr_idx, groups=groups)):
        m = build(cfg, dims).to(device)
        vl = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False, indices=b_idx)
        train_model(m, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"],
                                              shuffle=True, indices=a_idx),
                    epochs=epochs, lr=tr["lr"], wd=tr.get("weight_decay", 0.0),
                    rep=rep, device=device, tag=f"cv {k+1}/{folds}", optimizer=tr.get("optimizer", "adam"),
                    val_loader=vl, early_stop=(None if args.no_early_stop else tr.get("early_stopping")),
                    reduce_lr=tr.get("reduce_lr_on_plateau"))
        acc, *_ = evaluate(m, vl, rep, device)
        fold_acc.append(acc)
        print(f"{C}    fold {k+1} val_acc = {acc:.4f}{R}")
    cv = {"folds": folds, "fold_accuracies": fold_acc,
          "mean": float(np.mean(fold_acc)), "std": float(np.std(fold_acc))}
    print(f"{C}  CV: mean={cv['mean']:.4f}  σ={cv['std']:.4f}{R}")

    # ---- refit on full train (profiled) -> the delivered model --------------
    with profile_run(device=device, label=run_name) as prof:
        model = build(cfg, dims).to(device)
        sample_loader = dataio.to_torch_loader(loaded, "val", batch=8, shuffle=False)
        s_img, s_tab, _ = next(iter(sample_loader))
        prof.set_model(model, sample=(s_img.to(device), s_tab.to(device)))
        val_loader = dataio.to_torch_loader(loaded, "val", batch=256, shuffle=False)
        refit_hist = train_model(
            model, dataio.to_torch_loader(loaded, "train", batch=tr["batch_size"], shuffle=True),
            epochs=epochs, lr=tr["lr"], wd=tr.get("weight_decay", 0.0),
            rep=rep, device=device, prof=prof, tag="refit", optimizer=tr.get("optimizer", "adam"),
            val_loader=val_loader, early_stop=(None if args.no_early_stop else tr.get("early_stopping")),
            reduce_lr=tr.get("reduce_lr_on_plateau"), show_curve=True)
        # seed the F5 inference-path timing
        with prof.block("predict", n=s_img.size(0)):
            with torch.no_grad():
                model.eval(); model(s_img.to(device), s_tab.to(device))

    val_m = _split_metrics(model, loaded, "val", rep, device)
    test_m = _split_metrics(model, loaded, "test", rep, device)
    target = cfg.get("paper_target", {})
    accept_min = target.get("accept_min", 0.75)
    passed = val_m["accuracy"] >= accept_min

    metrics = {
        "cv": cv, "val": val_m, "test": test_m,
        "acceptance": {"target_val_accuracy": target.get("val_accuracy"),
                       "accept_min": accept_min, "passed": passed},
    }

    print(f"{C}  VAL  acc={val_m['accuracy']:.4f}  macroF1={val_m['macro_f1']:.4f}{R}")
    print(f"{C}  TEST acc={test_m['accuracy']:.4f}  macroF1={test_m['macro_f1']:.4f}  (report-once){R}")
    mark = f"{A}PASS{R}" if passed else "\033[31mBELOW TARGET\033[0m"
    print(f"{C}  acceptance (val ≥ {accept_min}): {mark}{R}")

    # ---- bundle (born plug-in-ready) ----------------------------------------
    models_dir = os.path.join("projects/genre/models", run_name)
    phi_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False)
    binfo = write_bundle(models_dir, model, loaded, metrics, phi_loader, device=device)
    print(f"{C}  bundle -> {binfo['dir']}  φ{binfo['phi_shape']}{R}")

    # ---- round-trip: reload and assert predictions match --------------------
    rb = load_bundle(models_dir, device=device)
    vb = next(iter(dataio.to_torch_loader(loaded, "val", batch=64, shuffle=False)))
    inp, _ = _unpack(vb, rep, device)
    model.eval()
    with torch.no_grad():
        a = model(*inp); b = rb.model(*inp)
    assert torch.allclose(a, b, atol=1e-5), "bundle round-trip mismatch!"
    print(f"{C}  round-trip OK · max|Δ|={ (a-b).abs().max().item():.2e}{R}")

    # ---- run record (Model panel reads this) --------------------------------
    rec = RunRecord(
        project="genre", model=run_name, git_sha=_git_sha(),
        config=cfg, split_mode=cfg["split"]["mode"],
        epochs=refit_hist,
        final_metrics={
            "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
            "cv_mean_acc": cv["mean"], "cv_std_acc": cv["std"],
        },
        paper_target=target,
    )
    rec.compute = prof.record()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("projects/genre/runs", run_name, ts)
    rec.write(run_dir)
    print(f"{C}  run.json -> {run_dir}   {A}🐻 [{run_name} delivered]{R}")


if __name__ == "__main__":
    main()
