"""
FORGE · phonon · MODEL 2 training entrypoint  —  "info-theory + sweeps".

mk2 is mk1's architecture with a *modified, information-theory-aware objective*.
The reproduction (mk1) trains on plain bin-wise MSE, which is blind to the fact that
the 51 DOS bins live on a metric frequency axis: a peak predicted one bin low costs
the same as one predicted ten bins low, and MSE systematically over-smooths sharp
van Hove features (hedging toward the mean lowers MSE). mk2 fixes that by blending in
a 1-D Earth Mover / Wasserstein-1 term (EMD) that penalises mass at the *wrong
frequency* — exactly what the headline metric (omega_bar within 10%) rewards.

    L = (1 - alpha) * MSE  +  alpha * DIST  +  beta * |H[p_pred] - H[p_true]|

      DIST   = emd1d (default) or JS divergence   — the distributional term
      ENTROPY= spectral-entropy match              — optional sharpness regulariser
      alpha  = 0  -> pure MSE  (this IS the mk1 control backbone; train it for the
                                 FORGE two-backbone comparison)
      alpha  > 0  -> the mk2 treatment

All three info-theory terms are torch ports of src/metrics.py, made differentiable.
EMD in 1-D has a closed form — the integral of |CDF_pred - CDF_true| — so it needs
only a cumsum (a linear op): gradients flow cleanly, no Sinkhorn iterations.

Runs / sweeps:

    # control (pure MSE) and treatment (EMD-regularised) — two backbones for FORGE
    python -m src.train_mk2 --data-dir data/raw --epochs 64 --alpha 0.0  --run-name mk2_mse
    python -m src.train_mk2 --data-dir data/raw --epochs 64 --alpha 0.4  --run-name mk2_emd

    # smoke (CPU, ~1 min)
    python -m src.train_mk2 --data-dir data/raw --limit 120 --epochs 2 --alpha 0.4 --run-name mk2_smoke

    # the "sweeps" half: Optuna over {alpha, beta, lr, mul, layers}, maximising val frac_within
    python -m src.train_mk2 --data-dir data/raw --sweep 40 --sweep-epochs 12 --limit 400 \
                            --run-name mk2_sweep

Emits: runs/<run_name>.torch  (same schema as mk1 + a "loss_cfg" block)
       runs/<run_name>.mk2_sweep.json  (best trial, when --sweep is used)

NOTE ON SCALES: with EMD in bin units a single 1-bin peak shift gives EMD ~ 1.0, while
MSE is ~ 0.02 — EMD is ~50x larger in magnitude. So alpha is NOT a 50/50 knob; even
modest alpha lets EMD dominate. Don't over-read alpha — optimise val frac_within (the
sweep does this) and watch the per-component readout printed at each checkpoint.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import data as D
from .model import build_model
from . import metrics as M

BAR = "{l_bar}{bar:18}{r_bar}{bar:-18b}"
EPS = 1e-12


def _banner(msg):
    print(f"\n  ARES // FORGE-PHONON :: {msg}\n  " + "-" * 58)


# ============================================================================
# Differentiable info-theory losses (torch ports of src/metrics.py)
# ============================================================================
def _as_dist(x):
    """Row-normalise a (non-negative) DOS into a probability distribution over bins."""
    x = x.clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(EPS)


def emd1d_torch(pred, target, dx=1.0):
    """1-D Wasserstein-1 (Earth Mover) along the frequency axis, mean over the batch.

    Closed form on an ordered axis: EMD = sum_bins |CDF_pred - CDF_true| * dx.
    cumsum is linear, abs/sum are differentiable a.e. -> clean gradients, no solver.
    """
    p, q = _as_dist(pred), _as_dist(target)
    cdf = torch.cumsum(p - q, dim=-1)
    return cdf.abs().sum(dim=-1).mul(dx).mean()


def spectral_entropy_torch(x):
    """Shannon entropy (nats) of the normalised DOS, per row. Sharp -> low; broad -> high."""
    p = _as_dist(x)
    return -(p * p.clamp_min(EPS).log()).sum(dim=-1)


def entropy_match_torch(pred, target):
    """|H[p_pred] - H[p_true]| averaged over the batch — counters MSE over-smoothing."""
    return (spectral_entropy_torch(pred) - spectral_entropy_torch(target)).abs().mean()


def js_torch(pred, target):
    """Jensen-Shannon divergence (nats) between predicted and true DOS, mean over batch."""
    p, q = _as_dist(pred), _as_dist(target)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(EPS).log() - b.clamp_min(EPS).log())).sum(dim=-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


class Criterion:
    """The composite mk2 objective. Holds config; callable like a loss_fn.

    total = (1-alpha)*MSE + alpha*DIST + beta*entropy_match
    """

    def __init__(self, dist="emd", alpha=0.4, beta=0.0, dx=1.0):
        assert dist in ("emd", "js", "mse"), dist
        self.dist, self.alpha, self.beta, self.dx = dist, float(alpha), float(beta), float(dx)
        self._mse = nn.MSELoss()

    def _dist_term(self, out, target):
        if self.dist == "emd":
            return emd1d_torch(out, target, self.dx)
        if self.dist == "js":
            return js_torch(out, target)
        return self._mse(out, target)  # "mse": degenerate, pure-MSE control

    def __call__(self, out, target):
        mse = self._mse(out, target)
        total = (1.0 - self.alpha) * mse
        if self.alpha > 0.0:
            total = total + self.alpha * self._dist_term(out, target)
        if self.beta > 0.0:
            total = total + self.beta * entropy_match_torch(out, target)
        return total

    @torch.no_grad()
    def components(self, out, target):
        """Per-term values for logging (not for backprop)."""
        d = {"mse": self._mse(out, target).item(),
             "dist": self._dist_term(out, target).item()}
        d["ent_match"] = entropy_match_torch(out, target).item() if self.beta > 0 else 0.0
        return d

    def tag(self):
        return f"{self.dist} a={self.alpha:g} b={self.beta:g}"


# ============================================================================
# Data / eval helpers
# ============================================================================
def build_everything(args):
    """Load data + build periodic graphs + loaders ONCE (reused across sweep trials)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(D.DEFAULT_DTYPE)
    _banner(f"device={device}  dtype=float64")

    _banner("loading data + building periodic graphs")
    graphs, meta = D.load_dataset(args.data_dir, max_radius=args.max_radius, limit=args.limit)
    tr, te, va = D.load_splits(args.data_dir)
    if args.limit:
        keep = set(range(args.limit))
        tr = np.array([i for i in tr if i in keep]); va = np.array([i for i in va if i in keep])
        te = np.array([i for i in te if i in keep])
    num_neighbors = D.avg_neighbors(graphs, idx=tr)
    print(f"  train/val/test = {len(tr)}/{len(va)}/{len(te)}   num_neighbors={num_neighbors:.2f}")

    dl_tr, dl_va, dl_te = D.make_loaders(graphs, tr, te, va, batch_size=args.batch_size)
    return dict(device=device, loaders=(dl_tr, dl_va, dl_te), num_neighbors=num_neighbors,
                freq=meta["phfre"])


@torch.no_grad()
def eval_loader(model, loader, crit, freq, device):
    """Return (mse, mae, val_frac_within_10pct, mean_emd_cm) over a loader."""
    model.eval()
    P, T = [], []
    for d in loader:
        d = d.to(device)
        P.append(model(d).cpu().numpy()); T.append(d.phdos.cpu().numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    frac, _ = M.frac_within(P, T, freq, tol=0.10)
    mse = float(M.mse_per_example(P, T).mean())
    mae = float(np.abs(P - T).mean())
    emd = float(M.emd1d(P, T, freq).mean())
    return mse, mae, frac, emd


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - np.exp(-t * rate / step)))


# ============================================================================
# Core training (shared by the main path and every sweep trial)
# ============================================================================
def train_once(cfg, opt_cfg, crit, ctx, epochs, run_name=None, out="runs",
               save=False, quiet=False, patience=6, min_delta=0.0):
    """Train one model with criterion `crit`. Returns (model, history, val_frac, val_emd).

    cfg      : model-kwargs dict (em_dim, out_dim, layers, mul, lmax, max_radius, number_of_basis)
    opt_cfg  : dict(lr, weight_decay, gamma)
    ctx      : output of build_everything()
    patience : early-stop after this many *consecutive checkpoints* with no val frac_within
               improvement. <=0 disables stopping (still selects/saves the best checkpoint).
               NB cadence is loglinspace, so late in training one checkpoint ~ 5-6 epochs;
               patience is in checkpoints, not epochs.
    min_delta: a checkpoint counts as an improvement only if frac > best_frac + min_delta.

    val frac_within peaks then drifts down under overtraining, so the returned frac/emd and
    the on-disk `.torch` are the BEST checkpoint seen (by val frac_within), not the last.
    """
    device = ctx["device"]; freq = ctx["freq"]; num_neighbors = ctx["num_neighbors"]
    dl_tr, dl_va, dl_te = ctx["loaders"]

    model = build_model(cfg, num_neighbors=num_neighbors).to(device)
    if not quiet:
        print(f"  params = {sum(p.numel() for p in model.parameters()):,}   loss[{crit.tag()}]")

    opt = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                            weight_decay=opt_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=opt_cfg["gamma"])

    history = []
    cpgen = loglinspace(0.3, 5); checkpoint = next(cpgen)
    t0 = time.time()
    ckpt_path = Path(out) / f"{run_name}.torch" if save else None
    if save:
        Path(out).mkdir(parents=True, exist_ok=True)

    # best-checkpoint selection on val frac_within (peaks then drifts -> keep the peak)
    best_frac = -1.0
    best_emd = float("nan")
    best_epoch = -1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_stale = 0
    stopped_early = False

    iterator = range(epochs)
    for epoch in iterator:
        model.train()
        last = 0.0
        bar = dl_tr if quiet else tqdm(dl_tr, total=len(dl_tr), bar_format=BAR, ncols=88,
                                       desc=f"epoch {epoch+1:3d}/{epochs}")
        for d in bar:
            d = d.to(device)
            out_ = model(d)
            loss = crit(out_, d.phdos)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()

        if epoch == checkpoint or epoch == epochs - 1:
            checkpoint = next(cpgen)
            mse, mae, frac, emd = eval_loader(model, dl_va, crit, freq, device)
            wall = time.time() - t0
            history.append({"step": epoch, "wall": wall, "batch": {"loss": last},
                            "valid": {"loss": mse, "mean_abs": mae,
                                      "frac_within": frac, "emd_cm": emd}})

            improved = frac > best_frac + min_delta
            if improved:
                best_frac, best_emd, best_epoch = frac, emd, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                n_stale = 0
            else:
                n_stale += 1

            if save:
                # always persist the BEST weights seen, with the full running history
                torch.save({"state": best_state, "history": history,
                            "model_kwargs": cfg,
                            "loss_cfg": {"dist": crit.dist, "alpha": crit.alpha,
                                         "beta": crit.beta},
                            "best": {"frac_within": best_frac, "emd_cm": best_emd,
                                     "epoch": best_epoch}}, ckpt_path)
            if not quiet:
                star = "  *best*" if improved else f"  (stale {n_stale}/{patience})"
                print(f"  [ckpt] epoch {epoch+1:3d}  val MSE={mse:.4f}  "
                      f"frac@10%={frac:.3f}  EMD={emd:.2f}cm^-1  "
                      f"elapsed={time.strftime('%H:%M:%S', time.gmtime(wall))}{star}")

            if patience > 0 and n_stale >= patience:
                stopped_early = True
                if not quiet:
                    print(f"  [early-stop] no val frac_within gain in {patience} checkpoints — "
                          f"halting at epoch {epoch+1}; best={best_frac:.3f} @ epoch {best_epoch+1}")
                break
        sched.step()

    # restore the best checkpoint so the returned model (and run_single's TEST scorecard)
    # reflect the peak, not the drifted final epoch
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if not quiet and not stopped_early:
        print(f"  [done] best val frac@10%={best_frac:.3f} @ epoch {best_epoch+1}"
              f"  (ran full {epochs} epochs, no early stop)")
    return model, history, best_frac, best_emd


def _cfg_from_args(args, **overrides):
    cfg = dict(in_dim=118, em_dim=args.em_dim, out_dim=args.out_dim, layers=args.layers,
               mul=args.mul, lmax=args.lmax, max_radius=args.max_radius,
               number_of_basis=args.number_of_basis)
    cfg.update(overrides)
    return cfg


# ============================================================================
# Entry points: single run  /  Optuna sweep
# ============================================================================
def run_single(args):
    ctx = build_everything(args)
    crit = Criterion(dist=args.loss, alpha=args.alpha, beta=args.beta,
                     dx=_dx(args, ctx["freq"]))
    _banner(f"training {args.epochs} epochs  [{crit.tag()}]  run={args.run_name}")
    model, _, val_frac, val_emd = train_once(
        _cfg_from_args(args), _opt_from_args(args), crit, ctx,
        args.epochs, run_name=args.run_name, out=args.out, save=True,
        patience=args.patience, min_delta=args.min_delta)

    _banner("done — final scorecard on TEST")
    mse, mae, frac, emd = eval_loader(model, ctx["loaders"][2], crit, ctx["freq"], ctx["device"])
    print(f"  omega_bar within 10%  = {frac:.3f}   (paper target 0.70)")
    print(f"  mean per-example MSE  = {mse:.4f}")
    print(f"  mean EMD-1 (cm^-1)    = {emd:.2f}")
    print(f"  saved -> {Path(args.out) / (args.run_name + '.torch')}")


def run_sweep(args):
    try:
        import optuna
    except ImportError:
        raise SystemExit("  optuna not installed.  pip install optuna   (then re-run with --sweep)")

    ctx = build_everything(args)         # load data ONCE; every trial reuses these loaders
    dx = _dx(args, ctx["freq"])
    _banner(f"OPTUNA sweep — {args.sweep} trials x {args.sweep_epochs} epochs, "
            f"objective = maximise val frac_within")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.0, 0.9)
        beta = trial.suggest_float("beta", 0.0, 0.1)
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        mul = trial.suggest_categorical("mul", [16, 32, 64])
        layers = trial.suggest_categorical("layers", [2, 3])
        crit = Criterion(dist=args.loss, alpha=alpha, beta=beta, dx=dx)
        cfg = _cfg_from_args(args, mul=mul, layers=layers)
        opt_cfg = dict(lr=lr, weight_decay=args.weight_decay, gamma=args.gamma)
        _, _, val_frac, val_emd = train_once(cfg, opt_cfg, crit, ctx,
                                             args.sweep_epochs, save=False, quiet=True,
                                             patience=args.patience, min_delta=args.min_delta)
        trial.set_user_attr("val_emd", val_emd)
        print(f"  trial {trial.number:3d}  a={alpha:.2f} b={beta:.3f} lr={lr:.1e} "
              f"mul={mul} L={layers}  ->  frac={val_frac:.3f}  EMD={val_emd:.2f}")
        return val_frac

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.sweep)

    best = {"value_frac_within": study.best_value, "params": study.best_params,
            "val_emd": study.best_trial.user_attrs.get("val_emd")}
    out_json = Path(args.out) / f"{args.run_name}.mk2_sweep.json"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(best, indent=2))

    _banner("sweep complete")
    print(f"  best val frac_within = {study.best_value:.3f}")
    print(f"  best params          = {study.best_params}")
    print(f"  saved -> {out_json}")
    print(f"\n  retrain the winner full-length, e.g.:\n"
          f"    python -m src.train_mk2 --data-dir {args.data_dir} --epochs {args.epochs} \\\n"
          f"        --alpha {study.best_params['alpha']:.3f} --beta {study.best_params['beta']:.3f} \\\n"
          f"        --lr {study.best_params['lr']:.4f} --mul {study.best_params['mul']} "
          f"--layers {study.best_params['layers']} --run-name {args.run_name}_best")


def _dx(args, freq):
    if args.emd_units == "cm":
        return float(np.mean(np.diff(np.asarray(freq, float))))
    return 1.0  # bin units (default, stable, interpretable knob)


def _opt_from_args(args):
    return dict(lr=args.lr, weight_decay=args.weight_decay, gamma=args.gamma)


def get_parser():
    p = argparse.ArgumentParser(description="FORGE phonon MODEL 2 — info-theory loss + sweeps")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name", default="mk2_emd")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="truncate dataset (smoke / sweep)")
    # early stopping / best-checkpoint selection (on val frac_within)
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop after N consecutive checkpoints with no val frac_within "
                        "gain; 0 disables (still selects+saves the best checkpoint). NB late "
                        "checkpoints are ~5-6 epochs apart (loglinspace cadence).")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="min val frac_within gain to count a checkpoint as an improvement")
    # objective
    p.add_argument("--loss", choices=["emd", "js", "mse"], default="emd",
                   help="distributional term blended with MSE (mse = degenerate control)")
    p.add_argument("--alpha", type=float, default=0.4, help="weight on DIST term; 0 => pure MSE")
    p.add_argument("--beta", type=float, default=0.0, help="weight on spectral-entropy match")
    p.add_argument("--emd-units", choices=["bins", "cm"], default="bins",
                   help="EMD dx scale: bin index (default) or cm^-1")
    # optimiser (mk1 reference defaults)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.96)
    # architecture ladder
    p.add_argument("--em-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=51)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mul", type=int, default=32)
    p.add_argument("--lmax", type=int, default=1)
    p.add_argument("--max-radius", type=float, default=5.0)
    p.add_argument("--number-of-basis", type=int, default=10)
    # sweep
    p.add_argument("--sweep", type=int, default=0, metavar="N_TRIALS",
                   help="run an Optuna sweep of N trials instead of a single training run")
    p.add_argument("--sweep-epochs", type=int, default=12, help="epochs per sweep trial")
    return p


def main():
    args = get_parser().parse_args()
    if args.sweep > 0:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
