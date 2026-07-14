"""
FORGE · phonon · MODEL 1 training entrypoint (faithful Chen et al. replication).

Run on a GPU box (CERN/ATLAS cluster or ARES). CPU works but is slow.
Reference run: ~64 epochs, AdamW(lr=0.005, wd=0.05), ExponentialLR(gamma=0.96), MSE.

    python -m src.train --data-dir data/raw --epochs 64 --batch-size 1 \
                        --run-name e3nn_repro --out runs/

Emits: runs/<run_name>.torch  (state + history, same schema as the authors' checkpoint).
Progress: tqdm per-epoch bar + a status line each checkpoint (ARES style).
"""
from __future__ import annotations
import argparse
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


def _banner(msg):
    print(f"\n  ARES // FORGE-PHONON :: {msg}\n  " + "-" * 58)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - np.exp(-t * rate / step)))


@torch.no_grad()
def evaluate(model, loader, loss_fn, loss_mae, device):
    model.eval()
    tot = tot_mae = 0.0
    for d in loader:
        d = d.to(device)
        out = model(d)
        tot += loss_fn(out, d.phdos).item()
        tot_mae += loss_mae(out, d.phdos).item()
    n = max(len(loader), 1)
    return tot / n, tot_mae / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(D.DEFAULT_DTYPE)
    _banner(f"device={device}  dtype=float64  run={args.run_name}")

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
    model = build_model(vars(args), num_neighbors=num_neighbors).to(device)
    print(f"  params = {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    loss_fn, loss_mae = nn.MSELoss(), nn.L1Loss()

    out_path = Path(args.out); out_path.mkdir(parents=True, exist_ok=True)
    ckpt = out_path / f"{args.run_name}.torch"
    history = []
    cpgen = loglinspace(0.3, 5); checkpoint = next(cpgen)
    t0 = time.time()

    _banner(f"training {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        last = 0.0
        for d in tqdm(dl_tr, total=len(dl_tr), bar_format=BAR, ncols=88,
                      desc=f"epoch {epoch+1:3d}/{args.epochs}"):
            d = d.to(device)
            out = model(d)
            loss = loss_fn(out, d.phdos)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
        wall = time.time() - t0

        if epoch == checkpoint:
            checkpoint = next(cpgen)
            va_loss, va_mae = evaluate(model, dl_va, loss_fn, loss_mae, device)
            tr_loss, tr_mae = evaluate(model, dl_tr, loss_fn, loss_mae, device)
            history.append({"step": epoch, "wall": wall,
                            "batch": {"loss": last},
                            "valid": {"loss": va_loss, "mean_abs": va_mae},
                            "train": {"loss": tr_loss, "mean_abs": tr_mae}})
            torch.save({"state": model.state_dict(), "history": history,
                        "model_kwargs": _kwargs(args, num_neighbors)}, ckpt)
            print(f"  [ckpt] epoch {epoch+1:3d}  train MSE={tr_loss:.4f}  "
                  f"valid MSE={va_loss:.4f}  elapsed={time.strftime('%H:%M:%S', time.gmtime(wall))}")
        sched.step()

    _banner("done — final scorecard on TEST")
    _scorecard(model, dl_te, meta["phfre"], device)
    torch.save({"state": model.state_dict(), "history": history,
                "model_kwargs": _kwargs(args, num_neighbors)}, ckpt)
    print(f"  saved -> {ckpt}")


def _kwargs(args, num_neighbors):
    return dict(in_dim=118, em_dim=args.em_dim, out_dim=args.out_dim, layers=args.layers,
                mul=args.mul, lmax=args.lmax, max_radius=args.max_radius,
                number_of_basis=args.number_of_basis, num_neighbors=num_neighbors)


@torch.no_grad()
def _scorecard(model, loader, freq, device):
    model.eval()
    P, T = [], []
    for d in loader:
        d = d.to(device)
        P.append(model(d).cpu().numpy()); T.append(d.phdos.cpu().numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    frac, _ = M.frac_within(P, T, freq, tol=0.10)
    print(f"  omega_bar within 10%  = {frac:.3f}   (paper target 0.70)")
    print(f"  mean per-example MSE  = {M.mse_per_example(P, T).mean():.4f}")
    print(f"  mean JS divergence    = {M.js_divergence(P, T).mean():.4f}")
    print(f"  mean EMD-1 (cm^-1)    = {M.emd1d(P, T, freq).mean():.2f}")


def get_parser():
    p = argparse.ArgumentParser(description="FORGE phonon MODEL 1 - Chen et al. replication")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name", default="e3nn_repro")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.96)
    p.add_argument("--limit", type=int, default=None, help="truncate dataset (smoke runs)")
    p.add_argument("--em-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=51)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mul", type=int, default=32)
    p.add_argument("--lmax", type=int, default=1)
    p.add_argument("--max-radius", type=float, default=5.0)
    p.add_argument("--number-of-basis", type=int, default=10)
    return p


def main():
    train(get_parser().parse_args())


if __name__ == "__main__":
    main()
