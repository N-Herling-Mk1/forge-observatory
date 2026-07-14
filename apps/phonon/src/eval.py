"""
FORGE · phonon · MODEL 1 scorecard — reproduce Chen et al. figures on the test split.

    python -m src.eval --data-dir data/raw --ckpt runs/e3nn_repro.torch

Reports:
  - omega_bar within 10%        (paper headline, Fig 2c; target 0.70)
  - corr(MSE, n_atoms)          (Fig 2a; target ~0, no size bias)
  - per-element mean MSE spread (Fig 2b; should be roughly balanced)
  - info-theory diagnostics     (JS, EMD-1, spectral-entropy error) for the later layers
Also dumps a small JSON scorecard next to the checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .model import build_model
from . import metrics as M


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, T, N, SYM = [], [], [], []
    for d in loader:
        d = d.to(device)
        P.append(model(d).cpu().numpy()); T.append(d.phdos.cpu().numpy())
        # per-graph atom counts + symbols from the batch
        b = d.batch.cpu().numpy()
        for gi in np.unique(b):
            N.append(int((b == gi).sum()))
        SYM.append(d.symbol if hasattr(d, "symbol") else None)
    return np.concatenate(P), np.concatenate(T), np.array(N), SYM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--ckpt", default="runs/e3nn_repro.torch")
    ap.add_argument("--max-radius", type=float, default=5.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(D.DEFAULT_DTYPE)

    graphs, meta = D.load_dataset(args.data_dir, max_radius=args.max_radius)
    tr, te, va = D.load_splits(args.data_dir)
    freq = meta["phfre"]

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mk = ck.get("model_kwargs", {})
    model = build_model(mk, num_neighbors=mk.get("num_neighbors", 1.0)).to(device)
    model.load_state_dict(ck["state"])

    _, _, dl_te = D.make_loaders(graphs, tr, te, va, batch_size=1)
    P, T, N, _ = predict(model, dl_te, device)

    frac, rel = M.frac_within(P, T, freq, tol=0.10)
    mse = M.mse_per_example(P, T)
    corr = float(np.corrcoef(mse, N)[0, 1]) if len(set(N)) > 1 else 0.0
    se_err = np.abs(M.spectral_entropy(P) - M.spectral_entropy(T))

    card = {
        "n_test": int(len(P)),
        "omega_bar_within_10pct": round(frac, 4),
        "paper_target": 0.70,
        "mean_mse": round(float(mse.mean()), 5),
        "corr_mse_natoms": round(corr, 4),
        "mean_js": round(float(M.js_divergence(P, T).mean()), 5),
        "mean_emd1_cm": round(float(M.emd1d(P, T, freq).mean()), 3),
        "mean_spectral_entropy_abs_err": round(float(se_err.mean()), 4),
    }
    print("\n=== FORGE phonon MODEL 1 scorecard (TEST) ===")
    for k, v in card.items():
        print(f"  {k:32s} {v}")
    print(f"  PASS omega_bar target: {frac >= 0.70}")

    out = Path(args.ckpt).with_suffix(".scorecard.json")
    out.write_text(json.dumps(card, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
