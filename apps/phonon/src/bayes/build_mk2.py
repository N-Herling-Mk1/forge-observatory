"""
FORGE · phonon · mk2 — build the LLLA bundle on the frozen mk1 backbone.

Extracts phi (pooled penultimate scalar features) for every training material via a
forward hook on the last gated block, fits the Bayesian last-layer head, and saves the
bundle next to the mk1 checkpoint.

    python -m src.bayes.build_mk2 --ckpt runs/e3nn_repro.torch --tau 1.0 --out runs/mk2_llla.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch_geometric as tg
from tqdm import tqdm

from .. import data as D
from ..model import build_model
from . import llla


def _root():
    return Path(__file__).resolve().parents[2]            # projects/phonon


def extract_phi(model, graphs, device="cpu"):
    """phi = mean over atoms of the penultimate gated block's scalar (l=0) features."""
    irr = model.layers[-2].irreps_out
    n_scal = sum(mul for mul, ir in irr if ir.l == 0)     # leading scalar block
    cap = {}
    h = model.layers[-2].register_forward_hook(lambda m, i, o: cap.__setitem__("f", o.detach()))
    Phi = []
    model.eval()
    with torch.no_grad():
        for g in tqdm(graphs, desc="extract phi", ncols=80):
            batch = next(iter(tg.loader.DataLoader([g], batch_size=1))).to(device)
            _ = model(batch)
            per_atom = cap["f"][:, :n_scal]               # (atoms, n_scal) scalar part
            Phi.append(per_atom.mean(0).cpu().numpy())     # pool over atoms
    h.remove()
    return np.asarray(Phi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/e3nn_repro.torch")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out", default="runs/mk2_llla.npz")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = _root()
    torch.set_default_dtype(D.DEFAULT_DTYPE)
    graphs, meta = D.load_dataset(root / args.data_dir, max_radius=5.0, limit=args.limit)
    tr, te, va = D.load_splits(root / args.data_dir)
    if args.limit:
        keep = set(range(args.limit)); tr = np.array([i for i in tr if i in keep])

    ck = torch.load(root / args.ckpt, map_location="cpu", weights_only=False)
    mk = ck.get("model_kwargs", {})
    model = build_model(mk, num_neighbors=mk.get("num_neighbors", 1.0))
    model.load_state_dict(ck["state"])

    Phi = extract_phi(model, [graphs[i] for i in tr])
    Y = np.stack([graphs[i].phdos.numpy().ravel() for i in tr])     # (Ntr, 51)
    print(f"  phi {Phi.shape}  targets {Y.shape}  tau={args.tau}")

    bundle = llla.fit_bundle(Phi, Y, tau=args.tau)
    bundle["ckpt"] = Path(args.ckpt).name
    bundle["phi_train"] = Phi                       # for data-sweep / HMC / attribution
    bundle["y_train"] = Y
    bundle["capabilities"] = np.array(["posterior", "datasweep", "hmc", "attribution"])
    out = root / args.out
    llla.save_npz(out, bundle)
    lam = llla.eigenspectrum(bundle)
    print(f"  saved {out}")
    print(f"  GGN eigenspectrum: max={lam[0]:.2f}  median={np.median(lam):.3f}  "
          f"min={lam[-1]:.4f}  (small eigenvalues = prior-led directions)")


if __name__ == "__main__":
    main()
