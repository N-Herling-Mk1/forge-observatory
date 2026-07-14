"""Backfill y_train.npy into an existing bundle.

HMC (bayes/hmc.py) needs the training labels for the likelihood gradient, but
bundles written before the y_train change don't have them. This regenerates the
labels in the SAME order as the cached phi_train.npy and saves y_train.npy — and
it VERIFIES that re-extracted φ matches the cached φ before writing, so a config
mismatch can never silently produce misaligned labels.

Run on your machine (needs torch + the data + the training config):

    python -m projects.genre.src.bayes.backfill_y_train \
        --bundle projects/genre/models/beardown \
        --config projects/genre/configs/beardown.yaml

It mirrors train.py's data loading exactly (same split mode, seed, representation,
image settings; phi_loader is train split, batch 256, shuffle=False).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Backfill y_train.npy into a bundle")
    ap.add_argument("--bundle", required=True, help="bundle dir (has phi_train.npy, weights.pt, arch.json)")
    ap.add_argument("--config", required=True, help="the training config yaml used for this bundle")
    ap.add_argument("--data-root", default=None, help="override data_root (else from config)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-3, help="max |Δφ| allowed in the alignment check")
    args = ap.parse_args()

    # repo root on path (…/projects/genre/src/bayes/backfill_y_train.py -> repo root)
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo))

    import yaml
    import torch
    from projects.genre.src import dataio
    from projects.genre.src.models import REGISTRY
    from projects.genre.src.bundle import compute_phi

    bundle = Path(args.bundle)
    phi_cached = np.load(bundle / "phi_train.npy")
    print(f"  bundle: {bundle}  cached φ: {phi_cached.shape}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rep = cfg.get("representation", "fused")

    # mirror train.py: same loader config
    loaded = dataio.load(
        representation=rep,
        split_strategy=cfg["split"]["mode"],
        data_root=args.data_root or cfg.get("data_root", "projects/genre/data/raw"),
        seed=cfg["split"].get("seed", 0),
        standardize=True,
        drop_length=cfg.get("features", {}).get("drop_length", False),
        image_size=cfg.get("features", {}).get("image_size", 128),
        image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"),
    )
    dims = {"tab_in": len(loaded.feature_cols), "img_ch": 1}

    # rebuild the trained model and load the bundle weights
    model = REGISTRY[cfg["model"]](cfg, dims).to(args.device)
    model.load_state_dict(torch.load(bundle / "weights.pt", map_location=args.device))

    phi_loader = dataio.to_torch_loader(loaded, "train", batch=256, shuffle=False)
    phi_new, y = compute_phi(model, phi_loader, device=args.device, return_y=True)
    print(f"  re-extracted φ: {phi_new.shape}   y: {y.shape}")

    # alignment guard — never write misaligned labels
    if phi_new.shape != phi_cached.shape:
        raise SystemExit(f"shape mismatch (cached {phi_cached.shape} vs re-extracted {phi_new.shape}); "
                         "config/data differs from training — aborting.")
    dmax = float(np.abs(phi_new - phi_cached).max())
    print(f"  φ alignment check: max|Δφ| = {dmax:.2e}  (tol {args.tol})")
    if dmax > args.tol:
        raise SystemExit("φ mismatch above tolerance — the loader order differs from training. "
                         "Labels would be misaligned; aborting WITHOUT writing.")

    np.save(bundle / "y_train.npy", y.astype(np.int64))
    # quick sanity: label distribution
    vals, cnts = np.unique(y, return_counts=True)
    print(f"  ✓ wrote {bundle / 'y_train.npy'}  ({len(y)} labels, {len(vals)} classes, "
          f"counts {dict(zip(vals.tolist(), cnts.tolist()))})")
    print("  HMC is now runnable on this bundle.")


if __name__ == "__main__":
    main()
