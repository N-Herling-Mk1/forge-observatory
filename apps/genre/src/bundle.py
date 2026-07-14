"""The bundle = the save/load contract (PLAN §0). A trained model is "born
plug-in-ready" when write_bundle emits these seven files into
``projects/genre/models/<model>/``:

    weights.pt      deterministic backbone + final Linear (state_dict)
    arch.json       layer spec -> rebuild + the network-structure SVG
    scaler.json     train-fit mean/std/cols  (inference REUSES, never refits)
    label_map.json  genre -> 0..9            (inference must agree exactly)
    phi_train.npy   cached penultimate φ(x_train)   [N, d]
    ggn_eig.npz     Λ, U of the last-layer GGN input factor  (LLLA precompute)
    metrics.json    standard + reliability metrics (the metrics wall)

load_bundle is the deterministic inverse: rebuild from arch.json, load weights,
attach scaler + label_map + the φ/GGN precompute. Same function for the offline
scripts and the Flask backend.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from .models.beardown import BeardownNet


BUNDLE_FILES = ["weights.pt", "arch.json", "scaler.json", "label_map.json",
                "phi_train.npy", "y_train.npy", "ggn_eig.npz", "metrics.json"]


def _json_safe(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ------------------------------------------------------------------ φ extraction
@torch.no_grad()
def compute_phi(model: BeardownNet, loader, device="cpu", return_y=False):
    """Penultimate features over a loader (fused batches: image, tabular, y).
    With return_y, also returns the labels in the SAME order (for HMC / y_train)."""
    model.eval()
    out, ys = [], []
    for batch in loader:
        img, tab, y = batch
        phi = model.features(img.to(device), tab.to(device))
        out.append(phi.cpu().numpy())
        if return_y:
            ys.append(y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y).ravel())
    phi = np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)
    if return_y:
        y = np.concatenate(ys, axis=0).astype(np.int64) if ys else np.empty((0,), np.int64)
        return phi, y
    return phi


def ggn_eigenbasis(phi: np.ndarray):
    """Eigendecomposition of the last-layer GGN input factor  H = Σ_n φ_n φ_nᵀ.

    This is the input-side (Kronecker) factor of the last-layer Laplace precision;
    the §5 browser animation reads (Λ, U) and computes predictive variance as
    Σ_i (Uᵀφ)²_i / (λ_i + τ) in O(d). The output-side factor / full laplace-torch
    KFAC fit is finalized on Day 2; this precompute is the eigenbasis that path reuses.
    """
    H = phi.T @ phi                                   # [d, d], symmetric PSD
    lam, U = np.linalg.eigh(H)                         # ascending λ, orthonormal U
    return lam.astype(np.float64), U.astype(np.float64)


# ----------------------------------------------------------------------- writer
def write_bundle(out_dir: str, model: BeardownNet, loaded, metrics: dict,
                 phi_loader, device: str = "cpu") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / "weights.pt")
    (out / "arch.json").write_text(json.dumps(model.arch_spec(), indent=2), encoding="utf-8")
    (out / "scaler.json").write_text(json.dumps(_json_safe(loaded.scaler), indent=2), encoding="utf-8")
    (out / "label_map.json").write_text(json.dumps(loaded.label_map, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    phi, y = compute_phi(model, phi_loader, device=device, return_y=True)
    np.save(out / "phi_train.npy", phi)
    np.save(out / "y_train.npy", y)                          # labels for HMC, φ-aligned
    lam, U = ggn_eigenbasis(phi)
    np.savez(out / "ggn_eig.npz", Lambda=lam, U=U, n=phi.shape[0], d=phi.shape[1])

    # self-describing bundle: inputs.json (named input axes) + bundle.json
    # (capabilities). Torch-free; gates the FORGE attribution tab (spec §5). The
    # 'attribution' capability appears once attribution.py:precompute writes
    # attribution.npz and re-runs this — born attribution-ready, filled in later.
    from .attribution import write_self_description
    desc = write_self_description(out, loaded)

    return {"dir": str(out), "files": BUNDLE_FILES, "phi_shape": list(phi.shape),
            "capabilities": desc["capabilities"]}


# ----------------------------------------------------------------------- loader
@dataclass
class InferenceBundle:
    model: BeardownNet
    arch: dict
    scaler: dict
    label_map: dict[str, int]
    phi_train: np.ndarray
    ggn_lambda: np.ndarray
    ggn_U: np.ndarray
    metrics: dict
    dir: str

    @property
    def genres(self) -> list[str]:
        inv = {v: k for k, v in self.label_map.items()}
        return [inv[i] for i in range(len(inv))]

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor, tabular: torch.Tensor) -> np.ndarray:
        self.model.eval()
        logits = self.model(image, tabular)
        return torch.softmax(logits, dim=1).cpu().numpy()


def load_bundle(model_dir: str, device: str = "cpu") -> InferenceBundle:
    d = Path(model_dir)
    arch = json.loads((d / "arch.json").read_text(encoding="utf-8"))
    model = BeardownNet.from_spec(arch)
    model.load_state_dict(torch.load(d / "weights.pt", map_location=device))
    model.to(device).eval()

    ggn = np.load(d / "ggn_eig.npz")
    return InferenceBundle(
        model=model,
        arch=arch,
        scaler=json.loads((d / "scaler.json").read_text(encoding="utf-8")),
        label_map=json.loads((d / "label_map.json").read_text(encoding="utf-8")),
        phi_train=np.load(d / "phi_train.npy"),
        ggn_lambda=ggn["Lambda"],
        ggn_U=ggn["U"],
        metrics=json.loads((d / "metrics.json").read_text(encoding="utf-8")),
        dir=str(d),
    )
