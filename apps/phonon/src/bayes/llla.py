"""
FORGE · phonon · mk2 — Last-Layer Laplace (LLLA) core.

Pure numpy. Operates on phi = the frozen mk1 backbone's pooled penultimate features
(32-d) and the 51-bin DOS targets. Fits a linear read-out head with a Gaussian last-
layer Laplace posterior, and returns a per-bin credible band.

Predictive (Bayesian linear regression, per output bin k):
    mean_k(phi*) = w_k . phi*
    var_k(phi*)  = sigma2_k + phi*^T Lambda_k^{-1} phi*,   Lambda_k = G/sigma2_k + tau I
with G = Phi^T Phi. We store the eigendecomposition G = U diag(lam) U^T once, so
    phi*^T Lambda_k^{-1} phi* = sum_i c_i^2 / (lam_i/sigma2_k + tau),   c = U^T phi*
This makes tau a live knob (FORGE's prior-precision slider) with no refit, and the
eigenspectrum {lam_i} is exactly the FORGE posterior view.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def fit_bundle(Phi, Y, tau=1.0):
    """Phi: (N,d) penultimate features. Y: (N,K) DOS targets. Returns a numpy bundle."""
    Phi = np.asarray(Phi, float); Y = np.asarray(Y, float)
    mu = Phi.mean(0); sd = Phi.std(0) + EPS
    Z = (Phi - mu) / sd                                   # standardize for conditioning
    N, d = Z.shape
    G = Z.T @ Z                                           # (d,d) un-noised GGN
    # ridge MAP head (prior precision tau on standardized weights)
    A0 = G + tau * np.eye(d)
    W = np.linalg.solve(A0, Z.T @ Y)                      # (d,K)
    resid = Y - Z @ W
    sigma2 = np.clip((resid ** 2).mean(0), 1e-6, None)    # per-bin noise (K,)
    lam, U = np.linalg.eigh(G)                            # G = U diag(lam) U^T
    return {"W": W, "U": U, "lam": lam, "sigma2": sigma2,
            "mu": mu, "sd": sd, "tau": float(tau), "d": int(d), "K": int(Y.shape[1]),
            "n_train": int(N)}


def predict_one(bundle, phi, tau=None):
    """phi: (d,) penultimate vector -> (mean (K,), band (K,)) with the LLLA credible std."""
    b = bundle
    tau = b["tau"] if tau is None else float(tau)
    z = (np.asarray(phi, float) - b["mu"]) / b["sd"]
    mean = z @ b["W"]                                     # (K,)
    c = b["U"].T @ z                                      # project onto eigenbasis
    # input-variance factor per bin: sum_i c_i^2 / (lam_i/sigma2_k + tau)
    denom = b["lam"][:, None] / b["sigma2"][None, :] + tau     # (d,K)
    v = ((c[:, None] ** 2) / denom).sum(0)               # (K,)
    band = np.sqrt(b["sigma2"] + v)                       # per-bin std (K,)
    return mean, band


def eigenspectrum(bundle):
    """The GGN eigenvalues (descending) — FORGE's posterior view: which directions are
    data-determined (large lam) vs prior-led (small lam, near tau)."""
    return np.sort(bundle["lam"])[::-1]


def save_npz(path, bundle):
    np.savez(path, **{k: np.asarray(v) for k, v in bundle.items()})


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    b = {k: z[k] for k in z.files}
    for s in ("tau", "d", "K", "n_train"):
        if s in b:
            b[s] = float(b[s]) if s == "tau" else int(b[s])
    return b
