"""
FORGE · phonon · metrics.

MODEL 1 reproduction metrics (the ones Chen et al. report):
  - omega_bar           : mean phonon frequency  omega_bar = sum(f*g)/sum(g)
  - frac_within         : fraction of test materials within `tol` relative error on omega_bar
                          (paper headline: 70% within 10%)
  - mse_per_example     : per-material MSE (for the no-MSE-vs-atom-count figure 2a)

Forward-looking (MODEL 3 info-theory layer — defined now, wired later):
  - spectral_entropy    : Shannon entropy of the normalized DOS (a distribution)
  - js_divergence       : Jensen-Shannon between predicted and true normalized DOS
  - emd1d               : 1-D Earth Mover / Wasserstein-1 along the frequency axis
These treat the DOS as the probability distribution it is (the MSE the paper uses
ignores that the bins live on a metric frequency axis).
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def omega_bar(dos, freq):
    """Mean phonon frequency per row. dos: (N,51), freq: (51,)."""
    dos = np.asarray(dos, float); freq = np.asarray(freq, float)
    w = dos.sum(axis=1)
    return (dos * freq[None, :]).sum(axis=1) / np.clip(w, EPS, None)


def frac_within(pred, true, freq, tol=0.10):
    """Fraction of rows whose omega_bar relative error <= tol. Paper target: 0.70 @ tol=0.10."""
    op, ot = omega_bar(pred, freq), omega_bar(true, freq)
    rel = np.abs(op - ot) / np.clip(np.abs(ot), EPS, None)
    return float((rel <= tol).mean()), rel


def mse_per_example(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return ((pred - true) ** 2).mean(axis=1)


# ---------------------------------------------------------- info-theory (later)
def _as_dist(x):
    x = np.clip(np.asarray(x, float), 0, None)
    return x / np.clip(x.sum(axis=-1, keepdims=True), EPS, None)


def spectral_entropy(dos):
    """Shannon entropy (nats) of the normalized DOS. Sharp spectrum -> low; broad -> high."""
    p = _as_dist(dos)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=-1)


def js_divergence(pred, true):
    """Jensen-Shannon divergence (nats) between predicted and true DOS, per row."""
    p, q = _as_dist(pred), _as_dist(true)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.clip(a, EPS, None)) - np.log(np.clip(b, EPS, None)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def emd1d(pred, true, freq=None):
    """1-D Wasserstein-1 (Earth Mover) along the frequency axis, per row.
    Respects that a peak shifted by one bin is a small error (MSE cannot see this)."""
    p, q = _as_dist(pred), _as_dist(true)
    dx = 1.0 if freq is None else float(np.mean(np.diff(np.asarray(freq, float))))
    return np.abs(np.cumsum(p - q, axis=-1)).sum(axis=-1) * dx
