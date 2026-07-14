"""
FORGE · phonon · the four observatory aspects, computed on the mk1 backbone's
last-layer Laplace bundle. Pure numpy — these run server-side with a live tau-knob,
no torch in the path. Aspect 4's input half (dphi/dx) needs one torch backward pass
through the frozen backbone and lives in attribution_torch().

  1. posterior_view / tau_curve   — eigenspectrum, prior-led vs data-determined, tau slider
  2. data_fraction_sweep          — would more data tighten it
  3. hmc_validate                 — sample the head posterior, cross-check the analytic Laplace
  4. dsigma_dphi (+ attribution_torch) — push epistemic sigma back to named inputs
"""
from __future__ import annotations
import numpy as np
from . import llla

EPS = 1e-12


# ── aspect 1 ────────────────────────────────────────────────────────────────
def posterior_view(bundle):
    """Eigenspectrum + per-direction classification at the bundle's tau."""
    lam = np.asarray(bundle["lam"], float)
    tau = float(bundle["tau"])
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    # a direction is 'data-determined' when its data eigenvalue dominates the prior
    determined = lam > tau
    return {
        "lam": lam,
        "tau": tau,
        "n_determined": int(determined.sum()),
        "n_prior_led": int((~determined).sum()),
        "frac_determined": float(determined.mean()),
        "effective_dim": float((lam / (lam + tau)).sum()),   # Σ λ/(λ+τ)
    }


def tau_curve(bundle, phi, taus):
    """Mean credible-band width vs prior precision tau (the FORGE tau-knob)."""
    out = []
    for t in taus:
        _, band = llla.predict_one(bundle, phi, tau=float(t))
        out.append(float(band.mean()))
    return np.asarray(taus, float), np.asarray(out)


# ── aspect 2 ────────────────────────────────────────────────────────────────
def data_fraction_sweep(Phi, Y, fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
                        tau=1.0, probe=None, seed=0):
    """Refit the LLLA head on growing data fractions; track effective dimension and
    (optionally) the mean band on a held probe — the 'would more data help' curve."""
    Phi = np.asarray(Phi, float); Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    N = len(Phi)
    perm = rng.permutation(N)
    rows = []
    for f in fractions:
        k = max(8, int(round(f * N)))
        idx = perm[:k]
        b = llla.fit_bundle(Phi[idx], Y[idx], tau=tau)
        pv = posterior_view(b)
        row = {"frac": float(f), "n": int(k),
               "effective_dim": pv["effective_dim"],
               "frac_determined": pv["frac_determined"]}
        if probe is not None:
            _, band = llla.predict_one(b, probe, tau=tau)
            row["mean_band"] = float(band.mean())
        rows.append(row)
    return rows


# ── aspect 3 ────────────────────────────────────────────────────────────────
def hmc_validate(Phi, Y, tau=1.0, bin_k=0, n_samples=1500, burn=500,
                 step=0.3, leap=12, seed=0):
    """Preconditioned HMC over one output bin's last-layer weights under the Gaussian
    model, cross-checked against the analytic Laplace posterior N(m, A⁻¹),
    A = ZᵀZ/σ² + τI. The posterior is stiff (wide eigenvalue spread), so we use mass
    matrix M = A — standard for a Gaussian target. Agreement of the sample mean/cov
    with the analytic posterior is the validation (mirrors the ATLAS HMC check)."""
    Phi = np.asarray(Phi, float); Y = np.asarray(Y, float)[:, bin_k]
    mu = Phi.mean(0); sd = Phi.std(0) + EPS
    Z = (Phi - mu) / sd
    d = Z.shape[1]
    w_map = np.linalg.solve(Z.T @ Z + tau*np.eye(d), Z.T @ Y)
    s2 = float(((Y - Z @ w_map) ** 2).mean()) + 1e-6
    A = Z.T @ Z / s2 + tau * np.eye(d)               # posterior precision
    Ainv = np.linalg.inv(A)
    m = Ainv @ (Z.T @ Y) / s2                         # posterior mean
    L = np.linalg.cholesky(A)                         # A = L Lᵀ  (for p ~ N(0,A))

    def grad_U(w):                                    # ∇ negative-log-posterior
        return A @ (w - m)
    rng = np.random.default_rng(seed)
    w = m.copy(); samples = []; accepts = 0
    for it in range(n_samples + burn):
        p = L @ rng.standard_normal(d)               # momentum ~ N(0, A=M)
        w0, p0 = w.copy(), p.copy()
        p = p - 0.5 * step * grad_U(w)
        for i in range(leap):
            w = w + step * (Ainv @ p)                # mass-matrix position update
            if i != leap - 1:
                p = p - step * grad_U(w)
        p = p - 0.5 * step * grad_U(w)
        Hn = 0.5 * (w - m) @ A @ (w - m) + 0.5 * p @ (Ainv @ p)
        H0 = 0.5 * (w0 - m) @ A @ (w0 - m) + 0.5 * p0 @ (Ainv @ p0)
        if np.log(rng.random()) < (H0 - Hn):
            accepts += 1
        else:
            w = w0
        if it >= burn:
            samples.append(w.copy())
    S = np.asarray(samples)
    m_hmc, C_hmc = S.mean(0), np.cov(S.T)
    scale = np.sqrt(np.trace(Ainv))                  # posterior spread, floors the denom
    mean_err = float(np.linalg.norm(m_hmc - m) / max(np.linalg.norm(m), scale, EPS))
    var_err = float(np.linalg.norm(np.diag(C_hmc) - np.diag(Ainv)) /
                    (np.linalg.norm(np.diag(Ainv)) + EPS))
    return {"bin": int(bin_k), "n_eff": len(S), "accept": accepts / (n_samples + burn),
            "mean_rel_err": mean_err, "var_rel_err": var_err,
            "verdict": "GOOD" if (mean_err < 0.1 and var_err < 0.2) else "CHECK"}


# ── aspect 4 (phi-space half — closed form, numpy) ──────────────────────────
def dsigma_dphi(bundle, phi, tau=None, bin_k=None):
    """Closed-form gradient of the predictive band w.r.t. phi.
    band_k = sqrt(σ²_k + v),  v(z) = Σ_i c_i²/(λ_i/σ²_k + τ),  c = Uᵀz,  z=(phi-μ)/sd.
    Returns dσ/dφ in input-phi coordinates (chain through the standardizer)."""
    b = bundle
    tau = b["tau"] if tau is None else float(tau)
    z = (np.asarray(phi, float) - b["mu"]) / b["sd"]
    U, lam, s2 = b["U"], b["lam"], b["sigma2"]
    c = U.T @ z
    K = len(s2)
    bins = range(K) if bin_k is None else [bin_k]
    g = np.zeros_like(z)
    for k in bins:
        denom = lam / s2[k] + tau
        v = float((c**2 / denom).sum())
        band = np.sqrt(s2[k] + v)
        dv_dz = U @ (2 * c / denom)                  # ∂v/∂z
        g += dv_dz / (2 * band)                       # ∂band/∂z
    g = g / (len(list(bins)))
    return g / b["sd"]                                # back to phi coordinates


def capabilities(bundle):
    caps = bundle.get("capabilities")
    if caps is None:
        return ["posterior"]
    return [str(c) for c in np.asarray(caps).ravel()]
