"""FORGE — Last-Layer Laplace (LLLA) on the frozen φ.

The deterministic backbone stays frozen; we put a Gaussian posterior over ONLY the
final Linear (W ∈ R[C×d], b ∈ R[C]) and integrate it for predictive uncertainty.
This is the closed-form, real-time-knob path. HMC (gold-standard) lands beside it.

What the bundle already cached (see bundle.ggn_eigenbasis):
    ggn_eig.npz : Λ, U  —  eigendecomposition of the INPUT-side Kronecker factor
                           H = Φᵀ Φ  (Φ = phi_train, the penultimate features)
    phi_train.npy        —  Φ, used here to build the OUTPUT-side factor
    weights.pt           —  W_map, b  (the MAP last layer)

The math
--------
Last-layer GGN Laplace, KFAC-factored:  precision ≈ A ⊗ H  + prior τI.
  • INPUT factor (cached):  v(φ;τ) = φᵀ(H+τI)⁻¹φ = Σᵢ (uᵢᵀφ)² / (λᵢ+τ)   — O(d)
        epistemic scale of THIS input — large where φ(x) is unlike training feats.
        τ = prior precision: ↑τ → ↓v → tighter posterior (prior dominates).
  • OUTPUT factor:  A = (1/N) Σₙ (diag(pₙ) − pₙpₙᵀ),  pₙ = softmax(W_map φₙ + b)
        the average softmax Hessian — the class-confusability structure.
        Σ_A = (A + εI)⁻¹  (ε small jitter for invertibility).
  • Predictive logit law:  f ~ N(μ, Σ_f),   μ = W_map φ + b,   Σ_f = v(φ;τ)·Σ_A.
  • Softmax integration (no closed form): MC over the Gaussian (primary), plus the
        probit/MacKay closed form for the mean.

Everything below the loader is pure NumPy and torch-free, so the predictive math is
unit-testable without the training stack. `from_bundle` uses torch only to read the
last-layer weights.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


# --------------------------------------------------------------------- helpers
def _softmax(z, axis=-1, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
class LastLayerLaplace:
    """Closed-form last-layer Laplace posterior + predictive integration.

    Construct from the cached arrays directly (testable), or via `from_bundle`.
    All predictive methods take a penultimate feature vector φ (shape [d]) — get
    φ for a song from model.features(...), or use rows of phi_train as examples.
    """

    def __init__(self, Lambda, U, W_map, b, phi_train=None,
                 probs_train=None, jitter=1e-4):
        self.Lambda = np.asarray(Lambda, dtype=np.float64).ravel()      # [d]
        self.U = np.asarray(U, dtype=np.float64)                        # [d, d]
        self.W = np.asarray(W_map, dtype=np.float64)                    # [C, d]
        self.b = np.asarray(b, dtype=np.float64).ravel()               # [C]
        self.C, self.d = self.W.shape
        self.jitter = float(jitter)

        # output factor A = <diag(p) - p pᵀ> over train (avg softmax Hessian)
        if probs_train is None:
            if phi_train is None:
                raise ValueError("need phi_train or probs_train to build the output factor A")
            logits = phi_train @ self.W.T + self.b                      # [N, C]
            probs_train = _softmax(logits, axis=1)
        P = np.asarray(probs_train, dtype=np.float64)                  # [N, C]
        # (1/N) Σ diag(p) - p pᵀ
        A = np.diag(P.mean(0)) - (P.T @ P) / P.shape[0]
        self.A = 0.5 * (A + A.T)                                        # symmetrize
        self.n_train = P.shape[0]
        self.phi_train = np.asarray(phi_train, dtype=np.float64) if phi_train is not None else None

    # ---- factors -----------------------------------------------------------
    def input_variance(self, phi, tau):
        """v(φ;τ) = Σᵢ (uᵢᵀφ)² / (λᵢ+τ).  φ: [d] or [m,d] → scalar or [m]."""
        phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))         # [m, d]
        proj = phi @ self.U                                            # [m, d] in eigenbasis
        v = (proj ** 2 / (self.Lambda + tau)).sum(axis=1)              # [m]
        return v if v.shape[0] > 1 else float(v[0])

    def output_cov(self):
        """Σ_A = (A + εI)⁻¹  — the [C,C] class-covariance shape (τ-independent)."""
        return np.linalg.inv(self.A + self.jitter * np.eye(self.C))

    def logit_moments(self, phi, tau, temperature=1.0, sigma_scale=1.0):
        """Predictive logit Gaussian: mean μ [C], covariance Σ_f [C,C]."""
        phi = np.asarray(phi, dtype=np.float64).ravel()
        mu = (self.W @ phi + self.b) / temperature
        v = float(self.input_variance(phi, tau))
        Sigma = (sigma_scale ** 2) * v * self.output_cov() / (temperature ** 2)
        return mu, 0.5 * (Sigma + Sigma.T)

    # ---- predictive --------------------------------------------------------
    def predict_posterior(self, phi, tau, *, method="mc", n_samples=2000,
                          temperature=1.0, sigma_scale=1.0, seed=0):
        """Per-class predictive probability mean + epistemic σ at prior precision τ.

        method='mc'     : sample logits ~ N(μ,Σ_f), softmax, empirical mean/std
                          (honest predictive integration; also returns 5/95 band).
        method='probit' : MacKay probit closed form for the mean; σ via the
                          softmax delta method. Deterministic, fast.
        Returns a dict (NumPy arrays).
        """
        mu, Sigma = self.logit_moments(phi, tau, temperature, sigma_scale)
        diag = np.clip(np.diag(Sigma), 0, None)

        if method == "probit":
            kappa = 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * diag)
            mean_p = _softmax(mu * kappa)
            # delta method: σ_p ≈ sqrt(diag(J Σ Jᵀ)), J = softmax Jacobian at μ
            p = _softmax(mu)
            J = np.diag(p) - np.outer(p, p)
            cov_p = J @ Sigma @ J.T
            sigma_p = np.sqrt(np.clip(np.diag(cov_p), 0, None))
            lo = np.clip(mean_p - sigma_p, 0, 1)
            hi = np.clip(mean_p + sigma_p, 0, 1)
        else:
            rng = np.random.default_rng(seed)
            L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(self.C))
            z = rng.standard_normal((n_samples, self.C))
            F = mu[None, :] + z @ L.T                                  # [S, C]
            Pp = _softmax(F, axis=1)                                   # [S, C]
            mean_p = Pp.mean(0)
            sigma_p = Pp.std(0)
            lo = np.percentile(Pp, 5, axis=0)
            hi = np.percentile(Pp, 95, axis=0)

        return {"mean": mean_p, "sigma": sigma_p, "lo": lo, "hi": hi,
                "logit_mean": mu, "logit_var": diag,
                "input_variance": float(self.input_variance(phi, tau)),
                "tau": float(tau), "method": method}

    # ---- epistemic / aleatoric decomposition (the hypothesis answer) --------
    def predictive_decomposition(self, phi, tau, *, n_samples=2000,
                                 temperature=1.0, sigma_scale=1.0, seed=0):
        """Split predictive uncertainty into aleatoric (irreducible data noise) and
        epistemic (model ignorance, shrinks with data/capacity), in nats.

            total     H[ E_θ p(y|x,θ) ]   entropy of the predictive mean
            aleatoric E_θ[ H p(y|x,θ) ]   expected entropy over the weight posterior
            epistemic total − aleatoric   = mutual information (BALD); ≥ 0

        High epistemic ⇒ "more data / capacity would help". High aleatoric ⇒ "capped".
        """
        mu, Sigma = self.logit_moments(phi, tau, temperature, sigma_scale)
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(self.C))
        F = mu[None, :] + rng.standard_normal((n_samples, self.C)) @ L.T
        P = _softmax(F, axis=1)                               # [S, C]
        def ent(p):
            p = np.clip(p, 1e-12, 1.0)
            return -(p * np.log(p)).sum(axis=-1)
        mean_p = P.mean(0)
        total = float(ent(mean_p[None, :])[0])
        aleatoric = float(ent(P).mean())
        epistemic = max(0.0, total - aleatoric)
        return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic,
                "epistemic_frac": float(epistemic / total) if total > 1e-9 else 0.0,
                "mean": mean_p.tolist()}

    # ---- data-fraction sweep (would more data tighten the posterior?) -------
    def datasweep(self, phi, tau, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), seed=0):
        """Rebuild H = ΦᵀΦ from random subsets of the cached φ (NO retraining — the
        MAP weights stay fixed) and recompute the posterior spread at each fraction.
        A curve still dropping at 100% ⇒ more data would still tighten the posterior;
        a flat tail ⇒ the posterior is data-saturated. Torch-free."""
        if self.phi_train is None:
            raise ValueError("need phi_train for the data-fraction sweep")
        Phi = self.phi_train
        N, d = Phi.shape
        perm = np.random.default_rng(seed).permutation(N)
        phi = np.asarray(phi, dtype=np.float64).ravel()
        out = []
        for f in fractions:
            nf = max(2, int(round(f * N)))
            idx = perm[:nf]
            Hf = Phi[idx].T @ Phi[idx]
            w, U = np.linalg.eigh(Hf)
            w = np.clip(w, 0, None)
            proj = U.T @ phi
            out.append({
                "frac": float(f), "n": int(nf),
                "trace_cov": float((1.0 / (w + tau)).sum()),     # tr((H+τI)⁻¹)
                "starved": int((w < tau).sum()),
                "v_input": float((proj ** 2 / (w + tau)).sum()),  # this example's epistemic input var
            })
        return out

    # ---- payload for the FORGE frontend (real-time τ knob, no new math) -----
    def datasweep_payload(self, phi, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), seed=0):
        """Per data-fraction: the rebuilt-H eigenvalues λ_f and this example's
        projections c_f=(U_fᵀφ)², so the browser recomputes trace/starved/v_input
        live for ANY τ (no server round-trip on the τ knob). Torch-free."""
        if self.phi_train is None:
            raise ValueError("need phi_train for the data-fraction sweep")
        Phi = self.phi_train
        N = Phi.shape[0]
        perm = np.random.default_rng(seed).permutation(N)
        phi = np.asarray(phi, dtype=np.float64).ravel()
        out = []
        for f in fractions:
            nf = max(2, int(round(f * N)))
            idx = perm[:nf]
            w, U = np.linalg.eigh(Phi[idx].T @ Phi[idx])
            w = np.clip(w, 0, None)
            proj = U.T @ phi
            out.append({"frac": float(f), "n": int(nf),
                        "lam": w.tolist(), "c": (proj ** 2).tolist()})
        return out

    def posterior_payload(self, phi, genres, tau=None):
        """Everything the browser needs to recompute the posterior live on a τ-drag,
        in O(d): the per-input eigen-projections cᵢ=(uᵢᵀφ)², the eigenvalues λᵢ,
        the logit mean μ, and the output covariance Σ_A. Then for ANY τ:
            v(τ)=Σ cᵢ/(λᵢ+τ);  Σ_f=v·Σ_A;  sample/probit → bands.
        Also returns a server-computed posterior at a default τ for first paint."""
        phi = np.asarray(phi, dtype=np.float64).ravel()
        proj = phi @ self.U
        c = proj ** 2
        lam = self.Lambda
        # sensible default τ ~ a small fraction of the spectrum's top eigenvalue
        if tau is None:
            tau = max(1e-3, 1e-2 * float(lam.max()))
        post = self.predict_posterior(phi, tau, method="mc")
        return {
            "genres": list(genres),
            "tau": float(tau),
            "tau_range": [float(max(1e-4, 1e-4 * lam.max())), float(lam.max())],
            "eig": {"c": c.tolist(), "lam": lam.tolist()},      # for browser v(τ)
            "logit_mean": (self.W @ phi + self.b).tolist(),
            "output_cov": self.output_cov().tolist(),           # Σ_A [C,C]
            "posterior": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in post.items()},
        }


# ----------------------------------------------------------------- bundle load
def from_bundle(bundle_dir, device="cpu", last_linear_key=None) -> LastLayerLaplace:
    """Assemble the LLLA from a saved bundle. NumPy for Λ/U/φ; torch only to read
    the final Linear's weight+bias out of weights.pt."""
    d = Path(bundle_dir)
    eig = np.load(d / "ggn_eig.npz")
    Lambda, U = eig["Lambda"], eig["U"]
    phi_train = np.load(d / "phi_train.npy")

    try:
        import torch
    except ImportError:
        hz = d / "head.npz"          # torch-free fallback: exported final Linear
        if hz.exists():
            h = np.load(hz)
            return LastLayerLaplace(Lambda, U, h["W"], h["b"], phi_train=phi_train)
        raise ImportError("torch unavailable and no head.npz in bundle "
                          f"{d.name} - export one or install torch")
    sd = torch.load(d / "weights.pt", map_location=device)
    # find the final Linear: weight [C, d] with d == phi dim, prefer an explicit key
    dphi = phi_train.shape[1]
    W = bkey = None
    if last_linear_key and f"{last_linear_key}.weight" in sd:
        W = sd[f"{last_linear_key}.weight"]; bkey = f"{last_linear_key}.bias"
    else:
        for k, vv in sd.items():
            if k.endswith(".weight") and vv.ndim == 2 and vv.shape[1] == dphi:
                W, bkey = vv, k[:-7] + ".bias"            # last match wins (head is last)
    if W is None:
        raise ValueError(f"could not find a final Linear [C,{dphi}] in weights.pt")
    W = W.detach().cpu().numpy()
    b = sd[bkey].detach().cpu().numpy() if bkey in sd else np.zeros(W.shape[0])
    return LastLayerLaplace(Lambda, U, W, b, phi_train=phi_train)


# ------------------------------------------------------------------------ cli
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="LLLA posterior over a bundle's frozen head")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--index", type=int, default=0, help="which phi_train row to score")
    args = ap.parse_args()

    lap = from_bundle(args.bundle)
    label_map = json.loads((Path(args.bundle) / "label_map.json").read_text())
    inv = {v: k for k, v in label_map.items()}
    genres = [inv[i] for i in range(len(inv))]

    phi = np.load(Path(args.bundle) / "phi_train.npy")[args.index]
    tau = args.tau if args.tau is not None else max(1e-3, 1e-2 * float(lap.Lambda.max()))
    post = lap.predict_posterior(phi, tau, method="mc")
    print(f"LLLA · bundle={args.bundle} · τ={tau:.4g} · input_var={post['input_variance']:.4g}")
    order = np.argsort(post["mean"])[::-1]
    for i in order:
        print(f"  {genres[i]:10s}  {100*post['mean'][i]:5.1f}%  ± {100*post['sigma'][i]:4.1f}")


if __name__ == "__main__":
    _main()
