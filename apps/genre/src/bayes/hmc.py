"""FORGE — Last-Layer HMC.

The gold-standard counterpart to llla.py. Where the Laplace head *assumes* the
last-layer posterior is Gaussian, HMC *samples* the true posterior

    p(θ | D) ∝ exp( Σ_n log softmax(W φ_n + b)[y_n]  −  (τ/2)‖θ‖² ),   θ = (W, b)

with Hamiltonian Monte Carlo, then pushes the samples through softmax to get the
exact predictive. Comparing the two tells you where Laplace is honest and where it
over/under-states uncertainty — i.e. whether the epistemic number FORGE reports is
itself trustworthy.

The backbone stays frozen: this only touches the final Linear, conditioned on the
cached penultimate features φ. The gradient is analytic, so the sampler is pure
NumPy and unit-testable without the training stack. `from_bundle` uses torch only to
read the MAP weights, and needs y_train.npy in the bundle (see backfill note below).
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _logsoftmax(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


def split_rhat(x):
    """Split-R̂ (Gelman–Rubin) on a 1-D chain: split in half, compare the halves.
    ≈1.0 ⇒ the two halves agree (converged); >1.1 ⇒ not converged."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) // 2
    if n < 2:
        return float("nan")
    a, b = x[:n], x[n:2 * n]
    means = np.array([a.mean(), b.mean()])
    W = np.array([a.var(ddof=1), b.var(ddof=1)]).mean()
    if W <= 0:
        return float("nan")
    B = n * means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def ess(x):
    """Effective sample size via the initial-positive-sequence autocorrelation sum.
    Accounts for autocorrelation: ESS ≪ N ⇒ few independent draws."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * n)
    s = 1.0
    for t in range(1, n):
        if acf[t] < 0:
            break
        s += 2 * acf[t]
    return float(n / s) if s > 0 else float(n)


# ============================================================================
class LastLayerHMC:
    """HMC over the final Linear (W ∈ R[C×d], b ∈ R[C]) on frozen features φ."""

    def __init__(self, phi, y, n_classes=None, tau=1.0):
        self.phi = np.asarray(phi, dtype=np.float64)        # [N, d]
        self.y = np.asarray(y, dtype=np.int64).ravel()      # [N]
        self.N, self.d = self.phi.shape
        self.C = int(n_classes if n_classes is not None else self.y.max() + 1)
        self.tau = float(tau)
        self.Y = np.zeros((self.N, self.C))
        self.Y[np.arange(self.N), self.y] = 1.0
        self.D = self.C * self.d + self.C                   # flat param dim

    # ---- pack / unpack θ = (W, b) <-> flat vector --------------------------
    def _unpack(self, th):
        W = th[:self.C * self.d].reshape(self.C, self.d)
        b = th[self.C * self.d:]
        return W, b

    def _pack(self, W, b):
        return np.concatenate([W.ravel(), b])

    # ---- target: log posterior + analytic gradient ------------------------
    def log_post(self, th):
        W, b = self._unpack(th)
        logits = self.phi @ W.T + b                          # [N, C]
        ll = _logsoftmax(logits, axis=1)[np.arange(self.N), self.y].sum()
        prior = -0.5 * self.tau * (th @ th)
        return ll + prior

    def grad_log_post(self, th):
        W, b = self._unpack(th)
        P = _softmax(self.phi @ W.T + b, axis=1)             # [N, C]
        G = self.Y - P                                       # [N, C]
        dW = G.T @ self.phi - self.tau * W                   # [C, d]
        db = G.sum(0) - self.tau * b                         # [C]
        return self._pack(dW, db)

    # ---- MAP (good HMC init: a few gradient-ascent steps) ------------------
    def find_map(self, iters=400, lr=None, theta0=None):
        th = np.zeros(self.D) if theta0 is None else theta0.copy()
        lr = lr if lr is not None else 1.0 / self.N
        for _ in range(iters):
            th = th + lr * self.grad_log_post(th)
        return th

    # ---- HMC ---------------------------------------------------------------
    def sample(self, n_samples=500, n_warmup=200, step_size=None, n_leapfrog=25,
               theta0=None, seed=0, thin=1, target_accept=0.8, adapt=True,
               on_progress=None):
        """Hamiltonian Monte Carlo with dual-averaging step-size adaptation
        (Hoffman & Gelman 2014). During warmup the step size is adapted toward
        target_accept, then frozen at its running average for sampling. theta0
        defaults to the MAP. Returns (samples [S, D], info)."""
        rng = np.random.default_rng(seed)
        if theta0 is None:
            theta0 = self.find_map()
        if step_size is None:
            step_size = 0.25 / np.sqrt(self.N)      # data-scale-aware start
        th = theta0.copy()
        U = lambda t: -self.log_post(t)
        gradU = lambda t: -self.grad_log_post(t)

        # dual-averaging state
        mu = np.log(10 * step_size)
        log_eps, log_eps_bar, H_bar = np.log(step_size), 0.0, 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75

        samples, n_acc, n_div = [], 0, 0
        lp_trace = []                              # log-posterior every iter (incl warmup)
        total = n_warmup + n_samples * thin
        for it in range(total):
            eps = np.exp(log_eps) if (adapt and it < n_warmup) else step_size
            p0 = rng.standard_normal(self.D)
            th_new, p = th.copy(), p0.copy()
            p -= 0.5 * eps * gradU(th_new)
            for L in range(n_leapfrog):
                th_new += eps * p
                if L != n_leapfrog - 1:
                    p -= eps * gradU(th_new)
            p -= 0.5 * eps * gradU(th_new)
            dH = (U(th) + 0.5 * p0 @ p0) - (U(th_new) + 0.5 * p @ p)
            # divergence: energy not conserved by the integrator (step too big / bad geometry)
            if (not np.isfinite(dH)) or abs(dH) > 1000.0:
                n_div += 1
            a = min(1.0, np.exp(dH)) if np.isfinite(dH) else 0.0
            if np.log(rng.random()) < dH:
                th = th_new
                if it >= n_warmup:
                    n_acc += 1
            if adapt and it < n_warmup:
                m = it + 1
                H_bar = (1 - 1.0 / (m + t0)) * H_bar + (target_accept - a) / (m + t0)
                log_eps = mu - np.sqrt(m) / gamma * H_bar
                eta = m ** (-kappa)
                log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
                step_size = float(np.exp(log_eps_bar))
            lp_trace.append(float(self.log_post(th)))
            if it >= n_warmup and (it - n_warmup) % thin == 0:
                samples.append(th.copy())
            if on_progress and (total < 100 or it % (total // 100) == 0):
                frac_acc = n_acc / max(1, it - n_warmup + 1) if it >= n_warmup else a
                on_progress(it + 1, total, frac_acc)

        samples = np.asarray(samples)
        post = np.asarray(lp_trace[n_warmup:]) if len(lp_trace) > n_warmup else np.asarray(lp_trace)
        # downsample the trace for transport (keep the warmup boundary marker)
        keep = 240
        if len(lp_trace) > keep:
            idx = np.linspace(0, len(lp_trace) - 1, keep).round().astype(int)
            trace_ds = [lp_trace[i] for i in idx]
            warm_frac = n_warmup / len(lp_trace)
        else:
            trace_ds = lp_trace
            warm_frac = n_warmup / max(1, len(lp_trace))
        info = {"accept": n_acc / max(1, n_samples), "step_size": step_size,
                "n_leapfrog": n_leapfrog, "n_samples": len(samples),
                "n_warmup": n_warmup, "tau": self.tau,
                "n_divergences": int(n_div),
                "rhat": split_rhat(post), "ess": ess(post),
                "lp_trace": trace_ds, "warmup_frac": float(warm_frac)}
        return samples, info

    # ---- predictive --------------------------------------------------------
    def predictive(self, phi_x, samples):
        """Push HMC samples through softmax for one input φ → probs [S, C]."""
        phi_x = np.asarray(phi_x, dtype=np.float64).ravel()
        out = np.empty((len(samples), self.C))
        for i, th in enumerate(samples):
            W, b = self._unpack(th)
            out[i] = _softmax(W @ phi_x + b)
        return out

    def predictive_summary(self, phi_x, samples):
        P = self.predictive(phi_x, samples)
        mean = P.mean(0)
        return {"mean": mean, "sigma": P.std(0),
                "lo": np.percentile(P, 5, 0), "hi": np.percentile(P, 95, 0),
                "samples": P}


# ----------------------------------------------------------------- bundle load
def from_bundle(bundle_dir, tau=1.0, device="cpu"):
    """Build the HMC target from a bundle. NumPy for φ/y; torch only to read the
    MAP weights (used as the HMC init). Requires y_train.npy in the bundle."""
    d = Path(bundle_dir)
    phi = np.load(d / "phi_train.npy")
    yp = d / "y_train.npy"
    if not yp.exists():
        raise FileNotFoundError(
            "y_train.npy not found in the bundle. HMC needs the training labels for "
            "the likelihood gradient. Add y_train caching to the bundle writer (see "
            "bundle.write_bundle) and/or backfill the current bundle with "
            "`python -m projects.genre.src.bayes.backfill_y_train --bundle <dir>`.")
    y = np.load(yp)

    label_map = json.loads((d / "label_map.json").read_text())
    n_classes = len(label_map)
    hmc = LastLayerHMC(phi, y, n_classes=n_classes, tau=tau)

    # use the trained MAP weights as the HMC init (faster warmup)
    import torch
    sd = torch.load(d / "weights.pt", map_location=device)
    dphi = phi.shape[1]
    W = bkey = None
    for k, vv in sd.items():
        if k.endswith(".weight") and vv.ndim == 2 and vv.shape[1] == dphi:
            W, bkey = vv, k[:-7] + ".bias"
    if W is not None:
        W = W.detach().cpu().numpy()
        b = sd[bkey].detach().cpu().numpy() if bkey in sd else np.zeros(W.shape[0])
        hmc._theta0 = hmc._pack(W, b)
    return hmc


# ------------------------------------------------------------------------ cli
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Last-layer HMC over a bundle's frozen head")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--leapfrog", type=int, default=25)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--index", type=int, default=0, help="phi_train row to summarize")
    args = ap.parse_args()

    hmc = from_bundle(args.bundle, tau=args.tau)
    label_map = json.loads((Path(args.bundle) / "label_map.json").read_text())
    inv = {v: k for k, v in label_map.items()}
    genres = [inv[i] for i in range(len(inv))]

    t0 = time.time()
    def prog(it, total, acc):
        bar = "█" * int(30 * it / total) + "·" * (30 - int(30 * it / total))
        print(f"\r  HMC [{bar}] {it}/{total}  accept={acc:.2f}", end="", flush=True)
    theta0 = getattr(hmc, "_theta0", None)
    samples, info = hmc.sample(n_samples=args.samples, n_warmup=args.warmup,
                               step_size=args.step, n_leapfrog=args.leapfrog,
                               theta0=theta0, on_progress=prog)
    print(f"\n  done in {time.time()-t0:.1f}s · accept={info['accept']:.2f} · "
          f"step={info['step_size']:.2e} · {info['n_samples']} samples")

    phi = np.load(Path(args.bundle) / "phi_train.npy")[args.index]
    summ = hmc.predictive_summary(phi, samples)
    order = np.argsort(summ["mean"])[::-1]
    print(f"\n  HMC predictive · example #{args.index} (τ={args.tau})")
    for i in order:
        print(f"    {genres[i]:10s} {100*summ['mean'][i]:5.1f}%  ± {100*summ['sigma'][i]:4.1f}")


if __name__ == "__main__":
    _main()
