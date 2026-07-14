"""Reliability metric family for the mk2 (beardown_rrm) sweep — pure NumPy, torch-free.

Everything the sweep needs to turn a cohort of per-config measurements into a ranking
lives here, so the math is unit-testable without the training stack (the sandbox has
no torch). The selector is RRM; the rest (RAM, Pareto, Pearson, dCor) is recorded.

The metric (FORGE-native instantiation of the INFO 510 RRM, Fig B1), det head fixed:

    v   = [ 1 - A,  s / s_max,  U / U_max ]          # model penalty vector
    RRM = 1 - ||v||                                  # reward form;  range [1-√3, 1]

  A     = CV-mean held-out accuracy           (train.py already computes this)
  s     = std of the K fold accuracies        (fold-to-fold instability)
  U     = mean LLLA predictive epistemic σ    (the Bayesian-layer term)
  s_max = max s   over the cohort             (worst instability)
  U_max = max U   over the cohort             (worst epistemic spread)

s_max / U_max are COHORT quantities — RRM is undefined for a lone model. That is the
structural reason mk2 is a sweep, not a single config.

U is computed at a shared prior-strength α (τ = α·λ_max per config) so the penalty
reflects φ-geometry, not each config's flattering τ-fit. `optimal_tau_mackay` provides
the recorded per-config empirical-Bayes τ (the B-column), never the selector.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- confusion / off-diag
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Integer confusion matrix C[i,j] = #(true=i, pred=j)."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(C, (yt, yp), 1)
    return C


def off_diagonal_rate(y_true, y_pred, n_classes: int) -> dict:
    """Off-diagonal confusion mass as a RATE (not raw counts) so it is comparable
    across datasets/folds of different size — forward-compat with mk3's larger val.

    Returns the rate, the raw off/total counts, and the single most-confused pair
    (the symmetric C[i,j]+C[j,i] argmax) — a cheap diagnostic column, not part of RRM.
    """
    C = confusion_matrix(y_true, y_pred, n_classes)
    total = int(C.sum())
    diag = int(np.trace(C))
    off = total - diag
    # most-confused unordered pair
    S = C + C.T
    np.fill_diagonal(S, 0)
    flat = int(np.argmax(S))
    i, j = divmod(flat, n_classes)
    pair = (min(i, j), max(i, j))
    return {
        "rate": float(off / total) if total else 0.0,
        "off": off, "total": total,
        "top_pair": [int(pair[0]), int(pair[1])],
        "top_pair_count": int(S[i, j]),
    }


def pool_off_diagonal(fold_results: list[dict]) -> dict:
    """Pool per-fold off_diagonal_rate dicts into one CV-level rate (Σoff / Σtotal)."""
    off = sum(r["off"] for r in fold_results)
    total = sum(r["total"] for r in fold_results)
    # pooled top pair: sum the per-fold top-pair counts by pair key
    from collections import Counter
    ctr: Counter = Counter()
    for r in fold_results:
        ctr[tuple(r["top_pair"])] += r["top_pair_count"]
    top = max(ctr.items(), key=lambda kv: kv[1]) if ctr else ((0, 0), 0)
    return {"rate": float(off / total) if total else 0.0, "off": off, "total": total,
            "top_pair": [int(top[0][0]), int(top[0][1])], "top_pair_count": int(top[1])}


# ------------------------------------------------------------------------ the metric
def rrm(A: float, s: float, U: float, s_max: float, U_max: float) -> float:
    """RRM = 1 - √((1-A)² + (s/s_max)² + (U/U_max)²).  A degenerate cohort axis
    (s_max or U_max == 0 → every model identical on it) contributes 0 penalty."""
    pa = 1.0 - float(A)
    ps = (float(s) / s_max) if s_max > 0 else 0.0
    pu = (float(U) / U_max) if U_max > 0 else 0.0
    return float(1.0 - np.sqrt(pa * pa + ps * ps + pu * pu))


def rrm_cohort(A, s, U) -> dict:
    """Vectorized RRM over the whole cohort. Returns per-config RRM + the s_max/U_max
    used (so the sweep.json records exactly what normalized the penalties)."""
    A = np.asarray(A, float); s = np.asarray(s, float); U = np.asarray(U, float)
    s_max = float(s.max()) if s.size else 0.0
    U_max = float(U.max()) if U.size else 0.0
    out = np.array([rrm(a, si, ui, s_max, u_max := U_max) for a, si, ui in zip(A, s, U)])
    return {"rrm": out, "s_max": s_max, "U_max": U_max}


# ------------------------------------------------------------ recorded global rankings
def _rank_desc(x) -> np.ndarray:
    """Competition rank, best (largest) = 1. Ties share the lower rank."""
    x = np.asarray(x, float)
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties so tie-handling is symmetric
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def rank_aggregation(A, RRM, macro, micro) -> dict:
    """RAM(i) = rank_A + rank_R + rank_M + rank_U  (each best=1; lower RAM = better)."""
    rA, rR = _rank_desc(A), _rank_desc(RRM)
    rM, rU = _rank_desc(macro), _rank_desc(micro)
    ram = rA + rR + rM + rU
    return {"ram": ram, "rank_A": rA, "rank_R": rR, "rank_M": rM, "rank_U": rU}


def pareto_front(A, R) -> np.ndarray:
    """PFR tier per config on (accuracy A, reliability R). i dominates j iff
    A_i≥A_j and R_i≥R_j with at least one strict. Tier 1 = non-dominated; peel & repeat."""
    A = np.asarray(A, float); R = np.asarray(R, float)
    n = len(A)
    tier = np.zeros(n, dtype=int)
    remaining = set(range(n))
    t = 1
    while remaining:
        idx = list(remaining)
        front = []
        for i in idx:
            dominated = any(
                (A[j] >= A[i] and R[j] >= R[i] and (A[j] > A[i] or R[j] > R[i]))
                for j in idx if j != i)
            if not dominated:
                front.append(i)
        for i in front:
            tier[i] = t
            remaining.discard(i)
        t += 1
        if not front:                       # safety: numerical pathology
            for i in idx:
                tier[i] = t
            break
    return tier


# ----------------------------------------------------------------- correlation block
def pearson_matrix(columns: dict) -> dict:
    """Pearson r matrix over the named cohort columns (the Fig-7 object). Constant
    columns yield NaN r (reported honestly, not silently zeroed)."""
    labels = list(columns.keys())
    M = np.vstack([np.asarray(columns[k], float) for k in labels])
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(M)
    R = np.atleast_2d(R)
    return {"labels": labels, "matrix": R}


def distance_correlation(x, y) -> float:
    """Distance correlation ∈ [0,1]; 0 iff x⊥y (catches nonlinear coupling Pearson
    misses). Used only on the (A,RRM) and (off_diag,RRM) pairs to harden the
    orthogonality claim. Same tool as the HELIX/ABCD decorrelated-axis selection."""
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    n = x.shape[0]
    if n < 2:
        return 0.0
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    denom = np.sqrt(dvarx2 * dvary2)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))


# ---------------------------------------------------------- recorded per-config τ (B)
def optimal_tau_mackay(Lambda, w_energy: float, n_classes: int,
                       iters: int = 100, tau0: float = 1.0) -> float:
    """MacKay empirical-Bayes prior precision τ for the last layer (the RECORDED
    B-column; never the selector). Treats the C output heads as sharing the input
    Hessian H=ΦᵀΦ (eigvals Λ). Fixed point:

        γ = C · Σ_i λ_i/(λ_i+τ)      (effective # well-determined directions)
        τ = γ / Σ_c‖w_c‖²            (Σ_c‖w_c‖² = ‖W_map‖_F², U orthonormal)

    Approximate (ignores the output-factor coupling) but standard and τ>0, cheap."""
    lam = np.asarray(Lambda, float).ravel()
    lam = np.clip(lam, 0.0, None)
    w_energy = max(float(w_energy), 1e-12)
    tau = float(tau0)
    for _ in range(iters):
        gamma = n_classes * float((lam / (lam + tau)).sum())
        tau_new = gamma / w_energy
        if not np.isfinite(tau_new) or tau_new <= 0:
            break
        if abs(tau_new - tau) < 1e-9 * max(tau, 1.0):
            tau = tau_new
            break
        tau = tau_new
    return float(max(tau, 1e-8))


# ------------------------------------------------------- weighted RRM (mk2.5)
def rrm_weighted(A, s, U, s_max, U_max, w) -> float:
    """Simplex-weighted RRM:  RRM_w = 1 - sqrt(w0(1-A)^2 + w1(s/s_max)^2 + w2(U/U_max)^2),
    w in Delta^2 (nonneg, sums to 1). The centroid w=(1/3,1/3,1/3) yields the SAME RANKING
    as the locked equal-weight rrm() — the 1/sqrt(3) scaling is monotone — so the centroid
    winner == the locked-metric winner (the absolute scalar shifts, the order does not)."""
    pa = 1.0 - float(A)
    ps = (float(s) / s_max) if s_max > 0 else 0.0
    pu = (float(U) / U_max) if U_max > 0 else 0.0
    w = np.asarray(w, float)
    return float(1.0 - np.sqrt(w[0] * pa * pa + w[1] * ps * ps + w[2] * pu * pu))


def penalty_components(A, s, U) -> np.ndarray:
    """The three normalized penalty axes v=[1-A, s/s_max, U/U_max] over the cohort -> [n,3]."""
    A = np.asarray(A, float); s = np.asarray(s, float); U = np.asarray(U, float)
    s_max = float(s.max()) if s.size else 1.0
    U_max = float(U.max()) if U.size else 1.0
    v0 = 1.0 - A
    v1 = s / s_max if s_max > 0 else np.zeros_like(s)
    v2 = U / U_max if U_max > 0 else np.zeros_like(U)
    return np.column_stack([v0, v1, v2])


def rrm_weighted_cohort(A, s, U, w) -> dict:
    """Vectorized RRM_w over the cohort at one weighting w. Returns per-config RRM_w and
    the winner (argmax)."""
    V = penalty_components(A, s, U)
    pen = (np.asarray(w, float) * V ** 2).sum(axis=1)        # sum_k w_k v_k^2
    rrm_w = 1.0 - np.sqrt(np.clip(pen, 0, None))
    return {"rrm": rrm_w, "winner": int(rrm_w.argmax())}


def variance_weights(A, s, U, eps: float = 1e-12) -> np.ndarray:
    """Principled default w: weight each penalty axis by how much it VARIES across the
    cohort (variance of v_i), normalized to the simplex. A near-constant axis (e.g. a flat
    U) gets ~0 weight — it cannot discriminate, so it should not vote. Computed from CV
    columns ONLY (no test set): data-derived, not result-derived."""
    V = penalty_components(A, s, U)
    var = V.var(axis=0)
    return var / (var.sum() + eps)


def dirichlet_winner_sweep(A, s, U, n_samples: int = 20000,
                           alpha=(1.0, 1.0, 1.0), seed: int = 0) -> dict:
    """Sample w ~ Dirichlet(alpha) over the simplex; for each w find the RRM_w winner.
    Returns per-config win-FRACTION (robustness across ALL weightings) + the modal winner.
    A config that wins across most of the simplex is the ROBUST choice — not a w-cherry-pick.
    This integrates over w instead of fixing it, answering the 'arbitrary / game-able' worry
    directly: report how often each config wins, not which w you happened to choose."""
    V = penalty_components(A, s, U)
    Vsq = V ** 2                                             # [n,3]
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha, size=n_samples)                # [m,3]
    pen = W @ Vsq.T                                          # [m,n]  weighted penalty
    winners = pen.argmin(axis=1)                            # argmin pen == argmax RRM_w
    counts = np.bincount(winners, minlength=len(np.asarray(A, float)))
    frac = counts / n_samples
    return {"win_fraction": frac, "modal_winner": int(frac.argmax()),
            "counts": counts.astype(int).tolist(), "n_samples": int(n_samples)}


# --------------------------------------------------------------------------- selftest
def _selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok' if cond else 'XX'}] {name}")

    # off-diagonal: a clean diagonal has rate 0; a fully-wrong matrix has rate 1
    yt = np.array([0, 1, 2, 0, 1, 2])
    chk("off_diag perfect = 0", off_diagonal_rate(yt, yt, 3)["rate"] == 0.0)
    yp = np.array([1, 2, 0, 1, 2, 0])
    od = off_diagonal_rate(yt, yp, 3)
    chk("off_diag all-wrong = 1", abs(od["rate"] - 1.0) < 1e-12)

    # top confused pair: build a matrix that smears 0<->2
    yt2 = np.array([0, 0, 2, 2, 1, 1, 1])
    yp2 = np.array([2, 2, 0, 0, 1, 1, 1])
    od2 = off_diagonal_rate(yt2, yp2, 3)
    chk("top_pair = (0,2)", od2["top_pair"] == [0, 2])

    # rrm: A=1, s=0, U=0 -> perfect 1
    chk("rrm best = 1", abs(rrm(1.0, 0.0, 0.0, 0.1, 0.1) - 1.0) < 1e-12)
    # rrm: A=0, s=s_max, U=U_max -> 1-sqrt(3)
    chk("rrm worst = 1-√3", abs(rrm(0.0, 0.1, 0.1, 0.1, 0.1) - (1 - np.sqrt(3))) < 1e-9)
    # degenerate cohort axis contributes 0
    chk("rrm s_max=0 -> no s penalty", abs(rrm(1.0, 0.0, 0.0, 0.0, 0.0) - 1.0) < 1e-12)

    # cohort rrm matches scalar
    A = [0.70, 0.74, 0.69]; s = [0.02, 0.05, 0.01]; U = [0.30, 0.20, 0.40]
    coh = rrm_cohort(A, s, U)
    man = rrm(0.74, 0.05, 0.20, max(s), max(U))
    chk("cohort rrm[1] == scalar", abs(coh["rrm"][1] - man) < 1e-12)

    # pareto: a strictly-best point is tier 1; a strictly-dominated one is >1
    Av = [0.9, 0.8, 0.5]; Rv = [0.9, 0.4, 0.3]
    tier = pareto_front(Av, Rv)
    chk("pareto best = tier 1", tier[0] == 1)
    chk("pareto dominated > 1", tier[2] > 1)

    # rank aggregation: the all-best config has the lowest RAM
    A2 = [0.9, 0.7, 0.5]; R2 = [0.9, 0.6, 0.4]; M2 = [0.9, 0.6, 0.4]; U2 = [0.9, 0.6, 0.4]
    ram = rank_aggregation(A2, R2, M2, U2)["ram"]
    chk("RAM argmin = best config", int(np.argmin(ram)) == 0)

    # dCor: identical signals -> 1; a nonlinear dep (y=x²) is detected (> Pearson|r|)
    x = np.linspace(-1, 1, 21)
    chk("dCor(x,x) = 1", abs(distance_correlation(x, x) - 1.0) < 1e-9)
    yq = x ** 2
    pear = abs(np.corrcoef(x, yq)[0, 1])
    dc = distance_correlation(x, yq)
    chk("dCor catches y=x² (dCor > |pearson|)", dc > pear + 0.2)

    # pearson matrix: diagonal is 1
    cols = {"A": A, "s": s, "U": U}
    pm = pearson_matrix(cols)
    chk("pearson diag = 1", np.allclose(np.diag(pm["matrix"]), 1.0))

    # mackay tau: converges, positive, and rises when weights shrink (stronger prior)
    lam = np.array([10.0, 5.0, 1.0, 0.1, 0.01])
    t_big = optimal_tau_mackay(lam, w_energy=100.0, n_classes=10)
    t_small = optimal_tau_mackay(lam, w_energy=1.0, n_classes=10)
    chk("mackay tau > 0", t_big > 0 and t_small > 0)
    chk("mackay tau rises as ‖W‖ shrinks", t_small > t_big)

    # mk2.5 weighted RRM: centroid ranking == locked equal-weight ranking
    A2 = [0.72, 0.77, 0.70, 0.75]; s2 = [0.007, 0.004, 0.030, 0.020]; U2 = [0.026, 0.027, 0.029, 0.034]
    locked = rrm_cohort(A2, s2, U2)["rrm"]
    cen = rrm_weighted_cohort(A2, s2, U2, [1/3, 1/3, 1/3])["rrm"]
    chk("centroid winner == locked winner", int(np.argmax(cen)) == int(np.argmax(locked)))
    chk("centroid order == locked order", list(np.argsort(-cen)) == list(np.argsort(-locked)))

    # variance weights: sum to 1; the FLATTEST axis gets the SMALLEST weight
    # build a cohort where U is near-constant and s varies a lot
    Af = [0.70, 0.75, 0.72, 0.77]; sf = [0.005, 0.045, 0.020, 0.035]; Uf = [0.026, 0.0261, 0.0259, 0.0262]
    vw = variance_weights(Af, sf, Uf)
    chk("variance weights sum to 1", abs(vw.sum() - 1.0) < 1e-9)
    chk("flat U axis gets smallest weight", vw[2] < vw[0] and vw[2] < vw[1])

    # dirichlet sweep: fractions sum to 1, modal winner is a valid index
    dw = dirichlet_winner_sweep(A2, s2, U2, n_samples=5000, seed=1)
    chk("dirichlet win_fraction sums to 1", abs(dw["win_fraction"].sum() - 1.0) < 1e-9)
    chk("dirichlet modal winner valid", 0 <= dw["modal_winner"] < len(A2))

    print("ALL PASS" if ok else "FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
