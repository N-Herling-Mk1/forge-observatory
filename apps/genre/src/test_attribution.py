#!/usr/bin/env python
"""Torch-free tests for the input-feature attribution spine (src/attribution.py).

Proves the parts the GPU precompute relies on, WITHOUT torch:
  1. dv/dφ closed form == finite-difference gradient of v(φ;τ)         (exact math)
  2. v_input == llla.LastLayerLaplace.input_variance                   (consistency)
  3. τ-live cache identity: dv_dx_from_M(M) == Jᵀ·dv/dφ  ∀τ            (the live-knob algebra)
  4. aggregate_per_genre means are correct
  5. inputs.json / capabilities / bundle.json gate correctly           (spec §5)
  6. InputAttribution.from_cache + payload ranking round-trips

    python projects/genre/src/test_attribution.py     # run from repo root

The torch precompute (dφ/dx VJP) runs on the GPU box; test 3 verifies the algebra
it plugs into using a SYNTHETIC Jacobian, so the only unverified-here piece is the
backward pass itself (standard autograd).
"""
from __future__ import annotations
import os, sys, json, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import attribution as ATTR          # noqa: E402
from projects.genre.src.bayes.llla import LastLayerLaplace  # noqa: E402

C_, A_, R_ = "\033[36m", "\033[33m", "\033[0m"
_n = 0
def ok(m): 
    global _n; _n += 1; print(f"  {C_}[ok {_n:02d}]{R_} {m}")


def _orthonormal(d, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q


def test_dv_dphi_vs_finite_diff():
    rng = np.random.default_rng(1)
    d, tau = 24, 0.137
    U = _orthonormal(d, 2)
    lam = np.sort(rng.uniform(0.01, 5.0, d))
    phi = rng.standard_normal(d)
    ana = ATTR.dv_dphi(phi, lam, U, tau)
    eps = 1e-6
    num = np.empty(d)
    for k in range(d):
        e = np.zeros(d); e[k] = eps
        num[k] = (ATTR.v_input(phi + e, lam, U, tau) - ATTR.v_input(phi - e, lam, U, tau)) / (2 * eps)
    err = np.abs(ana - num).max()
    assert err < 1e-5, f"dv/dφ closed form != finite diff (max err {err:.2e})"
    # and dv/dφ == 2(H+τI)⁻¹φ
    H = U @ np.diag(lam) @ U.T
    closed2 = 2.0 * np.linalg.solve(H + tau * np.eye(d), phi)
    assert np.allclose(ana, closed2, atol=1e-8), "dv/dφ != 2(H+τI)⁻¹φ"
    ok(f"dv/dφ exact: max|analytic−finitediff|={err:.1e}, matches 2(H+τI)⁻¹φ")


def test_v_input_matches_llla():
    rng = np.random.default_rng(3)
    d, C, N = 16, 5, 200
    U = _orthonormal(d, 4); lam = np.sort(rng.uniform(0.05, 3, d))
    W = rng.standard_normal((C, d)); b = rng.standard_normal(C)
    phi_tr = rng.standard_normal((N, d))
    lap = LastLayerLaplace(lam, U, W, b, phi_train=phi_tr)
    phi = rng.standard_normal(d)
    for tau in (1e-2, 0.5, 3.0):
        a = float(lap.input_variance(phi, tau)); c = ATTR.v_input(phi, lam, U, tau)
        assert abs(a - c) < 1e-9, f"v mismatch @τ={tau}: {a} vs {c}"
    ok("v_input == llla.input_variance across τ (the two paths agree by construction)")


def test_tau_live_identity():
    """The cached-M live-τ recompute must equal the direct VJP at every τ."""
    rng = np.random.default_rng(5)
    d, K = 20, 57
    U = _orthonormal(d, 6); lam = np.sort(rng.uniform(0.02, 4, d))
    phi = rng.standard_normal(d)
    J = rng.standard_normal((d, K))                # SYNTHETIC backbone Jacobian ∂φ/∂x
    M = ATTR.project_M(phi, J, U)                  # τ-independent cache piece [d,K]
    for tau in (1e-3, 0.05, 0.5, 2.0, 10.0):
        direct = ATTR.dv_dx_direct(J, ATTR.dv_dphi(phi, lam, U, tau))   # Jᵀ·dv/dφ
        live = ATTR.dv_dx_from_M(M, lam, tau)                            # 2Σ M/(λ+τ)
        err = np.abs(direct - live).max()
        assert err < 1e-9, f"τ-live identity broke @τ={tau} (err {err:.2e})"
    ok("dv_dx_from_M(M,τ) == Jᵀ·dv/dφ(τ)  ∀τ  → live-τ knob is exact, no Jacobian recompute")


def test_aggregate_per_genre():
    rng = np.random.default_rng(7)
    C, K = 4, 6
    y = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    dv = rng.standard_normal((y.size, K))
    signed, absm, cnt = ATTR.aggregate_per_genre(dv, y, C)
    assert np.allclose(signed[1], dv[y == 1].mean(0)), "signed mean wrong"
    assert np.allclose(absm[3], np.abs(dv[y == 3]).mean(0)), "abs mean wrong"
    assert list(cnt) == [2, 3, 1, 2], f"counts wrong: {cnt}"
    ok(f"aggregate_per_genre: per-genre signed/abs means + counts correct ({list(cnt)})")


def test_self_description_gating():
    rng = np.random.default_rng(9)
    class Stub:                       # minimal Loaded surface for the manifest
        representation = "fused3"
        scaler = {"cols": [f"feat{i}" for i in range(57)]}
        feature_cols = scaler["cols"]
        image_size = 128
        label_map = {g: i for i, g in enumerate("a b c d e f g h i j".split())}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        np.save(out / "phi_train.npy", rng.standard_normal((50, 64)).astype(np.float32))
        np.save(out / "y_train.npy", rng.integers(0, 10, 50))
        (out / "label_map.json").write_text(json.dumps(Stub.label_map))
        desc = ATTR.write_self_description(out, Stub)
        man = json.loads((out / "inputs.json").read_text())
        assert man["fused"] is True and man["representation"] == "fused3"
        assert man["tabular"] == Stub.scaler["cols"] and "image" in man
        assert desc["phi_dim"] == 64 and desc["n_classes"] == 10
        # before attribution.npz: hmc yes (y_train present), attribution NO
        assert "hmc" in desc["capabilities"] and "attribution" not in desc["capabilities"]
        # after attribution.npz lands: capability flips on
        np.savez(out / "attribution.npz", dvdx_abs=np.zeros((10, 57)))
        caps2 = ATTR.bundle_capabilities(out)
        assert "attribution" in caps2, "attribution capability didn't gate on after npz"
    ok("inputs.json names 57 axes · capabilities gate hmc/attribution on file presence")


def test_payload_roundtrip():
    rng = np.random.default_rng(11)
    C, K, d = 10, 57, 32
    names = [f"mfcc{i}_var" for i in range(K)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        absm = np.abs(rng.standard_normal((C, K))); signed = rng.standard_normal((C, K))
        M = rng.standard_normal((C, d, K)); lam = np.sort(rng.uniform(0.1, 3, d))
        np.savez(out / "attribution.npz", dvdx_abs=absm, dvdx_signed=signed,
                 counts=np.full(C, 25), lam=lam, tau_default=np.array(0.05), per_genre_M=M)
        (out / "inputs.json").write_text(json.dumps({"tabular": names, "fused": True,
                                                     "representation": "fused3"}))
        (out / "label_map.json").write_text(json.dumps(
            {g: i for i, g in enumerate("blues classical country disco hiphop jazz "
                                        "metal pop reggae rock".split())}))
        ia = ATTR.InputAttribution.from_cache(out)
        assert ia.tau_live, "per_genre_M present → should be tau_live"
        pay = ia.payload(genre="disco", top=5)
        bars = pay["per_genre"][0]["bars"]
        assert pay["per_genre"][0]["genre"] == "disco" and len(bars) == 5
        # bars sorted by |dv/dx| desc, and names resolve
        vals = [bbar["abs"] for bbar in bars]
        assert vals == sorted(vals, reverse=True), "bars not ranked by |dv/dx|"
        assert all(bbar["feature"] in names for bbar in bars)
        # live-τ changes the numbers (different τ → different bars)
        p0 = ia.payload(genre="disco", tau=0.01, top=5)
        p1 = ia.payload(genre="disco", tau=5.0, top=5)
        assert p0["tau"] == 0.01 and p1["tau"] == 5.0
        assert p0["per_genre"][0]["bars"][0]["abs"] != p1["per_genre"][0]["bars"][0]["abs"]
    ok("InputAttribution.from_cache → ranked named payload · live-τ recompute moves the bars")


def main():
    print(f"\n{'='*70}\n  ATTRIBUTION SPINE — torch-free tests\n{'='*70}")
    t0 = time.time()
    test_dv_dphi_vs_finite_diff()
    test_v_input_matches_llla()
    test_tau_live_identity()
    test_aggregate_per_genre()
    test_self_description_gating()
    test_payload_roundtrip()
    print(f"\n  {A_}{_n}/{_n} passed in {time.time()-t0:.2f}s{R_}\n")


if __name__ == "__main__":
    main()
