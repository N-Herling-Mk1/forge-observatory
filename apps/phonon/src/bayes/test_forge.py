"""
FORGE · phonon · sandbox vetting for the four aspects (numpy-only halves).
Runs off the saved bundle — no torch needed. The aspect-4 torch backward pass
(attribution.attribute_material) is verified separately with the model loaded.

    python -m src.bayes.test_forge            # uses runs/mk2_llla.npz
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from . import llla, forge


def run(bundle_path=None):
    root = Path(__file__).resolve().parents[2]
    bp = Path(bundle_path) if bundle_path else root / "runs" / "mk2_llla.npz"
    B = llla.load_npz(bp)
    Phi, Y = B["phi_train"], B["y_train"]
    probe = Phi[0]
    ok = True

    print(f"[forge] bundle {bp.name}  ·  phi {Phi.shape}  ·  caps {forge.capabilities(B)}")

    pv = forge.posterior_view(B)
    print(f"  1 posterior   eff_dim={pv['effective_dim']:.2f}  "
          f"determined={pv['n_determined']}/{pv['n_determined']+pv['n_prior_led']}")
    _, bw = forge.tau_curve(B, probe, [0.1, 1, 10, 100, 1000])
    mono = bool(np.all(np.diff(bw) <= 1e-9))
    print(f"    tau-knob band tightens with tau: {mono}"); ok &= mono

    sweep = forge.data_fraction_sweep(Phi, Y, probe=probe)
    grows = sweep[-1]["effective_dim"] >= sweep[0]["effective_dim"]
    print(f"  2 datasweep   eff_dim {sweep[0]['effective_dim']:.1f} -> "
          f"{sweep[-1]['effective_dim']:.1f}  (grows: {grows})"); ok &= grows

    verds = [forge.hmc_validate(Phi, Y, bin_k=k, n_samples=1000, burn=300)
             for k in (10, 25, 40)]
    good = all(v["verdict"] == "GOOD" for v in verds)
    print(f"  3 hmc         bins 10/25/40 -> "
          f"{', '.join(v['verdict'] for v in verds)}  (var_err "
          f"{np.mean([v['var_rel_err'] for v in verds]):.3f})"); ok &= good

    g = forge.dsigma_dphi(B, probe)
    finite = bool(np.all(np.isfinite(g))) and np.linalg.norm(g) > 0
    print(f"  4 attribution dsigma/dphi finite & nonzero: {finite}  "
          f"(||g||={np.linalg.norm(g):.3f})"); ok &= finite

    print(f"[forge] {'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(*(sys.argv[1:2])) else 1)
