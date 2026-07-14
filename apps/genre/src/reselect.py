"""mk2.5 — Dirichlet re-selection over an EXISTING sweep (no retraining).

    python projects/genre/src/reselect.py --config projects/genre/configs/beardown_rrm.yaml

Re-ranks the 20 configs already in sweep.json under different simplex weightings
w in Delta^2, where  RRM_w = 1 - sqrt(w0(1-A)^2 + w1(s/s_max)^2 + w2(U/U_max)^2).

  - CENTROID  w=(1/3,1/3,1/3)  == the locked equal-weight RRM (same ranking) -> mk2 winner.
  - VARIANCE  w_i ∝ Var_cohort(v_i)  -> a near-constant axis (flat U) gets ~0 weight; the
    data-derived default (CV columns only, NO test set).
  - DIRICHLET robustness: w ~ Dir(alpha) over the whole simplex; report each config's
    win-FRACTION. Integrating over w (instead of fixing it) answers the "weighting is
    arbitrary / game-able" objection: report how often a config wins, not which w you chose.

Honest reading: a ~50/50 split between two configs means the choice is weighting-dependent
and the centroid is arbitrary — NOT that one config dominates. State it that way.

mk2.5 is a SELECTION LENS over sweep.json. It produces no model and no bundle, touches no
weights, and changes no capability — so it cannot affect FORGE wrapping or the input-feature
attribution channel. It only justifies WHICH existing bundle is the deployed mk2.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import rrm as R                       # noqa: E402

C, A_, Rz = "\033[36m", "\033[33m", "\033[0m"


def _find_sweep(run_name):
    cands = glob.glob(f"projects/genre/runs/{run_name}/*/sweep.json")
    if not cands:
        sys.exit(f"no sweep.json under projects/genre/runs/{run_name}/")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", default=None, help="sweep.json (default: newest for this run)")
    ap.add_argument("--dirichlet-n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run_name = cfg.get("run_name", "beardown_rrm")
    sweep_path = args.sweep or _find_sweep(run_name)
    sweep = json.load(open(sweep_path, encoding="utf-8"))
    rows = sorted(sweep["configs"], key=lambda r: r["cfg_index"])

    idx = np.array([r["cfg_index"] for r in rows])
    A = np.array([r["A"] for r in rows]); s = np.array([r["s"] for r in rows])
    U = np.array([r["U"] for r in rows]); off = np.array([r["off_diag"] for r in rows])
    name = lambda i: f"cfg{idx[i]+1}"

    print(f"{C}🐻 mk2.5 re-selection · {sweep_path}{Rz}")
    print(f"{C}   {len(rows)} configs · RRM_w = 1 - sqrt(w0(1-A)^2 + w1(s/smax)^2 + w2(U/Umax)^2){Rz}\n")

    cen = R.rrm_weighted_cohort(A, s, U, [1/3, 1/3, 1/3])
    print(f"{A_}CENTROID{Rz}  w=(.333,.333,.333)   winner = {A_}{name(cen['winner'])}{Rz}  "
          f"(== locked equal-weight RRM; the recorded mk2 winner)")

    vw = R.variance_weights(A, s, U)
    var = R.rrm_weighted_cohort(A, s, U, vw)
    print(f"{A_}VARIANCE{Rz}  w=({vw[0]:.3f},{vw[1]:.3f},{vw[2]:.3f})   winner = {A_}{name(var['winner'])}{Rz}")
    print(f"          axis variances -> 1-A:{vw[0]:.3f}  s/smax:{vw[1]:.3f}  U/Umax:{vw[2]:.3f}  "
          f"(flat axis gets ~0 weight)")

    dw = R.dirichlet_winner_sweep(A, s, U, n_samples=args.dirichlet_n, alpha=(1, 1, 1), seed=args.seed)
    order = np.argsort(-dw["win_fraction"])
    print(f"\n{A_}DIRICHLET robustness{Rz}  ({args.dirichlet_n} uniform-simplex draws):")
    for i in order:
        if dw["win_fraction"][i] > 0.0005:
            print(f"   {name(i):>6}  wins {100*dw['win_fraction'][i]:5.1f}%   "
                  f"A={A[i]:.3f} s={s[i]:.4f} U={U[i]:.4f} off={off[i]:.4f}")
    print(f"   modal winner = {A_}{name(dw['modal_winner'])}{Rz}")

    acc = R.rrm_weighted_cohort(A, s, U, [1, 0, 0])
    print(f"\n   reference corners:  pure-A -> {name(acc['winner'])}   "
          f"pure-s -> {name(R.rrm_weighted_cohort(A,s,U,[0,1,0])['winner'])}   "
          f"pure-U -> {name(R.rrm_weighted_cohort(A,s,U,[0,0,1])['winner'])}")

    # honest verdict line
    top2 = order[:2]
    f0, f1 = dw["win_fraction"][top2[0]], dw["win_fraction"][top2[1]]
    if f1 > 0.20:
        verdict = (f"two robust contenders {name(top2[0])} ({100*f0:.0f}%) and "
                   f"{name(top2[1])} ({100*f1:.0f}%); choice is weighting-dependent. "
                   f"Principled (variance) tiebreak -> {name(var['winner'])}.")
    else:
        verdict = f"{name(top2[0])} wins {100*f0:.0f}% of weightings — a robust choice."
    print(f"\n{C}   VERDICT: {verdict}{Rz}")

    out = {
        "sweep_path": sweep_path, "run_name": run_name,
        "centroid": {"w": [1/3, 1/3, 1/3], "winner_cfg": int(idx[cen["winner"]] + 1)},
        "variance": {"w": vw.tolist(), "winner_cfg": int(idx[var["winner"]] + 1)},
        "dirichlet": {"n_samples": dw["n_samples"], "alpha": [1, 1, 1],
                      "win_fraction": {int(idx[i] + 1): float(dw["win_fraction"][i]) for i in order},
                      "modal_winner_cfg": int(idx[dw["modal_winner"]] + 1)},
        "corners": {"accuracy": int(idx[acc["winner"]] + 1)},
        "verdict": verdict,
    }
    out_path = os.path.join(os.path.dirname(sweep_path), "reselect.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{C}   reselect.json -> {out_path}{Rz}")


if __name__ == "__main__":
    main()
