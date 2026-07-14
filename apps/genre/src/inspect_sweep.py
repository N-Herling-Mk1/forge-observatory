"""Quick sweep.json inspector — usage:
    python projects/genre/src/inspect_sweep.py            # newest run auto-found
    python projects/genre/src/inspect_sweep.py <path/to/sweep.json>
"""
import json, sys, glob, os

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    cands = glob.glob("projects/genre/runs/beardown_rrm/*/sweep.json")
    if not cands:
        print("no sweep.json found under projects/genre/runs/beardown_rrm/"); sys.exit(1)
    path = max(cands, key=os.path.getmtime)
print(f"reading {path}\n")

d = json.load(open(path, encoding="utf-8"))
rows = sorted(d["configs"], key=lambda r: -r["RRM"])
print("  rank  cfg   RRM      A       s        U       off     macroF1  tier")
for k, r in enumerate(rows, 1):
    print(f"  {k:>4}  {r['cfg_index']+1:>3}  {r['RRM']:+.4f}  {r['A']:.4f}  "
          f"{r['s']:.4f}  {r['U']:.4f}  {r['off_diag']:.4f}  {r['macro_f1']:.4f}  "
          f"{r.get('pareto_tier','?')}")
sel = d["selection"]
print(f"\n  s_max={sel['s_max']:.5f}  U_max={sel['U_max']:.5f}  "
      f"winner=cfg{sel['winner_cfg_index']+1}  winner_RRM={sel['winner_rrm']:+.4f}")

# manual RRM recompute for the top 3, to verify internal consistency
import math
smax, umax = sel["s_max"], sel["U_max"]
print("\n  manual RRM check (top 3 by recorded RRM):")
for r in rows[:3]:
    pa = 1 - r["A"]; ps = r["s"]/smax if smax else 0; pu = r["U"]/umax if umax else 0
    rrm_manual = 1 - math.sqrt(pa*pa + ps*ps + pu*pu)
    flag = "" if abs(rrm_manual - r["RRM"]) < 1e-6 else "  <-- MISMATCH"
    print(f"    cfg{r['cfg_index']+1:>2}: recorded={r['RRM']:+.4f}  recomputed={rrm_manual:+.4f}{flag}")

# does any config dominate the winner on all three penalty axes?
w = next(r for r in d["configs"] if r["cfg_index"] == sel["winner_cfg_index"])
doms = [r for r in d["configs"]
        if r["cfg_index"] != w["cfg_index"]
        and r["A"] >= w["A"] and r["s"] <= w["s"] and r["U"] <= w["U"]
        and (r["A"] > w["A"] or r["s"] < w["s"] or r["U"] < w["U"])]
print(f"\n  configs that DOMINATE the winner (A≥, s≤, U≤, strict in one): "
      f"{[r['cfg_index']+1 for r in doms] or 'none (winner is non-dominated — OK)'}")
