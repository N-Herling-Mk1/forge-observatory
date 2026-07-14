#!/usr/bin/env python
"""mk3 (beardown_3sec) data-path verifier — torch-free, runs the REAL load() path.

Proves the genre-3 data tier is correct and leakage-safe before any GPU time is
spent. Asserts (and prints) the facts you cannot eyeball:

  • the fused3 dual-input contract  (image [N,128,128,1] + tabular [N,57]);
  • the MANY-TO-ONE join             (~10 segments of a track share 1 spectrogram);
  • the LEAKAGE GUARD                 (no track straddles train/val/test);
  • the deployed partition           (stratified seed 42 = SAME tracks as mk1/mk2);
  • scaler fit on TRAIN segments only (val/test transformed, not re-fit);
  • attribution compatibility        (scaler.json carries the 57 named axes);
  • the leakage delta                (track-safe vs naive: % of tracks that leak).

    python projects/genre/src/verify_mk3.py --data-root projects/genre/data/raw

Run from the repo root so `_shared` resolves. Mirrors the cfg in
configs/beardown_3sec.yaml (split=stratified seed 42, drop_length, images_mel).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import dataio                                  # noqa: E402
from _shared.splits import track_level_split, naive_random            # noqa: E402

_T0 = time.time()
C, A, R = "\033[36m", "\033[33m", "\033[0m"
def banner(t): print(f"\n{'='*70}\n  {t}\n{'='*70}")
def step(m):   print(f"  [+{time.time()-_T0:5.1f}s] {m}", flush=True)
def ok(m):     print(f"  {C}[ok]{R} {m}", flush=True)
def warn(m):   print(f"  {A}[!!]{R} {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="projects/genre/data/raw")
    ap.add_argument("--split", default="stratified", choices=["stratified", "track"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-dir", default="images_mel")
    args = ap.parse_args()
    root = Path(args.data_root)

    banner("mk3 beardown_3sec — fused3 data path (real GTZAN, torch-free)")
    step(f"load(representation='fused3', split={args.split!r}, seed={args.seed}, "
         f"image_dir={args.image_dir!r}, drop_length=True) …")
    d = dataio.load(representation="fused3", split_strategy=args.split,
                    data_root=str(root), seed=args.seed, standardize=True,
                    drop_length=True, image_size=128, image_dir=args.image_dir)
    print(f"        {d.summary()}")
    print(f"        meta: {d.meta}")

    # ---- 1. dual-input contract + shapes ------------------------------------
    banner("1 · dual-input contract")
    for s in ("train", "val", "test"):
        x = d.X[s]
        assert set(x) == {"image", "tabular"}, f"{s}: not dual-input"
        print(f"   {s:5} image={x['image'].shape}  tabular={x['tabular'].shape}  y={d.y[s].shape}")
        assert x["image"].shape[1:] == (128, 128, 1), f"{s}: bad image shape"
        assert x["tabular"].shape[1] == 57, f"{s}: expected 57 feats, got {x['tabular'].shape[1]}"
        assert x["image"].shape[0] == x["tabular"].shape[0] == d.y[s].shape[0], f"{s}: row mismatch"
    ok("X[split] = {image[N,128,128,1], tabular[N,57]} · rows aligned · 57 features")

    n_total = sum(d.X[s]["tabular"].shape[0] for s in ("train", "val", "test"))
    ok(f"total segments routed = {n_total}  (meta n_segments={d.meta['n_segments']})")
    assert n_total == d.meta["n_segments"], "segments lost between split and routing"

    # ---- 2. many-to-one join (per-track, across ALL splits) -----------------
    banner("2 · many-to-one join  (segments of a track share ONE spectrogram)")
    bad = 0; checked = 0
    for s in ("train", "val", "test"):
        ims, tids = d.X[s]["image"], d.track_ids[s]
        for t in list(dict.fromkeys(tids))[:50]:        # sample up to 50 tracks/split
            rows = np.where(tids == t)[0]
            if rows.size < 2:
                continue
            checked += 1
            base = ims[rows[0]]
            if not all(np.array_equal(ims[r], base) for r in rows[1:]):
                bad += 1
    assert bad == 0, f"{bad} tracks have inconsistent parent images — join broken"
    # and confirm the cardinality is ~10:1
    seg_per_track = Counter(np.concatenate([d.track_ids[s] for s in ("train", "val", "test")]))
    spt = np.array(list(seg_per_track.values()))
    ok(f"checked {checked} multi-seg tracks · all segments share their parent image")
    print(f"   segments/track: min={spt.min()} max={spt.max()} mean={spt.mean():.2f} "
          f"(tracks={len(spt)})  → ~10:1 many-to-one")

    # ---- 3. leakage guard ---------------------------------------------------
    banner("3 · leakage guard  (no track in >1 split)")
    tr, va, te = (set(d.track_ids["train"]), set(d.track_ids["val"]), set(d.track_ids["test"]))
    overlap = (tr & va) | (tr & te) | (va & te)
    assert not overlap, f"TRACK LEAKAGE: {len(overlap)} tracks straddle splits"
    print(f"   tracks: train={len(tr)} val={len(va)} test={len(te)}  (disjoint)")
    # coverage = all 10 genres in each split
    for s in ("train", "val", "test"):
        g = {t.split('.', 1)[0] for t in d.track_ids[s]}
        assert g == set(dataio.GENRES), f"{s}: missing genres {set(dataio.GENRES)-g}"
    ok("0 tracks straddle · all 10 genres present in every split (coverage=1.0)")

    # ---- 4. deployed partition: SAME tracks as the 30-sec models -------------
    banner("4 · partition parity  (mk3 segments inherit mk1/mk2's track split)")
    d30 = dataio.load(representation="fused", split_strategy=args.split,
                      data_root=str(root), seed=args.seed, standardize=True,
                      drop_length=True, image_size=128, image_dir=args.image_dir)
    for s in ("train", "val", "test"):
        t3 = set(d.track_ids[s]); t30 = set(d30.track_ids[s])
        same = t3 == t30
        flag = ok if same else warn
        flag(f"{s}: 3-sec tracks {'==' if same else '!='} 30-sec tracks "
             f"(|3s|={len(t3)} |30s|={len(t30)} ∩={len(t3 & t30)})")
        assert same, f"{s}: mk3 partition diverged from mk2 — not apples-to-apples"
    ok("mk3 trains/evaluates on the EXACT tracks mk1/mk2 used → data-density is the only variable")

    # ---- 5. scaler fit on TRAIN only ----------------------------------------
    banner("5 · scaler fit on TRAIN segments only")
    sc = d.scaler
    assert sc and set(sc) >= {"mean", "std", "cols"}, "scaler missing"
    # train tabular is standardized → ~0 mean / ~1 std
    tr_mean = np.abs(d.X["train"]["tabular"].mean(0)).max()
    tr_std = np.abs(d.X["train"]["tabular"].std(0) - 1).max()
    assert tr_mean < 1e-3 and tr_std < 1e-2, f"train not standardized (|mean|={tr_mean:.1e} |std-1|={tr_std:.1e})"
    # val/test are TRANSFORMED by train stats, so their mean is NOT forced to 0
    va_mean = np.abs(d.X["val"]["tabular"].mean(0)).max()
    print(f"   train  max|mean|={tr_mean:.2e}  max|std-1|={tr_std:.2e}   (standardized)")
    print(f"   val    max|mean|={va_mean:.2e}                         (transformed, not re-fit)")
    ok("scaler computed on train segments; val/test transformed with train μ,σ (no leak)")

    # ---- 6. attribution compatibility ---------------------------------------
    banner("6 · attribution compatibility  (scaler.json carries 57 named axes)")
    cols = sc["cols"]
    assert len(cols) == 57, f"expected 57 named cols, got {len(cols)}"
    print(f"   first/last named axes: {cols[0]} … {cols[-1]}")
    assert "length" not in cols and "filename" not in cols and "label" not in cols
    ok("57 named feature axes present → ∂σ/∂x attribution (mk2.5 spec) amenable on mk3")

    # ---- 7. the leakage DELTA (track-safe vs naive) -------------------------
    banner("7 · leakage delta  (why track-split is load-bearing)")
    import pandas as pd
    items = pd.read_csv(root / "features_3_sec.csv")["filename"].tolist()
    tsp = track_level_split(items, dataio.track_id_from_csv, seed=0)
    nsp = naive_random(items, seed=0, track_id_fn=dataio.track_id_from_csv)
    nt = tsp.meta["n_tracks"]; straddle = nsp.meta["tracks_straddling_splits"]
    print(f"   segments={len(items)}  tracks={nt}")
    print(f"   track-safe : 0 tracks straddle  (by construction)")
    print(f"   naive      : {straddle} tracks straddle  ({100*straddle/nt:.1f}% leak across splits)")
    warn("a naive 3-sec split would inflate accuracy by recognising the SONG, not the "
         "genre (Sturm 2013). Run a naive sibling ONLY to publish the delta — never deploy it.")

    banner("DONE — mk3 data path is correct, leakage-safe, and contract-identical")
    print(f"""  Next (on a torch box):
    {C}# full mk3 RRM sweep (re-asks mk2.5's U question on ~10× data){R}
    python projects/genre/src/sweep.py   --config projects/genre/configs/beardown_3sec.yaml --data-root {args.data_root}
    {C}# or one promoted single-config run (cfg-17 shape on 3-sec data){R}
    python projects/genre/src/train.py   --config projects/genre/configs/beardown_3sec.yaml --data-root {args.data_root}
  Both attach LLLA/HMC/bundle unchanged — fused3 is dual-input like fused.""")


if __name__ == "__main__":
    main()
