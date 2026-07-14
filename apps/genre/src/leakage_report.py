#!/usr/bin/env python
"""Quantify GTZAN segment leakage → data/leakage.json (feeds the Leakage Guard panel).

Compares the honest track-level split against the naive segment shuffle on the 3s
table and reports how many tracks would straddle splits under the naive scheme —
the Sturm-2013 inflation, as a number, per genre.

    python projects/genre/src/leakage_report.py --data-root projects/genre/data/raw
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import dataio
from _shared.splits import track_level_split, naive_random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="projects/genre/data/raw")
    ap.add_argument("--out", default=None, help="default <data-root>/../leakage.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    import pandas as pd
    df = pd.read_csv(root / "features_3_sec.csv")
    items = df["filename"].tolist()
    tid = np.array([dataio.track_id_from_csv(f) for f in items])

    tsp = track_level_split(items, dataio.track_id_from_csv, seed=args.seed)
    nsp = naive_random(items, seed=args.seed, track_id_fn=dataio.track_id_from_csv)

    # per-genre straddle under naive: a track straddles if its segments span >1 split
    home = {"train": set(tsp.train.tolist()), "val": set(tsp.val.tolist()),
            "test": set(tsp.test.tolist())}  # not used per-genre; recompute via naive map
    per_genre = {g: {"tracks": 0, "straddling": 0} for g in dataio.GENRES}
    membership = {}
    for name, idx in (("train", nsp.train), ("val", nsp.val), ("test", nsp.test)):
        for i in idx:
            membership.setdefault(tid[i], set()).add(name)
    for t, homes in membership.items():
        g = t.split(".", 1)[0]
        per_genre[g]["tracks"] += 1
        if len(homes) > 1:
            per_genre[g]["straddling"] += 1

    n_tracks = tsp.meta["n_tracks"]
    straddle = nsp.meta["tracks_straddling_splits"]
    report = {
        "schema": "leakage/1.0",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "GTZAN 3-sec segments",
        "n_segments": len(items),
        "n_tracks": n_tracks,
        "seed": args.seed,
        "ratios": tsp.meta["ratios"],
        "track_guard": {"tracks_straddling": 0, "note": "0 by construction — segments of a track stay together"},
        "naive_shuffle": {
            "tracks_straddling": straddle,
            "pct_tracks_leaking": round(100 * straddle / n_tracks, 1),
            "note": "segment-level random shuffle — the leaky baseline most GTZAN papers use",
        },
        "per_genre": per_genre,
        "interpretation": (
            f"Under a naive segment shuffle, {straddle}/{n_tracks} tracks "
            f"({100*straddle/n_tracks:.1f}%) leak across train/val/test — the model can "
            f"memorize the song and score it again at test. The track guard makes this 0, "
            f"so any accuracy gap between the two splits is leakage inflation, not skill."),
    }
    out = Path(args.out) if args.out else root.parent / "leakage.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[leakage] {straddle}/{n_tracks} tracks leak under naive shuffle "
          f"({report['naive_shuffle']['pct_tracks_leaking']}%) -> {out}")


if __name__ == "__main__":
    main()
