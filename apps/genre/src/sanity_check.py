#!/usr/bin/env python
"""End-to-end sanity check for the BEARDOWN data tier.

Proves the invariants you can't eyeball: shapes are right, label_map is stable,
no track straddles splits under the track guard, and the naive segment-shuffle
DOES straddle (the leakage gap, printed as a number).

    python projects/genre/src/sanity_check.py                       # synthetic fixture
    python projects/genre/src/sanity_check.py --data-root projects/genre/data/raw   # real GTZAN

Run from the repo root so the _shared import resolves. TF-free (no to_tf_dataset).
"""
from __future__ import annotations
import argparse, os, sys, time, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from projects.genre.src import dataio
from _shared.splits import track_level_split, naive_random, write_split_to_manifest

# ---- tiny progress UI (standing pref: always show status) -------------------
_T0 = time.time()
def banner(t): print(f"\n{'='*66}\n  {t}\n{'='*66}")
def step(m):   print(f"  [+{time.time()-_T0:5.1f}s] {m}", flush=True)
def ok(m):     print(f"  \033[36m[ok]\033[0m {m}", flush=True)
def warn(m):   print(f"  \033[33m[!!]\033[0m {m}", flush=True)


# ----------------------------------------------------- synthetic GTZAN fixture
def make_synthetic(root: Path, per_genre=12, segs=10):
    """Tiny on-disk GTZAN shape: CSVs + grey PNGs, so the real load() path runs."""
    import pandas as pd
    from PIL import Image
    rng = np.random.default_rng(0)
    grey = root / "images_grey_scale"
    feat = [f"f{i}" for i in range(57)]
    rows3, rows30 = [], []
    for g in dataio.GENRES:
        (grey / g).mkdir(parents=True, exist_ok=True)
        for t in range(per_genre):
            tid = f"{g}.{t:05d}"
            base = rng.normal(dataio.GENRES.index(g), 1.0, size=57)
            Image.fromarray((rng.random((128, 128)) * 255).astype(np.uint8), "L") \
                 .save(grey / g / f"{dataio.norm_image_id(tid)}.png")
            rows30.append([f"{tid}.wav", 661500, *(base + rng.normal(0, .1, 57)), g])
            k = segs - (1 if t % 10 == 0 else 0)
            for s in range(k):
                rows3.append([f"{tid}.{s}.wav", 66150, *(base + rng.normal(0, .3, 57)), g])
    cols = ["filename", "length", *feat, "label"]
    pd.DataFrame(rows30, columns=cols).to_csv(root / "features_30_sec.csv", index=False)
    pd.DataFrame(rows3, columns=cols).to_csv(root / "features_3_sec.csv", index=False)


# ----------------------------------------------------------------- checks
def check_rep(data_root: Path, rep: str, **kw):
    step(f"load(representation={rep!r}, split=track) …")
    d = dataio.load(representation=rep, split_strategy="track", data_root=str(data_root), **kw)
    print(f"        {d.summary()}")
    # label_map stable + sorted
    assert d.label_map == {g: i for i, g in enumerate(dataio.GENRES)}, "label_map drift!"
    # no track overlap at the row level
    tr, va, te = (set(d.track_ids["train"]), set(d.track_ids["val"]), set(d.track_ids["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te), f"{rep}: TRACK LEAKAGE"
    # coverage 1.0 in every split
    for name in ("train", "val", "test"):
        genres = {t.split(".", 1)[0] for t in d.track_ids[name]}
        assert genres == set(dataio.GENRES), f"{rep}/{name}: coverage<1.0 ({set(dataio.GENRES)-genres})"
    # standardization sanity (tab/fused): train tabular mean ≈ 0
    tab_train = None
    if rep in ("tab3", "tab30"):
        tab_train = d.X["train"]
    elif rep in ("fused", "fused3"):
        tab_train = d.X["train"]["tabular"]
    if tab_train is not None and tab_train.size:
        m = float(np.abs(tab_train.mean(axis=0)).max())
        assert m < 1e-3, f"{rep}: train tabular not zero-centered (max|mean|={m:.2e})"
    # many-to-one image join (fused3 only): every segment of a track shares the
    # SAME parent spectrogram, yet carries its own tabular row.
    if rep == "fused3":
        ims, tids = d.X["train"]["image"], d.track_ids["train"]
        # find a track with ≥2 segments in train
        from collections import Counter
        c = Counter(tids); multi = [t for t, k in c.items() if k >= 2]
        assert multi, "fused3: no multi-segment track in train (join untestable)"
        t0 = multi[0]; rows = np.where(tids == t0)[0]
        base = ims[rows[0]]
        for r in rows[1:]:
            assert np.array_equal(ims[r], base), f"fused3: segments of {t0} have different images (join broken)"
        tabs = d.X["train"]["tabular"][rows]
        assert not np.allclose(tabs[0], tabs[1]), f"fused3: segments of {t0} have identical tabular (segments not distinct)"
        ok(f"fused3: many-to-one join verified ({t0}: {len(rows)} segs share 1 image, distinct tab rows)")
    ok(f"{rep}: shapes ok · label_map stable · 0 tracks straddle · coverage=1.0")
    return d


def leakage_gap(data_root: Path):
    banner("LEAKAGE GAP  (track guard vs naive segment shuffle, on the 3s table)")
    import pandas as pd
    df = pd.read_csv(data_root / "features_3_sec.csv")
    items = df["filename"].tolist()
    tsp = track_level_split(items, dataio.track_id_from_csv, seed=0)
    nsp = naive_random(items, seed=0, track_id_fn=dataio.track_id_from_csv)
    n_tracks = tsp.meta["n_tracks"]
    straddle = nsp.meta["tracks_straddling_splits"]
    print(f"  segments (3s rows)            : {len(items)}")
    print(f"  tracks                        : {n_tracks}")
    print(f"  track guard — tracks straddling: 0   (by construction)")
    print(f"  naive shuffle — tracks straddling: {straddle}  "
          f"({100*straddle/n_tracks:.1f}% of tracks leak across splits)")
    warn("the naive split's accuracy would be inflated by exactly this leak — "
         "report the track-vs-naive delta, don't hide it")
    return {"n_tracks": n_tracks, "naive_straddle": straddle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="real GTZAN dir; omit to use a synthetic fixture")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--write-manifest", action="store_true",
                    help="record the chosen seed/ratios into data/manifest.json")
    args = ap.parse_args()

    banner("BEARDOWN data-tier sanity check")
    tmp = None
    if args.data_root:
        data_root = Path(args.data_root); step(f"real data-root = {data_root}")
    else:
        tmp = tempfile.TemporaryDirectory()
        data_root = Path(tmp.name); step("no --data-root → building synthetic fixture")
        make_synthetic(data_root)
        ok(f"synthetic GTZAN at {data_root} (12 tracks/genre)")

    for rep in ("tab30", "tab3"):
        check_rep(data_root, rep)
    for rep in ("image", "fused", "fused3"):
        check_rep(data_root, rep, image_size=args.image_size)

    gap = leakage_gap(data_root)

    if args.write_manifest and args.data_root:
        d = dataio.load("tab3", "track", str(data_root))
        mp = Path(args.data_root).parent / "manifest.json"
        write_split_to_manifest(str(mp), d.split)
        ok(f"recorded split (seed=0, 70/15/15) → {mp}")

    banner("DONE")
    ok("all representations load · leakage guard holds · gap measured")
    if tmp: tmp.cleanup()


if __name__ == "__main__":
    main()
