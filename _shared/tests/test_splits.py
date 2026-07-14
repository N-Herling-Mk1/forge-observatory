"""Invariant tests for the leakage guard. Pure numpy — no TF needed.

    python -m pytest _shared/tests/test_splits.py -q      # from repo root

The no-overlap test is load-bearing: if it ever goes red, GTZAN leakage is back.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from _shared.splits import (  # noqa: E402
    track_level_split, naive_random, artist_split,
    assert_no_track_overlap, write_split_to_manifest, _genre_of,
)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def _synth_segments(tracks_per_genre=100, segs=10):
    """A GTZAN-shaped fixture: 10 genres × N tracks × ~10 segments. Returns the
    segment filenames; track_id_fn strips the segment index + extension."""
    items = []
    for g in GENRES:
        for t in range(tracks_per_genre):
            k = segs - (1 if (t % 10 == 0) else 0)   # a few tracks have 9 segments
            for s in range(k):
                items.append(f"{g}.{t:05d}.{s}.wav")
    return items


def _track_id(seg_name: str) -> str:
    """blues.00000.3.wav -> blues.00000 ; blues.00000.wav -> blues.00000."""
    base = seg_name[:-4] if seg_name.endswith(".wav") else seg_name
    parts = base.split(".")
    return ".".join(parts[:2])


# -------------------------------------------------------- the load-bearing one
def test_no_track_overlap_track_split():
    """Under the track guard, no track id appears in more than one split (row-level too)."""
    items = _synth_segments()
    sp = track_level_split(items, _track_id, seed=0)
    tr = {t for t, s in sp.track_split.items() if s == "train"}
    va = {t for t, s in sp.track_split.items() if s == "val"}
    te = {t for t, s in sp.track_split.items() if s == "test"}
    assert tr and va and te
    assert_no_track_overlap(tr, va, te)            # must not raise
    # and at the row level: no track id appears in two index sets
    tid = np.array([_track_id(i) for i in items])
    s_tr, s_va, s_te = set(tid[sp.train]), set(tid[sp.val]), set(tid[sp.test])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)


def test_assert_no_track_overlap_fires_on_leak():
    """The overlap assertion fails loud when a track straddles splits."""
    with pytest.raises(AssertionError):
        assert_no_track_overlap(["a", "b"], ["b"], ["c"])   # 'b' straddles


# ------------------------------------------------------------ coverage & ratio
def test_every_split_has_all_ten_genres():
    """Stratification gives coverage = 1.0: every split holds all 10 genres."""
    items = _synth_segments()
    sp = track_level_split(items, _track_id, seed=1)
    tid = np.array([_track_id(i) for i in items])
    for idx in (sp.train, sp.val, sp.test):
        genres = {_genre_of(t) for t in tid[idx]}
        assert genres == set(GENRES), f"coverage != 1.0: missing {set(GENRES)-genres}"


def test_stratified_counts_exact_for_n100():
    """For 100 tracks/genre the per-genre split is exactly 70/15/15."""
    items = _synth_segments(tracks_per_genre=100)
    sp = track_level_split(items, _track_id, seed=0, ratios=(0.7, 0.15, 0.15))
    for stratum, c in sp.meta["coverage_per_stratum"].items():
        assert (c["train"], c["val"], c["test"]) == (70, 15, 15), (stratum, c)


def test_track_counts_sum_back():
    """Rounded splits sum back to the stratum size; train absorbs the remainder."""
    items = _synth_segments(tracks_per_genre=37)   # forces rounding remainders
    sp = track_level_split(items, _track_id, seed=3)
    for c in sp.meta["coverage_per_stratum"].values():
        assert c["train"] + c["val"] + c["test"] == 37
        assert c["train"] >= c["val"] and c["train"] >= c["test"]


# ------------------------------------------------------------------ determinism
def test_determinism_same_seed():
    """Same seed → identical split (reproducible)."""
    items = _synth_segments()
    a = track_level_split(items, _track_id, seed=7)
    b = track_level_split(items, _track_id, seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)


def test_different_seed_changes_assignment():
    """A different seed changes the assignment (it is actually randomized)."""
    items = _synth_segments()
    a = track_level_split(items, _track_id, seed=1)
    b = track_level_split(items, _track_id, seed=2)
    assert not np.array_equal(a.train, b.train)


# ----------------------------------------------------- the leakage measurement
def test_naive_random_straddles_tracks():
    """The whole point of keeping naive_random: on segment-level shuffling many
    tracks land in >1 split. That straddle count is the leakage being measured."""
    items = _synth_segments()
    sp = naive_random(items, seed=0, track_id_fn=_track_id)
    assert sp.meta["tracks_straddling_splits"] > 0
    # track split has zero straddles, by construction
    tsp = track_level_split(items, _track_id, seed=0)
    assert tsp.meta.get("n_tracks") == 1000


# ------------------------------------------------------------------- ratios bad
def test_bad_ratios_raise():
    """Ratios that don't sum to 1.0 (or wrong length) are rejected."""
    items = _synth_segments(tracks_per_genre=5)
    with pytest.raises(ValueError):
        track_level_split(items, _track_id, ratios=(0.5, 0.3, 0.3))   # sums to 1.1
    with pytest.raises(ValueError):
        track_level_split(items, _track_id, ratios=(0.7, 0.3))        # wrong length


# ------------------------------------------------------------------ artist stub
def test_artist_split_is_explicit_notimplemented():
    """artist_split fails explicitly (no GTZAN artist metadata) rather than splitting wrong."""
    items = _synth_segments(tracks_per_genre=3)
    with pytest.raises(NotImplementedError):
        artist_split(items, artist_id_fn=lambda x: "unknown")


# --------------------------------------------------------------- manifest write
def test_write_split_to_manifest(tmp_path):
    """The split records seed/ratios/sizes into manifest.json, preserving existing keys."""
    items = _synth_segments(tracks_per_genre=10)
    sp = track_level_split(items, _track_id, seed=0)
    mp = tmp_path / "manifest.json"
    mp.write_text('{"existing": true}')
    out = write_split_to_manifest(str(mp), sp)
    assert out["existing"] is True                 # preserved
    assert out["split"]["mode"] == "track"
    assert out["split"]["seed"] == 0
    assert out["split"]["train"] + out["split"]["val"] + out["split"]["test"] == len(items)
