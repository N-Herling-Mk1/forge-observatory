"""BEARDOWN data loader — one loader, four representations, one leakage guard.

    from projects.genre.src.dataio import load, to_tf_dataset
    d = load(representation="tab3", split_strategy="track", data_root="projects/genre/data/raw")
    Xtr, ytr = d.X["train"], d.y["train"]          # numpy, ready to train
    ds = to_tf_dataset(d, "train", batch=32)        # tf.data (lazy TF import)

Design (see DATA_PIPELINE.md):
  • Split on TRACKS, never segments. Stratified per genre → 70/15/15 @ coverage 1.0.
  • Fit scaler + label_map on the TRAIN split only; apply to val/test.
  • representation ∈ {tab30, tab3, image, fused, fused3}; split_strategy ∈ {track, naive}.
  • Returns a self-describing ``Loaded`` (arrays + label_map + feature_cols + Split).

The numpy core runs WITHOUT TensorFlow. ``to_tf_dataset`` imports TF lazily, so
sanity checks and tests don't need the GPU stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys, os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.splits import track_level_split, naive_random, stratified_split_seed42, Split, DEFAULT_RATIOS  # noqa: E402

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
LABEL_MAP = {g: i for i, g in enumerate(GENRES)}        # persisted; inference agrees
NON_FEATURE = {"filename", "label"}                      # always dropped


# --------------------------------------------------------------- id normalizers
def track_id_from_csv(filename: str) -> str:
    """``blues.00000.3.wav`` (3s) or ``blues.00000.wav`` (30s) -> ``blues.00000``."""
    base = str(filename)
    if base.endswith(".wav"):
        base = base[:-4]
    parts = base.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else base


def norm_image_id(track_id: str) -> str:
    """``blues.00000`` -> ``blues00000`` (the undotted png stem)."""
    return track_id.replace(".", "")


def image_path(data_root: Path, track_id: str, image_dir: str = "images_grey_scale") -> Path:
    g = track_id.split(".", 1)[0]
    return data_root / image_dir / g / f"{norm_image_id(track_id)}.png"


# ---------------------------------------------------------------- return type
@dataclass
class Loaded:
    representation: str
    X: dict[str, np.ndarray]                 # split -> features (tab: [N,F]; image: [N,H,W,1])
    y: dict[str, np.ndarray]                 # split -> int labels [N]
    track_ids: dict[str, np.ndarray]         # split -> track id per row [N]
    label_map: dict[str, int]
    feature_cols: list[str] = field(default_factory=list)   # tab/fused only
    scaler: dict[str, Any] = field(default_factory=dict)    # {'mean','std','cols'} | {}
    split: Split | None = None
    image_size: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _rows(v) -> int:
        return int(next(iter(v.values())).shape[0]) if isinstance(v, dict) else int(v.shape[0])

    @property
    def n(self) -> dict[str, int]:
        return {k: self._rows(v) for k, v in self.X.items()}

    @property
    def input_shape(self):
        any_split = next(iter(self.X.values()))
        if isinstance(any_split, dict):     # fused: {'image':..., 'tabular':...}
            return {k: tuple(a.shape[1:]) for k, a in any_split.items()}
        return tuple(any_split.shape[1:])

    def summary(self) -> str:
        parts = [f"representation={self.representation}",
                 f"n={self.n}", f"input_shape={self.input_shape}",
                 f"n_classes={len(self.label_map)}"]
        if self.feature_cols:
            parts.append(f"n_features={len(self.feature_cols)}")
        if self.split is not None:
            parts.append(f"split={self.split.meta.get('mode')}")
        return "  ".join(parts)


# ----------------------------------------------------------------- tabular core
def _feature_columns(df: pd.DataFrame, drop_length: bool) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE]
    if drop_length and "length" in cols:
        cols.remove("length")               # 58 -> 57 (length ~constant for 30s clips)
    return cols


def _fit_scaler(X_train: np.ndarray, cols: list[str]) -> dict[str, Any]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)    # guard zero-variance columns
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64), "cols": cols}


def _apply_scaler(X: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _load_tabular(data_root: Path, which: str, split_strategy: str, seed: int,
                  ratios, standardize: bool, drop_length: bool) -> Loaded:
    csv = data_root / ("features_3_sec.csv" if which == "tab3" else "features_30_sec.csv")
    df = pd.read_csv(csv)
    feature_cols = _feature_columns(df, drop_length)
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_all).all():
        raise ValueError(f"non-finite values in {csv.name} features (EDA said zero — assert failed)")
    y_all = df["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    tid_all = df["filename"].map(track_id_from_csv).to_numpy()

    sp = _make_split(df["filename"].tolist(), track_id_from_csv,
                     split_strategy, seed, ratios, data_root=data_root)

    scaler = {}
    if standardize:
        scaler = _fit_scaler(X_all[sp.train], feature_cols)   # TRAIN ONLY

    X, y, tids = {}, {}, {}
    for name, idx in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        Xi = X_all[idx]
        X[name] = _apply_scaler(Xi, scaler) if standardize else Xi.astype(np.float32)
        y[name] = y_all[idx]
        tids[name] = tid_all[idx]
    return Loaded(representation=which, X=X, y=y, track_ids=tids, label_map=dict(LABEL_MAP),
                  feature_cols=feature_cols, scaler=scaler, split=sp,
                  meta={"csv": csv.name, "standardized": standardize, "drop_length": drop_length})


# ------------------------------------------------------------------- image core
def _load_one_image(path: Path, size: int) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L")
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return a[..., None]                       # [H,W,1]


def _enumerate_tracks(data_root: Path, image_dir: str = "images_grey_scale") -> list[str]:
    """One track id per grey spectrogram (the image rep is track-level)."""
    grey = data_root / image_dir
    out = []
    for g in GENRES:
        for p in sorted((grey / g).glob("*.png")):
            stem = p.stem                     # blues00000
            out.append(f"{g}.{stem[len(g):]}")   # -> blues.00000
    return out


def _load_image(data_root: Path, split_strategy: str, seed: int, ratios,
                standardize: bool, image_size: int, image_dir: str = "images_grey_scale") -> Loaded:
    track_ids = _enumerate_tracks(data_root, image_dir)
    sp = _make_split(track_ids, lambda t: t, split_strategy, seed, ratios, data_root=data_root)
    tid_all = np.array(track_ids)
    y_all = np.array([LABEL_MAP[t.split(".", 1)[0]] for t in track_ids], dtype=np.int64)

    cache: dict[str, np.ndarray] = {}
    def img(t):
        if t not in cache:
            cache[t] = _load_one_image(image_path(data_root, t, image_dir), image_size)
        return cache[t]

    X, y, tids = {}, {}, {}
    for name, idx in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        X[name] = np.stack([img(tid_all[i]) for i in idx]).astype(np.float32) if idx.size else \
                  np.empty((0, image_size, image_size, 1), np.float32)
        y[name] = y_all[idx]
        tids[name] = tid_all[idx]
    # image standardize = per-pixel z-score over train (optional; default /255 only)
    scaler = {}
    if standardize and X["train"].size:
        mean = X["train"].mean(axis=0); std = X["train"].std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        for name in X:
            if X[name].size:
                X[name] = ((X[name] - mean) / std).astype(np.float32)
        scaler = {"per_pixel": True}
    return Loaded(representation="image", X=X, y=y, track_ids=tids, label_map=dict(LABEL_MAP),
                  scaler=scaler, split=sp, image_size=image_size,
                  meta={"standardized": standardize})


# -------------------------------------------------------------------- fused core
def _load_fused(data_root: Path, split_strategy: str, seed: int, ratios,
                standardize: bool, drop_length: bool, image_size: int, image_dir: str = "images_grey_scale") -> Loaded:
    """tab30 ⨝ image, 1:1 per track (BEARDOWN's gated/concat head). The split is
    computed once on track ids so the tabular and image rows stay aligned."""
    df = pd.read_csv(data_root / "features_30_sec.csv")
    feature_cols = _feature_columns(df, drop_length)
    df["track_id"] = df["filename"].map(track_id_from_csv)
    df = df.set_index("track_id")

    track_ids = [t for t in _enumerate_tracks(data_root, image_dir) if t in df.index]
    sp = _make_split(track_ids, lambda t: t, split_strategy, seed, ratios, data_root=data_root)
    tid_all = np.array(track_ids)
    Xtab_all = df.loc[tid_all, feature_cols].to_numpy(dtype=np.float64)
    y_all = np.array([LABEL_MAP[t.split(".", 1)[0]] for t in track_ids], dtype=np.int64)

    scaler = {}
    if standardize:
        scaler = _fit_scaler(Xtab_all[sp.train], feature_cols)

    cache: dict[str, np.ndarray] = {}
    def img(t):
        if t not in cache:
            cache[t] = _load_one_image(image_path(data_root, t, image_dir), image_size)
        return cache[t]

    X, y, tids = {}, {}, {}
    for name, idx in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        tab = _apply_scaler(Xtab_all[idx], scaler) if standardize else Xtab_all[idx].astype(np.float32)
        ims = np.stack([img(tid_all[i]) for i in idx]).astype(np.float32) if idx.size else \
              np.empty((0, image_size, image_size, 1), np.float32)
        X[name] = {"image": ims, "tabular": tab}      # dict per split (BEARDOWN dual input)
        y[name] = y_all[idx]
        tids[name] = tid_all[idx]
    return Loaded(representation="fused", X=X, y=y, track_ids=tids, label_map=dict(LABEL_MAP),
                  feature_cols=feature_cols, scaler=scaler, split=sp, image_size=image_size,
                  meta={"join": "tab30 x image (1:1 per track)", "standardized": standardize,
                        "drop_length": drop_length})


# ------------------------------------------------------------- fused-3sec core
def _load_fused3(data_root: Path, split_strategy: str, seed: int, ratios,
                 standardize: bool, drop_length: bool, image_size: int,
                 image_dir: str = "images_grey_scale") -> Loaded:
    """mk3: tab3 ⨝ parent-track image, MANY-TO-ONE (~10 segments : 1 spectrogram).

    Each 3-sec segment row is its own training sample; all ~10 segments of a track
    share that track's single mel image (no spectrogram cutting needed — the image
    is track-level). The split is on TRACK ids, so every segment of a track lands in
    exactly one split — this is the load-bearing leakage guard (Sturm-2013). The
    scaler is fit on TRAIN segments only. Structurally identical dual-input contract
    to ``_load_fused`` (``X[split] = {'image','tabular'}``) so train.py / LLLA / HMC /
    bundle all attach unchanged; only the row count (~10×) and clip length differ.
    """
    df = pd.read_csv(data_root / "features_3_sec.csv")
    feature_cols = _feature_columns(df, drop_length)
    df["track_id"] = df["filename"].map(track_id_from_csv)

    # keep only segments whose parent track actually has an image on disk
    avail = set(_enumerate_tracks(data_root, image_dir))
    df = df[df["track_id"].isin(avail)].reset_index(drop=True)

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_all).all():
        raise ValueError("non-finite values in features_3_sec.csv features (assert failed)")
    y_all = df["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    tid_all = df["track_id"].to_numpy()

    # split on the per-segment track id → segments of a track never straddle splits
    sp = _make_split(df["filename"].tolist(), track_id_from_csv,
                     split_strategy, seed, ratios, data_root=data_root)

    scaler = {}
    if standardize:
        scaler = _fit_scaler(X_all[sp.train], feature_cols)        # TRAIN segments only

    cache: dict[str, np.ndarray] = {}                              # 1 load per unique track
    def img(t):
        if t not in cache:
            cache[t] = _load_one_image(image_path(data_root, t, image_dir), image_size)
        return cache[t]

    X, y, tids = {}, {}, {}
    for name, idx in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        tab = _apply_scaler(X_all[idx], scaler) if standardize else X_all[idx].astype(np.float32)
        ims = np.stack([img(tid_all[i]) for i in idx]).astype(np.float32) if idx.size else \
              np.empty((0, image_size, image_size, 1), np.float32)
        X[name] = {"image": ims, "tabular": tab}                  # parent image repeated per segment
        y[name] = y_all[idx]
        tids[name] = tid_all[idx]
    return Loaded(representation="fused3", X=X, y=y, track_ids=tids, label_map=dict(LABEL_MAP),
                  feature_cols=feature_cols, scaler=scaler, split=sp, image_size=image_size,
                  meta={"join": "tab3 x parent image (~10:1 many-to-one)",
                        "standardized": standardize, "drop_length": drop_length,
                        "tab_csv": "features_3_sec.csv", "n_segments": int(len(df)),
                        "n_tracks": int(df["track_id"].nunique())})


# --------------------------------------------------------------------- dispatch
def _make_split(items, track_id_fn, strategy: str, seed: int, ratios, data_root=None) -> Split:
    if strategy == "track":
        return track_level_split(items, track_id_fn, seed=seed, ratios=ratios)
    if strategy == "naive":
        return naive_random(items, seed=seed, ratios=ratios, track_id_fn=track_id_fn)
    if strategy == "stratified":
        # BEARDOWN gtzan_eda.py split (seed 42 by default). Build the canonical track
        # order + genre from features_30_sec.csv (CSV row order -> bit-exact partition).
        import pandas as pd
        df = pd.read_csv(Path(data_root) / "features_30_sec.csv")
        ordered, genre = [], {}
        for fn, lab in zip(df["filename"].astype(str), df["label"].astype(str)):
            t = fn[:-4] if fn.lower().endswith(".wav") else fn
            if t not in genre:
                ordered.append(t); genre[t] = lab.lower()
        return stratified_split_seed42(items, track_id_fn, ordered,
                                       lambda t: genre[t], seed=seed)
    raise ValueError(f"split_strategy must be 'track', 'naive', or 'stratified'; got {strategy!r}")


def load(representation: str = "tab3",
         split_strategy: str = "track",
         data_root: str = "projects/genre/data/raw",
         seed: int = 0,
         ratios=DEFAULT_RATIOS,
         standardize: bool = True,
         drop_length: bool = False,            # keep 58 to match the dashboard arch; True -> 57
         image_size: int = 128,
         image_dir: str = "images_grey_scale") -> Loaded:  # image_dir: images_grey_scale (Kaggle) | images_mel (clean)
    """Load GTZAN into split numpy arrays with the leakage guard applied.

    Returns a ``Loaded``: ``.X`` / ``.y`` / ``.track_ids`` keyed by split, plus
    ``.label_map``, ``.feature_cols`` (tab/fused), ``.scaler`` (train-fit) and the
    ``.split`` provenance. Feed ``.X['train']`` straight to numpy, or hand the whole
    thing to ``to_tf_dataset`` for batched ``tf.data``.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root.resolve()}")
    if representation in ("tab3", "tab30"):
        return _load_tabular(root, representation, split_strategy, seed, ratios,
                             standardize, drop_length)
    if representation == "image":
        return _load_image(root, split_strategy, seed, ratios, standardize, image_size, image_dir)
    if representation == "fused":
        return _load_fused(root, split_strategy, seed, ratios, standardize, drop_length, image_size, image_dir)
    if representation == "fused3":
        return _load_fused3(root, split_strategy, seed, ratios, standardize, drop_length, image_size, image_dir)
    raise ValueError(f"representation must be tab3|tab30|image|fused|fused3; got {representation!r}")


# --------------------------------------------------------------- tf.data wrapper
def to_tf_dataset(loaded: Loaded, split: str = "train", batch: int = 32,
                  shuffle: bool | None = None, seed: int = 0):
    """Build a batched ``tf.data.Dataset`` for one split. TF is imported lazily
    here so the numpy path stays TF-free. Train shuffles (buffer ≥ split size)."""
    import tensorflow as tf
    if shuffle is None:
        shuffle = (split == "train")
    X, y = loaded.X[split], loaded.y[split]
    if loaded.representation in ("fused", "fused3"):
        ds = tf.data.Dataset.from_tensor_slices(((X["image"], X["tabular"]), y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    n = int(y.shape[0])
    if shuffle and n:
        ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------- torch.data wrapper
def to_torch_loader(loaded: Loaded, split: str = "train", batch: int = 32,
                    shuffle: bool | None = None, seed: int = 0,
                    num_workers: int = 0, indices=None):
    """Wrap one split's arrays in a torch ``TensorDataset`` + ``DataLoader``.

    Torch mirror of ``to_tf_dataset``. The numpy core is unchanged — this only
    adapts shape/dtype: images are transposed NHWC ``[N,H,W,1]`` -> NCHW
    ``[N,1,H,W]`` for conv layers; labels -> ``long``. The fused rep yields
    ``(image, tabular, y)`` batches; image/tabular reps yield ``(x, y)``.

    ``indices`` (optional) selects a subset of rows *within* the split — used by
    the k-fold CV to carve train/val folds out of the train split without
    rebuilding the loader. Train shuffles by default.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if shuffle is None:
        shuffle = (split == "train")
    X, y = loaded.X[split], loaded.y[split]

    def _img_nchw(a):
        t = torch.as_tensor(np.asarray(a), dtype=torch.float32)
        if t.ndim == 4:                      # [N,H,W,C] -> [N,C,H,W]
            t = t.permute(0, 3, 1, 2).contiguous()
        return t

    def _sel(t):
        return t if indices is None else t[indices]

    yt = _sel(torch.as_tensor(np.asarray(y), dtype=torch.long))
    if loaded.representation in ("fused", "fused3"):
        img = _sel(_img_nchw(X["image"]))
        tab = _sel(torch.as_tensor(np.asarray(X["tabular"]), dtype=torch.float32))
        ds = TensorDataset(img, tab, yt)
    elif loaded.representation == "image":
        ds = TensorDataset(_sel(_img_nchw(X)), yt)
    else:                                    # tab3 | tab30
        ds = TensorDataset(_sel(torch.as_tensor(np.asarray(X), dtype=torch.float32)), yt)

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=num_workers, generator=g, drop_last=False)
