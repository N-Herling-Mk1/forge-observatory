#!/usr/bin/env python
"""data_doctor.py - patch the live GTZAN set (data/raw/) in place.

Minimal by design. The EDA surfaced exactly one genuinely broken track, so rather
than build a parallel fixed/ tree we patch the two repaired files directly into
raw/ -- the path the data loader reads -- giving the loader a single, branch-free
read path. The short-clip / long-clip findings are documented, not fabricated
(see data/README.md).

Repairs:
  R1  place the known-good jazz.00054.wav in raw/genres_original/jazz/  (--jazz-src),
      verifying it's readable and ~30 s. Overwrites the corrupt original.
  R2  generate jazz.00054's missing grey-scale spectrogram so it matches the corpus:
        raw/images_grey_scale/jazz/jazz00054.png  ->  128x128, mode L

      Recipe (reverse-engineered + verified against the existing images):
        - mel render: 432x288 magma figure (axis-off, default subplot margins),
          librosa.feature.melspectrogram -> power_to_db(ref=max);
        - grey = render.convert('L').resize((128,128), BICUBIC)  [BICUBIC reproduces
          the existing greys bit-exact];
        - tone-calibration (default ON): the vanilla render lands ~7.5 sigma brighter
          than this corpus (grey mean 120 +/- 5, vanilla ~162) because Kaggle's exact
          renderer is unrecoverable. We histogram-match the grey to the existing jazz
          greys. Tone only - spectral structure (the real audio) is preserved
          (corr ~0.98); --no-tone-match to skip.

  We do NOT regenerate the colour image (images_original): it isn't consumed and the
  EDA tracks the grey representation only.

  Honesty note: raw/ is no longer byte-for-byte the original GTZAN after this runs.
  The pre-fix state is recorded in data/before/eda_stats.json; this script also
  writes data/manifest.fixed.json (the repair receipt: actions + sha256).

Usage (Windows, from repo root):
  python projects\\genre\\src\\data_doctor.py ^
      --jazz-src "C:\\Users\\natha\\Downloads\\jazz.00054.fixed (1).wav"
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np

JAZZ_WAV  = ("genres_original", "jazz", "jazz.00054.wav")    # dotted id, .wav
JAZZ_GREY = ("images_grey_scale", "jazz", "jazz00054.png")   # undotted id, .png
GENRES = ["blues","classical","country","disco","hiphop",
          "jazz","metal","pop","reggae","rock"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------------------------ ui
def banner(t):  print(f"\n{'='*64}\n  {t}\n{'='*64}")
def step(m):    print(f"  [..] {m}", flush=True)
def ok(m):      print(f"  [ok] {m}", flush=True)
def warn(m):    print(f"  [!!] {m}", flush=True)
def die(m):     print(f"\n  [XX] {m}\n"); sys.exit(1)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------- spectro
def mel_db(wav: Path, sr: int, n_mels: int):
    import librosa
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)   # n_fft 2048, hop 512
    return librosa.power_to_db(S, ref=np.max)


def _match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Histogram-match: remap src intensities so their CDF matches ref's."""
    sv, si, sc = np.unique(src, return_inverse=True, return_counts=True)
    rv, rc = np.unique(ref, return_counts=True)
    s_cdf = np.cumsum(sc) / src.size
    r_cdf = np.cumsum(rc) / ref.size
    return np.round(np.interp(s_cdf, r_cdf, rv)[si]).astype(np.uint8)


def render_grey(SdB, grey_out: Path, tone_match: bool = True):
    """Render the 128x128 L grey spectrogram in the dataset's style. Returns
    (size, matched: bool, mean_intensity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    from PIL import Image

    fig = plt.figure(figsize=(4.32, 2.88), dpi=100)    # -> 432 x 288 canvas
    ax = fig.add_subplot(111)                           # default subplot margins
    librosa.display.specshow(SdB, sr=22050, ax=ax, cmap="magma")
    ax.set_axis_off()                                   # heatmap inset, white margins
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    color = Image.fromarray(buf, "RGBA")
    if color.size != (432, 288):
        color = color.resize((432, 288))
    grey = np.asarray(color.convert("L").resize((128, 128), Image.BICUBIC))

    matched = False
    if tone_match:
        refs = []
        for p in sorted(grey_out.parent.glob("*.png")):
            if p.name == grey_out.name:
                continue
            try:
                a = np.asarray(Image.open(p).convert("L"))
            except Exception:
                continue
            if a.shape == (128, 128):
                refs.append(a)
        if len(refs) >= 5:
            pool = np.concatenate([r.ravel() for r in refs])
            grey = _match(grey.ravel(), pool).reshape(128, 128)
            matched = True
        else:
            warn(f"tone-match skipped: only {len(refs)} usable reference jazz greys. "
                 "Saving the raw render.")

    grey_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grey.astype(np.uint8), "L").save(grey_out)
    return (128, 128), matched, float(grey.mean())


# -------------------------------------------------------------------------- verify
def _count(d: Path, ext: str):
    return {g: len(list((d / g).glob(f"*.{ext}"))) for g in GENRES}

def verify(root: Path):
    banner("VERIFY  raw/ (wav + grey representations)")
    full = True
    for name, sub, ext in [("genres_original (wav)", "genres_original", "wav"),
                           ("images_grey_scale (png)", "images_grey_scale", "png")]:
        c = _count(root / sub, ext)
        short = [f"{g}={n}" for g, n in c.items() if n != 100]
        if short: full = False
        flag = "OK" if not short else "short: " + ", ".join(short)
        print(f"  {name:26} total={sum(c.values()):4}  jazz={c['jazz']:3}  [{flag}]")
    print()
    (ok if full else warn)("wav + grey both at 1000/1000" if full
                           else "still short - check above")
    return full


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Patch raw/ in place (jazz.00054 repairs).")
    ap.add_argument("--jazz-src", required=True, help="path to the known-good jazz.00054 wav")
    ap.add_argument("--dst", default=str(DATA_DIR / "raw"),
                    help="dataset root to patch (default data/raw - what the loader reads)")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--force", action="store_true", help="overwrite an existing grey")
    ap.add_argument("--no-tone-match", action="store_true",
                    help="skip histogram tone-calibration (ship the raw, brighter render)")
    args = ap.parse_args()

    dst = Path(args.dst)
    jazz_src = Path(args.jazz_src)

    banner("data_doctor - patching raw/ in place (jazz.00054)")
    print(f"  dst (raw) : {dst}")
    print(f"  jazz src  : {jazz_src}")

    if not jazz_src.exists():
        die(f"--jazz-src not found: {jazz_src}")
    if not dst.exists():
        die(f"dataset root not found: {dst}  (point --dst at your GTZAN Data dir)")

    manifest = {
        "_note": "Repair receipt for raw/ - emitted by data_doctor.py, not hand-edited.",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patched_root": str(dst),
        "recipe": {
            "sr": args.sr, "n_mels": args.n_mels, "n_fft": 2048, "hop_length": 512,
            "power_to_db_ref": "max", "cmap": "magma",
            "grey_png": "magma render -> convert('L').resize((128,128), BICUBIC)",
            "tone_match": (not args.no_tone_match) and "histogram-matched to existing jazz greys" or False,
        },
        "repairs": [],
        "documented_not_fixed": {
            "short_3sec_tracks_9_segments": 10,
            "off_duration_tracks_30_649s": 10,
            "color_images_original_jazz": "left at 99 (colour not consumed; EDA tracks grey)",
            "policy": "kept as-is; no fabrication (see data/README.md).",
        },
    }

    # ---- R1: place the known-good jazz wav -----------------------------------
    banner("R1  place known-good jazz.00054.wav (overwrites corrupt original)")
    dst_wav = dst.joinpath(*JAZZ_WAV)
    step(f"validating replacement is readable + ~30 s: {jazz_src.name}")
    try:
        import librosa
        y, _ = librosa.load(str(jazz_src), sr=args.sr, mono=True)
        dur = len(y) / args.sr
    except Exception as e:
        die(f"replacement wav not readable by librosa: {e}")
    (warn if not (25.0 <= dur <= 35.0) else ok)(
        f"readable, duration {dur:.2f}s" + ("" if 25.0 <= dur <= 35.0 else " (outside 25-35s - check it)"))
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jazz_src, dst_wav)
    ok(f"placed -> {dst_wav}")
    manifest["repairs"].append({
        "id": "R1", "action": "place_known_good_wav",
        "target": str(Path(*JAZZ_WAV)), "from": str(jazz_src),
        "duration_s": round(dur, 3), "sha256": sha256(dst_wav),
    })

    # ---- R2: grey spectrogram -------------------------------------------------
    banner("R2  generate jazz.00054 grey spectrogram (128x128 L)")
    grey_out = dst.joinpath(*JAZZ_GREY)
    if grey_out.exists() and not args.force:
        warn("grey already exists - pass --force to overwrite (skipping R2)")
    else:
        step("computing mel spectrogram (dB)")
        SdB = mel_db(dst_wav, args.sr, args.n_mels)
        ok(f"mel-dB shape {SdB.shape}  (n_mels x frames)")
        step("rendering + tone-matching grey" if not args.no_tone_match else "rendering grey")
        size, matched, mean = render_grey(SdB, grey_out, tone_match=not args.no_tone_match)
        ok(f"grey {size} mode=L  mean={mean:.1f}  tone_matched={matched}  -> {grey_out.name}")
        if not (110 <= mean <= 135):
            warn(f"grey mean {mean:.1f} outside corpus band 112-130 - looks like an outlier")
        manifest["repairs"].append({
            "id": "R2", "action": "generate_grey_spectrogram",
            "target": str(Path(*JAZZ_GREY)), "size": list(size),
            "mean_intensity": round(mean, 1), "tone_matched": matched,
            "sha256": sha256(grey_out),
        })

    # ---- manifest + verify ----------------------------------------------------
    man_path = DATA_DIR / "manifest.fixed.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    banner("MANIFEST"); ok(f"wrote {man_path}")

    full = verify(dst)
    banner("DONE" if full else "DONE (with warnings)")
    print(f"  next: python eda/run_eda.py --phase after --data-root {dst}\n")


if __name__ == "__main__":
    main()
