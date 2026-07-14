"""GTZAN feature extraction for the song drop-in path. Mel-spectrograms + the
58-feature tabular tier — computed with the SAME librosa recipe that produced the
shipped GTZAN artifacts, so a dropped song lands in the training distribution.
Mismatch here fails silently (plausible-looking but wrong scores), so the recipe
params live in beardown.yaml (the `mel:` block) and the tabular column order is
driven by the bundle's scaler.json `cols`.

Heavy imports (librosa) live inside functions so this module imports without them.
"""
from __future__ import annotations
import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128

# canonical 58-feature schema (matches features_30_sec.csv, minus filename/label).
# `length` included -> 58; drop it -> 57. Order here is the CSV order; the actual
# emit order is whatever the bundle's scaler["cols"] says (we assemble to match).
_AUDIO_COLS = [
    "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
]
_AUDIO_COLS += [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")]  # 40


def load_audio(path, sr: int = SR):
    """Mono float32 at sr. Returns (y, sr)."""
    import librosa
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def window_segments(y, sr: int = SR, seconds: float = 30.0):
    """Chunk into non-overlapping `seconds`-long windows. Songs shorter than one
    window -> a single right-padded window (so the model always sees full length)."""
    w = int(round(seconds * sr))
    if len(y) < w:
        return [np.pad(y, (0, w - len(y)))]
    n = len(y) // w
    return [y[i * w:(i + 1) * w] for i in range(n)]


def extract_mel(y, sr: int = SR, n_mels: int = N_MELS, n_fft: int = N_FFT,
                hop_length: int = HOP, image_size: int = 224) -> np.ndarray:
    """Mel power -> dB -> min-max [0,1] -> resize to (image_size, image_size).

    Returns [image_size, image_size] float32 in [0,1], comparable to the shipped
    grey PNG / 255. NOTE: the exact normalization/colormap of the Kaggle grey PNGs
    is not guaranteed identical — validate with validate_mel_recipe() before trusting.
    """
    import librosa
    from PIL import Image
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    rng = S_db.max() - S_db.min()
    norm = (S_db - S_db.min()) / (rng if rng > 1e-8 else 1.0)        # [0,1], [n_mels, T]
    im = Image.fromarray((norm * 255).astype(np.uint8)).convert("L")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.BICUBIC)
    return (np.asarray(im, dtype=np.float32) / 255.0)               # [H, W]


def extract_tabular(y, sr: int = SR, n_fft: int = N_FFT, hop_length: int = HOP,
                    cols: list[str] | None = None) -> np.ndarray:
    """Compute the GTZAN engineered features for one segment, returned in `cols`
    order (the bundle's scaler["cols"]). Raises if a requested column isn't known,
    so feature/scaler misalignment fails loud instead of silently mis-scaling."""
    import librosa
    feats: dict[str, float] = {}

    def mv(name, arr):
        feats[f"{name}_mean"] = float(np.mean(arr)); feats[f"{name}_var"] = float(np.var(arr))

    mv("chroma_stft", librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    mv("rms", librosa.feature.rms(y=y, hop_length=hop_length))
    mv("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    mv("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    mv("rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    mv("zero_crossing_rate", librosa.feature.zero_crossing_rate(y, hop_length=hop_length))
    y_harm, y_perc = librosa.effects.hpss(y)
    mv("harmony", y_harm); mv("perceptr", y_perc)
    tempo = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)[0]
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    for i in range(20):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_var"] = float(np.var(mfcc[i]))
    feats["length"] = float(len(y))

    order = cols if cols is not None else (_AUDIO_COLS + ["length"])
    missing = [c for c in order if c not in feats]
    if missing:
        raise KeyError(f"extract_tabular: unknown/uncomputed columns {missing[:5]}"
                       f"{'…' if len(missing) > 5 else ''} (feature/scaler mismatch)")
    return np.array([feats[c] for c in order], dtype=np.float32)


def validate_mel_recipe(train_wav_path, shipped_png_path, image_size: int = 128,
                        **mel_kw) -> dict:
    """Sanity gate: re-extract a TRAINING song's mel and correlate it against its
    shipped grey PNG. High correlation => the recipe matches the training
    distribution. Low => the mel params (n_mels/n_fft/hop/normalization) are off and
    drop-in scores will be silently wrong. Run once per dataset before trusting /infer.
    """
    from PIL import Image
    y, sr = load_audio(train_wav_path)
    mine = extract_mel(y, sr, image_size=image_size, **mel_kw)
    shipped = np.asarray(Image.open(shipped_png_path).convert("L").resize(
        (image_size, image_size)), dtype=np.float32) / 255.0
    a, b = mine.ravel(), shipped.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    return {"pearson_r": corr, "mae": float(np.abs(a - b).mean()),
            "verdict": "match" if corr > 0.9 else ("weak" if corr > 0.6 else "MISMATCH")}
