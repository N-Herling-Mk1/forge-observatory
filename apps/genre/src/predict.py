"""Song drop-in -> per-genre scores + uncertainty. Reuses the proven bundle:
load_bundle -> window the audio -> per-window (mel + 58 tabular) -> scale with the
bundle's scaler -> forward -> epistemic σ -> aggregate to song level.

Epistemic σ is MC-dropout here (works with the deterministic backbone TODAY). The
LLLA fast path (Day-2 `bayes` extra) replaces this with the closed-form last-layer
posterior the bundle's ggn_eig.npz precomputes; the /api contract below is unchanged
when that lands — only the σ source swaps.
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import torch

from .bundle import load_bundle
from . import features as F


def _scale(x_tab: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x_tab - mean) / std).astype(np.float32)


def _enable_mc_dropout(model):
    """Eval mode (BN uses running stats) but Dropout left ON for MC sampling."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def _mc_forward(model, img, tab, T: int = 30):
    """T stochastic passes -> (mean_probs [C], epistemic_sigma [C]) per window."""
    probs = []
    for _ in range(T):
        probs.append(torch.softmax(model(img, tab), dim=1).cpu().numpy())
    P = np.stack(probs, axis=0)            # [T, N, C]
    return P.mean(0), P.std(0)             # [N, C], [N, C]


def predict_song(audio_path, model_dir="projects/genre/models/beardown",
                 mel_cfg: dict | None = None, image_size: int = 224,
                 mc_samples: int = 30, device: str = "cpu",
                 clip_seconds: float = 30.0, image_mode: str = "per_window",
                 pre_audio: tuple | None = None, n_jobs: int | None = None) -> dict:
    """Drop a wav -> song-level per-genre scores + uncertainty + per-window detail.

    clip_seconds : window length for the tabular branch. 30.0 for the 30-sec models
                   (mk1/mk2, rep=fused); 3.0 for mk3 (rep=fused3), whose tabular was
                   trained on 3-sec segments.
    image_mode   : 'per_window' computes a mel per window (mk1/mk2). 'full_song' computes
                   ONE whole-song mel and reuses it for every clip — matching mk3's
                   many-to-one training (each 3-sec segment saw the full-track image).
    pre_audio    : (y, sr) already decoded — skips re-loading when several models score
                   the SAME upload (the multi-model path loads once, shares here).
    n_jobs       : threads for the per-window librosa extraction (the mk3 bottleneck —
                   ~10 independent windows). None -> min(cpu, n_windows). librosa's heavy
                   ops (FFT/numba) release the GIL, so this parallelizes for real. The
                   per-window result is identical to serial — no fidelity cost.
    """
    mel_cfg = mel_cfg or {}
    b = load_bundle(model_dir, device=device)
    genres = b.genres
    cols = b.scaler.get("cols")
    nmels = mel_cfg.get("n_mels", F.N_MELS)
    nfft = mel_cfg.get("n_fft", F.N_FFT)
    hop = mel_cfg.get("hop_length", F.HOP)

    y, sr = pre_audio if pre_audio is not None else F.load_audio(audio_path)
    # mk3: one full-song spectrogram, reused per 3-sec clip (the many-to-one join)
    full_mel = (F.extract_mel(y, sr, image_size=image_size, n_mels=nmels, n_fft=nfft,
                              hop_length=hop) if image_mode == "full_song" else None)
    windows = F.window_segments(y, sr, seconds=clip_seconds)

    def _features_for(w):
        """(mel[H,W], tab[F]) for one window — the librosa-heavy part, run per-thread."""
        mel = full_mel if full_mel is not None else \
            F.extract_mel(w, sr, image_size=image_size, n_mels=nmels, n_fft=nfft, hop_length=hop)
        tab = F.extract_tabular(w, sr, n_fft=nfft, hop_length=hop, cols=cols)
        return mel, _scale(tab[None, :], b.scaler)

    # extract all windows up front (parallel) — this is where mk3's ~10× cost lives
    if n_jobs is None:
        n_jobs = min(len(windows), os.cpu_count() or 1)
    if n_jobs > 1 and len(windows) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            feats = list(ex.map(_features_for, windows))
    else:
        feats = [_features_for(w) for w in windows]

    seg_mean, seg_sig, seg_rows = [], [], []
    for i, (mel, tab) in enumerate(feats):
        img_t = torch.from_numpy(mel[None, None, :, :]).to(device)          # [1,1,H,W]
        tab_t = torch.from_numpy(tab).to(device)                            # [1,F]
        _enable_mc_dropout(b.model)
        mean_p, sig = _mc_forward(b.model, img_t, tab_t, T=mc_samples)
        mean_p, sig = mean_p[0], sig[0]
        seg_mean.append(mean_p); seg_sig.append(sig)
        top = int(mean_p.argmax())
        seg_rows.append({"window": i, "top_genre": genres[top],
                         "top_prob": float(mean_p[top]),
                         "probs": {g: float(p) for g, p in zip(genres, mean_p)}})

    seg_mean = np.stack(seg_mean); seg_sig = np.stack(seg_sig)
    song_prob = seg_mean.mean(0)                                            # [C]
    # aggregate uncertainty: within-window epistemic (mean σ) + across-window spread
    song_sig = np.sqrt((seg_sig**2).mean(0) + seg_mean.var(0))
    order = np.argsort(song_prob)[::-1]

    return {
        "model_dir": str(model_dir),
        "n_windows": len(windows),
        "sigma_source": "mc_dropout",   # -> "llla" once the Day-2 bayes path lands
        "per_genre": [{"genre": genres[i], "prob": float(song_prob[i]),
                       "sigma": float(song_sig[i])} for i in order],
        "top": {"genre": genres[int(order[0])], "prob": float(song_prob[order[0]]),
                "sigma": float(song_sig[order[0]])},
        "segments": seg_rows,
    }


def has_model(model_dir="projects/genre/models/beardown") -> bool:
    d = Path(model_dir)
    return (d / "weights.pt").exists() and (d / "arch.json").exists()


def predict_upload(file_path, exp_root, device: str = "cpu") -> dict:
    """Uniform entrypoint the generic /api/predict route calls (one per experiment).
    Resolves the SELECTED bundle from the model registry (sticky active model), so
    song drop-in always runs whatever you picked in the Model panel; falls back to
    the legacy flat models/<run_name> slot if the registry isn't available."""
    import yaml
    exp_root = Path(exp_root)
    cfg = yaml.safe_load(open(exp_root / "configs" / "beardown.yaml", encoding="utf-8"))
    run_name = cfg.get("run_name", "beardown")
    models_root = exp_root / "models"
    model_dir = None
    try:
        from . import registry
        model_dir = registry.selected_dir(str(models_root), fallback_family=run_name)
    except Exception:
        model_dir = None
    if not model_dir:
        model_dir = str(models_root / run_name)
    if not has_model(str(model_dir)):
        return {"error": "no trained model yet — run train.py first", "model_dir": str(model_dir)}
    return predict_song(str(file_path), model_dir=str(model_dir),
                        mel_cfg=cfg.get("mel", {}),
                        image_size=cfg.get("features", {}).get("image_size", 224),
                        device=device)


# ----------------------------------------------------------------- multi-model
# The drop-in predictor fans a single uploaded song across the BEST-ACCURACY
# bundle of each generation and returns one result block per model (three graphs,
# not a pick-one). Edit DEFAULT_FAMILY to swap which bundles are compared — e.g.
# point mk3 at "beardown_3sec_cfg17" (the accuracy-best config) once it's promoted.
DEFAULT_FAMILY = [
    {"label": "mk1 · beardown",      "run": "beardown",      "cfg": "beardown.yaml"},
    {"label": "mk2 · beardown_rrm",  "run": "beardown_rrm",  "cfg": "beardown_rrm.yaml"},
    {"label": "mk3 · beardown_3sec", "run": "beardown_3sec", "cfg": "beardown_3sec.yaml"},
]


def _infer_params(cfg: dict):
    """Derive per-bundle inference settings from its training config. fused3 (mk3) →
    3-sec clips + one full-song image (its many-to-one training); fused (mk1/mk2) →
    30-sec windows + per-window images."""
    rep = cfg.get("representation", "fused")
    feat = cfg.get("features", {})
    image_size = int(feat.get("image_size", 128))
    mel = cfg.get("mel", {})
    if rep == "fused3":
        return image_size, mel, 3.0, "full_song", rep
    return image_size, mel, 30.0, "per_window", rep


def _headline_metrics(model_dir: Path, rep: str) -> dict:
    """Test accuracy for the label. For fused3 prefer voted-to-track (vote_eval.json,
    the apples-to-apples number) over per-segment metrics.json."""
    import json as _json
    out = {"test_acc": None, "test_macro_f1": None, "acc_kind": None}
    vp = model_dir / "vote_eval.json"
    if rep == "fused3" and vp.exists():
        try:
            v = _json.loads(vp.read_text(encoding="utf-8"))["voted_to_track"]
            out.update(test_acc=v.get("acc"), test_macro_f1=v.get("macro_f1"),
                       acc_kind="voted-to-track")
            return out
        except Exception:
            pass
    mp = model_dir / "metrics.json"
    if mp.exists():
        try:
            t = _json.loads(mp.read_text(encoding="utf-8")).get("test", {})
            out.update(test_acc=t.get("accuracy"), test_macro_f1=t.get("macro_f1"),
                       acc_kind="per-segment" if rep == "fused3" else "per-track")
        except Exception:
            pass
    return out


def predict_upload_multi(file_path, exp_root, device: str | None = None,
                         models: list | None = None, mc_samples: int = 20) -> dict:
    """Run the uploaded song through every model in DEFAULT_FAMILY and return one
    block per model: {label, run, per_genre[], top, test_acc, ...}. Models with no
    bundle yet degrade gracefully (an error block) instead of failing the whole call.
    Auto-selects CUDA when available (much faster across three models)."""
    import yaml
    exp_root = Path(exp_root)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    family = models or DEFAULT_FAMILY
    models_root = exp_root / "models"

    # decode the upload ONCE and share across all models (was 1 librosa.load per model)
    pre_audio = None
    try:
        from . import features as F
        pre_audio = F.load_audio(str(file_path))
    except Exception:
        pre_audio = None        # fall back to per-model load inside predict_song

    out = []
    for spec in family:
        mdir = models_root / spec["run"]
        block = {"label": spec["label"], "run": spec["run"], "model_dir": str(mdir)}
        if not has_model(str(mdir)):
            block["error"] = "no bundle yet — train/promote this model first"
            out.append(block); continue
        cfgp = exp_root / "configs" / spec["cfg"]
        cfg = yaml.safe_load(open(cfgp, encoding="utf-8")) if cfgp.exists() else {}
        image_size, mel, clip, mode, rep = _infer_params(cfg)
        try:
            r = predict_song(str(file_path), model_dir=str(mdir), mel_cfg=mel,
                             image_size=image_size, mc_samples=mc_samples, device=device,
                             clip_seconds=clip, image_mode=mode, pre_audio=pre_audio)
        except Exception as e:
            block["error"] = f"prediction failed: {e}"
            out.append(block); continue
        r.update(block, representation=rep, clip_seconds=clip,
                 **_headline_metrics(mdir, rep))
        out.append(r)
    return {"models": out, "device": device, "filename": Path(file_path).name}
