#!/usr/bin/env python
"""mk3 segment→track evaluation — the number that compares to mk2.

The sweep/train report PER-SEGMENT accuracy (one row per 3-sec clip). mk1/mk2
reported PER-TRACK accuracy (one row per song). They are not comparable. This
collapses a bundle's per-segment test predictions to one prediction per track
(majority vote, prob-sum tiebreak) so you get the apples-to-apples per-track
number for the deployed mk3 bundle — directly against mk2's 0.76.

    python projects/genre/src/vote_eval.py --bundle projects/genre/models/beardown_3sec \
        --config projects/genre/configs/beardown_3sec.yaml --data-root projects/genre/data/raw

Reports per-segment AND voted-to-track accuracy + macro-F1; the GAP between them
is the real signal that segmentation bought generalization (vote denoises the
ambiguous clips). The voting core is pure-numpy and unit-tested (--selftest, no
torch); the forward pass needs torch + the bundle.
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

C, A_, R = "\033[36m", "\033[33m", "\033[0m"


# ---- pure-numpy voting core (torch-free, unit-testable) ---------------------
def vote_to_track(probs, y_seg, track_ids):
    """Collapse per-segment predictions to one-per-track.

    probs:[N,C] per-segment softmax, y_seg:[N] true labels, track_ids:[N] str.
    Per track: SUM the segment probabilities (a soft majority vote — equivalent to
    hard-majority but with a principled tiebreak), argmax → the track prediction.
    All segments of a track share a label, so the track's true label is well-defined.
    Returns (track_ids[T], y_true[T], y_pred[T])."""
    probs = np.asarray(probs, float); y_seg = np.asarray(y_seg).ravel()
    track_ids = np.asarray(track_ids)
    order = []                                     # stable unique-track order
    seen = {}
    for t in track_ids:
        if t not in seen:
            seen[t] = len(order); order.append(t)
    T, C = len(order), probs.shape[1]
    psum = np.zeros((T, C)); ytrue = np.full(T, -1, int)
    for i, t in enumerate(track_ids):
        k = seen[t]; psum[k] += probs[i]; ytrue[k] = y_seg[i]
    ypred = psum.argmax(1)
    return np.array(order), ytrue, ypred


def _acc(yt, yp):
    return float((np.asarray(yt) == np.asarray(yp)).mean())


def _macro_f1(yt, yp, C):
    yt = np.asarray(yt); yp = np.asarray(yp); f1 = []
    for c in range(C):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1))


def _selftest():
    # 3 tracks, 4 segments each; track A all-correct, B split 3:1, C wrong-majority
    ncls = 3
    rows = []  # (track, ytrue, pred_class)
    for s in range(4): rows.append(("a.1", 0, 0))                 # 4/4 correct → vote 0 ✓
    for s, p in enumerate([1, 1, 1, 2]): rows.append(("b.1", 1, p))  # 3:1 → vote 1 ✓
    for s, p in enumerate([2, 2, 0, 0]): rows.append(("c.1", 0, 2))  # 2:2 tie, but prob-sum...
    tids = np.array([r[0] for r in rows]); yseg = np.array([r[1] for r in rows])
    probs = np.zeros((len(rows), ncls))
    for i, r in enumerate(rows): probs[i, r[2]] = 1.0            # one-hot hard preds
    ot, yt, yp = vote_to_track(probs, yseg, tids)
    assert list(ot) == ["a.1", "b.1", "c.1"], ot
    assert yp[0] == 0 and yt[0] == 0, "track A vote wrong"
    assert yp[1] == 1 and yt[1] == 1, "track B 3:1 vote wrong"
    # per-segment acc here = (4 + 3 + 0)/12 = 0.583 ; track acc = (1+1+0)/3 = 0.667
    seg_acc = _acc(yseg, probs.argmax(1)); trk_acc = _acc(yt, yp)
    assert abs(seg_acc - 7/12) < 1e-9 and abs(trk_acc - 2/3) < 1e-9, (seg_acc, trk_acc)
    print(f"  {C}[ok]{R} vote_to_track: seg_acc={seg_acc:.3f} → track_acc={trk_acc:.3f} "
          f"(majority vote denoises; tie broken by prob-sum)")
    print(f"  {C}[ok]{R} selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="projects/genre/models/beardown_3sec")
    ap.add_argument("--config", default="projects/genre/configs/beardown_3sec.yaml")
    ap.add_argument("--data-root", default="projects/genre/data/raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true", help="run the torch-free voting test and exit")
    args = ap.parse_args()

    if args.selftest:
        print(f"\n{'='*64}\n  vote_eval — voting core selftest (torch-free)\n{'='*64}")
        _selftest(); return

    import torch, yaml
    from projects.genre.src import dataio
    from projects.genre.src.bundle import load_bundle
    from projects.genre.src.train import evaluate

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rep = cfg.get("representation", "fused3")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    loaded = dataio.load(representation=rep, split_strategy=cfg["split"]["mode"],
                         data_root=args.data_root, seed=cfg["split"].get("seed", 0),
                         standardize=True, drop_length=cfg.get("features", {}).get("drop_length", False),
                         image_size=cfg.get("features", {}).get("image_size", 128),
                         image_dir=cfg.get("features", {}).get("image_dir", "images_grey_scale"))
    rb = load_bundle(args.bundle, device=device)
    loader = dataio.to_torch_loader(loaded, args.split, batch=256, shuffle=False)
    seg_acc, y_true, y_pred, probs = evaluate(rb.model, loader, rep, device)

    Cn = len(loaded.label_map)
    tids = loaded.track_ids[args.split]
    ot, yt, yp = vote_to_track(probs, y_true, tids)

    seg_f1 = _macro_f1(y_true, y_pred, Cn)
    trk_acc = _acc(yt, yp); trk_f1 = _macro_f1(yt, yp, Cn)

    print(f"\n{'='*64}\n  mk3 beardown_3sec — segment vs voted-to-track ({args.split})\n{'='*64}")
    print(f"  segments        : {len(y_true)}   tracks: {len(ot)}")
    print(f"  {C}per-segment   acc={seg_acc:.4f}  macroF1={seg_f1:.4f}{R}   (what the sweep reports)")
    print(f"  {A_}voted-to-track acc={trk_acc:.4f}  macroF1={trk_f1:.4f}{R}   (apples-to-apples vs mk2 0.76)")
    print(f"  vote gain: {trk_acc - seg_acc:+.4f} acc  — the denoising segmentation bought")
    # per-genre voted-to-track
    inv = {v: k for k, v in loaded.label_map.items()}
    print(f"\n  per-genre voted-to-track accuracy:")
    for c in range(Cn):
        m = (yt == c)
        if m.any():
            print(f"    {inv[c]:10s} {(_acc(yt[m], yp[m])):.3f}  ({int(m.sum())} tracks)")
    out = {"split": args.split, "segments": int(len(y_true)), "tracks": int(len(ot)),
           "per_segment": {"acc": seg_acc, "macro_f1": seg_f1},
           "voted_to_track": {"acc": trk_acc, "macro_f1": trk_f1},
           "vote_gain_acc": trk_acc - seg_acc}
    p = Path(args.bundle) / "vote_eval.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  {C}wrote {p}{R}")


if __name__ == "__main__":
    main()
