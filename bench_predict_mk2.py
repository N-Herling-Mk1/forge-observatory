#!/usr/bin/env python3
"""bench_predict_mk2.py - FORGE drop-path benchmark (repo root).

Times the exact chain /api/predict runs (predict_upload_multi), split by stage:
    decode | per-model: bundle load, feature extraction (mel/tabular), MC
    forwards, predict wall, n_windows
No Flask, no browser, no delta required - imports the serve stack directly and
wraps it in read-only timing shims, so the SAME script measures baseline and
every optimization after it. Hypothesis map:
    H1 bundle reload   -> load_ms  (+ --cache-bundles prices the fix today)
    H2 batch-1 MC      -> mc_fwd_ms vs n_windows (mk3 should dwarf mk1/mk2)
    H3 librosa/window  -> feat_tab_ms + feat_mel_ms vs mc_fwd_ms
    H4 cold import     -> [stack] line + cold repeat vs warm medians
Caveat: feat_* are per-call sums; extraction runs in a thread pool, so calls
overlap and the sum is CPU-cost-like, not wall. mc_fwd runs serially - its sum
IS wall. The features:forwards ratio per model is the H3 verdict either way.

Usage (repo root):
    python bench_predict_mk2.py path\\to\\track.wav -N 3 --label baseline
    python bench_predict_mk2.py track.wav -N 3 --label cached --cache-bundles
    python bench_predict_mk2.py --compare baseline cached

Protocol: same wav(s), same N, medians; repeat 1 of a fresh process = cold
(torch import + first disk reads), later repeats = warm. One change per label.
Every run appends to bench_results.json - the before/after record lives on disk.

--cache-bundles previews the bundle-cache fix (load once per process) WITHOUT
touching src - measures what that delta will buy before it exists.
"""
from __future__ import annotations
import argparse, importlib, json, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BAR = "-" * 66


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3


def _median(xs):
    return round(statistics.median(xs), 1) if xs else None


def load_stack(device_note: str):
    """Import the serve stack (apps. = deploy repo, projects. = dev repo),
    timing the import - repeat this in a fresh process and it IS the cold tax."""
    pkg = "apps" if (ROOT / "apps" / "genre" / "src" / "predict.py").exists() else "projects"
    t0 = time.perf_counter()
    predict = importlib.import_module(f"{pkg}.genre.src.predict")   # pulls torch
    feats = importlib.import_module(f"{pkg}.genre.src.features")
    imp_ms = _ms(t0)
    print(f"[stack] {pkg}.genre.src imported in {imp_ms:.0f}ms (torch et al.) | device={device_note}")
    return pkg, predict, feats, imp_ms


def main():
    ap = argparse.ArgumentParser(description="FORGE drop-path benchmark")
    ap.add_argument("wav", nargs="*", help="audio file(s) to drop")
    ap.add_argument("-N", type=int, default=3, help="repeats per file (default 3)")
    ap.add_argument("--label", default="run", help="name this measurement (baseline / cached / batched-mc / ...)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-bundles", action="store_true",
                    help="preview the bundle-cache fix: load each bundle once per process")
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="print median table A vs B from the results file and exit")
    args = ap.parse_args()

    outp = ROOT / args.out
    if args.compare:
        return compare(outp, *args.compare)
    if not args.wav:
        ap.error("give at least one wav (or use --compare)")

    pkg, predict, feats, import_ms = load_stack(args.device)
    exp_root = ROOT / pkg / "genre"

    # ---- timing shims: transparent passthroughs, record every call ----------
    loads, songs, decodes = [], [], []
    mels, tabs, fwds = [], [], []          # per-call ms; bracketed per model in timed_song
    real_load, real_song, real_dec = predict.load_bundle, predict.predict_song, feats.load_audio
    real_mel, real_tab, real_fwd = feats.extract_mel, feats.extract_tabular, predict._mc_forward
    cache = {}

    def timed_load(model_dir, device="cpu"):
        key = (str(model_dir), device)
        if args.cache_bundles and key in cache:
            loads.append({"model_dir": str(model_dir), "ms": 0.0, "cached": True})
            return cache[key]
        t0 = time.perf_counter()
        b = real_load(model_dir, device=device)
        loads.append({"model_dir": str(model_dir), "ms": round(_ms(t0), 1), "cached": False})
        if args.cache_bundles:
            cache[key] = b
        return b

    def timed_song(audio_path, **kw):
        m0, t0i, f0 = len(mels), len(tabs), len(fwds)      # bracket this model's calls
        t0 = time.perf_counter()
        r = real_song(audio_path, **kw)
        songs.append({"model_dir": str(kw.get("model_dir", "?")), "ms": round(_ms(t0), 1),
                      "n_windows": r.get("n_windows"),
                      "feat_mel_ms": round(sum(mels[m0:]), 1), "n_mel": len(mels) - m0,
                      "feat_tab_ms": round(sum(tabs[t0i:]), 1), "n_tab": len(tabs) - t0i,
                      "mc_fwd_ms": round(sum(fwds[f0:]), 1), "n_fwd": len(fwds) - f0})
        return r

    def timed_mel(*a, **kw):
        t0 = time.perf_counter()
        r = real_mel(*a, **kw)
        mels.append(_ms(t0)); return r

    def timed_tab(*a, **kw):
        t0 = time.perf_counter()
        r = real_tab(*a, **kw)
        tabs.append(_ms(t0)); return r

    def timed_fwd(*a, **kw):
        t0 = time.perf_counter()
        r = real_fwd(*a, **kw)
        fwds.append(_ms(t0)); return r

    def timed_dec(path, sr=feats.SR):
        t0 = time.perf_counter()
        y, s = real_dec(path, sr=sr)
        decodes.append(round(_ms(t0), 1))
        return y, s

    predict.load_bundle = timed_load
    predict.predict_song = timed_song
    feats.load_audio = timed_dec
    feats.extract_mel = timed_mel
    feats.extract_tabular = timed_tab
    predict._mc_forward = timed_fwd

    # ---- run ----------------------------------------------------------------
    run = {"label": args.label, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "device": args.device, "cache_bundles": args.cache_bundles,
           "import_ms": round(import_ms, 1), "files": []}

    for wav in args.wav:
        wp = Path(wav)
        if not wp.exists():
            print(f"[x] missing: {wav}"); continue
        frec = {"wav": wp.name, "size_kb": round(wp.stat().st_size / 1024, 1), "repeats": []}
        print(f"\n{BAR}\n=== {wp.name} ({frec['size_kb']:.0f} kB) | N={args.N} | label={args.label} ===")
        for i in range(1, args.N + 1):
            loads.clear(); songs.clear(); decodes.clear()
            print(f"[{i}/{args.N}] dropping...", flush=True)
            t0 = time.perf_counter()
            res = predict.predict_upload_multi(str(wp), exp_root, device=args.device)
            total = round(_ms(t0), 1)
            kind = "cold" if i == 1 else "warm"
            rec = {"kind": kind, "total_ms": total, "decode_ms": decodes[0] if decodes else None,
                   "models": []}
            print(f"    decode {rec['decode_ms']}ms | total {total}ms ({kind})")
            for blk in res.get("models", []):
                mdir = blk.get("model_dir", "?")
                l = next((x["ms"] for x in loads if x["model_dir"] == mdir), None)
                s = next((x for x in songs if x["model_dir"] == mdir), {})
                m = {"label": blk.get("label", Path(mdir).name), "load_ms": l,
                     "predict_ms": s.get("ms"), "n_windows": s.get("n_windows"),
                     "feat_mel_ms": s.get("feat_mel_ms"), "feat_tab_ms": s.get("feat_tab_ms"),
                     "mc_fwd_ms": s.get("mc_fwd_ms"), "n_fwd": s.get("n_fwd"),
                     "error": blk.get("error")}
                rec["models"].append(m)
                if m["error"]:
                    print(f"      {m['label']:<24} ERROR: {m['error']}")
                else:
                    print(f"      {m['label']:<24} load {m['load_ms']}ms | "
                          f"predict {m['predict_ms']}ms | windows {m['n_windows']}")
                    print(f"      {'':<24} feat mel {m['feat_mel_ms']}ms + tab {m['feat_tab_ms']}ms (cpu-sum) | "
                          f"mc_fwd {m['mc_fwd_ms']}ms / {m['n_fwd']} calls")
            frec["repeats"].append(rec)

        # medians (warm-only where there are enough repeats, else all)
        pool = [r for r in frec["repeats"] if r["kind"] == "warm"] or frec["repeats"]
        med = {"basis": "warm" if len(frec["repeats"]) > 1 else "all",
               "total_ms": _median([r["total_ms"] for r in pool]),
               "decode_ms": _median([r["decode_ms"] for r in pool if r["decode_ms"] is not None]),
               "models": {}}
        for m in (pool[0]["models"] if pool else []):
            lab = m["label"]
            def _mmed(key):
                return _median([x.get(key) for r in pool for x in r["models"]
                                if x["label"] == lab and x.get(key) is not None])
            med["models"][lab] = {
                "load_ms": _mmed("load_ms"), "predict_ms": _mmed("predict_ms"),
                "feat_mel_ms": _mmed("feat_mel_ms"), "feat_tab_ms": _mmed("feat_tab_ms"),
                "mc_fwd_ms": _mmed("mc_fwd_ms"), "n_windows": m["n_windows"]}
        frec["median"] = med
        run["files"].append(frec)

        print(f"\n--- medians ({med['basis']}) --- decode {med['decode_ms']}ms | total {med['total_ms']}ms")
        for lab, m in med["models"].items():
            print(f"    {lab:<24} load {m['load_ms']}ms | predict {m['predict_ms']}ms | windows {m['n_windows']}")
            print(f"    {'':<24} feat mel {m['feat_mel_ms']}ms + tab {m['feat_tab_ms']}ms (cpu-sum) | mc_fwd {m['mc_fwd_ms']}ms")

    # ---- append to results file --------------------------------------------
    hist = []
    if outp.exists():
        try:
            hist = json.loads(outp.read_text(encoding="utf-8"))
        except Exception:
            hist = []
    hist.append(run)
    outp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\n[saved] run '{args.label}' appended -> {outp.name} ({len(hist)} runs on record)")
    print(f"[next]  compare any two: python {Path(__file__).name} --compare <labelA> <labelB>")


def compare(outp: Path, a: str, b: str):
    if not outp.exists():
        print(f"[x] no {outp.name} yet"); return
    hist = json.loads(outp.read_text(encoding="utf-8"))
    ra = next((r for r in reversed(hist) if r["label"] == a), None)
    rb = next((r for r in reversed(hist) if r["label"] == b), None)
    if not ra or not rb:
        have = sorted({r['label'] for r in hist})
        print(f"[x] label missing; on record: {have}"); return
    print(f"\n=== {a}  vs  {b} ===  (medians; latest run per label)")
    for fa in ra["files"]:
        fb = next((f for f in rb["files"] if f["wav"] == fa["wav"]), None)
        if not fb:
            continue
        ma, mb = fa["median"], fb["median"]
        def line(name, va, vb):
            if va is None or vb is None:
                print(f"  {name:<28} {va} -> {vb}"); return
            d = (vb - va) / va * 100 if va else 0.0
            print(f"  {name:<28} {va:>9.1f} -> {vb:>9.1f} ms   ({d:+.0f}%)")
        print(f"\n[{fa['wav']}]")
        line("total", ma["total_ms"], mb["total_ms"])
        line("decode", ma["decode_ms"], mb["decode_ms"])
        for lab in ma["models"]:
            if lab in mb["models"]:
                A, B = ma["models"][lab], mb["models"][lab]
                line(f"{lab} load", A.get("load_ms"), B.get("load_ms"))
                line(f"{lab} predict", A.get("predict_ms"), B.get("predict_ms"))
                for k, nm in (("feat_mel_ms", "feat mel"), ("feat_tab_ms", "feat tab"),
                              ("mc_fwd_ms", "mc_fwd")):
                    if A.get(k) is not None and B.get(k) is not None:
                        line(f"{lab} {nm}", A[k], B[k])


if __name__ == "__main__":
    main()
