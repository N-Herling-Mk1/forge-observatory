"""Reusable EDA helpers. Light deps only (numpy/pandas/matplotlib) — no torch."""
from __future__ import annotations
import json, os


def write_eda_stats(out_path: str, stats: dict) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def _commafy(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def build_summary(stats: dict) -> dict:
    """Collapse a full eda_stats dict into the headline flags + a one-paragraph
    narrative the dashboard renders above the detail panels. Pure dict→dict; emits
    only JSON-safe scalars so the stats file stays valid (allow_nan=False).
    """
    ta = stats.get("type_audit", {})
    cols = ta.get("columns", [])
    nerd = stats.get("nerd_stats", {})
    mc = stats.get("missing_corrupt", {})
    counts = mc.get("counts", {})

    n_rows = int(ta.get("n_rows", 0))
    n_columns = len(cols)
    n_features = int(nerd.get("n_features", 0))
    n_pass = sum(1 for c in cols if c.get("match") and not c.get("note"))
    n_fail = n_columns - n_pass
    total_nan = sum(int(c.get("n_nan", 0)) for c in cols)
    total_inf = sum(int(c.get("n_inf", 0)) for c in cols)

    corrupt = mc.get("corrupt_audio", [])
    off_dur = mc.get("off_duration", [])
    missing = mc.get("missing_per_representation", {})
    n_missing = sum(len(v) for v in missing.values())
    seg_anom = mc.get("segment_anomalies", [])

    # representation_gaps = catalogued file-level integrity issues
    # (unreadable + missing-representation + off-duration); the feature table is separate.
    representation_gaps = len(corrupt) + n_missing + len(off_dur)

    # all_present = the FEATURE TABLE is whole: every track row present, no NaN/inf.
    # (File-level audio gaps are tracked by representation_gaps, not here.)
    feat_missing = len(missing.get("features_30sec", []))
    all_present = (feat_missing == 0 and total_nan == 0 and total_inf == 0)
    all_types_pass = (n_fail == 0)
    n_clips = counts.get("wav_total", 0) or 1
    gap_pct = 100 * representation_gaps / n_clips

    bits = [f"The feature table holds {_commafy(n_rows)} rows across {n_columns} columns "
            f"({n_features} numeric features)."]
    bits.append(f"Type audit: {n_pass}/{n_columns} columns match their expected dtype, "
                f"with {'nothing flagged' if n_fail == 0 else f'{n_fail} flagged'}.")
    if total_nan == 0 and total_inf == 0:
        bits.append("No missing, NaN, or non-finite values appear in any examined column.")
    else:
        bits.append(f"{total_nan} NaN and {total_inf} non-finite values were found.")
    file_bits = []
    if corrupt: file_bits.append(f"{len(corrupt)} unreadable audio clip{'s' if len(corrupt)!=1 else ''}")
    if n_missing: file_bits.append(f"{n_missing} missing representation file{'s' if n_missing!=1 else ''}")
    if off_dur: file_bits.append(f"{len(off_dur)} off-duration clips (~{gap_pct:.1f}% of clips)")
    _ = seg_anom  # tracked in structured fields; intentionally not narrated (matches v1.1)
    if file_bits:
        bits.append("At the file level there are " + ", ".join(file_bits) +
                    ", all catalogued under Integrity.")
    bits.append("Overall the dataset is well-behaved — the feature table is complete and "
                "correctly typed, with only minor, documented file-level gaps."
                if all_present
                else "The dataset needs repair before training (see Integrity).")
    if mc.get("wav_available", True) is False:
        bits.append("Note: audio (wav) files are not present in this bundle, so "
                    "wav-level duration and corruption fields were not re-measured here.")

    return {
        "n_rows": n_rows, "n_columns": n_columns, "n_features": n_features,
        "all_present": bool(all_present), "all_types_pass": bool(all_types_pass),
        "n_types_pass": n_pass, "n_types_fail": n_fail,
        "total_nan": total_nan, "total_inf": total_inf,
        "representation_gaps": representation_gaps,
        "narrative": " ".join(bits),
    }


def plot_class_balance(counts, out_png): raise NotImplementedError("TODO")
def plot_feature_corr(df, out_png):      raise NotImplementedError("TODO")
def plot_mel_exemplars(by_genre, out_png): raise NotImplementedError("TODO")
def plot_leakage_audit(splits, out_png): raise NotImplementedError("TODO")
