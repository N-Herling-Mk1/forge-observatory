"""Model registry — every train run is saved as an immutable, versioned bundle and
indexed here, so you can keep N models, compare them, and SELECT one as the active
model for song drop-in inference and (later) FORGE.

Layout (projects/genre/models/):
    registry.json          index + `selected` pointer (this module owns it)
    <family>__<ts>/        one immutable bundle per run (weights, arch, …, metrics)
    <family>/              legacy flat slot — auto-migrated into the index

An *entry* is one saved bundle. `selected` is the id every downstream consumer
(predict, FORGE) resolves to. The index is rebuilt from disk on demand (`scan`),
so it can never drift from the bundles that actually exist; fields the scan can't
recompute (git_sha, run_id) are carried over by id.
"""
from __future__ import annotations
import json, time
from pathlib import Path

REGISTRY = "registry.json"


# ------------------------------------------------------------------ id helpers
def entry_id(family: str, ts: str) -> str:
    return f"{family}__{ts}"


def family_of(eid: str) -> str:
    return eid.split("__", 1)[0]


def _is_bundle(d: Path) -> bool:
    return d.is_dir() and (d / "weights.pt").exists() and (d / "arch.json").exists()


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------------------------------------------- comparison view
def summarize(metrics: dict, metrics_full: dict | None = None) -> dict:
    """Pull the cross-model scorecard from a bundle's metrics.json (+ optional
    metrics_full.json for calibration). Everything a 'which model do I pick'
    decision needs, in one flat dict."""
    metrics = metrics or {}
    cv = metrics.get("cv") or {}
    val = metrics.get("val") or {}
    test = metrics.get("test") or {}
    acc = metrics.get("acceptance") or {}
    ece = None
    if metrics_full:
        ece = (((metrics_full.get("splits") or {}).get("test") or {})
               .get("calibration") or {}).get("ece")
    return {
        "val_acc": val.get("accuracy"), "val_macro_f1": val.get("macro_f1"),
        "test_acc": test.get("accuracy"), "test_macro_f1": test.get("macro_f1"),
        "cv_mean": cv.get("mean"), "cv_std": cv.get("std"),
        "test_ece": ece,
        "passed": acc.get("passed"),
        "accept_min": acc.get("accept_min"),
        "target": acc.get("target_val_accuracy"),
    }


# ------------------------------------------------------------------- index i/o
def load_index(models_dir) -> dict:
    idx = _read_json(Path(models_dir) / REGISTRY) or {}
    idx.setdefault("selected", None)
    idx.setdefault("entries", [])
    return idx


def save_index(models_dir, idx) -> None:
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY).write_text(json.dumps(idx, indent=2), encoding="utf-8")


def _entry_from_dir(d: Path) -> dict | None:
    if not _is_bundle(d):
        return None
    eid = d.name
    return {
        "id": eid,
        "family": family_of(eid),
        "dir": str(d).replace("\\", "/"),
        "created": int(d.stat().st_mtime),
        "summary": summarize(_read_json(d / "metrics.json") or {},
                             _read_json(d / "metrics_full.json")),
        "has_metrics_full": (d / "metrics_full.json").exists(),
    }


# ----------------------------------------------------------------------- scan
def scan(models_dir, write: bool = True) -> dict:
    """Rebuild the index from whatever bundle dirs exist on disk. Preserves the
    current `selected` if it still resolves, else auto-selects the newest.
    Idempotent — safe to call on every dashboard load. Carries over git_sha /
    run_id (which the scan can't recompute) by id."""
    root = Path(models_dir)
    prior = {e["id"]: e for e in load_index(root).get("entries", [])}
    sel = load_index(root).get("selected")

    entries = []
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            e = _entry_from_dir(d)
            if e is None:
                continue
            for k in ("git_sha", "run_id"):
                if k not in e and prior.get(e["id"], {}).get(k):
                    e[k] = prior[e["id"]][k]
            entries.append(e)
    entries.sort(key=lambda e: e["created"], reverse=True)

    ids = {e["id"] for e in entries}
    if sel not in ids:
        sel = entries[0]["id"] if entries else None
    idx = {"selected": sel, "entries": entries,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if write:
        save_index(root, idx)
    return idx


# ------------------------------------------------------------------- mutations
def register(models_dir, bundle_dir, *, git_sha=None, run_id=None,
             select: str = "if_none") -> dict:
    """Add/refresh the entry for `bundle_dir` and persist. `select`:
      'if_none' (default) → make it active only if nothing is selected yet
      'always'            → force it active
      'never'             → just register, leave selection alone
    """
    root = Path(models_dir)
    idx = scan(root, write=False)
    e = _entry_from_dir(Path(bundle_dir))
    if e is None:
        raise ValueError(f"not a bundle dir (need weights.pt + arch.json): {bundle_dir}")
    if git_sha:
        e["git_sha"] = git_sha
    if run_id:
        e["run_id"] = run_id
    idx["entries"] = [x for x in idx["entries"] if x["id"] != e["id"]] + [e]
    idx["entries"].sort(key=lambda x: x["created"], reverse=True)
    if select == "always" or (select == "if_none" and not idx.get("selected")):
        idx["selected"] = e["id"]
    save_index(root, idx)
    return idx


def select(models_dir, entry_id) -> dict:
    """Set the active model. Raises if the id isn't in the registry."""
    idx = scan(models_dir, write=False)
    if entry_id not in {e["id"] for e in idx["entries"]}:
        raise ValueError(f"no such model in registry: {entry_id}")
    idx["selected"] = entry_id
    save_index(models_dir, idx)
    return idx


# ----------------------------------------------------------------- resolution
def selected_dir(models_dir, fallback_family: str | None = None) -> str | None:
    """Resolve the selected bundle dir for downstream consumers (predict / FORGE).
    Falls back to legacy flat models/<family>, then the newest bundle."""
    root = Path(models_dir)
    idx = scan(root, write=False)
    by_id = {e["id"]: e for e in idx["entries"]}
    sel = idx.get("selected")
    if sel and sel in by_id:
        return by_id[sel]["dir"]
    if fallback_family and _is_bundle(root / fallback_family):
        return str(root / fallback_family).replace("\\", "/")
    return idx["entries"][0]["dir"] if idx["entries"] else None
