"""FORGE per-experiment mini-stack — backend (canonical template).

Reads the artifacts an experiment emits and serves them to the TRON-Ares
dashboard. Trains nothing. Two EDA snapshots per experiment:

    <root>/data/<phase>/eda_stats.json        phase in {before, after}
    <root>/eda/figures/<phase>/*.png

'before' is the pre-fix compendium; 'after' is regenerated after the
fix/imputation pass. The dashboard toggles between them.

Dual context — ONE file, two homes:
  • In the repo, this lives at projects/<exp>/app/ and reads its SIBLINGS
    (../data, ../eda) so the app and the training tier share one source of truth.
  • In an exported standalone bundle, server.py sits at the bundle root with
    data/ and eda/ beside it.
The root is auto-detected below, so the same file works in both.

Clone discipline: copy this folder unchanged into a new experiment. The
experiment name is auto-derived from the folder; keep route names + eda_stats.json
keys identical across genre/phonon/atlas so the eventual merge stays mechanical.

    pip install flask
    python server.py            # -> http://127.0.0.1:5000
"""
from __future__ import annotations
import json, os
from pathlib import Path
from flask import Flask, jsonify, send_file, send_from_directory, request, abort

import config

APP_DIR = Path(__file__).resolve().parent
# self-contained bundle: data/ sits beside server.py; repo: data/ is in the parent
EXP_ROOT = APP_DIR if (APP_DIR / "data").exists() else APP_DIR.parent
EXPERIMENT = config.EXPERIMENT or EXP_ROOT.name
DATA = EXP_ROOT / "data"
FIGS = EXP_ROOT / "eda" / "figures"
RUNS = EXP_ROOT / "runs"
MODELS = EXP_ROOT / "models"
REPO_ROOT = EXP_ROOT.parent.parent          # projects/<exp> -> repo root (for _shared)

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.after_request
def _no_cache(resp):
    # dev convenience: never serve stale page/app code (figures are cache-busted by ?v=)
    if resp.mimetype in ("text/html", "text/css", "application/javascript", "text/javascript"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp


def _phase():
    p = request.args.get("phase", config.DEFAULT_PHASE)
    return p if p in config.PHASES else config.DEFAULT_PHASE


def _stats(phase):
    return DATA / phase / "eda_stats.json"


def _fig_count(phase):
    d = FIGS / phase
    return len(list(d.glob("*.png"))) if d.exists() else 0


def _logo():
    """Project logo dropped into static/assets/ — any image that isn't a FORGE brand
    mark or part of the favicon/app-icon family (those are tab/OS icons, not the logo)."""
    d = APP_DIR / "static" / "assets"
    brand = {"forge-mark.svg", "favicon.svg"}
    icon_prefixes = ("favicon", "icon-", "apple-touch")   # the browser/OS icon set
    if d.exists():
        imgs = [p for p in d.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
                and p.name not in brand
                and not p.name.lower().startswith(icon_prefixes)]
        imgs.sort(key=lambda p: (p.suffix.lower() == ".svg", p.name))  # prefer raster
        if imgs:
            return f"/static/assets/{imgs[0].name}"
    return None


# ---- routes (KEEP NAMES IDENTICAL ACROSS EXPERIMENTS) -----------------------
@app.get("/")
def index():
    return send_file(APP_DIR / "templates" / "index.html")


@app.get("/welcome")
def welcome():
    return send_file(APP_DIR / "templates" / "welcome.html")


@app.get("/experiment")
def experiment_page():
    return send_file(APP_DIR / "templates" / "experiment.html")


@app.get("/eda")
def eda_page():
    return send_file(APP_DIR / "templates" / "eda.html")


@app.get("/glossary")
def glossary_page():
    return send_file(APP_DIR / "templates" / "glossary.html")


# phonon-specific panels (no genre analogue yet) — serve the themed stub until built
@app.get("/derived")
def derived_page():
    tpl = APP_DIR / "templates" / "derived.html"
    return send_file(tpl if tpl.exists() else APP_DIR / "templates" / "stub.html")


@app.get("/alloy")
def alloy_page():
    tpl = APP_DIR / "templates" / "alloy.html"
    return send_file(tpl if tpl.exists() else APP_DIR / "templates" / "stub.html")


@app.get("/data")
def data_page():
    return send_file(APP_DIR / "templates" / "data.html")


@app.get("/train")
def train_page():
    # the Runs / metrics observatory ("kitchen sink") — falls back to the stub if absent
    tpl = APP_DIR / "templates" / "train.html"
    return send_file(tpl if tpl.exists() else APP_DIR / "templates" / "stub.html")


@app.get("/infer")
def infer_page():
    return send_file(APP_DIR / "templates" / "infer.html")


# ---- phonon DOS Explorer endpoints -----------------------------------------
def _phonon_predict_module():
    """Lazy-import projects/phonon/src/predict.py (pulls the torch stack only here)."""
    import sys, importlib
    repo_root = EXP_ROOT.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module(f"projects.{EXPERIMENT}.src.predict")


@app.get("/api/phonon/catalog")
def phonon_catalog():
    """Element coverage map + material catalog for the periodic table (no torch)."""
    f = EXP_ROOT / "data" / "phonon_catalog.json"
    if not f.exists():
        return jsonify(error="phonon_catalog.json missing — run the catalog builder"), 404
    return send_file(f)


@app.get("/api/phonon/status")
def phonon_status():
    try:
        return jsonify(_phonon_predict_module().status())
    except Exception as e:
        return jsonify(ready=False, reason=str(e)), 200


@app.get("/api/phonon/predict")
def phonon_predict():
    """Predict the DOS for dataset material ?idx=N against its DFPT truth."""
    try:
        idx = int(request.args.get("idx", ""))
    except ValueError:
        return jsonify(error="pass ?idx=<int>"), 400
    try:
        mod = _phonon_predict_module()
    except Exception as e:
        return jsonify(error=f"inference stack unavailable: {e}"), 501
    try:
        return jsonify(mod.predict_index(idx, ckpt_name=request.args.get("ckpt")))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/phonon/predict_cif")
def phonon_predict_cif():
    """Predict the DOS for a pasted CIF (no ground truth)."""
    cif = (request.get_json(silent=True) or {}).get("cif") or request.data.decode("utf-8", "ignore")
    if not cif.strip():
        return jsonify(error="empty CIF"), 400
    try:
        mod = _phonon_predict_module()
    except Exception as e:
        return jsonify(error=f"inference stack unavailable: {e}"), 501
    try:
        return jsonify(mod.predict_cif(cif, ckpt_name=request.args.get("ckpt")))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/predict")
def api_predict():
    """Song drop-in -> per-genre scores + uncertainty for EVERY model in the family
    (mk1/mk2/mk3), returned as {models:[...]} for the three-graph compare view.
    Generic across experiments: lazy-imports projects/<exp>/src/predict.py and calls
    its predict_upload_multi(...). Lazy so the dashboard boots without the torch stack."""
    import sys, os, tempfile, importlib
    f = request.files.get("audio")
    if f is None or not f.filename:
        return jsonify(error="no audio file uploaded (field 'audio')"), 400
    repo_root = EXP_ROOT.parent.parent          # projects/<exp> -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module(f"projects.{EXPERIMENT}.src.predict")
    except Exception as e:
        return jsonify(error=f"no predict module for '{EXPERIMENT}': {e}"), 501
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name); tmp.close()
    try:
        out = mod.predict_upload_multi(tmp.name, EXP_ROOT)
    except Exception as e:
        return jsonify(error=f"prediction failed: {e}"), 500
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass
    code = 404 if isinstance(out, dict) and out.get("error") else 200
    return jsonify(out), code


@app.get("/model")
def model_page():
    return send_file(APP_DIR / "templates" / "model.html")


@app.get("/api/config")
def cfg():
    return jsonify(experiment=EXPERIMENT, phases=config.PHASES,
                   default_phase=config.DEFAULT_PHASE, logo=_logo(),
                   available={p: _stats(p).exists() for p in config.PHASES})


@app.get("/api/health")
def health():
    return jsonify(experiment=EXPERIMENT, ok=True,
                   figures={p: _fig_count(p) for p in config.PHASES},
                   eda_present={p: _stats(p).exists() for p in config.PHASES})


@app.get("/api/eda")
def eda():
    phase = _phase()
    fp = _stats(phase)
    if not fp.exists():
        return jsonify(error=f"no '{phase}' EDA yet — run eda/run_eda.py --phase {phase}",
                       experiment=EXPERIMENT, phase=phase), 404
    return jsonify(json.loads(fp.read_text(encoding="utf-8")))


@app.get("/api/glossary")
def glossary_data():
    """Static reference content (phase-independent): <root>/data/glossary.json."""
    fp = DATA / "glossary.json"
    if not fp.exists():
        return jsonify(error="no glossary.json in data/", experiment=EXPERIMENT), 404
    return jsonify(json.loads(fp.read_text(encoding="utf-8")))


@app.get("/api/provenance")
def provenance_data():
    """Canonical data-provenance record (phase-independent): <root>/data/provenance.json.
    The Data panel bakes a copy for offline use but prefers this when present."""
    fp = DATA / "provenance.json"
    if not fp.exists():
        return jsonify(error="no provenance.json in data/", experiment=EXPERIMENT), 404
    return jsonify(json.loads(fp.read_text(encoding="utf-8")))


@app.get("/api/data")
def data_artifacts():
    """Bundle the data-pipeline artifacts for the Data panel (phase-independent):
    leakage measurement, the recorded split, and the split-test results. Each piece
    is optional — the panel degrades gracefully and tells you which script to run."""
    def _load(name):
        fp = DATA / name
        if fp.exists():
            try:
                return json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    manifest = _load("manifest.json") or {}
    return jsonify(
        experiment=EXPERIMENT,
        leakage=_load("leakage.json"),
        split=manifest.get("split"),
        tests=_load("splits_report.json"),
        hints={
            "leakage": "python projects/genre/src/leakage_report.py --data-root projects/genre/data/raw",
            "tests": "python projects/genre/src/split_tests_report.py",
            "split": "python projects/genre/src/sanity_check.py --data-root projects/genre/data/raw --write-manifest",
        },
    )


def _repo_on_path():
    """Make `_shared` importable (usage ledger). Lazy + idempotent so the
    dashboard still boots if the training stack isn't installed."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@app.get("/api/runs")
def runs():
    """Training tier — the kitchen sink. One entry per run.json (full RunRecord:
    config, per-epoch curve, final_metrics, embedded compute). Each is enriched
    in-place with a `usage` block (energy/carbon/$ derived from its compute record)
    so the panel never has to join across files. Empty until train.py emits."""
    out = []
    try:
        _repo_on_path()
        from _shared.usage import estimate_usage, UsageAssumptions
        asmp = UsageAssumptions.load(str(REPO_ROOT / "_shared" / "usage_assumptions.yaml"))
    except Exception:
        estimate_usage = None; asmp = None
    if RUNS.exists():
        for rj in sorted(RUNS.glob("*/*/run.json")):
            try:
                rec = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            rec["run_id"] = str(rj.parent.relative_to(RUNS)).replace("\\", "/")
            if estimate_usage is not None and isinstance(rec.get("compute"), dict):
                try:
                    rec["usage"] = estimate_usage(rec["compute"], asmp).as_row()
                except Exception:
                    rec["usage"] = None
            out.append(rec)
    out.sort(key=lambda r: r.get("created", 0), reverse=True)
    return jsonify(runs=out, experiment=EXPERIMENT, count=len(out))


@app.get("/api/usage")
def usage_ledger():
    """Roll-up energy/carbon/$ across every run (compute.json). Totals +
    per-experiment breakdown + per-run rows, with measured-vs-modeled basis.
    Reuses _shared/usage.aggregate_runs so the panel and the CLI agree."""
    if not RUNS.exists():
        return jsonify(error="no runs/ yet — run train.py", experiment=EXPERIMENT,
                       totals={}, runs=[]), 404
    try:
        _repo_on_path()
        from _shared.usage import aggregate_runs, UsageAssumptions
        asmp = UsageAssumptions.load(str(REPO_ROOT / "_shared" / "usage_assumptions.yaml"))
        return jsonify(aggregate_runs(str(RUNS), asmp))
    except Exception as e:
        return jsonify(error=f"usage ledger unavailable: {e}", experiment=EXPERIMENT), 500


@app.get("/api/model_card")
def model_card():
    """Frozen-bundle metrics — the canonical model card(s). Reads each
    models/<name>/{metrics,arch,label_map}.json. Available even with zero runs,
    so the panel always has the delivered Model-1 scorecard to show."""
    cards = []
    if MODELS.exists():
        for md in sorted(p for p in MODELS.iterdir() if p.is_dir()):
            def _load(name):
                fp = md / name
                if fp.exists():
                    try:
                        return json.loads(fp.read_text(encoding="utf-8"))
                    except Exception:
                        return None
                return None
            card = {"name": md.name, "metrics": _load("metrics.json"),
                    "arch": _load("arch.json"), "label_map": _load("label_map.json")}
            if card["metrics"] or card["arch"]:
                cards.append(card)
    return jsonify(cards=cards, experiment=EXPERIMENT, count=len(cards))


def _registry():
    """Lazy import of the genre model registry (projects/<exp>/src/registry.py)."""
    _repo_on_path()
    import importlib
    return importlib.import_module(f"projects.{EXPERIMENT}.src.registry")


@app.get("/api/registry")
def registry_list():
    """The model registry — every saved run-bundle + the `selected` (active) id.
    Drives the Model panel's comparison + selection UI. Read-only (no disk write)."""
    try:
        idx = _registry().scan(str(MODELS), write=False)
        return jsonify(idx)
    except Exception as e:
        return jsonify(error=f"registry unavailable: {e}", selected=None, entries=[]), 500


@app.post("/api/model/select")
def registry_select():
    """Set the active model. Body: {"id": "<family>__<ts>"}. The selected bundle is
    what song drop-in inference and FORGE resolve to."""
    eid = (request.get_json(silent=True) or {}).get("id") or request.form.get("id")
    if not eid:
        return jsonify(error="missing 'id'"), 400
    try:
        idx = _registry().select(str(MODELS), eid)
        return jsonify(ok=True, selected=idx["selected"])
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 404
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ----------------------------------------------------------------------- FORGE
_FORGE = {}

def _forge_lap():
    """Load + cache the last-layer Laplace for the SELECTED bundle (keyed by dir +
    weights mtime). The heavy load (weights + φ) happens once; τ-knob recomputes
    are then O(d) in the browser off the payload."""
    _repo_on_path()
    import importlib
    reg = importlib.import_module(f"projects.{EXPERIMENT}.src.registry")
    bundle = reg.selected_dir(str(MODELS))
    if not bundle:
        raise RuntimeError("no active model — select one in the Model panel")
    wp = Path(bundle) / "weights.pt"
    key = (bundle, wp.stat().st_mtime if wp.exists() else 0)
    if _FORGE.get("key") != key:
        llla = importlib.import_module(f"projects.{EXPERIMENT}.src.bayes.llla")
        _FORGE.clear()
        _FORGE["key"] = key
        _FORGE["lap"] = llla.from_bundle(bundle)
        _FORGE["bundle"] = bundle
    return _FORGE["lap"], _FORGE["bundle"]


@app.get("/forge")
def forge_page():
    return send_file(APP_DIR / "templates" / "forge.html")


@app.get("/api/forge/meta")
def forge_meta():
    """Everything the FORGE tab needs to set up: genres, the real architecture (for
    the stylized model), the example index (predicted genre + input-variance per
    train example, so the UI can pick interesting ones), τ defaults, and the
    selected model's metric card."""
    try:
        import numpy as np
        lap, bundle = _forge_lap()
        bp = Path(bundle)
        lm = json.loads((bp / "label_map.json").read_text(encoding="utf-8"))
        inv = {v: k for k, v in lm.items()}
        genres = [inv[i] for i in range(len(inv))]
        arch = json.loads((bp / "arch.json").read_text(encoding="utf-8"))
        metrics = {}
        mf = bp / "metrics_full.json"
        mj = bp / "metrics.json"
        if mj.exists():
            metrics = json.loads(mj.read_text(encoding="utf-8"))
        ece = None
        confusion = None
        conf_labels = None
        if mf.exists():
            mfj = json.loads(mf.read_text(encoding="utf-8"))
            test_blk = (mfj.get("splits") or {}).get("test") or {}
            ece = (test_blk.get("calibration") or {}).get("ece")
            cf = test_blk.get("confusion") or {}
            confusion = cf.get("row_normalized")     # recall view, the aleatoric-overlap signal
            conf_labels = cf.get("labels")

        phi = lap.phi_train
        n = int(phi.shape[0]) if phi is not None else 0
        predicted, input_var = [], []
        if n:
            logits = phi @ lap.W.T + lap.b
            predicted = logits.argmax(1).astype(int).tolist()
            tau0 = max(1e-3, 1e-2 * float(lap.Lambda.max()))
            input_var = np.round(lap.input_variance(phi, tau0), 6).tolist()
        else:
            tau0 = 1.0

        card = {
            "name": Path(bundle).name,
            "cv": metrics.get("cv"), "val": metrics.get("val"),
            "test": metrics.get("test"), "acceptance": metrics.get("acceptance"),
            "test_ece": ece,
        }
        return jsonify(genres=genres, arch=arch, n_examples=n,
                       predicted=predicted, input_var=input_var,
                       tau_default=tau0,
                       tau_range=[max(1e-4, 1e-4 * float(lap.Lambda.max())), float(lap.Lambda.max())],
                       confusion=confusion, conf_labels=conf_labels,
                       card=card, bundle=Path(bundle).name)
    except Exception as e:
        return jsonify(error=f"forge meta unavailable: {e}"), 500


@app.get("/api/forge/datasweep")
def forge_datasweep():
    """Data-fraction sweep payload for one example: per fraction, the rebuilt-H
    eigenvalues + this example's projections, so the browser draws the learning
    curve live for any τ. No retraining — H is rebuilt from subsets of cached φ."""
    try:
        lap, _ = _forge_lap()
        if lap.phi_train is None or not lap.phi_train.shape[0]:
            return jsonify(error="no cached φ for this bundle"), 404
        idx = int(request.args.get("index", 0))
        idx = max(0, min(idx, lap.phi_train.shape[0] - 1))
        return jsonify(fractions=lap.datasweep_payload(lap.phi_train[idx]),
                       d=int(lap.d), n_train=int(lap.phi_train.shape[0]), index=idx)
    except Exception as e:
        return jsonify(error=f"forge datasweep unavailable: {e}"), 500


@app.get("/api/forge/hmc")
def forge_hmc():
    """Run last-layer HMC (gold standard) and return its predictive for one example,
    alongside the LLLA predictive at the same τ for the overlay. The chain depends
    only on (bundle, τ, sampler params) — not the example — so it's cached and reused
    across examples; changing τ re-samples. Needs y_train.npy in the bundle."""
    try:
        import importlib
        import numpy as np
        lap, bundle = _forge_lap()
        if lap.phi_train is None or not lap.phi_train.shape[0]:
            return jsonify(error="no cached φ for this bundle"), 404
        idx = int(request.args.get("index", 0))
        idx = max(0, min(idx, lap.phi_train.shape[0] - 1))
        tau = float(request.args.get("tau", max(1e-3, 1e-2 * float(lap.Lambda.max()))))
        n_samples = max(50, min(int(request.args.get("samples", 250)), 1500))
        n_leap = max(5, min(int(request.args.get("leapfrog", 15)), 60))

        hmc_mod = importlib.import_module(f"projects.{EXPERIMENT}.src.bayes.hmc")
        key = (bundle, round(tau, 6), n_samples, n_leap)
        if _FORGE.get("hmc_key") != key:
            H = hmc_mod.from_bundle(bundle, tau=tau)
            theta0 = getattr(H, "_theta0", None)
            samples, info = H.sample(n_samples=n_samples, n_warmup=max(150, n_samples // 2),
                                     n_leapfrog=n_leap, theta0=theta0, seed=0)
            _FORGE["hmc_key"] = key
            _FORGE["hmc_H"] = H
            _FORGE["hmc_samples"] = samples
            _FORGE["hmc_info"] = info
        H, samples, info = _FORGE["hmc_H"], _FORGE["hmc_samples"], _FORGE["hmc_info"]

        bp = Path(bundle)
        lm = json.loads((bp / "label_map.json").read_text(encoding="utf-8"))
        inv = {v: k for k, v in lm.items()}
        genres = [inv[i] for i in range(len(inv))]

        Ppred = H.predictive(lap.phi_train[idx], samples)          # [S, C] in chain order
        S = Ppred.shape[0]
        stepd = max(1, S // 400)
        pred = Ppred[::stepd]                                       # cap payload for the animation
        llla = lap.predict_posterior(lap.phi_train[idx], tau, method="mc", n_samples=2000)
        # MAP-anchor: the trained model's own prediction for this example (HMC's top
        # class should sit near this if the chain converged on the confident class)
        import numpy as _np
        _z = lap.W @ lap.phi_train[idx] + lap.b
        _z = _z - _z.max()
        _e = _np.exp(_z)
        map_pred = _e / _e.sum()

        return jsonify(
            genres=genres, index=idx, tau=tau,
            hmc={"pred": pred.tolist(),
                 "mean": Ppred.mean(0).tolist(), "sigma": Ppred.std(0).tolist(),
                 "accept": float(info["accept"]), "step": float(info["step_size"]),
                 "n_samples": int(S), "n_leapfrog": int(info["n_leapfrog"]),
                 "n_warmup": int(info["n_warmup"]),
                 "rhat": (None if info["rhat"] != info["rhat"] else float(info["rhat"])),
                 "ess": float(info["ess"]), "n_divergences": int(info["n_divergences"]),
                 "lp_trace": info["lp_trace"], "warmup_frac": float(info["warmup_frac"])},
            llla={"mean": llla["mean"].tolist(), "sigma": llla["sigma"].tolist()},
            map_pred=map_pred.tolist(),
        )
    except FileNotFoundError as e:
        return jsonify(error=str(e), needs_y_train=True), 400
    except Exception as e:
        return jsonify(error=f"hmc unavailable: {e}"), 500


@app.get("/api/forge/posterior")
def forge_posterior():
    """Posterior payload for one example: the per-input eigen-projections + Σ_A + μ,
    so the browser recomputes the τ-knob posterior live with no further calls."""
    try:
        lap, bundle = _forge_lap()
        if lap.phi_train is None or not lap.phi_train.shape[0]:
            return jsonify(error="no cached φ for this bundle"), 404
        idx = int(request.args.get("index", 0))
        idx = max(0, min(idx, lap.phi_train.shape[0] - 1))
        bp = Path(bundle)
        lm = json.loads((bp / "label_map.json").read_text(encoding="utf-8"))
        inv = {v: k for k, v in lm.items()}
        genres = [inv[i] for i in range(len(inv))]
        payload = lap.posterior_payload(lap.phi_train[idx], genres)
        payload["index"] = idx
        return jsonify(payload)
    except Exception as e:
        return jsonify(error=f"forge posterior unavailable: {e}"), 500


@app.get("/api/metrics_full")
def metrics_full():
    """The full metrics wall (src/metrics.py output) for the SELECTED model — the
    Training panel shows the active bundle in depth. Falls back to globbing
    models/* if the registry isn't available. 404 (gracefully) until metrics.py runs."""
    candidates = []
    try:
        sd = _registry().selected_dir(str(MODELS))
        if sd:
            candidates = [Path(sd)]
    except Exception:
        candidates = []
    if not candidates and MODELS.exists():
        candidates = [p for p in MODELS.iterdir() if p.is_dir()]
    cards = []
    for md in candidates:
        fp = md / "metrics_full.json"
        if fp.exists():
            try:
                cards.append({"name": md.name, "full": json.loads(fp.read_text(encoding="utf-8"))})
            except Exception:
                pass
    if not cards:
        return jsonify(cards=[], experiment=EXPERIMENT, count=0,
                       hint="python projects/genre/src/metrics.py --config projects/genre/configs/beardown.yaml --bundle <selected bundle dir>"), 404
    return jsonify(cards=cards, experiment=EXPERIMENT, count=len(cards))


@app.get("/figures/<phase>/<path:name>")
def figures(phase, name):
    if phase not in config.PHASES:
        abort(404)
    d = FIGS / phase
    if not d.exists():
        abort(404)
    return send_from_directory(d, name)


@app.get("/favicon.ico")
def favicon_ico():
    """Browsers auto-request /favicon.ico; serve the .ico beside the SVG."""
    ico = APP_DIR / "static" / "assets" / "favicon.ico"
    if ico.exists():
        return send_file(ico)
    abort(404)


def _free_port(host, start, attempts=50):
    """Return the first bindable port >= start (so genre/phonon/atlas can all run at
    once without colliding). Falls back to letting the OS choose if none in range."""
    import socket
    for p in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # last resort: any free port
        s.bind((host, 0))
        return s.getsockname()[1]


def _banner(port):
    C, A, R = "\033[36m", "\033[33m", "\033[0m"
    line = "═" * 50
    print(f"{C}╔{line}╗{R}")
    print(f"{C}║  FORGE mini-stack   experiment: {A}{EXPERIMENT:<17}{C}║{R}")
    print(f"{C}╚{line}╝{R}")
    print(f"  root         : {EXP_ROOT}")
    for p in config.PHASES:
        mark = "FOUND" if _stats(p).exists() else "—    "
        print(f"  EDA [{p:<6}] : {mark}   figures: {_fig_count(p)}")
    if port != config.PORT:
        print(f"  port {config.PORT} busy  -> using {port}")
    print(f"  serving      : http://{config.HOST}:{port}\n")


if __name__ == "__main__":
    # The reloader re-execs __main__; let the child reuse the parent's choice (via env)
    # so both halves agree on one port instead of each probing separately.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        port = int(os.environ.get("FORGE_PORT", config.PORT))
    else:
        port = _free_port(config.HOST, config.PORT)
        os.environ["FORGE_PORT"] = str(port)   # pin for the reloader child
        _banner(port)
    app.run(host=config.HOST, port=port, debug=config.DEBUG)
