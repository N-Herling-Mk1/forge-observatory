"""
FORGE · phonon · inference (mk1 point predictions).

Lazy-loaded by the dashboard's /api/phonon/* endpoints — imports the torch/e3nn/ase
stack only when a prediction is actually requested, so the app boots without it.

Builds the graph for ONE requested material on demand (not all 1524), runs the frozen
mk1 checkpoint, and returns the predicted 51-bin DOS alongside the DFPT ground truth
(which we have for every dataset material) plus the reproduction metrics.

mk2 will extend predict_index() with a credible-interval band; the return shape already
leaves room for it (`band` is None until then).
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

_STATE = {"model": None, "ckpt": None, "raw": None, "enc": None, "freq": None}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]          # projects/phonon


def _pick_ckpt(name=None) -> Path:
    runs = _root() / "runs"
    if name:
        p = runs / name
        if p.exists():
            return p
    for pref in ("e3nn_repro.torch",):                  # preferred mk1 run
        if (runs / pref).exists():
            return runs / pref
    cks = sorted(runs.glob("*.torch"))
    if not cks:
        raise FileNotFoundError(f"no checkpoint (*.torch) in {runs}")
    return cks[0]


def _ensure(ckpt_name=None):
    """Load (and cache) the model, raw dataset dict, and encodings."""
    import torch
    from . import data as D
    from .model import build_model

    if _STATE["raw"] is None:
        with open(_root() / "data" / "raw" / "phdos_e3nn_len51max1000_fwin101ord3.pkl", "rb") as f:
            _STATE["raw"] = pickle.load(f)
        _STATE["enc"] = D.build_encodings()
        _STATE["freq"] = np.asarray(_STATE["raw"]["phfre"], float)

    ckpt = _pick_ckpt(ckpt_name)
    if _STATE["model"] is None or _STATE["ckpt"] != str(ckpt):
        torch.set_default_dtype(D.DEFAULT_DTYPE)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        mk = ck.get("model_kwargs", {})
        m = build_model(mk, num_neighbors=mk.get("num_neighbors", 1.0))
        m.load_state_dict(ck["state"]); m.eval()
        _STATE["model"] = m; _STATE["ckpt"] = str(ckpt)
    return _STATE


def _forward(graph):
    import torch
    import torch_geometric as tg
    from . import data as D
    s = _STATE
    batch = next(iter(tg.loader.DataLoader([graph], batch_size=1)))
    with torch.no_grad():
        pred = s["model"](batch).cpu().numpy()[0]       # (51,)
    return pred


def predict_index(idx, ckpt_name=None):
    """Predict the DOS for dataset material `idx`; compare to its DFPT truth."""
    from . import data as D
    from . import metrics as M
    s = _ensure(ckpt_name)
    raw = s["raw"]; freq = s["freq"]
    atoms = D.cif_to_atoms(raw["cif"][idx])
    graph = D.build_graph(atoms, raw["phdos"][idx], s["enc"], max_radius=5.0)
    pred = _forward(graph)
    truth = np.asarray(raw["phdos"][idx], float)

    P, T = pred[None, :], truth[None, :]
    obp = float(M.omega_bar(P, freq)[0]); obt = float(M.omega_bar(T, freq)[0])
    return {
        "idx": int(idx),
        "id": str(raw["material_id"][idx]),
        "freq": freq.tolist(),
        "pred": pred.tolist(),
        "truth": truth.tolist(),
        "band": None,                                   # mk2 credible interval slots here
        "omega_bar_pred": round(obp, 1),
        "omega_bar_true": round(obt, 1),
        "rel_err": round(abs(obp - obt) / max(abs(obt), 1e-9), 4),
        "js": round(float(M.js_divergence(P, T)[0]), 4),
        "emd_cm": round(float(M.emd1d(P, T, freq)[0]), 2),
        "ckpt": Path(s["ckpt"]).name,
    }


def predict_cif(cif_text, ckpt_name=None):
    """Predict the DOS for a pasted CIF (no ground truth)."""
    import io
    from ase.io import read as ase_read
    from . import data as D
    s = _ensure(ckpt_name)
    atoms = ase_read(io.StringIO(cif_text), format="cif")
    dummy = np.zeros(51, float)
    graph = D.build_graph(atoms, dummy, s["enc"], max_radius=5.0)
    pred = _forward(graph)
    from . import metrics as M
    obp = float(M.omega_bar(pred[None, :], s["freq"])[0])
    return {"freq": s["freq"].tolist(), "pred": pred.tolist(), "truth": None,
            "band": None, "omega_bar_pred": round(obp, 1),
            "formula": atoms.get_chemical_formula(), "ckpt": Path(s["ckpt"]).name}


def status():
    """Which checkpoint will serve, without loading the torch stack if possible."""
    try:
        ckpt = _pick_ckpt()
        return {"ready": True, "ckpt": ckpt.name}
    except FileNotFoundError as e:
        return {"ready": False, "reason": str(e)}
