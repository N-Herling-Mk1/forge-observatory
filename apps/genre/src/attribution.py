"""FORGE — Input-feature attribution (rung 2 of the attribution ladder).

Push the last-layer Laplace epistemic scale BACKWARD through the frozen backbone
onto NAMED inputs: "how much does mfcc4_var drive THIS prediction's epistemic
uncertainty." Names the latent features INFO 510's Table A3 could only rank
anonymously (see FORGE_input_feature_attribution_SPEC.txt §7).

The math (spec §2, rung 2)
--------------------------
The input's entire epistemic contribution flows through the LLLA input-variance
scalar (llla.input_variance):

    v(φ;τ) = Σᵢ (uᵢᵀφ)² / (λᵢ+τ) = φᵀ(H+τI)⁻¹φ          [H = ΦᵀΦ, Λ=λ, U cached]

v is a differentiable scalar of φ, and φ = f_backbone(x) is differentiable in the
inputs. So the per-feature epistemic SENSITIVITY is a plain chain rule:

    dv/dx_k = (dv/dφ)ᵀ (dφ/dx_k),     dv/dφ = 2·U·(proj/(λ+τ)) = 2(H+τI)⁻¹φ   (exact)

dv/dφ is closed-form (below, pure numpy); dφ/dx is ONE vector-Jacobian product
(backprop the cotangent g=dv/dφ through the frozen torch backbone) → dv/dx in a
single backward pass per example. It is a SENSITIVITY (saliency-style), not a
posterior over input weights — honest about what it is (spec §8).

τ factors out cleanly. With the projected Jacobian UtJ = Uᵀ(dφ/dx) cached as
M_ik = projᵢ·(UtJ)_ik, FORGE recomputes attribution for ANY τ in O(d·K):

    dv/dx_k(τ) = 2 Σᵢ M_ik / (λᵢ+τ)

so the attribution tab gets the same live-τ knob as every other FORGE tab.

Compute split (spec §3, §4 — Option 1)
--------------------------------------
The torch backward is INFERENCE-TIME on the already-trained frozen model, run ONCE
at bundle-write on the GPU box (`precompute`). It writes attribution.npz; FORGE then
reads + renders in pure numpy/JS, no torch in the serving path — exactly the
ggn_eig.npz pattern. Everything below `precompute` is torch-free and unit-testable
in the sandbox (like rrm.py); torch is imported lazily inside `precompute` only.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


# ============================================================================
# torch-free bundle self-description (spec §5 — the "design move to make now")
# ============================================================================
BASE_CAPABILITIES = ["posterior", "datasweep"]
BASE_FILES = ["weights.pt", "arch.json", "scaler.json", "label_map.json",
              "phi_train.npy", "y_train.npy", "ggn_eig.npz", "metrics.json"]


def build_inputs_manifest(loaded) -> dict:
    """inputs.json — NAMES the input axes so attribution bars are labeled. This is
    THE thing a generic FORGE must demand of any uploaded model (spec §5): without
    names you get sensitivities with no labels. Tabular names come from
    scaler["cols"] (the 57 audio features, in scaler order)."""
    rep = getattr(loaded, "representation", "fused")
    fused = rep in ("fused", "fused3")
    cols = list((getattr(loaded, "scaler", None) or {}).get("cols",
                getattr(loaded, "feature_cols", None) or []))
    man: dict = {"representation": rep, "fused": bool(fused)}
    if fused or rep in ("tab3", "tab30"):
        man["tabular"] = cols                                   # 57 names, scaler order
    if fused or rep == "image":
        sz = int(getattr(loaded, "image_size", None) or 128)
        man["image"] = {"shape": [sz, sz, 1], "kind": "log_mel"}
    return man


def bundle_capabilities(out_dir, has_y_train: bool = True,
                        has_attribution: bool | None = None) -> list[str]:
    """What this bundle can drive. Base (posterior/datasweep) always; hmc iff
    y_train.npy present; attribution iff attribution.npz + inputs.json present.
    A bundle missing a capability still loads — the tab just stays dark (spec §5)."""
    out = Path(out_dir)
    caps = list(BASE_CAPABILITIES)
    if has_y_train and (out / "y_train.npy").exists():
        caps.append("hmc")
    if has_attribution is None:
        has_attribution = (out / "attribution.npz").exists() and (out / "inputs.json").exists()
    if has_attribution:
        caps.append("attribution")
    return caps


def write_self_description(out_dir, loaded) -> dict:
    """Emit inputs.json + bundle.json. Torch-free. Call AFTER the base files exist
    so capability probing sees them; idempotent, so `precompute` re-calls it once
    attribution.npz lands (then 'attribution' joins the capability list)."""
    out = Path(out_dir)
    man = build_inputs_manifest(loaded)
    (out / "inputs.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    phi_p = out / "phi_train.npy"
    phi_dim = int(np.load(phi_p, mmap_mode="r").shape[1]) if phi_p.exists() else None
    desc = {
        "representation": man["representation"],
        "n_classes": len(getattr(loaded, "label_map", {})) or None,
        "phi_dim": phi_dim,
        "capabilities": bundle_capabilities(out, has_y_train=(out / "y_train.npy").exists()),
        "inputs": "inputs.json",
        "base_files": BASE_FILES,
        "optional_files": [f for f in ("attribution.npz", "inputs.json") if (out / f).exists()],
    }
    (out / "bundle.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    return desc


# ============================================================================
# numpy core — the attribution math (testable, torch-free)
# ============================================================================
def v_input(phi, lam, U, tau):
    """LLLA input-variance v(φ;τ)=Σᵢ(uᵢᵀφ)²/(λᵢ+τ). φ:[d] or [m,d] → scalar or [m].
    (Mirrors llla.LastLayerLaplace.input_variance so the two agree by construction.)"""
    proj = np.atleast_2d(np.asarray(phi, float)) @ np.asarray(U, float)     # [m,d]
    v = (proj ** 2 / (np.asarray(lam, float) + tau)).sum(axis=1)
    return v if v.shape[0] > 1 else float(v[0])


def dv_dphi(phi, lam, U, tau):
    """Exact gradient of v wrt φ:  dv/dφ = 2·U·(proj/(λ+τ)) = 2(H+τI)⁻¹φ, proj=Uᵀφ.
    This is the cotangent fed to the single backbone VJP. Pure numpy."""
    phi = np.asarray(phi, float).ravel()
    lam = np.asarray(lam, float); U = np.asarray(U, float)
    proj = U.T @ phi                                            # [d] = (uᵢᵀφ)
    return 2.0 * (U @ (proj / (lam + tau)))                     # [d]


def dv_dx_direct(J, dvdphi):
    """dv/dx = (dφ/dx)ᵀ (dv/dφ) = Jᵀ g.  J=[d,K] backbone Jacobian, g=[d]. The
    numpy analog of the torch VJP φ.backward(gradient=g) → exact dv/dx [K]."""
    return np.asarray(J, float).T @ np.asarray(dvdphi, float)


def dv_dx_from_M(M, lam, tau):
    """τ-live recompute:  dv/dx_k(τ) = 2 Σᵢ M_ik/(λᵢ+τ),  M_ik=projᵢ·(UᵀJ)_ik.
    M:[...,d,K] (per-genre or per-example), lam:[d] → [...,K]. Pure numpy, O(d·K)."""
    w = 1.0 / (np.asarray(lam, float) + tau)                    # [d]
    return 2.0 * np.tensordot(w, np.asarray(M, float), axes=([0], [-2]))   # [...,K]


def project_M(phi, J, U):
    """Build the τ-independent cache piece M=[d,K] for one example:
    M_ik = (Uᵀφ)_i · (Uᵀ J)_ik. With it, dv/dx(τ) needs no Jacobian recompute."""
    phi = np.asarray(phi, float).ravel(); U = np.asarray(U, float)
    proj = U.T @ phi                                            # [d]
    UtJ = U.T @ np.asarray(J, float)                            # [d,K]
    return proj[:, None] * UtJ                                  # [d,K]


def aggregate_per_genre(dv_dx, y, n_classes):
    """Per-genre mean signed and mean-|·| sensitivity → ([C,K], [C,K], counts[C]).
    Signed shows direction (does the feature raise or lower epistemic v); abs is
    the magnitude ranking that names Table A3's features."""
    dv_dx = np.atleast_2d(np.asarray(dv_dx, float)); y = np.asarray(y).ravel()
    C, K = int(n_classes), dv_dx.shape[1]
    signed = np.zeros((C, K)); absm = np.zeros((C, K)); cnt = np.zeros(C, int)
    for c in range(C):
        m = (y == c)
        if m.any():
            signed[c] = dv_dx[m].mean(0)
            absm[c] = np.abs(dv_dx[m]).mean(0)
            cnt[c] = int(m.sum())
    return signed, absm, cnt


# ============================================================================
# numpy serve-side: load the cache, render the FORGE payload (testable)
# ============================================================================
class InputAttribution:
    """Serve-side reader for attribution.npz. Pure numpy — the FORGE backend builds
    the bar-chart payload from this with the live-τ knob, no torch."""

    def __init__(self, feature_names, lam, genres, dvdx_abs, dvdx_signed,
                 counts, tau_default, per_genre_M=None):
        self.feature_names = list(feature_names)
        self.lam = np.asarray(lam, float)
        self.genres = list(genres)
        self.dvdx_abs = np.asarray(dvdx_abs, float)             # [C,K] at tau_default
        self.dvdx_signed = np.asarray(dvdx_signed, float)       # [C,K] at tau_default
        self.counts = np.asarray(counts, int)
        self.tau_default = float(tau_default)
        self.per_genre_M = None if per_genre_M is None else np.asarray(per_genre_M, float)  # [C,d,K]

    @property
    def tau_live(self) -> bool:
        return self.per_genre_M is not None

    @classmethod
    def from_cache(cls, bundle_dir):
        d = Path(bundle_dir)
        z = np.load(d / "attribution.npz")
        man = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
        lm = json.loads((d / "label_map.json").read_text(encoding="utf-8"))
        genres = [g for g, _ in sorted(lm.items(), key=lambda kv: kv[1])]
        names = man.get("tabular") or [f"x{i}" for i in range(z["dvdx_abs"].shape[1])]
        M = z["per_genre_M"] if "per_genre_M" in z.files else None
        return cls(names, z["lam"], genres, z["dvdx_abs"], z["dvdx_signed"],
                   z["counts"], float(z["tau_default"][()]), per_genre_M=M)

    def at_tau(self, tau=None):
        """[C,K] (signed, abs) per-genre dv/dx at τ. Uses the cached τ-independent M
        when present (any τ); otherwise returns the cached fixed-τ_default matrices."""
        if tau is None or not self.tau_live:
            return self.dvdx_signed, self.dvdx_abs
        signed = dv_dx_from_M(self.per_genre_M, self.lam, tau)  # [C,K]
        return signed, np.abs(signed)

    def payload(self, genre=None, tau=None, top=15) -> dict:
        """Ranked named bars for one genre (or all). The exact thing the FORGE
        attribution tab renders: top-|dv/dx| named features with sign + magnitude."""
        signed, absm = self.at_tau(tau)
        tau_eff = self.tau_default if (tau is None or not self.tau_live) else float(tau)
        names = self.feature_names

        def one(ci):
            order = np.argsort(absm[ci])[::-1][:top]
            return {
                "genre": self.genres[ci], "n_examples": int(self.counts[ci]),
                "bars": [{"feature": names[k], "abs": float(absm[ci, k]),
                          "signed": float(signed[ci, k])} for k in order],
            }

        gsel = range(len(self.genres)) if genre is None else \
            [self.genres.index(genre)] if isinstance(genre, str) else [int(genre)]
        return {
            "tau": tau_eff, "tau_live": self.tau_live, "top": int(top),
            "feature_names": names, "genres": self.genres,
            "per_genre": [one(ci) for ci in gsel],
            "note": "epistemic sensitivity dv/dx per 1σ feature move (standardized "
                    "inputs); SENSITIVITY not a posterior over input weights (spec §8).",
        }


# ============================================================================
# torch precompute — the ONE-TIME GPU-box step (lazy torch import)
# ============================================================================
def precompute(bundle_dir, loaded, *, device="cpu", tau=None, split="train",
               per_genre=25, tau_live=False, seed=0, last_linear_key=None):
    """Backward dφ/dx through the frozen backbone, combine with closed-form dv/dφ →
    per-example dv/dx; aggregate per genre; write attribution.npz + refresh
    bundle.json (adds the 'attribution' capability). Run once on the GPU box.

    per_genre : exemplars/genre to attribute (subset keeps the precompute cheap and
                the per-genre aggregate is the thesis deliverable). None → all rows.
    tau_live  : also cache the per-genre projected-Jacobian M ([C,d,K]) so FORGE's
                τ-knob drives attribution live. Costs ~d backward passes/example
                (a full Jacobian) — opt-in. Default off (fixed-τ bar chart only).
    """
    import torch
    from .bayes.llla import from_bundle

    lap = from_bundle(bundle_dir, device=device, last_linear_key=last_linear_key)
    lam, U = lap.Lambda, lap.U
    if tau is None:
        tau = max(1e-3, 1e-2 * float(lam.max()))               # llla CLI default

    # rebuild the frozen net (torch) for the backbone VJP
    from .bundle import load_bundle
    model = load_bundle(bundle_dir, device=device).model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    Xi = loaded.X[split]; y_all = np.asarray(loaded.y[split]).ravel()
    img_all = np.asarray(Xi["image"], np.float32)              # [N,H,W,1]
    tab_all = np.asarray(Xi["tabular"], np.float32)            # [N,K]
    N, K = tab_all.shape
    C = len(loaded.label_map)

    # per-genre exemplar subset (balanced, reproducible)
    rng = np.random.default_rng(seed)
    if per_genre is None:
        idx = np.arange(N)
    else:
        idx = np.concatenate([rng.permutation(np.where(y_all == c)[0])[:per_genre]
                              for c in range(C)]) if N else np.array([], int)
    idx = idx.astype(int)

    def nchw(a):                                               # [H,W,1] -> [1,1,H,W]
        t = torch.as_tensor(a, dtype=torch.float32, device=device)
        return t.permute(2, 0, 1).unsqueeze(0).contiguous()

    dvdx = np.zeros((idx.size, K), np.float64)
    M_acc = np.zeros((C, lam.size, K), np.float64) if tau_live else None
    M_cnt = np.zeros(C, int)

    for r, i in enumerate(idx):
        img = nchw(img_all[i])
        tab = torch.as_tensor(tab_all[i], dtype=torch.float32, device=device)[None, :]
        tab.requires_grad_(True)
        phi = model.features(img, tab)[0]                      # [d]
        g = dv_dphi(phi.detach().cpu().numpy(), lam, U, tau)   # [d] exact cotangent
        if tab.grad is not None:
            tab.grad = None
        phi.backward(gradient=torch.as_tensor(g, dtype=torch.float32, device=device))
        dvdx[r] = tab.grad[0].detach().cpu().numpy()           # = Jᵀg = dv/dx  [K]
        if tau_live:
            J = torch.autograd.functional.jacobian(
                lambda t: model.features(img, t[None, :])[0], tab[0].detach(),
                vectorize=True).detach().cpu().numpy()         # [d,K]
            c = int(y_all[i])
            M_acc[c] += project_M(phi.detach().cpu().numpy(), J, U)
            M_cnt[c] += 1

    signed, absm, counts = aggregate_per_genre(dvdx, y_all[idx], C)
    save = {"dvdx_signed": signed, "dvdx_abs": absm, "counts": counts,
            "lam": np.asarray(lam, float), "tau_default": np.array(float(tau))}
    if tau_live:
        with np.errstate(invalid="ignore", divide="ignore"):
            M_mean = M_acc / np.maximum(M_cnt, 1)[:, None, None]
        save["per_genre_M"] = M_mean
    np.savez(Path(bundle_dir) / "attribution.npz", **save)

    desc = write_self_description(bundle_dir, loaded)           # now caps += attribution
    return {"dir": str(bundle_dir), "n_attributed": int(idx.size),
            "per_genre": int(per_genre) if per_genre else None, "tau": float(tau),
            "tau_live": bool(tau_live), "capabilities": desc["capabilities"]}
