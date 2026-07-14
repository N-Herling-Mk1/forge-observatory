"""
FORGE · phonon · aspect 4 input half — input-feature attribution (torch precompute).

Chain rule  dσ/dx_k = (dσ/dφ) · (dφ/dx_k):
  - dσ/dφ : closed form from the LLLA bundle (forge.dsigma_dphi, pure numpy).
  - dφ/dx : one backward pass through the FROZEN mk1 backbone at a given crystal.
For phonon the named input x_k is an atom's mass channel, so aggregating |dσ/dx| by
element answers "which element drives this prediction's epistemic uncertainty" — the
phonon analogue of the genre tabular bar chart, and the first real run of this metric.

This is a SENSITIVITY (saliency-style), not a posterior over input weights — stated
honestly per the spec. Inference-time: reads the trained model, does not train.
"""
from __future__ import annotations
import numpy as np


def attribute_material(model, graph, bundle, n_scal, tau=None):
    """Return per-element epistemic sensitivity for one crystal graph."""
    import torch
    import torch_geometric as tg
    from . import forge

    batch = next(iter(tg.loader.DataLoader([graph], batch_size=1)))
    x = batch.x.detach().clone().requires_grad_(True)     # leaf input
    batch.x = x
    z = batch.z if hasattr(batch, "z") else None
    if z is not None:
        batch.z = z.detach()

    cap = {}
    h = model.layers[-2].register_forward_hook(lambda m, i, o: cap.__setitem__("f", o))
    model.eval()
    _ = model(batch)                                       # mutates batch.x in place; x stays the leaf
    phi = cap["f"][:, :n_scal].mean(0)                     # pooled penultimate scalars (torch)
    h.remove()

    g = forge.dsigma_dphi(bundle, phi.detach().cpu().numpy(), tau=tau)   # dσ/dφ (numpy)
    surrogate = (torch.tensor(g, dtype=phi.dtype) * phi).sum()           # gᵀφ
    surrogate.backward()                                  # → x.grad = dφ/dx · g = dσ/dx

    grad = x.grad.detach().cpu().numpy()                  # (n_atoms, 118)
    onehot = (graph.z if hasattr(graph, "z") else batch.z).detach().cpu().numpy()
    cols = onehot.argmax(1)                                # element column per atom (Z-1)
    from ase.data import chemical_symbols
    per_el = {}
    for i, c in enumerate(cols):
        s = float(grad[i, c])                             # sensitivity at this atom's mass channel
        sym = chemical_symbols[int(c) + 1]
        per_el.setdefault(sym, []).append(s)
    attribution = {k: float(np.mean(v)) for k, v in per_el.items()}
    return attribution
