"""
FORGE · phonon · data pipeline (MODEL 1, faithful replication).

Loads the Chen et al. bundled dataset (1524 materials) + their train/test/val splits,
turns each CIF (stored as a list of text lines) into a periodic crystal graph in the
exact representation the authors' network expects:
    x = mass-weighted one-hot (118-d)   -> node feature  (embedded 118->em_dim)
    z = type one-hot (118-d)            -> node attribute (embedded 118->em_dim)
    edges from ASE neighbor_list("ijS", cutoff=max_radius, self_interaction=True)
    edge_vec includes periodic image shifts.

No Materials Project key, no torch_cluster/torch_scatter. ASE does the neighbor search.
"""
from __future__ import annotations
import io
import pickle
from pathlib import Path

import numpy as np
import torch
import torch_geometric as tg
from ase import Atom
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from tqdm import tqdm

DEFAULT_DTYPE = torch.float64

# ----------------------------------------------------------------- encodings
def build_encodings():
    """118-element type encoding + mass and type one-hot tables (ASE atomic masses)."""
    type_encoding, specie_am = {}, []
    for Z in range(1, 119):
        sp = Atom(Z)
        type_encoding[sp.symbol] = Z - 1
        specie_am.append(sp.mass)
    type_onehot = torch.eye(118, dtype=DEFAULT_DTYPE)
    am_onehot = torch.diag(torch.tensor(specie_am, dtype=DEFAULT_DTYPE))  # mass-weighted
    return type_encoding, type_onehot, am_onehot


# ------------------------------------------------------------- graph builder
def cif_to_atoms(cif_entry):
    """A dataset `cif` entry is a list of CIF lines (or a raw CIF string)."""
    s = "\n".join(cif_entry) if isinstance(cif_entry, (list, tuple)) else cif_entry
    return ase_read(io.StringIO(s), format="cif")


def build_graph(atoms, phdos_row, enc, max_radius=5.0):
    type_encoding, type_onehot, am_onehot = enc
    symbols = list(atoms.symbols)
    pos = torch.tensor(atoms.get_positions(), dtype=DEFAULT_DTYPE)
    lattice = torch.tensor(atoms.cell.array, dtype=DEFAULT_DTYPE).unsqueeze(0)

    src, dst, shift = neighbor_list("ijS", a=atoms, cutoff=max_radius, self_interaction=True)
    ebatch = pos.new_zeros(pos.shape[0], dtype=torch.long)[torch.from_numpy(src)]
    edge_vec = (pos[torch.from_numpy(dst)] - pos[torch.from_numpy(src)]
                + torch.einsum("ni,nij->nj", torch.tensor(shift, dtype=DEFAULT_DTYPE), lattice[ebatch]))

    return tg.data.Data(
        pos=pos, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[s] for s in symbols]],
        z=type_onehot[[type_encoding[s] for s in symbols]],
        edge_index=torch.stack([torch.LongTensor(src), torch.LongTensor(dst)], dim=0),
        edge_vec=edge_vec,
        phdos=torch.tensor(np.asarray(phdos_row), dtype=DEFAULT_DTYPE).unsqueeze(0),
    )


# ------------------------------------------------------------------- loaders
def load_dataset(data_dir, max_radius=5.0, limit=None):
    """Returns (graphs, meta). graphs[i] aligns with the pickle's row i."""
    data_dir = Path(data_dir)
    with open(data_dir / "phdos_e3nn_len51max1000_fwin101ord3.pkl", "rb") as f:
        d = pickle.load(f)
    cif, phdos = d["cif"], np.asarray(d["phdos"])
    phfre = np.asarray(d["phfre"])
    n = len(cif) if limit is None else min(limit, len(cif))

    enc = build_encodings()
    graphs = []
    for i in tqdm(range(n), desc="building graphs", ncols=80):
        graphs.append(build_graph(cif_to_atoms(cif[i]), phdos[i], enc, max_radius))
    meta = {"phfre": phfre, "material_id": d.get("material_id"),
            "phdos_gt": np.asarray(d["phdos_gt"]), "phfre_gt": np.asarray(d["phfre_gt"])}
    return graphs, meta


def load_splits(data_dir):
    """Authors' train/test/val index arrays (1220 / 152 / 152)."""
    with open(Path(data_dir) / "trteva_indices.pkl", "rb") as f:
        tr, te, va = pickle.load(f)
    return np.asarray(tr), np.asarray(te), np.asarray(va)


def avg_neighbors(graphs, idx=None):
    """num_neighbors scaling factor (mean edges/atom over a set of graphs)."""
    sel = range(len(graphs)) if idx is None else idx
    vals = [graphs[i].edge_index.shape[1] / graphs[i].num_nodes for i in sel]
    return float(np.mean(vals))


def make_loaders(graphs, tr, te, va, batch_size=1):
    from torch_geometric.loader import DataLoader
    g = lambda idx: [graphs[i] for i in idx]
    return (DataLoader(g(tr), batch_size=batch_size, shuffle=True),
            DataLoader(g(va), batch_size=batch_size, shuffle=False),
            DataLoader(g(te), batch_size=batch_size, shuffle=False))
