"""
FORGE · phonon · MODEL 1 — faithful replication of Chen et al. 2021 (Adv. Sci. 2004214).

E(3)-equivariant network (e3nn `gate_points_2101`) wrapped for periodic crystals.
Architecture pinned to the authors' released reference checkpoint:
    em_dim=64, irreps_in=64x0e, irreps_out=51x0e, irreps_node_attr=64x0e,
    layers=2, mul=32, lmax=1, max_radius=5, number_of_basis=10.

Vendored from ninarina12/phononDoS_tutorial (same authors, modern e3nn) with the
compiled torch_scatter/torch_cluster deps removed: we feed edges precomputed by ASE,
and pool with a plain segment-mean. No behavioural change vs. the tutorial network.
"""
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists


class CustomCompose(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class Network(nn.Module):
    """E(3)-equivariant message-passing network (gate_points_2101 lineage)."""
    def __init__(self, irreps_in, irreps_out, irreps_node_attr, layers, mul, lmax,
                 max_radius, number_of_basis=10, radial_layers=1, radial_neurons=100,
                 num_neighbors=1., num_nodes=1., reduce_output=True):
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")
        act = {1: F.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.layers = nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                        if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                      if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(m, ir) for m, _ in irreps_gated])
            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Convolution(irreps, self.irreps_node_attr, self.irreps_edge_attr,
                               gate.irreps_in, number_of_basis, radial_layers, radial_neurons, num_neighbors)
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(irreps, self.irreps_node_attr, self.irreps_edge_attr, self.irreps_out,
                        number_of_basis, radial_layers, radial_neurons, num_neighbors)
        )

    def preprocess(self, data):
        batch = data['batch'] if 'batch' in data else data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data):
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius, number=self.number_of_basis,
            basis='gaussian', cutoff=False).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        x = data['x'] if (self.input_has_node_in and 'x' in data) else data['pos'].new_ones((data['pos'].shape[0], 1))
        z = data['z'] if (self.input_has_node_attr and 'z' in data) else data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0, reduce='sum').div(self.num_nodes ** 0.5)
        return x


class PeriodicNetwork(Network):
    """Embeds the 118-d mass/type one-hots to em_dim scalars, pools per-atom, peak-normalizes."""
    def __init__(self, in_dim, em_dim, **kwargs):
        self.pool = False
        if kwargs['reduce_output']:
            kwargs['reduce_output'] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em = nn.Linear(in_dim, em_dim)

    def forward(self, data):
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = torch.relu(output)
        if self.pool:
            output = scatter(output, data.batch, dim=0, reduce='mean')
        maxima, _ = torch.max(output, dim=1)
        output = output.div(maxima.unsqueeze(1))
        return output


def build_model(cfg, num_neighbors=1.0):
    """Construct the reference-spec PeriodicNetwork from a config dict."""
    return PeriodicNetwork(
        in_dim=cfg.get("in_dim", 118),
        em_dim=cfg.get("em_dim", 64),
        irreps_in=f"{cfg.get('em_dim', 64)}x0e",
        irreps_out=f"{cfg.get('out_dim', 51)}x0e",
        irreps_node_attr=f"{cfg.get('em_dim', 64)}x0e",
        layers=cfg.get("layers", 2),
        mul=cfg.get("mul", 32),
        lmax=cfg.get("lmax", 1),
        max_radius=cfg.get("max_radius", 5.0),
        number_of_basis=cfg.get("number_of_basis", 10),
        num_neighbors=num_neighbors,
        reduce_output=True,
    )
