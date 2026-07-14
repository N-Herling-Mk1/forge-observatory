"""BEARDOWN reproduction — PyTorch dual encoder (spec-CNN ⨝ tab-MLP → fusion → head).

Faithful Model-1 architecture. Deterministic backbone; the FINAL layer is a plain
``nn.Linear`` so the Last-Layer Laplace (LLLA) attaches there with nothing else to
change. ``features(image, tabular)`` returns the penultimate vector φ — the cache the
bundle needs for ``phi_train.npy`` and the last-layer GGN eigenbasis.

Everything is cfg-driven (configs/beardown.yaml, the single source of truth). The
cfg_14 numbers (conv filters/kernels, dense widths, dropout, fusion type) are OPEN
ITEMS to overwrite from the BEARDOWN report slide_7 — only the values change, not
this code. ``arch_spec()`` serializes the resolved plan to arch.json; ``from_spec``
rebuilds from it (the deterministic inverse used by bundle.load_bundle).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register


# --------------------------------------------------------------------- branches
class SpecCNN(nn.Module):
    """Mel-spectrogram image [N,1,H,W] -> embedding [N, embed].

    Conv blocks (Conv2d + BN + ReLU + 2x2 MaxPool), then AdaptiveAvgPool(1) so the
    branch is image-size-agnostic (native 128 or BEARDOWN's upsampled 224 both work).
    """
    def __init__(self, in_ch=1, conv=((32, 3), (64, 3), (128, 3)),
                 embed=128, dropout=0.3):
        super().__init__()
        layers = []
        c = in_ch
        for out_ch, k in conv:
            layers += [
                nn.Conv2d(c, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c, embed)
        self.drop = nn.Dropout(dropout)
        self.out_dim = embed

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.drop(F.relu(self.proj(x)))


class TabMLP(nn.Module):
    """58 engineered features [N,F] -> embedding [N, embed]."""
    def __init__(self, in_dim=58, hidden=(128, 64), embed=64, dropout=0.3):
        super().__init__()
        dims = [in_dim, *hidden]
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Linear(dims[-1], embed)
        self.out_dim = embed

    def forward(self, x):
        return F.relu(self.proj(self.body(x)))


# ----------------------------------------------------------------------- network
class BeardownNet(nn.Module):
    """spec-CNN ⨝ tab-MLP → fusion (concat|gated) → φ → final Linear(φ, n_classes)."""

    def __init__(self, spec_cnn: dict, tab_mlp: dict, fusion: str,
                 penultimate: int, head_dropout: float, n_classes: int):
        super().__init__()
        self.spec = SpecCNN(**spec_cnn)
        self.tab = TabMLP(**tab_mlp)
        self.fusion = fusion
        es, et = self.spec.out_dim, self.tab.out_dim

        if fusion == "concat":
            fused_dim = es + et
        elif fusion == "gated":
            # project both branches to a common dim, gate the spec stream by the tab stream
            self.gate_dim = min(es, et)
            self.gp_spec = nn.Linear(es, self.gate_dim)
            self.gp_tab = nn.Linear(et, self.gate_dim)
            self.gate = nn.Linear(et, self.gate_dim)
            fused_dim = self.gate_dim
        else:
            raise ValueError(f"fusion must be 'concat' or 'gated'; got {fusion!r}")

        self.fuse = nn.Linear(fused_dim, penultimate)
        self.fuse_drop = nn.Dropout(head_dropout)
        # LLLA attaches HERE — keep it a plain Linear, nothing fancy after φ.
        self.classifier = nn.Linear(penultimate, n_classes)

        self._spec_kw = spec_cnn
        self._tab_kw = tab_mlp
        self._penultimate = penultimate
        self._head_dropout = head_dropout
        self._n_classes = n_classes

    # -- the φ cache the Bayesian layer consumes -----------------------------
    def features(self, image, tabular):
        es = self.spec(image)
        et = self.tab(tabular)
        if self.fusion == "concat":
            z = torch.cat([es, et], dim=1)
        else:  # gated
            g = torch.sigmoid(self.gate(et))
            z = g * self.gp_spec(es) + (1.0 - g) * self.gp_tab(et)
        phi = self.fuse_drop(F.relu(self.fuse(z)))
        return phi

    def forward(self, image, tabular):
        return self.classifier(self.features(image, tabular))

    # -- save/load contract ---------------------------------------------------
    def arch_spec(self) -> dict:
        """Everything needed to rebuild this exact network (-> arch.json)."""
        return {
            "model": "beardown",
            "spec_cnn": self._spec_kw,
            "tab_mlp": self._tab_kw,
            "fusion": self.fusion,
            "penultimate": self._penultimate,
            "head_dropout": self._head_dropout,
            "n_classes": self._n_classes,
            "phi_dim": self._penultimate,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "BeardownNet":
        return cls(
            spec_cnn=spec["spec_cnn"],
            tab_mlp=spec["tab_mlp"],
            fusion=spec["fusion"],
            penultimate=spec["penultimate"],
            head_dropout=spec["head_dropout"],
            n_classes=spec["n_classes"],
        )


# ----------------------------------------------------------------------- builder
@register("beardown")
def build_model(cfg: dict, dims: dict | None = None) -> BeardownNet:
    """Build from the resolved config's ``arch`` block. ``dims`` (optional) injects
    the ACTUAL data shapes from the Loaded object so the model always matches the
    real feature count / image channels (overrides the cfg placeholders)."""
    a = cfg["arch"]
    spec_cnn = {
        "in_ch": a["spec_cnn"].get("in_ch", 1),
        "conv": [tuple(c) for c in a["spec_cnn"]["conv"]],
        "embed": a["spec_cnn"]["embed"],
        "dropout": a["spec_cnn"].get("dropout", 0.3),
    }
    tab_mlp = {
        "in_dim": a["tab_mlp"]["in_dim"],
        "hidden": list(a["tab_mlp"]["hidden"]),
        "embed": a["tab_mlp"]["embed"],
        "dropout": a["tab_mlp"].get("dropout", 0.3),
    }
    if dims:                                   # real data wins over cfg placeholders
        if "tab_in" in dims:
            tab_mlp["in_dim"] = int(dims["tab_in"])
        if "img_ch" in dims:
            spec_cnn["in_ch"] = int(dims["img_ch"])
    return BeardownNet(
        spec_cnn=spec_cnn,
        tab_mlp=tab_mlp,
        fusion=a.get("fusion", "concat"),
        penultimate=a["head"]["penultimate"],
        head_dropout=a["head"].get("dropout", 0.3),
        n_classes=a.get("n_classes", 10),
    )
