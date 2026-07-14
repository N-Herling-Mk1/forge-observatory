"""Transformer / attention reproduction — STUB.
Blocked on: which paper? EAViT (3s mel segments) / improved-ViT / attention-CNN.
The choice sets features (configs/transformer.yaml) and the patch/token scheme here.
"""
from . import register

@register("transformer")
def build_model(cfg):
    raise NotImplementedError("Confirm transformer paper, then implement.")
