"""phonon_net reproduction — registers the e3nn replication (Chen et al. 2021)."""
from . import register


@register("phonon_net")
def build_model(cfg, **kw):
    # defer heavy e3nn import until actually building
    from ..model import build_model as _build
    arch = cfg.get("architecture", cfg)
    return _build(arch, num_neighbors=kw.get("num_neighbors", 1.0))
