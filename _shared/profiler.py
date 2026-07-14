"""Compute logging -> ComputeRecord. Source of all cost estimates (Feature F5).

The deployed cost is dominated by *inference*, not training, so the profiler times
the four compute paths individually via ``block(...)`` (llla_fit | llla_knob |
predict | hmc_resample) on top of the whole-run wall/throughput.

Usage:
    from _shared.profiler import profile_run, profile_model
    with profile_run(device="cuda") as prof:
        prof.set_model(model)                  # optional: records n_params / est_flops
        for step in ...:
            ... train ...
            prof.tick(n_samples_processed)     # call each step/epoch
        with prof.block("predict", n=batch):   # per-path timing -> ComputeRecord.ops
            ... forward ...
    rec = prof.record()                        # -> ComputeRecord

Heavy imports (torch, psutil, thop) live inside methods so EDA can import _shared
without a GPU/training stack present.
"""
from __future__ import annotations
import time, contextlib
from _shared.schema import ComputeRecord


def _rss_mb() -> float:
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        # psutil-free fallback: Linux statm (resident pages * page size)
        try:
            import os, resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0


def _vram_mb(device: str) -> float | None:
    if "cuda" not in device:
        return None
    try:
        import torch
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def profile_model(model, sample) -> tuple[int, int | None]:
    """(n_params, est_flops). FLOPs via thop if available, else None.

    ``sample`` is one forward-ready input (tensor, or tuple/list/dict of tensors for
    the dual-input fused model). Param count is always available.
    """
    n_params = int(sum(p.numel() for p in model.parameters()))
    est_flops = None
    try:
        import copy, torch
        from thop import profile as thop_profile
        probe = copy.deepcopy(model)      # thop registers total_ops/total_params buffers;
        probe.eval()                       # profile a throwaway so weights.pt stays clean
        with torch.no_grad():
            inp = sample if isinstance(sample, (tuple, list)) else (sample,)
            macs, _ = thop_profile(probe, inputs=inp, verbose=False)
        est_flops = int(macs) * 2          # MACs -> FLOPs
        del probe
    except Exception:
        pass
    return n_params, est_flops


class _Profiler:
    def __init__(self, device: str = "cpu", label: str = ""):
        self.device = device
        self.label = label
        self._t0 = None
        self._samples = 0
        self._peak_rss = 0.0
        self._peak_vram = None
        self._n_params = None
        self._est_flops = None
        self.ops: list[dict] = []

    # -- model metadata -------------------------------------------------------
    def set_model(self, model, sample=None) -> None:
        if sample is not None:
            self._n_params, self._est_flops = profile_model(model, sample)
        else:
            self._n_params = int(sum(p.numel() for p in model.parameters()))

    # -- sampling -------------------------------------------------------------
    def tick(self, n_samples: int) -> None:
        self._samples += int(n_samples)
        rss = _rss_mb()
        if rss > self._peak_rss:
            self._peak_rss = rss
        v = _vram_mb(self.device)
        if v is not None and (self._peak_vram is None or v > self._peak_vram):
            self._peak_vram = v

    # -- per-path timing (F5) -------------------------------------------------
    @contextlib.contextmanager
    def block(self, label: str, n: int = 0):
        """Time one compute path; append {label, wall_s, peak_rss_mb, peak_vram_mb, n}
        to ``ops``. Use for llla_fit / llla_knob / predict / hmc_resample."""
        if "cuda" in self.device:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            self.ops.append({
                "label": label,
                "wall_s": wall,
                "peak_rss_mb": _rss_mb(),
                "peak_vram_mb": _vram_mb(self.device),
                "n": int(n),
            })

    def record(self) -> ComputeRecord:
        wall = (time.perf_counter() - self._t0) if self._t0 else 0.0
        tput = self._samples / wall if wall > 0 else 0.0
        if self._peak_rss <= 0.0:
            self._peak_rss = _rss_mb()
        return ComputeRecord(
            wall_seconds=wall,
            peak_rss_mb=self._peak_rss,
            peak_vram_mb=self._peak_vram,
            throughput_samples_per_s=tput,
            est_flops=self._est_flops,
            n_params=self._n_params,
            device=self.device,
            ops=list(self.ops),
        )


@contextlib.contextmanager
def profile_run(device: str = "cpu", label: str = ""):
    p = _Profiler(device, label)
    p._t0 = time.perf_counter()
    tag = f"[{label}] " if label else ""
    print(f"\033[36m▸ profiler armed {tag}· device={device}\033[0m", flush=True)
    try:
        yield p
    finally:
        wall = (time.perf_counter() - p._t0) if p._t0 else 0.0
        print(f"\033[36m▸ profiler done {tag}· {wall:.2f}s · "
              f"{p._samples} samples · peak_rss={p._peak_rss:.0f}MB\033[0m", flush=True)


# --------------------------------------------------------------------------- F5
class CostModel:
    """Least-squares fit  wall ≈ a + b·n_params + c·n_samples  over a sweep's
    compute.json records (per device). Feeds resource_report.json and the
    proposal's deploy-sizing tables (6/7). One fit per device family.
    """
    def __init__(self, coef=None, device="cpu"):
        self.coef = coef          # (a, b, c)
        self.device = device

    @classmethod
    def fit(cls, records: list[ComputeRecord], device: str = "cpu") -> "CostModel":
        import numpy as np
        rows = [(1.0, float(r.n_params or 0), float(r.throughput_samples_per_s and
                 r.wall_seconds and (r.throughput_samples_per_s * r.wall_seconds) or 0.0),
                 float(r.wall_seconds))
                for r in records if r.device == device and r.wall_seconds > 0]
        if len(rows) < 3:
            return cls(coef=None, device=device)   # underdetermined
        A = np.array([[a, b, c] for (a, b, c, _) in rows], dtype=float)
        y = np.array([w for (*_, w) in rows], dtype=float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return cls(coef=tuple(coef), device=device)

    def project(self, n_params: int, n_samples: int) -> float:
        if self.coef is None:
            return float("nan")
        a, b, c = self.coef
        return float(a + b * n_params + c * n_samples)
