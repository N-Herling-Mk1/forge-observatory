"""The artifact contract. The ONE coupling between Tier R and Tier S.

Every run emits a RunRecord (run.json) and a ComputeRecord (compute.json).
EDA emits eda_stats.json. Downstream renderers (software backend, doc site)
depend on these shapes — change them deliberately, version them when you do.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json, time

SCHEMA_VERSION = "1.1"   # 1.1: ComputeRecord.ops[] (per-path timings for F5)


@dataclass
class ComputeRecord:
    wall_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_mb: float | None = None      # None on CPU
    throughput_samples_per_s: float = 0.0
    est_flops: int | None = None           # via ptflops/thop on the model
    n_params: int | None = None
    device: str = "cpu"
    # F5: per-compute-path timings (train | llla_fit | llla_knob | predict | hmc_resample).
    # Each entry: {label, wall_s, peak_rss_mb, peak_vram_mb, n}. Populated by
    # profiler.block(...). Sizes the deployed box from the inference paths, not just train.
    ops: list[dict] = field(default_factory=list)

    # cost estimate helper: throughput x dataset x epochs -> projected wall time
    def project_wall_seconds(self, n_samples: int, epochs: int) -> float:
        if self.throughput_samples_per_s <= 0:
            return float("nan")
        return (n_samples * epochs) / self.throughput_samples_per_s


@dataclass
class RunRecord:
    schema_version: str = SCHEMA_VERSION
    project: str = ""                      # "genre" | "phonon" | "atlas"
    model: str = ""                        # "beardown" | "transformer" | ...
    git_sha: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""                 # ties run to a specific data manifest
    split_mode: str = ""                   # "naive_random" | "track" | "artist"
    epochs: list[dict] = field(default_factory=list)   # per-epoch {loss, acc, ...}
    final_metrics: dict[str, float] = field(default_factory=dict)
    paper_target: dict[str, float] = field(default_factory=dict)  # for the scorecard
    compute: ComputeRecord = field(default_factory=ComputeRecord)
    created: float = field(default_factory=time.time)

    def write(self, run_dir: str) -> None:
        import os
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)
        with open(os.path.join(run_dir, "compute.json"), "w") as f:
            json.dump(asdict(self.compute), f, indent=2)
