"""ML usage accounting -> energy, carbon, and cost, on top of ComputeRecord (F5).

Two layers, kept deliberately separate so a modeled number is never mistaken for a
metered one:

  MEASURED  wall_seconds, FLOPs, params, peak RSS/VRAM, device   (from the profiler)
            + actual energy IF the OS exposes it (Linux Intel RAPL / nvidia-smi).
  MODELED   energy = power x time   (power from a TDP table when not metered),
            carbon = energy x grid_intensity,
            cost   = energy x electricity_rate   (local)
                   = device_hours x cloud_rate   (cloud).

Every derived figure carries a ``basis`` of "measured" or "modeled". Every constant
that feeds a modeled figure lives in ``UsageAssumptions`` and is written next to the
ledger, so the estimate is auditable and reproducible. When you move to real paid
hardware, replace the assumptions (or capture real power) and the same ledger recomputes.

  Honesty note: modeled energy from TDP x time is a ROUGH proxy (real draw depends on
  utilization and clocks). Treat modeled kWh as order-of-magnitude until metered. The
  whole module is built so the measured path supersedes the modeled one wherever available.

CLI:
    python _shared/usage.py --runs projects/genre/runs [--assumptions usage_assumptions.yaml]
        -> prints a ledger table, writes usage_ledger.json next to --runs' parent.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json, time, contextlib

# ----------------------------------------------------------------- assumptions
@dataclass
class UsageAssumptions:
    """Everything a modeled figure depends on. Replace with your real values IRL.

    Defaults are approximate, clearly-placeholder, US/Arizona-ish starting points —
    NOT authoritative current prices. Override via ``usage_assumptions.yaml``.
    """
    # device active power (watts) when metering is unavailable. Keyed by a substring
    # matched against ComputeRecord.device (e.g. "cpu", "cuda", "a100", "t4").
    tdp_watts: dict[str, float] = field(default_factory=lambda: {
        "cpu":  65.0,     # generic desktop CPU package TDP — set to YOUR chip's TDP
        "cuda": 250.0,    # generic discrete GPU — overridden by more specific keys below
        "a100": 400.0, "h100": 700.0, "v100": 300.0,
        "t4": 70.0, "l4": 72.0, "rtx4090": 450.0, "rtx3090": 350.0,
    })
    utilization: float = 0.70           # fraction of TDP actually drawn during mixed train/infer
    pue: float = 1.0                    # power-usage-effectiveness (1.0 local desktop; ~1.1-1.5 datacenter)

    grid_intensity_kg_per_kwh: float = 0.40   # ~AZ/US-Southwest grid avg (EPA eGRID-ish). REPLACE for your region.
    electricity_rate_usd_per_kwh: float = 0.15  # local $/kWh. REPLACE with your bill.

    # example cloud LIST prices ($/device-hour) — illustrative, replace with your provider/contract.
    cloud_rate_usd_per_hour: dict[str, float] = field(default_factory=lambda: {
        "cpu": 0.05, "t4": 0.35, "l4": 0.70, "v100": 2.48,
        "a100": 3.00, "h100": 8.00, "rtx4090": 0.70,
    })

    notes: str = "Defaults are placeholders. Replace before quoting any number as real."

    @classmethod
    def load(cls, path: str | None) -> "UsageAssumptions":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        raw = p.read_text(encoding="utf-8")
        data = (json.loads(raw) if p.suffix == ".json"
                else __import__("yaml").safe_load(raw))
        base = cls()
        for k, v in (data or {}).items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def power_for(self, device: str) -> tuple[float, str]:
        """(watts, key) — most specific TDP-table match for a device string."""
        d = (device or "cpu").lower()
        best, bestkey = None, None
        for key, w in self.tdp_watts.items():
            if key in d and (bestkey is None or len(key) > len(bestkey)):
                best, bestkey = w, key
        if best is None:
            best, bestkey = self.tdp_watts.get("cpu", 65.0), "cpu"
        return float(best), bestkey

    def cloud_rate_for(self, device: str) -> float:
        d = (device or "cpu").lower()
        best, bestkey = None, None
        for key, r in self.cloud_rate_usd_per_hour.items():
            if key in d and (bestkey is None or len(key) > len(bestkey)):
                best, bestkey = r, key
        return float(best if best is not None else self.cloud_rate_usd_per_hour.get("cpu", 0.05))


# ------------------------------------------------------------------- estimate
@dataclass
class UsageEstimate:
    device: str
    wall_seconds: float
    device_hours: float
    energy_kwh: float
    energy_basis: str            # "measured" | "modeled"
    carbon_kg: float
    cost_local_usd: float
    cost_cloud_usd: float
    avg_power_watts: float
    power_basis: str             # "measured" | "modeled(tdp:<key>)"
    est_flops: int | None = None
    n_params: int | None = None
    peak_rss_mb: float = 0.0
    peak_vram_mb: float | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def estimate_usage(rec: dict | Any,
                   asmp: UsageAssumptions | None = None,
                   measured_energy_kwh: float | None = None) -> UsageEstimate:
    """Derive energy/carbon/cost for ONE ComputeRecord (dataclass or dict from compute.json).

    measured_energy_kwh: if you captured real energy (RAPL/nvidia-smi via PowerSampler),
    pass it here and it supersedes the TDP model.
    """
    asmp = asmp or UsageAssumptions()
    g = (lambda k, d=None: getattr(rec, k, d)) if not isinstance(rec, dict) else (lambda k, d=None: rec.get(k, d))
    device = g("device", "cpu") or "cpu"
    wall = float(g("wall_seconds", 0.0) or 0.0)
    hours = wall / 3600.0

    if measured_energy_kwh is not None:
        energy = float(measured_energy_kwh)
        energy_basis = "measured"
        avg_power = (energy * 1000.0) / hours if hours > 0 else 0.0
        power_basis = "measured"
    else:
        watts, key = asmp.power_for(device)
        avg_power = watts * asmp.utilization
        energy = (avg_power * asmp.pue * hours) / 1000.0   # kWh
        energy_basis = "modeled"
        power_basis = f"modeled(tdp:{key}@{asmp.utilization:.0%})"

    carbon = energy * asmp.grid_intensity_kg_per_kwh
    cost_local = energy * asmp.electricity_rate_usd_per_kwh
    cost_cloud = hours * asmp.cloud_rate_for(device)

    return UsageEstimate(
        device=device, wall_seconds=wall, device_hours=hours,
        energy_kwh=energy, energy_basis=energy_basis,
        carbon_kg=carbon, cost_local_usd=cost_local, cost_cloud_usd=cost_cloud,
        avg_power_watts=avg_power, power_basis=power_basis,
        est_flops=g("est_flops"), n_params=g("n_params"),
        peak_rss_mb=float(g("peak_rss_mb", 0.0) or 0.0), peak_vram_mb=g("peak_vram_mb"),
    )


# --------------------------------------------------- actual-power capture (opt)
class PowerSampler:
    """Context manager that captures REAL energy where the OS allows, else yields None.

      Linux + Intel RAPL: reads /sys/class/powercap/intel-rapl:*/energy_uj deltas (CPU pkg).
      NVIDIA GPU:         samples `nvidia-smi --query-gpu=power.draw` in a thread, integrates.

    On Windows CPU (no RAPL) this yields measured_energy_kwh=None -> caller falls back to
    the TDP model. Use:

        with PowerSampler(device="cuda") as ps:
            ... run ...
        est = estimate_usage(rec, asmp, measured_energy_kwh=ps.energy_kwh)
    """
    def __init__(self, device: str = "cpu", sample_hz: float = 2.0):
        self.device = (device or "cpu").lower()
        self.sample_hz = sample_hz
        self.energy_kwh: float | None = None
        self._rapl_paths: list[Path] = []
        self._rapl_start: list[int] = []
        self._gpu_thread = None
        self._gpu_stop = False
        self._gpu_energy_j = 0.0
        self._t0 = None

    def _rapl_read(self) -> list[int]:
        vals = []
        for p in self._rapl_paths:
            try:
                vals.append(int((p / "energy_uj").read_text().strip()))
            except Exception:
                vals.append(0)
        return vals

    def __enter__(self):
        self._t0 = time.perf_counter()
        # CPU: Intel RAPL (Linux)
        base = Path("/sys/class/powercap")
        if base.exists():
            self._rapl_paths = sorted(d for d in base.glob("intel-rapl:*")
                                      if (d / "energy_uj").exists() and ":" in d.name and d.name.count(":") == 1)
            if self._rapl_paths:
                self._rapl_start = self._rapl_read()
        # GPU: nvidia-smi power sampler thread
        if "cuda" in self.device or "gpu" in self.device:
            import shutil
            if shutil.which("nvidia-smi"):
                import threading
                self._gpu_stop = False
                self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
                self._gpu_thread.start()
        return self

    def _gpu_loop(self):
        import subprocess, time as _t
        dt = 1.0 / max(self.sample_hz, 0.5)
        last = _t.perf_counter()
        while not self._gpu_stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                watts = sum(float(x) for x in out.stdout.split("\n") if x.strip())
            except Exception:
                watts = 0.0
            now = _t.perf_counter()
            self._gpu_energy_j += watts * (now - last)
            last = now
            _t.sleep(dt)

    def __exit__(self, *exc):
        joules = 0.0
        have = False
        if self._rapl_paths:
            end = self._rapl_read()
            # energy_uj is a wrapping counter; assume no wrap for run-length intervals
            joules += sum(max(e - s, 0) for s, e in zip(self._rapl_start, end)) / 1e6
            have = True
        if self._gpu_thread is not None:
            self._gpu_stop = True
            self._gpu_thread.join(timeout=2)
            joules += self._gpu_energy_j
            have = True
        self.energy_kwh = (joules / 3.6e6) if have else None   # J -> kWh
        return False


# ------------------------------------------------------------------ aggregate
def aggregate_runs(runs_dir: str, asmp: UsageAssumptions | None = None) -> dict[str, Any]:
    """Walk runs_dir for compute.json (+ sibling run.json for project/model labels),
    estimate each, and roll up totals + per-experiment / per-model breakdowns."""
    asmp = asmp or UsageAssumptions()
    root = Path(runs_dir)
    rows, total = [], dict(wall_seconds=0.0, energy_kwh=0.0, carbon_kg=0.0,
                           cost_local_usd=0.0, cost_cloud_usd=0.0, device_hours=0.0, n=0)
    by_exp: dict[str, dict] = {}
    any_measured = False

    for cj in sorted(root.rglob("compute.json")):
        try:
            rec = json.loads(cj.read_text(encoding="utf-8"))
        except Exception:
            continue
        rj = cj.with_name("run.json")
        proj, model = "?", "?"
        if rj.exists():
            try:
                r = json.loads(rj.read_text(encoding="utf-8"))
                proj, model = r.get("project") or "?", r.get("model") or "?"
            except Exception:
                pass
        est = estimate_usage(rec, asmp)
        any_measured = any_measured or (est.energy_basis == "measured")
        row = {"run": str(cj.parent.relative_to(root)), "project": proj, "model": model, **est.as_row()}
        rows.append(row)
        for k in ("wall_seconds", "energy_kwh", "carbon_kg", "cost_local_usd", "cost_cloud_usd", "device_hours"):
            total[k] += row[k]
        total["n"] += 1
        b = by_exp.setdefault(proj, dict(wall_seconds=0.0, energy_kwh=0.0, carbon_kg=0.0,
                                         cost_local_usd=0.0, cost_cloud_usd=0.0, n=0))
        for k in ("wall_seconds", "energy_kwh", "carbon_kg", "cost_local_usd", "cost_cloud_usd"):
            b[k] += row[k]
        b["n"] += 1

    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs_dir": str(root), "assumptions": asdict(asmp),
            "all_energy_measured": any_measured,
            "totals": total, "by_experiment": by_exp, "runs": rows}


def project_suite(cost_model, plan: list[dict], asmp: UsageAssumptions | None = None,
                  device: str = "cpu") -> dict[str, Any]:
    """Project energy/carbon/$ for a planned set of runs BEFORE running them.

    plan: [{"name","n_params","n_samples","epochs","count"}...]. Uses a fitted CostModel
    to predict wall time, then the assumptions to turn time into energy/carbon/cost.
    """
    asmp = asmp or UsageAssumptions()
    watts, key = asmp.power_for(device)
    avg_power = watts * asmp.utilization
    items, tot = [], dict(wall_seconds=0.0, energy_kwh=0.0, carbon_kg=0.0,
                          cost_local_usd=0.0, cost_cloud_usd=0.0)
    for it in plan:
        count = int(it.get("count", 1))
        per_wall = cost_model.project(int(it.get("n_params", 0)),
                                      int(it.get("n_samples", 0)) * int(it.get("epochs", 1)))
        wall = (per_wall if per_wall == per_wall else 0.0) * count   # nan-guard
        hours = wall / 3600.0
        energy = (avg_power * asmp.pue * hours) / 1000.0
        row = dict(name=it.get("name", "?"), count=count, wall_seconds=wall,
                   energy_kwh=energy, carbon_kg=energy * asmp.grid_intensity_kg_per_kwh,
                   cost_local_usd=energy * asmp.electricity_rate_usd_per_kwh,
                   cost_cloud_usd=hours * asmp.cloud_rate_for(device))
        items.append(row)
        for k in tot:
            tot[k] += row[k]
    return {"device": device, "power_basis": f"modeled(tdp:{key}@{asmp.utilization:.0%})",
            "items": items, "totals": tot, "assumptions": asdict(asmp)}


# ------------------------------------------------------------------------- CLI
_C = "\033[36m"; _A = "\033[38;5;214m"; _D = "\033[2m"; _R = "\033[0m"

def _fmt_table(ledger: dict) -> str:
    t = ledger["totals"]
    lines = []
    lines.append(f"{_C}╔══ ML USAGE LEDGER ═══════════════════════════════════════════{_R}")
    lines.append(f"{_C}║{_R} {ledger['runs_dir']}   ·   {t['n']} runs   ·   {ledger['generated']}")
    basis = "measured" if ledger["all_energy_measured"] else f"{_A}modeled (TDP){_R}"
    lines.append(f"{_C}║{_R} energy basis: {basis}   {_D}(see assumptions block in usage_ledger.json){_R}")
    lines.append(f"{_C}╠══ per experiment ════════════════════════════════════════════{_R}")
    lines.append(f"{_C}║{_R} {'exp':<10}{'runs':>5} {'wall(h)':>9} {'kWh':>9} {'kgCO2e':>9} {'$local':>9} {'$cloud':>9}")
    for exp, b in sorted(ledger["by_experiment"].items()):
        lines.append(f"{_C}║{_R} {exp:<10}{b['n']:>5} {b['wall_seconds']/3600:>9.3f} "
                     f"{b['energy_kwh']:>9.4f} {b['carbon_kg']:>9.4f} "
                     f"{b['cost_local_usd']:>9.4f} {b['cost_cloud_usd']:>9.4f}")
    lines.append(f"{_C}╠══ TOTAL ═════════════════════════════════════════════════════{_R}")
    lines.append(f"{_C}║{_R} {'':<10}{t['n']:>5} {t['wall_seconds']/3600:>9.3f} "
                 f"{_A}{t['energy_kwh']:>9.4f}{_R} {t['carbon_kg']:>9.4f} "
                 f"{_A}{t['cost_local_usd']:>9.4f}{_R} {t['cost_cloud_usd']:>9.4f}")
    lines.append(f"{_C}╚══════════════════════════════════════════════════════════════{_R}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Roll up ML usage (energy/carbon/$) from compute.json records.")
    ap.add_argument("--runs", required=True, help="dir to walk for compute.json (e.g. projects/genre/runs)")
    ap.add_argument("--assumptions", default=None, help="usage_assumptions.yaml/json to override defaults")
    ap.add_argument("--out", default=None, help="ledger json path (default: <runs>/../usage_ledger.json)")
    args = ap.parse_args(argv)

    asmp = UsageAssumptions.load(args.assumptions)
    ledger = aggregate_runs(args.runs, asmp)
    print(_fmt_table(ledger))
    out = Path(args.out) if args.out else (Path(args.runs).parent / "usage_ledger.json")
    out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"{_D}ledger -> {out}{_R}")
    if not ledger["all_energy_measured"]:
        print(f"{_A}⚠ modeled energy (TDP×time). For metered numbers run on Linux (RAPL) or with a GPU "
              f"(nvidia-smi) via PowerSampler, or set real rates in --assumptions.{_R}")


if __name__ == "__main__":
    main()
