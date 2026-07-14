# `_shared/` — the transferable engine

Everything in here is **domain-agnostic** and shared by all three projects via
**import**. Rule of thumb: if `genre/` and `phonon/` would both need it, it lives here.
If you ever copy-paste between projects, the thing you copied belongs in this folder.

| Module | What it owns |
|--------|--------------|
| `schema.py` | ⭐ The artifact contract — dataclasses/validators for `run.json` and `compute.json`. The single source of truth for what a run emits. Change here = change everywhere. |
| `profiler.py` | A context manager wrapping training. Captures wall-clock, peak RSS/VRAM, throughput (samples/s), and est. FLOPs/params. Emits `compute.json`. This is where compute-cost estimates come from. |
| `splits.py` | Dataset splitting with the **track-level guard** — prevents segments of one recording landing in both train and test (the GTZAN 3-second-segment leakage trap). Also supports artist-aware splits. |
| `eda.py` | Reusable EDA: class-balance plots, correlation heatmaps, per-class exemplar grids, and a `write_eda_stats()` that emits `eda_stats.json` in the site-expected shape. |

## Import pattern

```python
from _shared.profiler import profile_run
from _shared.splits import track_level_split
from _shared.schema import RunRecord, ComputeRecord
```

Keep this package **dependency-light at import time** — heavy imports (torch) belong
inside the functions that need them, so `eda.py` can run without a GPU stack present.
