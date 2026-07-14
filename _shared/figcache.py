"""Incremental figure controller — render a plot only when its inputs changed.

The EDA produces ~640 panels. Re-running to fix one feature shouldn't redraw all
of them. FigCache hashes each figure's *inputs* (the data + a style version) and
skips the matplotlib render when a PNG with a matching hash already exists. It also
prunes orphan PNGs (figures no longer produced) so the directory can't accumulate
stale duplicates.

    cache = FigCache(fig_dir, style_version="1.1")     # loads .figcache.json
    cache.render("feature_01_x.png", {"x": arr, "title": t},
                 lambda path: _dist_fig(arr, t, path))   # draw_fn only runs on a miss
    cache.prune()                                          # delete orphans
    cache.save()                                           # persist hashes
    print(cache.report())                                  # rendered / skipped / pruned

Pure stdlib + numpy. No matplotlib import here — the draw_fn owns that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import json

import numpy as np


def _payload_bytes(v: Any) -> bytes:
    """Stable bytes for a hashable figure input. Arrays hash by content+shape+dtype;
    everything else by its repr — order-independent because keys are sorted upstream."""
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return f"ndarray|{a.dtype}|{a.shape}|".encode() + a.tobytes()
    if isinstance(v, (list, tuple)):
        return b"seq|" + b"|".join(_payload_bytes(x) for x in v)
    if isinstance(v, dict):
        return b"dict|" + b"|".join(
            k.encode() + b"=" + _payload_bytes(v[k]) for k in sorted(v))
    return ("scalar|" + repr(v)).encode()


class FigCache:
    """Content-addressed render gate for a single figure directory."""

    def __init__(self, fig_dir: str | Path, style_version: str = "1.1",
                 force: bool = False):
        self.dir = Path(fig_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / ".figcache.json"
        self.style_version = str(style_version)
        self.force = bool(force)
        try:
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            # invalidate everything if the style version moved
            self.prev = saved.get("figs", {}) if saved.get("style") == self.style_version else {}
        except Exception:
            self.prev = {}
        self.cur: dict[str, str] = {}
        self.stats = {"rendered": 0, "skipped": 0, "pruned": 0}

    def key(self, payload: dict) -> str:
        h = hashlib.sha1()
        h.update(f"style={self.style_version}\x00".encode())
        h.update(_payload_bytes(payload))
        return h.hexdigest()[:16]

    def render(self, fname: str, payload: dict,
               draw_fn: Callable[[str], None]) -> bool:
        """Render ``fname`` iff its input hash changed (or --force / missing PNG).
        ``draw_fn(path)`` does the matplotlib work. Returns True if it rendered."""
        k = self.key(payload)
        self.cur[fname] = k
        png = self.dir / fname
        if (not self.force) and png.exists() and self.prev.get(fname) == k:
            self.stats["skipped"] += 1
            return False
        draw_fn(str(png))
        self.stats["rendered"] += 1
        return True

    def prune(self) -> list[str]:
        """Delete PNGs in the dir that this run did NOT produce (orphans/dedup)."""
        keep = set(self.cur)
        removed = []
        for p in self.dir.glob("*.png"):
            if p.name not in keep:
                p.unlink()
                removed.append(p.name)
                self.stats["pruned"] += 1
        return removed

    def save(self) -> None:
        self.manifest_path.write_text(
            json.dumps({"style": self.style_version, "figs": self.cur}, indent=0),
            encoding="utf-8")

    def report(self) -> str:
        s = self.stats
        total = s["rendered"] + s["skipped"]
        return (f"figures: {s['rendered']} rendered, {s['skipped']} cached "
                f"(of {total}), {s['pruned']} pruned")
