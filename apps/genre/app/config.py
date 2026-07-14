"""Shared dashboard config. Identical across experiments — nothing to edit on
clone (EXPERIMENT auto-derives from the folder; export_bundle sets it for bundles).

HOST/PORT/DEBUG read the environment so the SAME server.py works both ways:
  • native local run   -> 127.0.0.1 (default)
  • inside Docker       -> FORGE_HOST=0.0.0.0 (set by compose) so the published
                           port is reachable from the host browser
"""
import os

EXPERIMENT = None             # None = auto-detect from folder name
DEFAULT_PHASE = "after"       # before | after  (post-fix is the default view now)
PHASES = ["before", "after"]

HOST = os.environ.get("FORGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FORGE_PORT", "5000"))
DEBUG = os.environ.get("FORGE_DEBUG", "1") == "1"
