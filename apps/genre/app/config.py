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
def _port_free(host, port):
    """True only if nothing answers on the port AND we can bind it exclusively.
    Windows trap: SO_REUSEADDR lets a bind "succeed" on an actively-held port,
    so a bare bind-probe lies. Counter: (1) connect-probe - anything listening
    means busy on every OS; (2) SO_EXCLUSIVEADDRUSE bind - Windows-strict."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return False              # something answered -> occupied
    except OSError:
        pass                          # nothing listening on loopback
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):   # Windows: defeat REUSEADDR steal
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            s.bind((host if host != "0.0.0.0" else "", port))
        return True
    except OSError:
        return False

def _pick_port(host, base=5000, tries=50):
    """Smart port: first genuinely-free port scanning UP from base."""
    for cand in range(base, base + tries):
        if _port_free(host, cand):
            return cand
    raise OSError(f"no free port in {base}..{base + tries - 1}")

_env_port = os.environ.get("FORGE_PORT")
if _env_port:                             # pinned (compose, explicit dev)
    PORT = int(_env_port)
    PORT_NOTE = ""
else:                                     # smart scan; pin result so the
    PORT = _pick_port(HOST)               # debug-reloader child reuses it
    os.environ["FORGE_PORT"] = str(PORT)
    PORT_NOTE = "" if PORT == 5000 else f"(5000 busy -> auto-picked {PORT})"
DEBUG = os.environ.get("FORGE_DEBUG", "1") == "1"

# Site-shell mode: when 1, this Flask process ALSO serves the repo-root splash/
# floor/assets so localhost mirrors production end-to-end (splash -> floor ->
# select -> dashboard). Compose pins 0: the public genre subdomain keeps / = select.
SITE = os.environ.get("FORGE_SITE", "1") == "1"
