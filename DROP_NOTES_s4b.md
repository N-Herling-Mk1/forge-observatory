# DROP - s4b: smart port

2 files over s4a. `python apps\genre\app\server.py` with FORGE_PORT unset now
scans UP from 5000 and binds the first free port; the banner prints the pick
(e.g. "5000 busy -> auto-picked 5001"). The result is pinned into the env so
the debug reloader child keeps the SAME port across auto-restarts (no hopping
mid-session). Explicit FORGE_PORT still pins exactly - compose stays
deterministic (container 5000 -> host 5001 map unchanged).

Sandbox-verified: 5000 occupied -> picked 5001, chain 200; FORGE_PORT=5077 -> exact.
