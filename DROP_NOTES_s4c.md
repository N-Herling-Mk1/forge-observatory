# DROP - s4c: port probe corrected (Windows)

1 file over s4b. The s4b probe set SO_REUSEADDR before test-binding; on Windows
that flag lets a bind SUCCEED on an actively-held port, so busy ports reported
free (and Flask then bound alongside the other server instead of erroring).

New probe, per candidate port:
  1. connect() to 127.0.0.1:port - anything answering = busy, on every OS
  2. exclusive bind check (SO_EXCLUSIVEADDRUSE on Windows) - defeats the
     REUSEADDR steal entirely

Verified with 5000 (REUSEADDR listener) + 5001 both occupied -> picked 5002.
Explicit FORGE_PORT still pins exactly (compose unchanged).
