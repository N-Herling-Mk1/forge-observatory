# DROP - s5: the Observatory (bay 02, live)

5 files over s4c. The dashboard OBSERVATORY bay is no longer a stub - it is the
thesis centerpiece, running the real LLLA math.

| file | change |
|---|---|
| apps/genre/app/templates/dashboard.html | bay 02 live: tau log-slider (every plot recomputes in-browser via v(tau)=sum c_i/(lambda_i+tau) + probit), example picker (index / most-starved / most-anchored / random from meta.input_var), posterior bars (live LLLA mean+-sigma, MC reference @ tau0, MAP softmax ticks, HMC overlay), GGN eigenspectrum with tau line + prior-dominated shading + "k/d data-dominated" readout, data-fraction sweep v(tau), test-confusion heat, HMC console (samples/leapfrog, diag chips accept/rhat/ess/divergences, log-posterior trace with warmup shading). Canvas only, no libs. |
| apps/genre/src/bayes/llla.py | from_bundle: torch ImportError -> head.npz fallback (exported final Linear). Your box keeps the canonical weights.pt path - zero behavior change with torch installed. |
| apps/genre/src/bayes/hmc.py | same fallback for the MAP-init (or clean no-init if absent). |
| apps/genre/app/server.py | /api/forge/hmc gated by FORGE_HMC env (403 when "0") - the pre-public hardening item. Dev default stays on. |
| docker-compose.yml | FORGE_HMC: "0" pinned on all services (public face never runs live chains). |

Sandbox-verified against the REAL bundles (torch-free): meta / posterior /
datasweep / hmc all live (hmc test chain: accept 0.87, rhat 1.04); /dashboard
200; observatory JS passes node --check.

Dev: python apps\genre\app\server.py -> dashboard -> OBSERVATORY bay.
HMC on your box uses full weights.pt via torch as before.
