# `_app_template/` — the canonical per-experiment mini-stack

Copy this folder to `projects/<experiment>/app/`. It reads that experiment's
artifacts and serves the TRON-Ares dashboard. Trains nothing.

- `server.py` auto-detects its root: in the repo it reads sibling `../data` and
  `../eda`; in an exported standalone bundle it reads `./data` and `./eda`.
- The experiment name auto-derives from the folder — no per-clone edit.
- Keep route names and `eda_stats.json` keys identical across experiments; that
  rigidity is what makes the eventual merge into one shared backend mechanical.

Run from inside an experiment's `app/`:
```bash
pip install flask
python server.py            # -> http://127.0.0.1:5000
```
