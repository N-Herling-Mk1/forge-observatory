# DROP — session 3 (2026-07-15): genre insertion + dashboard mk1

Delta against `forge-observatory` @ session-2 close. Unzip at repo root; every path
is repo-relative. 8 files: 4 new, 4 rewrites.

## Contents

| file | status | what |
|---|---|---|
| `apps/genre/app/server.py` | REWRITE | (1) `PKG` autodetect — dynamic imports now resolve `apps.genre.src.*` in this repo and `projects.genre.src.*` in dev (the RUNBOOK-anticipated first-boot fix, verified). (2) `/` now serves the select page; new routes `/select`, `/dashboard`; old dev landing kept at `/hub`. Nothing else touched. |
| `apps/genre/app/templates/select.html` | NEW | Bay-01 landing: model plates from `/api/registry` (VAL/TEST/CV/F1, gate badge, mk lore), click = `POST /api/model/select`, ignition bar -> `/dashboard`. Floor aesthetic. |
| `apps/genre/app/templates/dashboard.html` | NEW | The analysis dashboard. Instrument rail: FEATURE STATS (**live**) / OBSERVATORY (staged, links `/forge` dev view) / GENEALOGY (queued stub) / ATTRIBUTION (gated stub — states the attribution.npz precompute dependency). Feature-stats module: expectation/QC chips, BEFORE/AFTER toggle, filterable sortable stats table with per-feature histogram sparklines (drawn from `nerd_stats.hist`, no images), figures grid (dist / balance / exemplars) + lightbox. Zero new API surface — reads `/api/eda`, `/figures/*`, `/api/registry` only. |
| `experiments.html` | REWRITE | Beardown plate flipped: `coming-soon.html?exp=genre` -> `https://genre.forge-observatory.com/`. **Dead link until tunnel P1** — inline comment shows the one-line revert if you push to Pages first. |
| `docker-compose.yml` | REWRITE | `FORGE_DEBUG: "0"` pinned on genre/phonon (and the commented atlas block). Unset, config.py defaults debug **on** — Werkzeug debugger behind a public tunnel is an RCE console. Local-only change otherwise. |
| `.gitignore` | REWRITE | + `apps/*/models/*` (README-excepted) — bundle discipline matches root `models/`. |
| `sync_genre_artifacts_mk1.ps1` | NEW | Mirrors dev -> deploy: `data\` (+prunes any `data\raw`), `eda\figures\`, 3 bundles (`beardown`, `beardown_rrm`, `beardown_3sec`). Robocopy /MIR, progress bars, `-WhatIfOnly` dry run. ASCII+BOM. Edit the two repo-path defaults at the top if yours differ. |
| `DROP_NOTES_s3.md` | NEW | this file |

## Bring-up (task 3, your box)

```
1  .\sync_genre_artifacts_mk1.ps1 -WhatIfOnly     # sanity
2  .\sync_genre_artifacts_mk1.ps1                 # data + figures + 3 bundles land in apps\genre\
3  docker compose build genre
4  docker compose up -d genre                     # tunnel NOT needed for local
5  docker compose logs -f genre                   # expect the FORGE banner, EDA FOUND x2
```

Smoke (browser or curl), in order:
```
http://127.0.0.1:5001/                 select page — 3 plates, pick one, chip flips
http://127.0.0.1:5001/dashboard        rail lights, chips populate, table + sparklines render
   toggle BEFORE/AFTER, open a figure  (proves Flask->disk->browser round trip)
http://127.0.0.1:5001/infer            drop a real wav/mp3 -> per-genre bars + sigma
                                       = "one model answering through Flask"
http://127.0.0.1:5001/forge            tau-knob dev view (uses the SELECTED bundle)
```
Sandbox pre-verification already done on the exact patched tree: `/`, `/select`,
`/dashboard`, `/hub`, `/api/registry`, select round-trip (persists), `/api/eda`
both phases, `/figures/*`, `/api/health` all 200; `/api/predict` resolves the
promoted module correctly (fails only on missing torch there — the compose image
installs the real env).

## Done-criteria map (today)

- full stack tested end-to-end locally .... steps 3-5 + smoke list
- one model answering through Flask ....... `/infer` -> `/api/predict`
- one dashboard component rendering ....... `/dashboard` FEATURE STATS (live, not a stub)

## Carried / decisions parked

- `apps/genre/data` (~2 MB) + `apps/genre/eda` (~31 MB) are now in the working tree
  and **git-tracked** by default. Fine for Docker (`COPY . .` bakes them). If you'd
  rather keep git slim: add `apps/*/data/` and `apps/*/eda/` to `.gitignore` — the
  sync script repopulates any clone.
- Floor push timing: see `experiments.html` note above.
- Observatory promotion (restyle `/forge` into the rail) = next session's module;
  genealogy board after; attribution stays gated on the GPU-box precompute.
