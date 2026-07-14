# forge-observatory
Deploy repo for forge-observatory.com — promoted apps + orchestration ONLY.
Dev/experiments never happen here (that is INFO_698_experiments).

Quickstart (the box):
  1. copy .env.example -> .env, paste TUNNEL_TOKEN
  2. docker compose up -d --build
  3. verify: http://127.0.0.1:5001 (genre), :5002 (phonon)

Promotion workflow: app matures in experiments mk-cycle -> locked version
copied into apps/<name>/ -> compose serves it. See RUNBOOK.md and
FORGE_stage_D_fullstack_LOCKIN.txt for the wiring and the why.
