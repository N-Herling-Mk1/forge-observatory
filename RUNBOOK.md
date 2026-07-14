# RUNBOOK — first bring-up (P1/P2)

## P1 — tunnel
1. one.dash.cloudflare.com -> Networks -> Tunnels -> Create tunnel
   (Cloudflared connector) -> name: forge -> copy the TOKEN only.
2. .env <- TUNNEL_TOKEN  (never git)
3. In the tunnel Public Hostname tab add:
     genre.forge-observatory.com  -> HTTP -> genre:5000
     phonon.forge-observatory.com -> HTTP -> phonon:5000
   (DNS records are auto-created; atlas added at birth)
4. docker compose up -d --build
5. progress/status:  docker compose ps ; docker compose logs -f cloudflared

## P2 — apex -> GitHub Pages (docs repo)
1. INFO_698_documentation repo -> Settings -> Pages -> Custom domain:
   forge-observatory.com  (commits a CNAME file) -> wait for check -> Enforce HTTPS
2. Cloudflare DNS -> add CNAME  @  -> n-herling-mk1.github.io  (DNS only
   until the GitHub cert issues, then proxy optional)
3. add CNAME  www -> n-herling-mk1.github.io

## Smoke test order
local 127.0.0.1 ports -> tunnel hostnames -> apex.
Known check: promoted apps ran under /workspace with repo-root PYTHONPATH;
if an import or data path 404s on first boot, fix the path in the app here
(deploy repo owns its copies) and note it below.

## Deviations log
- (none yet)
