# DROP - s7c: container ports out of the war zone

compose: genre 127.0.0.1:5601->5000, phonon 5602 (was 5001/5002 - your dev
servers and MaxEnt hub fight over the 5000 range; the container now lives
where nothing scans). The tunnel is unaffected: cloudflared reaches genre:5000
over the docker network, host ports are for local smoke only.
publish_mk2.ps1 replaces mk1 (delete mk1) - smoke target updated to 5601.

Apply: unzip at root, then:
  docker compose down
  .\publish_mk2.ps1 -NoBuild -NoPush
