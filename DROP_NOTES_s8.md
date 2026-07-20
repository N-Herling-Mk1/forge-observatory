# DROP - s8: torch into the serving image

Dockerfile only. Root cause of the public "No module named torch": pyproject
lists the TF training tier; the serve stack (predict/bundle/bayes + weights.pt)
is torch, which your host had installed ad hoc but the image never did. The
image now installs the CPU torch wheel explicitly (own cached layer, survives
code edits).

Apply + redeploy (full rebuild, ~3-6 min for the ~200MB wheel):
  Expand-Archive .\forge_observatory_delta_s8.zip -DestinationPath . -Force
  .\publish_mk2.ps1 -NoPush

Then re-test from the other PC/phone: Drop Deck song -> three verdict cards,
and the public Observatory bay (it needed torch too - same fix).
If the pytorch index ever fails on a build: swap that RUN line to plain
  uv pip install ... torch   (PyPI; bigger download, same result).
