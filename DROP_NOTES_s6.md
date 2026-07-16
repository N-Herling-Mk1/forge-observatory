# DROP - s6: Drop Deck (bay 03)

1 file over s5b (dashboard.html). New rail bay between Observatory and
Genealogy: DROP DECK - drag/click a track into the strike zone -> POST
/api/predict -> one card per family model (MK1/MK2/MK3): top-genre callout +
horizontal per-genre probability bars with sigma whiskers. Per-model error
blocks render in-card (a missing bundle degrades that card, not the page).

No server changes - rides the existing /api/predict. Native run needs the
host ML stack (torch + librosa); the container has it baked already.

Sandbox-verified: page 200, JS passes node --check, error path renders clean.
