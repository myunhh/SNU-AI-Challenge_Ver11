# Batch 5 (remaining dpo-introduced, all AUDITED=Y)

## aPKUZi (swimmer telemetry) — KEY RIGHT (1-2-3-4); **model ignored on-screen ground-truth numbers**
Every frame has a "live data" overlay (pace/100m, stroke rate). Actual values:
  elk(In1): easy 1:16 SR54 | hun(In2): easy 1:15 SR55 | idi(In3): steady 1:12 SR60 | zfo(In4): steady 1:05 SR69
Caption: "begins easy, gradually increases pace & stroke rate, steady by final frame."
Monotonic pace/SR ⇒ order is EXACTLY 1-2-3-4 (printed on screen). Model 2-1-3-4 swapped In1/In2 — the only pair separated by fine print (1:16 vs 1:15, SR54 vs55), fell back on visual similarity of the two "easy" frames.
**CAUSE: failed to read/use small on-screen telemetry that IS the answer. OCR-of-fine-print + numeric-monotonic reasoning weakness.** ver4 also wrong; SFT right. HIGH-VALUE: this class is fixable with better text grounding.

## eEcInl (canoe under bridge) — KEY RIGHT (1-3-2-4); establishing-shot bias again
Frames: fud=bridge ahead (approach), tfq=directly under bridge, lhe=downstream smooth drop, yaw=further downstream paddling.
Key: approach(fud)→under(tfq)→past(lhe)→far(yaw). Model 2-1-3-4 put lhe (a downstream frame) FIRST as if establishing shot.
CAUSE: same establishing-shot-first prior as 706QNR — model front-loads a wide/scenic frame instead of using bridge-relative geometry. SFT+ver4 right.

## qkSjok (cat clipper + kitchen) — key 3-4-2-1, model swapped the two cat close-ups
Dispute is only gui(clippers approaching paw) vs wlf(paw at blade, mid-clip); caption "paw approaches the clippers" ⇒ gui first. Model wlf→gui. Extreme-close-up near-duplicates, thin margin.
CAUSE: coin-flip near-duplicate (approach vs contact). ver4 also wrong.

## vIO9v9 (basketball) — key 4-2-1-3, model swapped two "running with ball" frames
xeq(blurry sprint) vs gou(dribble past chair) — both mid-run; key xeq→gou, model gou→xeq. Ends on tlo (ball loose, aftermath).
CAUSE: coin-flip near-duplicate running frames. ver4 also wrong.

## xhhhpg (ice rink: zamboni→wheelchairs) — key 1-4-3-2, model swapped two zamboni frames
Both agree eot(intro) first, ivv(wheelchair) last. Middle: ulc(zamboni at edge, just entered) vs ocm(zamboni central). Key ulc→ocm (enter→proceed); model ocm→ulc.
NOTE trajectory X===: dpo200 got this RIGHT, then 400/600/800/1000 all regressed identically → DPO drift away from a correct early state.
CAUSE: near-duplicate zamboni-position pair; DPO regression over training.

---
PATTERN across all 17 dpo-introduced: model was RIGHT after SFT, DPO flipped exactly one thin-margin near-duplicate adjacent pair. 15/17 are d=1 or d=2 (single local swap), not global scrambles. Root causes cluster:
 (a) near-duplicate adjacent frames / coin-flip pairs (majority)
 (b) establishing-shot-first prior (eEcInl, 706QNR)
 (c) caption-clause literalism vs visual state (NkMj3t, CbgVj3, Kk83WO)
 (d) pre-state vs event-result confusion (CbgVj3, S0cFZA put-in/pick-up)
 (e) **ignoring on-screen text that is the literal answer (aPKUZi)** — most fixable
