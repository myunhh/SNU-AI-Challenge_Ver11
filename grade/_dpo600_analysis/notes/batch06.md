# Batch 6 (both-wrong-diff: model & SFT both wrong, different answers) — the genuinely HARD tier

## 6fko8F (instructional pool video, 2 people) — key 1-2-3-4 (literal caption order)
blue-shirt student leans (hts) → overhead (nce) → yellow instructor stands+gestures (xhg) → instructor lines up (yph). Key follows caption clauses exactly. Model 1-4-2-3 and SFT 4-2-1-3 both scrambled the two-person multi-shot structure.
CAUSE: multi-actor instructional video; model failed to bind caption clause sequence to the right person's shots. Hard, but key is caption-supported.

## 8L7TfG (boy on monkey bars) — **LIKELY KEY ERROR; model plausibly RIGHT**
Frames: hyt=climbing UP onto bars at right (start), dkq/frh=hanging mid-crossing, hxa=at far platform (end).
Model 4-1-2-3 (hyt→dkq→frh→hxa = climb-up→cross→cross→arrive) is physically coherent. Key 1-2-3-4 ends on hyt (the climb-UP frame) which is backwards.
Key is FLAGGED low-confidence in grade.py (v10 fix "landed on a value matching NONE of key/ckpt1600/ver4"). **This is key noise, not a model failure.** Do NOT train against this.

## Qv2g53 (cement spray on wall) — hard monotonic-coverage; all 3 differ
All agree zpw first. Middle/end differ on which frame has more wet-cement coverage; camera pans so area is hard to compare. Key 4-1-2-3, model 4-2-1-3, SFT 4-2-3-1.
CAUSE: monotonic progress (coverage growth) estimation across camera-moving near-duplicates. Genuinely hard; low training value.

## YZuSuG (pumpkin carve #3) — **CONTESTED KEY** (hand-revised in v9)
Key was manually flipped [1,4,3,2]→[2,4,3,1] over a "draw vs carve / cut vs uncut" physical-impossibility argument. Model 1-4-2-3, ver4 same, SFT 4-1-2-3 — nobody agrees with anybody.
draw(fyn)/start(soy)/eyes-cut(oxc)/more-cut(hgs) ordering is ambiguous because "gloved hands on pumpkin" reads as either drawing or carving.
CAUSE: contested/hand-arbitrated key + genuine draw-vs-carve ambiguity. Weak signal.

## Z9CdGI (slackline→rock) — **KEY EXPLICITLY FLAGGED UNCERTAIN** in grade.py
"plausible counter-ordering for the two near-rock frames but no confident resolution." Model, SFT, ver4 all differ. Not clean signal.

## a90lEu (hair braiding) — no consensus (model/SFT/key all differ; dpo800/1000 → yet another value)
Fine-grained near-duplicate braiding frames; nobody stable. Hard, low value.

## b44PjE (girl on bars, "climbs back + jumps down midway") — d=5 large scramble
Model 3-1-4-2 vs key 2-1-4-3; SFT 3-4-1-2. Big disagreement on a fast playground-motion clip (cf 8L7TfG same domain). Possible key-fragility domain.

---
KEY TAKEAWAY for both-wrong-diff (7): at least 3 of 7 (8L7TfG, Z9CdGI, YZuSuG) are contested/flagged/likely-wrong KEYS, not model failures. The remaining are genuinely hard (multi-actor, monotonic-coverage, fast playground motion). This tier gives LITTLE clean training signal — much of it is answer-key noise on the hardest 4-frame clips.
