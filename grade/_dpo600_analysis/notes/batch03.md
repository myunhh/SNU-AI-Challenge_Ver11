# Batch 3 (dpo-introduced, unaudited)

## S0cFZA (baby in pool) — key favored, model chose symmetric alternate narrative
Frames: 1=woman in water holding baby upright at step corner; 2=baby swims near diagonal step border, adult at left edge; 3=baby swims mid-pool, woman at edge behind; 4=baby swims close to camera.
Key 1-3-4-2: put-in (1) → swims (3→4) → reaches border (2). Model 3-4-2-1: swims → mom lifts baby at end.
A held-baby still is ambiguous between "lowering in" (start) and "lifting out" (end); caption explicitly says "puts baby IN" and never mentions a pickup → holding frame maps to start. SFT/ver4/vote agree with key.
CAUSE: symmetric-event ambiguity (put-in vs pick-up); model picked the narrative NOT in the caption. DPO regression.

## UwdDWd (glass pan into oven) — KEY RIGHT (2-3-1-4), model rotated bookends (4-2-3-1)
Frames: 1=talking at stove, glass pan WITH pale food ON stovetop; 2=carrying pan near fridge; 3=bent at open oven door; 4=talking front-and-center, stovetop EMPTY.
Key: carry (2) → open oven (3) → pan waits on stove, talk (1) → pan in oven (unshown), closing talk with empty stove (4).
Model: treated empty-stove talk (4) as INTRO and pan-on-stove (1) as final — requires an uncaptioned pan-removal after the oven step (food in 1 still pale/unbaked, so 1 is pre-bake).
CAUSE: intro-vs-outro "talking head" bookend ambiguity; model ignored the object-state chain (pan on stove pre-oven / stove empty post-oven). DPO regression; SFT+ver4 right.
