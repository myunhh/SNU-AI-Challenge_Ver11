# Batch 1 (dpo-introduced, unaudited)

## 0AezL1 (pole vault + MBC idol show composite) — KEY RIGHT, model wrong
Frames: 1=4.70m attempt#3 (athlete at mat), 2=MBC red-shirt girl walking on mat, 3=4.41m mid-air vault, 4=MBC orange girl celebrating arms up.
Two-clip composite: clip A = pole vault (3→1: mid-air 4.41 → at mat, marker now 4.70), clip B = MBC (2→4).
Caption states A is continuous ("transitions from mid-air to sitting on mat as marker changes 4.41→4.70"), then pans to celebrating athlete.
Key 3-1-2-4 correct. Model 3-2-1-4 inserted MBC red-shirt frame *between* the two pole-vault frames — broke same-scene grouping that both caption and visual similarity dictate.
CAUSE: scene-grouping failure in 2-clip composite (DPO regression; SFT+ver4 right).

## 151Vxf (kid puts leaf in trash bin) — KEY plausible, model wrong
Frames: 1=close-up profile of kid carrying leaf (street bg), 2=kid standing in leaf pile (rake visible), 3=kid at trash bin by garage reaching up, 4=kid mid-shot walking toward camera grinning, holding leaf/stick.
Story: grabs leaf at pile (2) → walks toward camera (4) → passes close (1, close-up) → deposits at bin (3).
Key 2-4-1-3; model 2-1-4-3 (swapped the two "walking with leaf" near-duplicates).
Camera-distance continuity (mid-shot then close-up then turn to bin) supports key; SFT+ver4 agreed with key. Unaudited, mild uncertainty but key favored.
CAUSE: near-duplicate mid-sequence frames, order carried only by shot-flow continuity; DPO flipped a coin-flip pair.

## 706QNR (diving tower) — ambiguous near-duplicates, key weakly right
Frames: 1=tight top-of-tower shot, climber in orange shorts still on ladder; 2=wide tower shot, people on mid platform, figure standing at top; 3=diver mid-flip; 4=splash in water.
3→4 certain. 1 vs 2 first: climber mid-climb (1) → standing at top (2) suggests 1→2 (key 1-2-3-4). Model+ver4 say 2-1-3-4 (wide-establishing-shot-first prior).
Unaudited; key = vote (SFT side). Stills give only a weak pose cue (climbing vs standing).
CAUSE: near-duplicate platform frames; subtle pose cue; DPO moved to ver4's side of a coin flip. Flag: key confidence moderate.

## CbgVj3 (Strictly dance, 2 dresses = 2 clips) — KEY RIGHT, model wrong
Frames: 1=gold dress facing partner close (band bg), 2=gold dress back to camera, 3=red dress leg raised, chair on stage, 4=wide stage reveal red dress.
Composite: clip A gold (1→2), clip B red (3→4). Caption: "turns away, showcasing her back (zoom out), then lowers her leg, moves into close embrace, camera pans out to reveal stage."
Key 1-2-3-4 correct: 1 is the PRE-state (facing) before "turns away" produces 2 (back view); 3 (leg up) precedes lowering; 4 = stage reveal.
Model 2-1-3-4: matched the back-view frame to the caption's first clause literally and put it first, missing that frame 1 is the initial state before the described transition.
CAUSE: pre-state vs event-result confusion — caption's first named event describes a *change*; the frame showing the change's result belongs after the neutral initial-state frame. DPO regression (SFT right).
