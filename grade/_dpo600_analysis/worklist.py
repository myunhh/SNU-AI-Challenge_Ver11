"""Build per-item inspection blocks (sorted: dpo-introduced > both-wrong-diff >
inherited; unaudited first within each group) + positional-bias stats."""
import csv
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent.parent

rows = list(csv.DictReader(open(HERE / "wrong59.tsv", encoding="utf-8"), delimiter="\t"))

order = {"dpo-introduced(sft-was-right)": 0, "both-wrong-diff": 1, "inherited-same": 2}
rows.sort(key=lambda r: (order[r["origin"]], r["audited73"] == "Y", r["Id"]))

with open(HERE / "items.txt", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        sid = r["Id"]
        files = sorted(os.listdir(DATA / "test" / sid))
        f.write(f"### [{i:02d}] {sid}  origin={r['origin']}  type={r['type']}  audited={r['audited73']}\n")
        for j, fn in enumerate(files, 1):
            f.write(f"  Input_{j}: C:/vsc/data/test/{sid}/{fn}\n")
        f.write(f"  caption: {r['caption']}\n")
        f.write(f"  model temporal seq : {'-'.join(r['pred_seq'])}   (ranks {r['pred']})\n")
        f.write(f"  key   temporal seq : {'-'.join(r['truth_seq'])}   (ranks {r['truth']})\n")
        f.write(f"  sft1600 {r['sft1600']}  ver4 {r['ver4']}  dpo-traj {r['dpo_traj_200_400_800_1000']}\n\n")

print(f"wrote items.txt ({len(rows)} items)")

# positional bias across all 819 predictions vs key
import re


def load(path):
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
        ranks = tuple(int(c) for c in re.findall(r"[1-4]", row["Answer"] or ""))
        if len(ranks) == 4:
            out[row["Id"].strip()] = ranks
    return out


key = load(HERE.parent / "submission.csv")
dpo = load(HERE.parent / "submission-ver8-dpo-ckpt600_0.91099.csv")

print("\nrank distribution per input slot (key vs dpo600), all 819:")
for slot in range(4):
    kc = Counter(v[slot] for v in key.values())
    dc = Counter(v[slot] for v in dpo.values())
    print(f"  Input_{slot+1}: key {[kc[r] for r in (1,2,3,4)]}   dpo {[dc[r] for r in (1,2,3,4)]}")
