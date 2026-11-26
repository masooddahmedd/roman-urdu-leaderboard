"""Add word-level training examples: word pairs from the Dakshina word alignments, eight per line.

Reads the `lexicon_mapping` (word alignment) of every Dakshina sentence that is NOT in the test set,
collects the distinct (romanized word, Urdu word) pairs, and groups them eight at a time into
pseudo-sentences. Writes data/train/dakshina_train_lex.jsonl = training sentences + pseudo-sentences.

  python make_lex_train.py   (after build_tasks.py)

The exact file used in the paper (data/train/dakshina_train_lex.jsonl, 9,458 + 5,501 rows) is in the
repository. This script sorts the pairs before shuffling so it is deterministic; the original run did
not sort, so a rebuild has the same rows in a different order and grouping.
"""
import json
import random

import pandas as pd

random.seed(1)
d = pd.read_parquet("data/raw/dakshina_ur.parquet")
test = {json.loads(l)["ref"] for l in open("data/tasks/translit.jsonl", encoding="utf8")}
pairs = set()
for nat, lm in zip(d.native, d.lexicon_mapping):
    if nat in test or not lm:
        continue
    for w, roms in json.loads(lm).items():
        for r in roms:
            pairs.add((r, w))
pairs = sorted(pairs)
random.shuffle(pairs)
rows = [json.loads(l) for l in open("data/train/dakshina_train.jsonl", encoding="utf8")]
lex = [{"romanized": " ".join(r for r, _ in pairs[i:i + 8]), "native": " ".join(w for _, w in pairs[i:i + 8])}
       for i in range(0, len(pairs), 8)]
out = rows + lex
random.shuffle(out)
with open("data/train/dakshina_train_lex.jsonl", "w", encoding="utf8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), len(lex), len(out))
