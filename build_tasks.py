"""Build the Roman Urdu benchmark (v1) from human-made public sources.

Writes data/tasks/<task>.jsonl, one item per line: {id, task, prompt, ref, meta}.
Every source is human-written or human-romanized; nothing here is machine-generated.
"""
import json
import os
import random

import pandas as pd

RAW = "data/raw"
OUT = "data/tasks"
SEED = 2611
os.makedirs(OUT, exist_ok=True)


def write(task, items):
    with open(f"{OUT}/{task}.jsonl", "w", encoding="utf8") as fh:
        for i, it in enumerate(items):
            it = {"id": f"{task}-{i:04d}", "task": task, **it}
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(task, len(items))


TRANSLIT = (
    "Convert this Roman Urdu text into Urdu script (Nastaliq / Perso-Arabic letters). "
    "Keep the meaning and words exactly; write names and English words the way they are normally written in Urdu. "
    "Reply with the Urdu text only.\n\n{src}"
)

# 1. Transliteration: Dakshina (Urdu Wikipedia sentences romanized by native-speaker annotators).
# (Roman-Urdu-Parl was checked and dropped: its Roman side is machine-made and often garbled.)
d = pd.read_parquet(f"{RAW}/dakshina_ur.parquet")
d = d[(d.native_word_count >= 6) & (d.native_word_count <= 30)].drop_duplicates("native")
test = d.sample(n=300, random_state=SEED)
write("translit", [
    {"prompt": TRANSLIT.format(src=r.romanized), "ref": r.native, "meta": {"src": r.romanized, "source": "dakshina"}}
    for r in test.itertuples()
])
# Everything else in Dakshina is free for training our own model (never the test rows).
full = pd.read_parquet(f"{RAW}/dakshina_ur.parquet")
train = full[~full.native.isin(set(test.native))]
os.makedirs("data/train", exist_ok=True)
train[["romanized", "native"]].to_json("data/train/dakshina_train.jsonl", orient="records", lines=True,
                                       force_ascii=False)
print("dakshina train rows", len(train))

# 3. Reading comprehension: Belebele urd_Latn (romanized by annotators), 4-way multiple choice.
def load(name):
    return [json.loads(l) for l in open(f"{RAW}/belebele_{name}.jsonl", encoding="utf8")]

bl = load("urd_Latn")
rng = random.Random(SEED)
mc = rng.sample(bl, 300)
MC = (
    "Read the passage and answer the question. Both are in Roman Urdu.\n\n"
    "Passage: {p}\n\nQuestion: {q}\n\n1. {a1}\n2. {a2}\n3. {a3}\n4. {a4}\n\n"
    "Reply with the number of the correct answer only (1, 2, 3 or 4)."
)
write("reading", [
    {"prompt": MC.format(p=x["flores_passage"], q=x["question"], a1=x["mc_answer1"], a2=x["mc_answer2"],
                         a3=x["mc_answer3"], a4=x["mc_answer4"]),
     "ref": str(x["correct_answer_num"]), "meta": {"link": x["link"], "qn": x["question_number"]}}
    for x in mc
])

# 4. Translation to English: Belebele passages are FLORES passages, so urd_Latn and eng_Latn align by link.
en = {}
for x in load("eng_Latn"):
    en.setdefault(x["link"], x["flores_passage"])
passages = {}
for x in bl:
    passages.setdefault(x["link"], x["flores_passage"])
links = sorted(l for l in passages if l in en)
links = rng.sample(links, 120)
TR = "Translate this Roman Urdu text into natural English. Reply with the English translation only.\n\n{src}"
write("translate_en", [
    {"prompt": TR.format(src=passages[l]), "ref": en[l], "meta": {"link": l}} for l in links
])

# 5. Sentiment of real social-media comments (UCI Roman Urdu dataset; 0 pos, 1 neg, 2 neutral).
s = pd.read_parquet(f"{RAW}/roman_urdu_sent.parquet").dropna()
s["sentence"] = s.sentence.str.strip()
s = s[(s.sentence.str.split().str.len() >= 4) & (s.sentence.str.len() <= 400)].drop_duplicates("sentence")
names = {0: "positive", 1: "negative", 2: "neutral"}
parts = [s[s.sentiment == k].sample(n=100, random_state=SEED) for k in names]
s = pd.concat(parts).sample(frac=1, random_state=SEED)
SE = (
    "What is the sentiment of this Roman Urdu comment? "
    "Reply with one word: positive, negative or neutral.\n\n{src}"
)
write("sentiment", [
    {"prompt": SE.format(src=x.sentence), "ref": names[x.sentiment], "meta": {}} for x in s.itertuples()
])
