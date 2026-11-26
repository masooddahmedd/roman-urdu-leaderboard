"""Batched prompts for models we can only reach through a chat app (no API).

  python batches.py make                      -> data/batches/<task>_<k>.txt  (+ index.json)
  python batches.py parse <slug> <replies_dir> -> results/raw/<slug>.jsonl
The replies dir holds <task>_<k>.txt files with the model's raw reply to each batch.
"""
import glob
import json
import os
import re
import sys

SIZES = {"translit": 30, "reading": 15, "translate_en": 20, "sentiment": 50}
HEAD = {
    "translit": "Convert each numbered Roman Urdu sentence below into Urdu script (Perso-Arabic letters). "
                "Keep the words exactly; write names and English words the way they are normally written in Urdu.",
    "reading": "Each numbered item has a Roman Urdu passage, a question and four options. "
               "Answer each with the number of the correct option (1, 2, 3 or 4).",
    "translate_en": "Translate each numbered Roman Urdu passage into natural English. "
                    "Put each whole translation on a single line.",
    "sentiment": "Give the sentiment of each numbered Roman Urdu comment: positive, negative or neutral.",
}
TAIL = ("Reply with exactly {n} lines, one per item, in the format `<number>. <answer>`. "
        "No other text, no code block, no explanations.")


def load(task):
    return [json.loads(l) for l in open(f"data/tasks/{task}.jsonl", encoding="utf8")]


def body(task, it):
    if task == "translit":
        return it["meta"]["src"]
    if task == "reading":
        # strip the per-item instructions; keep passage, question, options
        p = it["prompt"].split("Passage: ", 1)[1].split("\n\nReply with")[0]
        return "Passage: " + p.replace("\n\n", "\n")
    src = it["prompt"].rsplit("\n\n", 1)[1]
    return src.replace("\n", " ")


def make():
    os.makedirs("data/batches", exist_ok=True)
    index = {}
    for task, size in SIZES.items():
        items = load(task)
        for k in range(0, len(items), size):
            chunk = items[k:k + size]
            parts = [HEAD[task], TAIL.format(n=len(chunk)), ""]
            for j, it in enumerate(chunk, 1):
                parts.append(f"{j}. {body(task, it)}")
                if task == "reading":
                    parts.append("")
            name = f"{task}_{k // size:02d}"
            open(f"data/batches/{name}.txt", "w", encoding="utf8").write("\n".join(parts).strip() + "\n")
            index[name] = [it["id"] for it in chunk]
    json.dump(index, open("data/batches/index.json", "w"), indent=0)
    print(len(index), "batches")


def parse(slug, rdir):
    index = json.load(open("data/batches/index.json"))
    rows, missing = [], 0
    for name, ids in index.items():
        f = f"{rdir}/{name}.txt"
        if not os.path.exists(f):
            missing += len(ids)
            continue
        ans = {}
        for line in open(f, encoding="utf8"):
            m = re.match(r"\s*\**(\d+)\**\s*[\.\):\-]\s*(.*)", line)
            if m and m.group(2).strip():
                ans.setdefault(int(m.group(1)), m.group(2).strip())
        for j, i in enumerate(ids, 1):
            ok = j in ans
            missing += not ok
            rows.append({"id": i, "task": i.rsplit("-", 1)[0], "ok": ok, "out": ans.get(j, ""), "batched": True,
                         "cost": 0})
    os.makedirs("results/raw", exist_ok=True)
    with open(f"results/raw/{slug}.jsonl", "w", encoding="utf8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(slug, "answers", sum(r["ok"] for r in rows), "missing", missing)


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else parse(sys.argv[2], sys.argv[3])
