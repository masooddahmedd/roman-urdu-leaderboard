"""Score every results/raw/*.jsonl and write results/leaderboard.json + a console table."""
import glob
import json
import os

from score import task_score

TASKS = ["translit", "reading", "translate_en", "sentiment"]
# Translation is shown but left out of Overall: its English references are public FLORES text,
# and at least one model reproduces many of them near-verbatim (see contamination check).
OVERALL = ["translit", "reading", "sentiment"]
refs = {}
for t in TASKS:
    for line in open(f"data/tasks/{t}.jsonl", encoding="utf8"):
        it = json.loads(line)
        refs[it["id"]] = it["ref"]

meta = json.load(open("models.json", encoding="utf8")) if os.path.exists("models.json") else {}
board = []
for f in sorted(glob.glob("results/raw/*.jsonl")):
    slug = os.path.basename(f)[:-6]
    ans = {}
    cost = 0.0
    for line in open(f, encoding="utf8"):
        r = json.loads(line)
        if r.get("ok"):
            ans[r["id"]] = r["out"]
            cost += r.get("cost") or 0
    row = {"slug": slug, **meta.get(slug, {}), "tasks": {}, "cost_usd": round(cost, 3)}
    complete = True
    for t in TASKS:
        ids = [i for i in refs if i.startswith(t + "-")]
        if not all(i in ans for i in ids):
            complete = False
            continue
        s, extra = task_score(t, [(ans[i], refs[i]) for i in ids])
        row["tasks"][t] = {"score": round(s, 1), **extra}
    row["complete"] = complete
    if complete:
        row["overall"] = round(sum(row["tasks"][t]["score"] for t in OVERALL) / len(OVERALL), 1)
    board.append(row)

board.sort(key=lambda r: -(r.get("overall") or -1))
json.dump({"tasks": TASKS, "overall_tasks": OVERALL, "models": board}, open("results/leaderboard.json", "w", encoding="utf8"), indent=1,
          ensure_ascii=False)
hdr = f"{'model':45s} {'overall':>7s} " + " ".join(f"{t[:10]:>10s}" for t in TASKS) + "   cost$"
print(hdr)
for r in board:
    cells = " ".join(f"{r['tasks'].get(t, {}).get('score', float('nan')):10.1f}" for t in TASKS)
    print(f"{r['slug'][:45]:45s} {r.get('overall', float('nan')):7.1f} {cells}   {r['cost_usd']:.2f}")
