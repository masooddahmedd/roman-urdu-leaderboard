"""Transliteration analysis for the paper: CIs, paired tests, raw vs normalised scores, error types.

  python paper/analysis.py   (from the repo root)  -> paper/analysis.json
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, ".")
from score import _lev, cer, clean, norm_ur, wer_ur  # noqa: E402

B = 2000
random.seed(2611)
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/tasks/translit.jsonl", encoding="utf8")}
ids = sorted(items)
models = json.load(open("models.json", encoding="utf8"))


def load(slug):
    rows = {}
    for l in open(f"results/raw/{slug}.jsonl", encoding="utf8"):
        r = json.loads(l)
        if r["task"] == "translit":
            rows[r["id"]] = clean(r.get("out") or "")
    return rows


def raw_cer(h, r):
    h, r = unicodedata.normalize("NFC", h).strip(), unicodedata.normalize("NFC", r).strip()
    return min(1.0, _lev(h, r) / max(1, len(r)))


def nospace_cer(h, r):
    h, r = norm_ur(h).replace(" ", ""), norm_ur(r).replace(" ", "")
    return min(1.0, _lev(h, r) / max(1, len(r)))


def boot(vals):
    n = len(vals)
    ms = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def boot_diff(a, b):
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    ms = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, ms[int(0.025 * B)], ms[int(0.975 * B)]


def align_ops(h, r):
    """Character substitutions (ref->hyp) from a Levenshtein alignment."""
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, subs = n, m, []
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]:
                subs.append((r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return subs


LATIN = re.compile(r"[A-Za-z]")
slugs = [s for s in sorted({f[:-6] for f in __import__("os").listdir("results/raw")}) if s in models]
out = {"n": len(ids), "models": {}}
per = {}
for s in slugs:
    rows = load(s)
    if len(rows) < len(ids):
        continue
    h = [rows[i] for i in ids]
    r = [items[i]["ref"] for i in ids]
    c = [cer(a, b) for a, b in zip(h, r)]
    per[s] = c
    w = [wer_ur(a, b) for a, b in zip(h, r)]
    rc = [raw_cer(a, b) for a, b in zip(h, r)]
    ns = [nospace_cer(a, b) for a, b in zip(h, r)]
    subs = Counter()
    for a, b in zip(h, r):
        subs.update(align_ops(norm_ur(a), norm_ur(b)))
    lo, hi = boot(c)
    out["models"][s] = {
        "name": models[s]["name"], "ours": bool(models[s].get("ours")),
        "cer": 100 * sum(c) / len(c), "cer_ci": [100 * lo, 100 * hi],
        "wer": 100 * sum(w) / len(w), "raw_cer": 100 * sum(rc) / len(rc), "nospace_cer": 100 * sum(ns) / len(ns),
        "exact": 100 * sum(x == 0 for x in c) / len(c),
        "latin_left": sum(bool(LATIN.search(a)) for a in h),
        "top_subs": [[a, b, k] for (a, b), k in subs.most_common(8)],
    }

best = max((s for s in per if out["models"][s]["ours"]), key=lambda s: -out["models"][s]["cer"])
out["best_ours"] = best
out["paired_vs_best_ours"] = {}
for s in per:
    if s != best:
        d, lo, hi = boot_diff(per[s], per[best])
        out["paired_vs_best_ours"][s] = {"cer_diff": 100 * d, "ci": [100 * lo, 100 * hi]}

# Length buckets (ref words) for the strongest systems
buckets = [(0, 12), (12, 20), (20, 30), (30, 999)]
lens = [len(norm_ur(items[i]["ref"]).split()) for i in ids]
out["by_length"] = {}
for s in per:
    out["by_length"][s] = []
    for lo, hi in buckets:
        v = [c for c, L in zip(per[s], lens) if lo <= L < hi]
        out["by_length"][s].append({"words": f"{lo}-{hi}", "n": len(v), "cer": 100 * sum(v) / max(1, len(v))})

json.dump(out, open("paper/analysis.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
for s, m in sorted(out["models"].items(), key=lambda x: x[1]["cer"]):
    p = out["paired_vs_best_ours"].get(s)
    print(f"{s:34s} CER {m['cer']:5.2f} [{m['cer_ci'][0]:.2f},{m['cer_ci'][1]:.2f}] WER {m['wer']:5.2f} "
          f"raw {m['raw_cer']:5.2f} nospace {m['nospace_cer']:5.2f} exact {m['exact']:4.1f}% "
          + (f"vs ours {p['cer_diff']:+.2f} [{p['ci'][0]:+.2f},{p['ci'][1]:+.2f}]" if p else "(best ours)"))
    print("   subs", m["top_subs"][:5])
