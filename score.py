"""Scorers. Every task score is 0-100, higher is better."""
import re
import unicodedata

import sacrebleu

# Urdu normalisation: drop diacritics (aerab), unify letter variants, strip punctuation/space differences.
_DIAC = re.compile("[ً-ٰٟۖ-ۭؐ-ؚ]")
_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ے": "ے",  # yeh forms -> farsi yeh (bari ye kept)
    "ك": "ک",  # arabic kaf -> keheh
    "ه": "ہ", "ھ": "ھ",  # heh -> heh goal (do-chashmi kept)
    "ة": "ۃ",
    "ۀ": "ۂ",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})
_PUNCT = re.compile(r"[\s\.\,\!\?\:\;\"\'\(\)\[\]\-۔،؟؛٫٬«»“”‘’]+")


def norm_ur(s):
    s = unicodedata.normalize("NFC", s)
    s = _DIAC.sub("", s).translate(_MAP)
    s = s.replace("‌", "").replace("‍", "").replace("ـ", "")
    return _PUNCT.sub(" ", s).strip()


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    h, r = norm_ur(hyp), norm_ur(ref)
    return min(1.0, _lev(h, r) / max(1, len(r)))


def wer_ur(hyp, ref):
    h, r = norm_ur(hyp).split(), norm_ur(ref).split()
    return min(1.0, _lev(h, r) / max(1, len(r)))


def clean(out):
    """Strip reasoning tags, code fences and labels models sometimes add."""
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S)
    out = re.sub(r"```[a-z]*", "", out).strip()
    return out


def parse_choice(out):
    m = re.search(r"\b([1-4])\b", clean(out))
    return m.group(1) if m else None


def parse_sentiment(out):
    o = clean(out).lower()
    hits = [(o.find(w), w) for w in ("positive", "negative", "neutral") if w in o]
    return min(hits)[1] if hits else None


def task_score(task, rows):
    """rows: list of (hyp, ref). Returns (score_0_100, extra dict)."""
    n = len(rows)
    if task.startswith("translit"):
        c = sum(cer(h, r) for h, r in rows) / n
        w = sum(wer_ur(h, r) for h, r in rows) / n
        return 100 * (1 - c), {"cer": round(100 * c, 2), "wer": round(100 * w, 2)}
    if task == "reading":
        acc = sum(parse_choice(h) == r for h, r in rows) / n
        return 100 * acc, {}
    if task == "sentiment":
        preds = [parse_sentiment(h) for h, _ in rows]
        f1s = []
        for lab in ("positive", "negative", "neutral"):
            tp = sum(p == lab and r == lab for p, (_, r) in zip(preds, rows))
            fp = sum(p == lab and r != lab for p, (_, r) in zip(preds, rows))
            fn = sum(p != lab and r == lab for p, (_, r) in zip(preds, rows))
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        acc = sum(p == r for p, (_, r) in zip(preds, rows)) / n
        return 100 * sum(f1s) / 3, {"acc": round(100 * acc, 1)}
    if task == "translate_en":
        hyps = [clean(h) for h, _ in rows]
        refs = [r for _, r in rows]
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        return chrf, {}
    raise ValueError(task)
