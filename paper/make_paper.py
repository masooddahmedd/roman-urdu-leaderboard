"""Write paper/numbers.tex and paper/tables.tex from results. Never type numbers into main.tex by hand.

  python paper/analysis.py && python paper/make_paper.py   (from the repo root)
"""
import json
import sys

import sacrebleu

sys.path.insert(0, ".")
from score import clean  # noqa: E402

A = json.load(open("paper/analysis.json", encoding="utf8"))
L = json.load(open("results/leaderboard.json", encoding="utf8"))
M = A["models"]
best = A["best_ours"]
paired = A["paired_vs_best_ours"]

SHORT = {
    "claude__fable-5.2": "Claude Fable 5.2",
    "chatgpt__gpt-6": "GPT-6",
    "ours__qwen3-8b-translit": "Ours: Qwen3-8B + QLoRA",
    "ours__qwen2.5-1.5b-translit-v2": "Ours: Qwen2.5-1.5B + QLoRA",
    "ollama__llama3.1_8b": "Llama 3.1 8B Instruct",
    "ollama__qwen2.5_7b-instruct": "Qwen2.5 7B Instruct",
    "ollama__qwen2.5_3b-instruct": "Qwen2.5 3B Instruct",
    "m2m100__rup_rur_to_ur": "m2m100 418M (Butt et al.)",
    "ollama__qwen3_14b": "Qwen3 14B",
    "ollama__gpt-oss_20b": "gpt-oss 20B",
}
GROUP = {"claude__fable-5.2": "Frontier (chat app)", "chatgpt__gpt-6": "Frontier (chat app)"}


def group(s):
    if s in GROUP:
        return GROUP[s]
    if M[s]["ours"]:
        return "Ours (fine-tuned)"
    if s.startswith("m2m100"):
        return "Specialist"
    return "Open, zero/few-shot"


def f(x, d=1):
    return f"{x:.{d}f}"


macros = {}


def mac(name, val):
    macros[name] = val


open_rows = [s for s in M if group(s) == "Open, zero/few-shot"]
best_open = min(open_rows, key=lambda s: M[s]["cer"])
frontier = [s for s in M if group(s).startswith("Frontier")]
best_front = min(frontier, key=lambda s: M[s]["cer"])
small = "ours__qwen2.5-1.5b-translit-v2"

mac("NTest", str(A["n"]))
mac("OursCER", f(M[best]["cer"]))
mac("OursScore", f(100 - M[best]["cer"]))
mac("OursWER", f(M[best]["wer"]))
mac("OursExact", f(M[best]["exact"], 0))
mac("OursCIlo", f(M[best]["cer_ci"][0]))
mac("OursCIhi", f(M[best]["cer_ci"][1]))
mac("SmallCER", f(M[small]["cer"]))
mac("SmallScore", f(100 - M[small]["cer"]))
mac("SmallWER", f(M[small]["wer"]))
mac("FableCER", f(M["claude__fable-5.2"]["cer"]))
mac("FableScore", f(100 - M["claude__fable-5.2"]["cer"]))
mac("FableWER", f(M["claude__fable-5.2"]["wer"]))
mac("GptCER", f(M["chatgpt__gpt-6"]["cer"]))
mac("GptScore", f(100 - M["chatgpt__gpt-6"]["cer"]))
mac("GptWER", f(M["chatgpt__gpt-6"]["wer"]))
mac("GapFront", f(M[best]["cer"] - M[best_front]["cer"]))
mac("GapGpt", f(M[best]["cer"] - M["chatgpt__gpt-6"]["cer"]))
mac("BestOpenName", SHORT.get(best_open, M[best_open]["name"]))
mac("BestOpenCER", f(M[best_open]["cer"]))
mac("BestOpenScore", f(100 - M[best_open]["cer"]))
mac("BestOpenWER", f(M[best_open]["wer"]))
mac("MarginOpen", f(M[best_open]["cer"] - M[best]["cer"]))
mac("ErrCut", f(100 * (1 - M[best]["cer"] / M[best_open]["cer"]), 0))
mac("NOpen", str(len(open_rows)))
mac("SmallToBigCER", f(M[small]["cer"] - M[best]["cer"]))
if "m2m100__rup_rur_to_ur" in M:
    mac("MtwoMCER", f(M["m2m100__rup_rur_to_ur"]["cer"]))
    mac("MtwoMWER", f(M["m2m100__rup_rur_to_ur"]["wer"]))
    mac("MarginMtwoM", f(M["m2m100__rup_rur_to_ur"]["cer"] - M[best]["cer"]))
if "zeroshot__qwen3-8b-base" in M:
    mac("BaseCER", f(M["zeroshot__qwen3-8b-base"]["cer"]))
    mac("BaseWER", f(M["zeroshot__qwen3-8b-base"]["wer"]))

# Training-set facts
tr = sum(1 for _ in open("data/train/dakshina_train.jsonl", encoding="utf8"))
trl = sum(1 for _ in open("data/train/dakshina_train_lex.jsonl", encoding="utf8"))
mac("NTrainSent", f"{tr:,}".replace(",", "{,}"))
mac("NTrainLex", f"{trl - tr:,}".replace(",", "{,}"))
mac("NTrainAll", f"{trl:,}".replace(",", "{,}"))

# Contamination check on the translation task: per-passage chrF of each model against the FLORES reference.
tasks = {json.loads(l)["id"]: json.loads(l) for l in open("data/tasks/translate_en.jsonl", encoding="utf8")}
contam = {}
for s in ["claude__fable-5.2", "chatgpt__gpt-6"]:
    outs = {}
    for l in open(f"results/raw/{s}.jsonl", encoding="utf8"):
        r = json.loads(l)
        if r["task"] == "translate_en":
            outs[r["id"]] = clean(r.get("out") or "")
    sc = [sacrebleu.sentence_chrf(outs[i], [tasks[i]["ref"]], word_order=2).score for i in tasks if i in outs]
    contam[s] = {"n": len(sc), "over90": sum(x > 90 for x in sc), "over95": sum(x > 95 for x in sc),
                 "corpus": next(m["tasks"]["translate_en"]["score"] for m in L["models"] if m["slug"] == s)}
mac("ContNTr", str(contam["claude__fable-5.2"]["n"]))
mac("ContFableNinety", str(contam["claude__fable-5.2"]["over90"]))
mac("ContGptNinety", str(contam["chatgpt__gpt-6"]["over90"]))
mac("ContFableNinetyFive", str(contam["claude__fable-5.2"]["over95"]))
mac("ContGptNinetyFive", str(contam["chatgpt__gpt-6"]["over95"]))
mac("ContFableChrf", f(contam["claude__fable-5.2"]["corpus"]))
mac("ContGptChrf", f(contam["chatgpt__gpt-6"]["corpus"]))

# Script leaks: outputs containing Devanagari or a Latin word
import re  # noqa: E402
def leaks(slug):
    o = [clean(json.loads(l)["out"] or "") for l in open(f"results/raw/{slug}.jsonl", encoding="utf8")
         if json.loads(l)["task"] == "translit"]
    return sum(bool(re.search("[ऀ-ॿ]", x)) for x in o), sum(bool(re.search("[A-Za-z]{2,}", x)) for x in o)
mac("QsevenDeva", str(leaks("ollama__qwen2.5_7b-instruct")[0]))
mac("QsevenLatin", str(leaks("ollama__qwen2.5_7b-instruct")[1]))
mac("QthreeLatin", str(leaks("ollama__qwen2.5_3b-instruct")[1]))
mac("OursDeva", str(leaks(best)[0]))
rawd = [M[s]["raw_cer"] - M[s]["cer"] for s in M]
mac("RawAddMin", f(min(rawd)))
mac("RawAddMax", f(max(rawd)))

with open("paper/numbers.tex", "w", encoding="utf8") as fh:
    fh.write("% Written by make_paper.py. Do not edit.\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")

# Main table
order = sorted(M, key=lambda s: M[s]["cer"])
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"System & CER $\downarrow$ & 95\% CI & WER $\downarrow$ & Exact \\", r"\midrule"]
for s in order:
    m = M[s]
    name = SHORT.get(s, m["name"])
    if m["ours"]:
        name = r"\textbf{" + name + "}"
    lines.append(f"{name} & {f(m['cer'], 2)} & {f(m['cer_ci'][0])}--{f(m['cer_ci'][1])} & {f(m['wer'])} & "
                 f"{f(m['exact'], 0)}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Roman Urdu to Urdu-script transliteration on \NTest Dakshina sentences. CER and WER in \%, "
          r"after Urdu normalization (\S\ref{sec:metric}); 95\% CI by bootstrap over sentences; Exact = share of "
          r"sentences with no character error. Bold: our fine-tuned models.}",
          r"\label{tab:main}", r"\end{table}", ""]

# Paired differences
lines += [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lrr@{}}", r"\toprule",
          r"System & $\Delta$CER vs.\ ours & 95\% CI \\", r"\midrule"]
for s in order:
    if s in paired:
        p = paired[s]
        lines.append(f"{SHORT.get(s, M[s]['name'])} & {p['cer_diff']:+.2f} & "
                     f"{p['ci'][0]:+.2f} to {p['ci'][1]:+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Paired bootstrap difference in CER (points) between each system and our Qwen3-8B model on "
          r"the same sentences. Positive: more errors than ours.}", r"\label{tab:paired}", r"\end{table}", ""]

# Metric robustness
lines += [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lrrr@{}}", r"\toprule",
          r"System & Norm.\ CER & Raw CER & No-space CER \\", r"\midrule"]
for s in order:
    m = M[s]
    lines.append(f"{SHORT.get(s, m['name'])} & {f(m['cer'], 2)} & {f(m['raw_cer'], 2)} & {f(m['nospace_cer'], 2)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{CER under three scorings: our normalized metric, raw Unicode (NFC only), and normalized with "
          r"all spaces removed (ignores word-boundary errors). The ranking does not change.}",
          r"\label{tab:metric}", r"\end{table}", ""]
open("paper/tables.tex", "w", encoding="utf8").write("\n".join(lines))
print(json.dumps(macros, indent=1, ensure_ascii=False))
print(json.dumps(contam, indent=1))
