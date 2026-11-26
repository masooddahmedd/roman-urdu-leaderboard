"""Baseline: the m2m100-418M Roman Urdu -> Urdu model released with Butt et al. (LoResMT 2025), on CPU.

  CUDA_VISIBLE_DEVICES= python run_m2m.py MODEL_DIR   -> results/raw/m2m100__rup_rur_to_ur.jsonl
MODEL_DIR holds local copies of Mavkif/m2m100_rup_rur_to_ur and Mavkif/m2m100_rup_tokenizer_both.
Same settings as the author's earlier retrieval paper: source code "ur", 4 beams.
"""
import json
import os
import sys
import time

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

d = sys.argv[1]
tok = M2M100Tokenizer.from_pretrained(os.path.join(d, "m2m100_rup_tokenizer_both"))
tok.src_lang = "ur"
model = M2M100ForConditionalGeneration.from_pretrained(os.path.join(d, "m2m100_rup_rur_to_ur")).eval()
items = [json.loads(l) for l in open("data/tasks/translit.jsonl", encoding="utf8")]
os.makedirs("results/raw", exist_ok=True)
out = open("results/raw/m2m100__rup_rur_to_ur.jsonl", "w", encoding="utf8")
for i in range(0, len(items), 16):
    chunk = items[i:i + 16]
    enc = tok([it["meta"]["src"] for it in chunk], return_tensors="pt", padding=True, truncation=True)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("ur"), num_beams=4, max_length=400)
    sec = (time.time() - t0) / len(chunk)
    for it, o in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
        out.write(json.dumps({"id": it["id"], "task": "translit", "ok": True, "out": o.strip(), "sec": round(sec, 3),
                              "cost": 0}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"{i + len(chunk)}/{len(items)}", flush=True)
print("done")
