"""Run our fine-tuned transliteration model on the benchmark translit items (greedy decoding).

Writes results/raw/<name>.jsonl in the same format run_eval.py uses.
  python eval_local.py --base <Qwen dir> --lora runs/qwen15_translit --name ours__qwen2.5-1.5b-translit
"""
import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--lora", required=True)
p.add_argument("--name", required=True)
p.add_argument("--tasks", default="data/tasks/translit.jsonl")
p.add_argument("--bs", type=int, default=16)
p.add_argument("--mem_frac", type=float, default=0.28)
p.add_argument("--qlora", action="store_true", help="load the base in 4-bit and keep LoRA unmerged (for 8B+)")
a = p.parse_args()
if a.mem_frac < 1:
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)

PROMPT = "Convert this Roman Urdu text into Urdu script. Reply with the Urdu text only.\n\n{src}"
tok = AutoTokenizer.from_pretrained(a.lora)
tok.padding_side = "left"
if a.qlora:
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=q, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.lora).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()

items = [json.loads(l) for l in open(a.tasks, encoding="utf8")]
os.makedirs("results/raw", exist_ok=True)
out = open(f"results/raw/{a.name}.jsonl", "w", encoding="utf8")
for i in range(0, len(items), a.bs):
    chunk = items[i:i + a.bs]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(src=it["meta"]["src"])}],
                                       tokenize=False, add_generation_prompt=True) for it in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
    sec = (time.time() - t0) / len(chunk)
    for it, g in zip(chunk, gen):
        text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out.write(json.dumps({"id": it["id"], "task": it["task"], "ok": True, "out": text, "sec": round(sec, 3),
                              "cost": 0}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"{i + len(chunk)}/{len(items)}", flush=True)
print("done")
