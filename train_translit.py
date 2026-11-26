"""LoRA fine-tune a small Qwen model for Roman Urdu -> Urdu script.

Trains only on Dakshina rows that are NOT in the benchmark test (build_tasks.py writes data/train/dakshina_train.jsonl).
GPU memory is capped so it can share the card with other services.

  python train_translit.py --base /path/to/Qwen2.5-1.5B-Instruct --out runs/qwen15_translit
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--data", default="data/train/dakshina_train.jsonl")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--max_len", type=int, default=320)
p.add_argument("--mem_frac", type=float, default=0.28)
p.add_argument("--val", type=int, default=200)
p.add_argument("--qlora", action="store_true")
a = p.parse_args()

torch.manual_seed(0)
random.seed(0)
if a.mem_frac < 1:
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
os.makedirs(a.out, exist_ok=True)

PROMPT = "Convert this Roman Urdu text into Urdu script. Reply with the Urdu text only.\n\n{src}"

tok = AutoTokenizer.from_pretrained(a.base)
rows = [json.loads(l) for l in open(a.data, encoding="utf8")]
rows = [r for r in rows if r["romanized"] and r["native"] and len(r["native"]) < 600]
random.shuffle(rows)
val, train = rows[: a.val], rows[a.val:]
print("train", len(train), "val", len(val), flush=True)


def encode(r):
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(src=r["romanized"])}],
                                     tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(r["native"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[: a.max_len]
    labels = ([-100] * len(p_ids) + a_ids)[: a.max_len]
    return ids, labels


data = [encode(r) for r in train]
if a.qlora:
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=q, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                         "up_proj", "down_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(data) * a.epochs / (a.bs * a.accum))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / steps))))


def batches():
    while True:
        order = sorted(range(len(data)), key=lambda i: len(data[i][0]) + random.random() * 40)
        chunks = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        random.shuffle(chunks)
        yield from chunks


pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
it = batches()
model.train()
t0 = time.time()
for step in range(steps):
    tot = 0.0
    for _ in range(a.accum):
        idx = next(it)
        L = max(len(data[i][0]) for i in idx)
        ids = torch.tensor([data[i][0] + [pad] * (L - len(data[i][0])) for i in idx]).cuda()
        lab = torch.tensor([data[i][1] + [-100] * (L - len(data[i][1])) for i in idx]).cuda()
        att = (torch.arange(L)[None, :] < torch.tensor([len(data[i][0]) for i in idx])[:, None]).cuda()
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum
        loss.backward()
        tot += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if step % 25 == 0 or step == steps - 1:
        print(f"step {step}/{steps} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)

model.save_pretrained(a.out)
tok.save_pretrained(a.out)
json.dump({"val": val, "args": vars(a)}, open(f"{a.out}/val.json", "w", encoding="utf8"), ensure_ascii=False)
print("saved", a.out, flush=True)
