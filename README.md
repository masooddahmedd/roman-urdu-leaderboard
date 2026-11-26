# roman-urdu-leaderboard

**Paper:** Masood Ahmed (2026). *Two Points from the Frontier: An Open 8B Model for Roman Urdu Transliteration, Fine-Tuned on One GPU.* Preprint (LaTeX source in [`paper/`](paper)).

Most Urdu typed on phones is Roman Urdu (*kya aap kal aa sakte hain?*), while most Urdu text worth searching, reading aloud or publishing is in Urdu script. This repository benchmarks how well AI models handle Roman Urdu, and releases an open model that converts it to Urdu script almost as well as the best frontier models.

Short version: **a Qwen3-8B fine-tuned for 68 minutes on one 16 GB GPU makes 3.6% character errors, within 2 points of GPT-6 and Claude Fable 5.2, and ahead of every open model we tested, up to 20B parameters, and of the dedicated Roman Urdu transliterator from prior work.**

## Transliteration: Roman Urdu → Urdu script

300 held-out sentences from Dakshina (Urdu Wikipedia sentences romanized by native speakers). Score = 100 − character error rate, after light Urdu normalization. Brackets are 95% bootstrap intervals.

| System | Score | Char. errors | Word errors | Exactly right |
| --- | --- | --- | --- | --- |
| Claude Fable 5.2 (claude.ai) | 98.6 | 1.4% [1.2, 1.7] | 6.0% | 55% |
| GPT-6 (ChatGPT, medium thinking) | 98.4 | 1.6% [1.3, 1.9] | 6.7% | 50% |
| **Ours: Qwen3-8B + QLoRA** | **96.4** | **3.6% [3.0, 4.2]** | **10.2%** | **37%** |
| **Ours: Qwen2.5-1.5B + QLoRA** | **95.3** | **4.7% [4.1, 5.4]** | **12.8%** | **25%** |
| m2m100 418M, Roman Urdu model of Butt et al. (2025) | 92.7 | 7.3% [6.4, 8.3] | 19.2% | 14% |
| gpt-oss 20B | 90.2 | 9.8% [8.2, 11.6] | 21.3% | 11% |
| Qwen3 14B | 90.1 | 9.9% [8.7, 11.3] | 22.4% | 9% |
| Llama 3.1 8B Instruct | 87.5 | 12.5% [11.0, 14.0] | 25.8% | 6% |
| Qwen2.5 7B Instruct | 72.4 | 27.6% [25.4, 30.0] | 51.9% | 0% |
| Qwen2.5 3B Instruct | 62.0 | 38.0% [35.9, 40.4] | 63.5% | 0% |

- Our 8B removes 63% of the errors of the best open model we ran (gpt-oss 20B), and the paired bootstrap interval for every open model is far from zero.
- Fine-tuning matters more than size: the 1.5B model already beats every general open model up to 20B.
- Its 10.2% word error rate compares well with published Dakshina Urdu results (12.1% for the best system on the official test half, 13.5% for GPT-4 Turbo), with the caveats in the paper.
- What remains is mostly letters Roman Urdu does not distinguish: *swad* vs *seen*, retroflex *ṭe* vs *te*, *hamza* on *yeh*.

## The wider benchmark

Version 1 has four tasks and 1,020 items, all written or romanized by people:

| Task | Items | Source | Score |
| --- | --- | --- | --- |
| Transliteration: Roman Urdu → Urdu script | 300 | Dakshina (Roark et al. 2020) | 100 − CER |
| Reading comprehension | 300 | Belebele `urd_Latn` (Bandarkar et al. 2024) | accuracy |
| Translation: Roman Urdu → English | 120 | Belebele passages with their FLORES English originals | chrF++ |
| Sentiment of real comments | 300 | UCI Roman Urdu dataset, 100 per class | macro F1 |

| Model | Overall | Transliteration | Reading | Translation | Sentiment |
| --- | --- | --- | --- | --- | --- |
| Claude Fable 5.2 | 85.7 | 98.6 | 88.7 | 86.3* | 69.8 |
| GPT-6 | 85.1 | 98.4 | 86.0 | 64.3 | 71.0 |
| Llama 3.1 8B Instruct | 59.3 | 87.5 | 33.3 | 46.2 | 57.0 |
| Qwen2.5 7B Instruct | 55.5 | 72.4 | 39.7 | 42.8 | 54.4 |
| Qwen2.5 3B Instruct | 48.7 | 62.0 | 29.7 | 36.2 | 54.3 |

Overall is the mean of transliteration, reading and sentiment. Translation is left out of it because of contamination: \*45 of Fable 5.2's 120 translations score above 90 sentence-level chrF++ against the FLORES reference, near-verbatim reproductions of the reference wording. GPT-6 has none.

Frontier models were run in their chat apps (fresh temporary chats, batched numbered items, the same instructions), because no API credit was used. Open models ran locally through Ollama at temperature 0.

## Use the model

The LoRA adapters (bf16) are attached to the [v1 release](https://github.com/masooddahmedd/roman-urdu-leaderboard/releases/tag/v1): download `qwen3-8b-translit.zip` (and `qwen2.5-1.5b-translit.zip` for the small model) and unzip it in the repository root, which fills `adapters/`.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/qwen3-8b-translit").eval()

prompt = "Convert this Roman Urdu text into Urdu script. Reply with the Urdu text only.\n\n{}"
msgs = [{"role": "user", "content": prompt.format("kya aap kal aa sakte hain?")}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

On a 16 GB GPU, load the base in 4-bit as in `eval_local.py --qlora`.

## Reproduce

```bash
bash fetch_data.sh                 # public sources into data/raw
python build_tasks.py              # data/tasks/*.jsonl and data/train/dakshina_train.jsonl
python make_lex_train.py           # adds word-alignment pseudo-sentences (the exact file used is included)

# our models (one GPU)
python train_translit.py --base Qwen/Qwen3-8B-Base --data data/train/dakshina_train_lex.jsonl \
  --out runs/qwen3_8b_translit --epochs 2 --lr 1.5e-4 --bs 8 --accum 2 --max_len 256 --mem_frac 0.95 --qlora
python eval_local.py --base Qwen/Qwen3-8B-Base --lora runs/qwen3_8b_translit --name ours__qwen3-8b-translit --qlora --mem_frac 0.95

# baselines
bash run_local.sh                  # small open models via Ollama
python run_ollama_native.py qwen3:14b false && python run_ollama_native.py gpt-oss:20b low
python run_m2m.py MODEL_DIR        # Butt et al. m2m100
python batches.py make             # chat-app batches; replies go in results/replies/<model>/
python batches.py parse chatgpt__gpt-6 results/replies/chatgpt-gpt-6

# tables, page, paper numbers
python leaderboard.py && python make_site.py
python paper/analysis.py && python paper/make_paper.py
```

Every model output is in `results/raw/`, and the raw chat-app replies are in `results/replies/`.

## Limits

- The 300 test sentences are a random sample of the public Dakshina sentence data, not its official test split. All sources are public, so any model may have seen them.
- Our training data comes from the same source as the test sentences, so our model is in-domain; informal chat-style Roman Urdu is more varied.
- Frontier models were queried once each, through consumer chat apps rather than an API.
- Roman-Urdu-Parl was considered for a casual-text test and dropped: its Roman side is machine-made and often garbled.

## License

Code: MIT. Benchmark items and model outputs are derived from Dakshina (CC BY-SA 4.0), Belebele (CC BY-SA 4.0) and the UCI Roman Urdu dataset (CC BY 4.0) and keep those licenses. The adapters inherit the licenses of their base models (Apache 2.0).
