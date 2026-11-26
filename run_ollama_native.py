"""Open-model baseline through Ollama's native API on CPU only (num_gpu 0), so it never touches the GPU.

  python run_ollama_native.py MODEL THINK   (from the repo root, on the machine running Ollama)   (THINK: false | low)  -> results/raw/ollama__<model>.jsonl
Same prompt as the other open models (the item's prompt, one sentence per request, temperature 0).
Resumable.
"""
import json
import os
import sys
import time
import urllib.request

model, think = sys.argv[1], sys.argv[2]
think = False if think == "false" else think
slug = "ollama__" + model.replace(":", "_").replace("/", "_")
items = [json.loads(l) for l in open("data/tasks/translit.jsonl", encoding="utf8")]
path = f"results/raw/{slug}.jsonl"
done = set()
if os.path.exists(path):
    done = {json.loads(l)["id"] for l in open(path, encoding="utf8")}
out = open(path, "a", encoding="utf8")
for k, it in enumerate(items):
    if it["id"] in done:
        continue
    body = {"model": model, "messages": [{"role": "user", "content": it["prompt"]}], "stream": False,
            "think": think, "options": {"num_gpu": 0, "temperature": 0, "num_ctx": 4096, "num_predict": 2000}}
    t0 = time.time()
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        text, ok = r["message"]["content"], True
    except Exception as e:  # noqa: BLE001
        text, ok = f"ERROR {e}", False
    out.write(json.dumps({"id": it["id"], "task": "translit", "ok": ok, "out": text,
                          "sec": round(time.time() - t0, 2), "cost": 0}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"{k + 1}/{len(items)} {time.time() - t0:.1f}s", flush=True)
print("done")
