"""Run one model over every task through an OpenAI-compatible endpoint (OpenRouter by default).

Resumable: answers are appended to results/raw/<slug>.jsonl and skipped on rerun.

  python run_eval.py --model openai/gpt-5.5
  python run_eval.py --model my-model --base_url http://host:8000/v1 --key_env NONE
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import threading
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--name", help="display slug (default: model id)")
p.add_argument("--base_url", default="https://openrouter.ai/api/v1")
p.add_argument("--key_env", default="OPENROUTER_API_KEY")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--max_tokens", type=int, default=4000)
p.add_argument("--effort", default="low", help="reasoning effort sent to OpenRouter; 'none' to omit")
p.add_argument("--tasks", default="*")
p.add_argument("--limit", type=int, default=0, help="items per task (0 = all), for smoke tests")
a = p.parse_args()


def load_env():
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
key = None if a.key_env == "NONE" else os.environ[a.key_env]
slug = (a.name or a.model).replace("/", "__").replace(":", "_")
os.makedirs("results/raw", exist_ok=True)
out_path = f"results/raw/{slug}.jsonl"
done = set()
if os.path.exists(out_path):
    for line in open(out_path, encoding="utf8"):
        r = json.loads(line)
        if r.get("ok"):
            done.add(r["id"])

items = []
for f in sorted(glob.glob(f"data/tasks/{a.tasks}.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf8")]
    items += rows[: a.limit] if a.limit else rows
todo = [it for it in items if it["id"] not in done]
print(f"{a.model}: {len(todo)} to run ({len(done)} done)", flush=True)

lock = threading.Lock()
sess = requests.Session()


def call(it):
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": it["prompt"]}],
        "temperature": 0,
        "max_tokens": a.max_tokens,
    }
    if a.effort != "none" and "openrouter" in a.base_url:
        body["reasoning"] = {"effort": a.effort, "exclude": True}
        body["usage"] = {"include": True}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    err = None
    for attempt in range(6):
        try:
            t0 = time.time()
            r = sess.post(f"{a.base_url}/chat/completions", json=body, headers=headers, timeout=300)
            if r.status_code == 200:
                j = r.json()
                if "choices" not in j:
                    raise RuntimeError(str(j)[:300])
                msg = j["choices"][0]["message"]
                u = j.get("usage") or {}
                return {"id": it["id"], "task": it["task"], "ok": True, "out": msg.get("content") or "",
                        "sec": round(time.time() - t0, 2), "in_tok": u.get("prompt_tokens"),
                        "out_tok": u.get("completion_tokens"), "cost": u.get("cost"),
                        "finish": j["choices"][0].get("finish_reason")}
            err = f"{r.status_code} {r.text[:300]}"
            if r.status_code in (400, 401, 402, 403, 404):
                break
        except Exception as e:  # network, json
            err = repr(e)[:300]
        time.sleep(2 ** attempt)
    return {"id": it["id"], "task": it["task"], "ok": False, "err": err}


n_ok = n_err = 0
with open(out_path, "a", encoding="utf8") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
    for res in ex.map(call, todo):
        with lock:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += res["ok"]
            n_err += not res["ok"]
            if not res["ok"] and n_err <= 3:
                print("ERR", res["id"], res["err"], flush=True)
            if (n_ok + n_err) % 100 == 0:
                print(f"  {n_ok + n_err}/{len(todo)} (errors {n_err})", flush=True)
print(f"done: ok {n_ok}, errors {n_err}")
