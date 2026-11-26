#!/usr/bin/env bash
# Open models through an Ollama server's OpenAI-compatible endpoint (default http://localhost:11434/v1).
cd "$(dirname "$0")"
OLLAMA=${OLLAMA:-http://localhost:11434/v1}
mkdir -p results/logs
for m in qwen2.5:7b-instruct llama3.1:8b qwen2.5:3b-instruct; do
  python run_eval.py --model "$m" --name "ollama__$m" --base_url "$OLLAMA" --key_env NONE --effort none --workers 2 --max_tokens 1500 > "results/logs/ollama_${m//:/_}.log" 2>&1
done
# Larger open models, CPU only (num_gpu 0), on the Ollama machine itself:
#   python run_ollama_native.py qwen3:14b false
#   python run_ollama_native.py gpt-oss:20b low
