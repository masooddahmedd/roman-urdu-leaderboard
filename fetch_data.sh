#!/usr/bin/env bash
# Download the public source datasets into data/raw (about 20 MB). Then run build_tasks.py.
set -e
cd "$(dirname "$0")"
mkdir -p data/raw && cd data/raw
HF=https://huggingface.co/datasets
# Dakshina (Roark et al. 2020), Urdu romanized sentences, via a Hugging Face copy of the Google release.
curl -sL -o dakshina_ur.parquet "$HF/Anvesh-Lankala/Copy_Dakshina_Google_research_dataset/resolve/main/data/ur-00000-of-00001.parquet"
# Belebele (Bandarkar et al. 2024): human-romanized Urdu, Urdu script and English.
for l in urd_Latn urd_Arab eng_Latn; do
  curl -sL -o "belebele_$l.jsonl" "$HF/facebook/belebele/resolve/main/data/$l.jsonl"
done
# UCI Roman Urdu sentiment comments.
curl -sL -o roman_urdu_sent.parquet "$HF/community-datasets/roman_urdu/resolve/main/data/train-00000-of-00001.parquet"
ls -la
