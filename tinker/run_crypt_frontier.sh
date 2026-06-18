#!/bin/bash
# #2 Frontier baseline: can a STRONG model solve cryptarithm IN-CONTEXT (no fine-tuning)?
# The competition prompt already contains the worked examples, so this is few-shot in-context.
# If strong models also cap ~0, the wall is task-intrinsic (not just a small-model limit).
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
N="${1:-20}"
for M in "deepseek-ai/DeepSeek-V3.1" "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"; do
  TAG="crypt-frontier-$(echo "$M" | tr '/:.' '___')"
  echo "=== FRONTIER BASELINE (in-context, raw base): $M  ($N rows) ==="
  $PY tinker/eval_oracle.py --base --base-model "$M" \
    --cats cryptarithm_deduce --rows-file pipeline/data/val.jsonl \
    --max-rows "$N" --tag "$TAG" 2>&1 | grep -viE "warning|HF_TOKEN" | tee "/tmp/${TAG}.log"
done
echo "=== FRONTIER BASELINE DONE ==="
