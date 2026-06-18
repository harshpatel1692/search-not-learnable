#!/bin/bash
# Architecture-control matrix runner: train the SAME crypt corpus on an arbitrary base
# model and eval held-out crypt with THAT model's own tokenizer (the fix: --base-model is
# passed to BOTH sft and eval; the first Llama run defaulted eval to the Nemotron tokenizer
# -> vocab overflow). Usage: run_crypt_arch_ctrl.sh <BASE_MODEL> [RUN_TAG] [EPOCHS]
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
BASE="$1"
RUN="${2:-arch-$(echo "$BASE" | tr '/:.' '___' )}"
EPOCHS="${3:-4}"
LOG=/tmp/${RUN}.log
echo "=== ARCH-CTRL: $BASE  run=$RUN  epochs=$EPOCHS ==="

$PY tinker/sft_warmstart.py --base-model "$BASE" \
  --data pipeline/data/crypt_honest_only.csv \
  --epochs "$EPOCHS" --run-name "$RUN" 2>&1 | tee "${LOG}.train"

SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' "${LOG}.train" | tail -1)
echo "=== SAMPLER = $SAMP ==="
[ -z "$SAMP" ] && { echo "NO SAMPLER -- abort"; exit 1; }

$PY tinker/eval_oracle.py --ckpt "$SAMP" --base-model "$BASE" \
  --cats cryptarithm_deduce,cryptarithm_guess \
  --rows-file pipeline/data/val.jsonl --tag "$RUN" 2>&1 | tee "${LOG}.eval"

echo "=== $RUN DONE -> tinker/evals/${RUN}.csv ==="
