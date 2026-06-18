#!/bin/bash
# crypt-twn-r9: train the trace_we_need corpus from base + eval on 200 held-out val.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python

echo "=== TRAIN crypt-twn-r9  (2 epochs, LR 2e-4 fresh from base) ==="
$PY tinker/sft_warmstart.py \
    --data pipeline/data/v16/train_crypt_twn.csv \
    --epochs 2 --run-name crypt-twn-r9 2>&1 | tee /tmp/twn_train.log

SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' /tmp/twn_train.log | tail -1)
echo "=== SAMPLER = $SAMP ==="

echo "=== EVAL crypt-twn-r9 on 200 held-out val (deduce + guess) ==="
$PY tinker/eval_oracle.py \
    --ckpt "$SAMP" \
    --cats cryptarithm_deduce,cryptarithm_guess \
    --rows-file pipeline/data/val.jsonl \
    --tag crypt-twn-r9 2>&1 | tee /tmp/twn_eval.log

echo "=== crypt-twn-r9 DONE -- per-row results in tinker/evals/crypt-twn-r9.csv ==="
