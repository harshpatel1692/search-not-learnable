#!/bin/bash
# crypt-twn-r10: train the trace_we_need corpus from base + eval on 200 held-out val.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python

echo "=== TRAIN crypt-twn-r10  (2 epochs, LR 2e-4 fresh from base) ==="
$PY tinker/sft_warmstart.py \
    --data pipeline/data/v16/train_crypt_twn_r10.csv \
    --epochs 2 --run-name crypt-twn-r10 2>&1 | tee /tmp/twn_r10_train.log

SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' /tmp/twn_r10_train.log | tail -1)
echo "=== SAMPLER = $SAMP ==="

echo "=== EVAL crypt-twn-r10 on 200 held-out val (deduce + guess) ==="
$PY tinker/eval_oracle.py \
    --ckpt "$SAMP" \
    --cats cryptarithm_deduce,cryptarithm_guess \
    --rows-file pipeline/data/val.jsonl \
    --tag crypt-twn-r10 2>&1 | tee /tmp/twn_r10_eval.log

echo "=== crypt-twn-r10 DONE -- per-row results in tinker/evals/crypt-twn-r10.csv ==="
