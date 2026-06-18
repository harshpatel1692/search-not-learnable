#!/bin/bash
# crypt-honest-${RUN:-r1}: train the HONEST (borrow-column + cast-9/11) crypt corpus from base + eval crypt on val.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
EPOCHS="${1:-6}"

echo "=== TRAIN crypt-honest-${RUN:-r1}  ($EPOCHS epochs, LR 2e-4 fresh from base) ==="
$PY tinker/sft_warmstart.py \
    --data pipeline/data/crypt_honest_only.csv \
    --epochs "$EPOCHS" --run-name crypt-honest-${RUN:-r1} 2>&1 | tee /tmp/crypt_honest_train.log

SAMP=$(grep -oP 'sampler( path)?\s*[:=]\s*\K\S+' /tmp/crypt_honest_train.log | tail -1)
echo "=== SAMPLER = $SAMP ==="

echo "=== EVAL crypt-honest-${RUN:-r1} on held-out val (deduce + guess) ==="
$PY tinker/eval_oracle.py \
    --ckpt "$SAMP" \
    --cats cryptarithm_deduce,cryptarithm_guess \
    --rows-file pipeline/data/val.jsonl \
    --tag crypt-honest-${RUN:-r1} 2>&1 | tee /tmp/crypt_honest_eval.log

echo "=== crypt-honest-${RUN:-r1} DONE -- per-row results in tinker/evals/crypt-honest-${RUN:-r1}.csv ==="
