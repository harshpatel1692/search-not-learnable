#!/bin/bash
# #3 frontier intervention on NEMOTRON: same crypt task, dial forward-derivability via key reveal.
# full-key (forward) and half-key trained; no-key (search) = existing crypt baseline (~0.03).
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
BASE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
for LEVEL in full half; do
  RUN=frontier-$LEVEL
  echo "=== TRAIN $RUN (Nemotron, 3 ep, train_forward_${LEVEL}.csv) ==="
  $PY tinker/sft_warmstart.py --base-model "$BASE" \
    --data pipeline/data/frontier/train_forward_${LEVEL}.csv \
    --epochs 3 --run-name "$RUN" 2>&1 | tee /tmp/${RUN}.log.train
  SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' /tmp/${RUN}.log.train | tail -1)
  echo "=== SAMPLER $RUN = $SAMP ==="
  [ -z "$SAMP" ] && { echo "no sampler, skip eval"; continue; }
  $PY tinker/eval_oracle.py --ckpt "$SAMP" --base-model "$BASE" \
    --cats cryptarithm_deduce --rows-file pipeline/data/frontier/val_forward_${LEVEL}.jsonl \
    --tag "$RUN" 2>&1 | tee /tmp/${RUN}.log.eval
done
echo "=== FRONTIER-EXP DONE -> tinker/evals/frontier-{full,half}.csv ==="
