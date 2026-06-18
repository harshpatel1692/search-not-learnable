#!/bin/bash
# EXP-CTRL (paper architecture control): train the SAME cryptarithm corpus on a DENSE
# transformer (Llama-3.2-3B, ~3B dense ~ Nemotron's 3.5B active) and eval held-out crypt.
# If the dense model ALSO caps near 0, the wall is a general small-model learnability limit;
# if it does markedly better, the wall is hybrid-Mamba/MoE-specific. Either result is in-scope.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
BASE="${1:-meta-llama/Llama-3.2-3B}"
EPOCHS="${2:-4}"
RUN=crypt-dense-ctrl
echo "rows: $(wc -l < pipeline/data/crypt_honest_only.csv)  base: $BASE  epochs: $EPOCHS"

echo "=== TRAIN $RUN ==="
$PY tinker/sft_warmstart.py \
  --base-model "$BASE" \
  --data pipeline/data/crypt_honest_only.csv \
  --epochs "$EPOCHS" --run-name "$RUN" 2>&1 | tee /tmp/ctrl_train.log

SAMP=$(grep -oP 'sampler( path)?\s*[:=]\s*\K\S+' /tmp/ctrl_train.log | tail -1)
echo "=== SAMPLER = $SAMP ==="

echo "=== EVAL $RUN on val.jsonl (crypt deduce+guess) ==="
$PY tinker/eval_oracle.py \
  --ckpt "$SAMP" \
  --cats cryptarithm_deduce,cryptarithm_guess \
  --rows-file pipeline/data/val.jsonl \
  --tag "$RUN" 2>&1 | tee /tmp/ctrl_eval.log

echo "=== EXP-CTRL DONE -> tinker/evals/${RUN}.csv ==="
