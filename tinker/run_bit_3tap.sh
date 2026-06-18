#!/bin/bash
# bit-3tap: train the genuine-3-input bit corpus from base + eval on the 500-row bit_eval500 (disjoint real rows).
# The lever: 3-tap = 34% of bit, currently 0.07. 1/2-tap must HOLD ~1.0; 3-tap slice is the test.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python

echo "=== TRAIN bit-3tap (2 epochs, LR 2e-4 fresh from base) ==="
$PY tinker/sft_warmstart.py \
    --data pipeline/data/v16/train_bit_3tap.csv \
    --epochs 2 --run-name bit-3tap 2>&1 | tee /tmp/bit_3tap_train.log

SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' /tmp/bit_3tap_train.log | tail -1)
echo "=== SAMPLER = $SAMP ==="

echo "=== EVAL bit-3tap on bit_eval500 (disjoint real rows) ==="
$PY tinker/eval_oracle.py \
    --ckpt "$SAMP" \
    --cats bit_manipulation \
    --rows-file pipeline/data/bit_eval500.jsonl \
    --tag bit-3tap 2>&1 | tee /tmp/bit_3tap_eval.log

echo "=== bit-3tap DONE -- per-row results in tinker/evals/bit-3tap.csv ==="
echo "=== tap-count split (1/2-tap must hold; 3-tap is the lever) ==="
$PY - <<'PY'
import json, csv
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
ev=[r for r in csv.DictReader(open('tinker/evals/bit-3tap.csv'))]
import collections
tot=collections.Counter(); ok=collections.Counter()
for r in ev:
    s=solved.get(r['id'])
    k=nv(s['expr']) if (s and s.get('correct')) else '?'
    tot[k]+=1; ok[k]+=int(r['correct'])
for k in sorted(tot, key=lambda x:str(x)):
    print(f"  {k}-tap: {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}")
print(f"  OVERALL bit: {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/sum(tot.values()):.3f}")
PY
