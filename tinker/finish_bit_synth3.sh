#!/bin/bash
# Finish bit-synth3: corpus already built (pipeline/data/v16/train_bit_synth3.csv, 1836 rows / 1619 distinct
# prompts / 1069 3-tap CoTs). Harvest is DONE & saved. This just TRAINS + EVALS (~$4-5, ~10 min).
# Run after Tinker billing is topped up:  bash tinker/finish_bit_synth3.sh
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python

echo "=== TRAIN bit-synth3 (2 ep, from base, 1836 rows) ==="
$PY tinker/sft_warmstart.py --data pipeline/data/v16/train_bit_synth3.csv \
    --epochs 2 --run-name bit-synth3 2>&1 | tee /tmp/bit_synth3_train.log

SAMP=$(grep -oP 'sampler path : \K\S+' /tmp/bit_synth3_train.log | head -1)
echo "SAMP=[$SAMP]"
[ -z "$SAMP" ] && { echo "NO SAMPLER (train failed?)"; exit 1; }

echo "=== EVAL bit-synth3 on bit_eval500 (real train rows) ==="
$PY tinker/eval_oracle.py --ckpt "$SAMP" --cats bit_manipulation \
    --rows-file pipeline/data/bit_eval500.jsonl --tag bit-synth3 2>&1 | tee /tmp/bit_synth3_eval.log

echo "=== tap-count split (real train 3-tap = the 169 slice) ==="
$PY - <<'PYEOF'
import json, csv, collections
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
ev=[r for r in csv.DictReader(open('tinker/evals/bit-synth3.csv'))]
tot=collections.Counter(); ok=collections.Counter()
for r in ev:
    s=solved.get(r['id']); k=nv(s['expr']) if (s and s.get('correct')) else '?'
    tot[k]+=1; ok[k]+=int(r['correct'])
for k in sorted(tot,key=str): print(f"  {k}-tap: {ok[k]}/{tot[k]} = {ok[k]/max(tot[k],1):.3f}")
print(f"  OVERALL: {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/sum(tot.values()):.3f}")
print("  baselines: r2=0.678 (1t.88/2t.76/3t.50) | r3-collapsed=0.04 | bit-3tap-handgrammar=0.06")
PYEOF
echo "BIT_SYNTH3_DONE"
