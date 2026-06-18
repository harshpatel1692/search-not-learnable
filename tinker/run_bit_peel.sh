#!/bin/bash
# bit-peel: train PEEL-ONLY corpus (peel->2-tap CoT for XNOR/XOR-headed 3-tap) from base, 3 epochs.
# Eval bit_eval500, split by FORM. Target = peelable-form accuracy vs bit-star-r2's 0.62 (XNOR 0.56 / XOR 0.76).
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python

echo "=== TRAIN bit-peel (3 epochs, from base, 2000 peel-only rows) ==="
$PY tinker/sft_warmstart.py --data pipeline/data/v16/train_bit_peel.csv \
    --epochs 3 --run-name bit-peel 2>&1 | tee /tmp/bit_peel_train.log
SAMP=$(grep -oP 'sampler path : \K\S+' /tmp/bit_peel_train.log | head -1)
echo "SAMP=[$SAMP]"; [ -z "$SAMP" ] && { echo "NO SAMPLER"; exit 1; }

echo "=== EVAL bit-peel on bit_eval500 ==="
$PY tinker/eval_oracle.py --ckpt "$SAMP" --cats bit_manipulation \
    --rows-file pipeline/data/bit_eval500.jsonl --tag bit-peel 2>&1 | tee /tmp/bit_peel_eval.log

echo "=== split by FORM (the peelable XNOR/XOR rows are the target) ==="
$PY - <<'PYEOF'
import json, csv, re, collections
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
def gform(e):
    o=[]
    for t in re.findall(r'\{[ABC]\}',e):
        if t not in o:o.append(t)
    g=e
    for k,t in enumerate(o):g=g.replace(t,'XYZ'[k])
    return g
PEEL={'XNOR(X, NOT_A_AND_B(Y, Z))','XOR(X, NOT_A_AND_B(Y, Z))'}
ev=[r for r in csv.DictReader(open('tinker/evals/bit-peel.csv'))]
tot=collections.Counter(); ok=collections.Counter()
for r in ev:
    s=solved.get(r['id'])
    if not (s and s.get('correct')): key='unsolved'
    elif nv(s['expr'])!=3: key=f"{nv(s['expr'])}-tap"
    else: key='3tap-PEELABLE' if gform(s['expr']) in PEEL else '3tap-other'
    tot[key]+=1; ok[key]+=int(r['correct'])
for k in sorted(tot):
    print(f"  {k:16s}: {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}")
print("  baseline bit-star-r2: 3tap-PEELABLE 0.62 (XNOR .56 / XOR .76)")
print("  (peel-only adapter is UNTRAINED on 1/2-tap & 3tap-other -> those will be low; judge on 3tap-PEELABLE)")
PYEOF
echo "BIT_PEEL_DONE"
