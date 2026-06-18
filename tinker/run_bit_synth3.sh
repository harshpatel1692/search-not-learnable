#!/bin/bash
# bit-synth3: STaR on FRESH SYNTH 3-tap prompts (unbounded 3-tap data) -> harvest model's own correct CoTs ->
# combine with r3 corpus -> retrain -> eval on REAL train 3-tap (bit_eval500 3-tap slice = 169 real rows).
# $1 = harvest ckpt (best bit sampler, e.g. bit-star-r3 or bit-star-r2).
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
HARVEST_CKPT="$1"
[ -z "$HARVEST_CKPT" ] && { echo "usage: run_bit_synth3.sh <harvest_sampler_ckpt>"; exit 1; }

echo "=== HARVEST synth-3tap correct CoTs from $HARVEST_CKPT (k=10, temp 0.8) ==="
$PY tinker/star_harvest.py --ckpt "$HARVEST_CKPT" \
    --rows-file pipeline/data/bit_synth3_src.jsonl --k 10 --temp 0.8 \
    --keep-per-row 2 --max-cot-tokens 4200 --tag bit-synth3harvest 2>&1 | tee /tmp/bit_synth3_harvest.log

echo "=== COMBINE ALL bit harvests (max diversity, fixes r3 exposure-bias collapse) + synth3 ==="
$PY - <<'PYEOF'
import csv, json, collections
csv.field_size_limit(10**9)
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
seen=set(); rows=[]
# all real-row harvests (r2harvest+rest+r3harvest = proven 0.678 diversity) + the NEW synth 3-tap CoTs
for f in ['pipeline/data/star/bit-star-r2harvest.csv','pipeline/data/star/bit-star-rest.csv',
          'pipeline/data/star/bit-star-r3harvest.csv','pipeline/data/star/bit-synth3harvest.csv']:
    try:
        for r in csv.DictReader(open(f)):
            k=(r['id'], r['raw_output'][:200])
            if k in seen: continue
            seen.add(k); rows.append(r)
    except FileNotFoundError: print("missing",f)
out='pipeline/data/v16/train_bit_synth3.csv'
with open(out,'w',newline='') as fo:
    w=csv.DictWriter(fo,fieldnames=['id','prompt','answer','category','raw_output','predicted','correct']); w.writeheader()
    for r in rows: w.writerow({k:r[k] for k in w.fieldnames})
nd=len(set(r['prompt'] for r in rows))
tc=collections.Counter(nv(solved[r['id']]['expr']) if (r['id'] in solved and solved[r['id']].get('correct')) else 'synth' for r in rows)
print(f"[combined] {len(rows)} rows ({nd} distinct prompts) -> {out} | tap-comp {dict(tc)}")
PYEOF

echo "=== TRAIN bit-synth3 (2 ep, from base) ==="
$PY tinker/sft_warmstart.py --data pipeline/data/v16/train_bit_synth3.csv --epochs 2 --run-name bit-synth3 2>&1 | tee /tmp/bit_synth3_train.log
SAMP=$($PY -c "import re;t=open('/tmp/bit_synth3_train.log').read();m=re.findall(r'tinker://\S*sampler\S*',t);print(m[-1] if m else '')")
echo "SAMP=$SAMP"; [ -z "$SAMP" ] && { echo NO_SAMPLER; exit 1; }

echo "=== EVAL bit-synth3 on bit_eval500 (real train rows) ==="
$PY tinker/eval_oracle.py --ckpt "$SAMP" --cats bit_manipulation --rows-file pipeline/data/bit_eval500.jsonl --tag bit-synth3 2>&1 | tee /tmp/bit_synth3_eval.log
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
print(f"  OVERALL: {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/sum(tot.values()):.3f}  (r2=0.678, 3t=0.503)")
PYEOF
echo "BIT_SYNTH3_DONE"
