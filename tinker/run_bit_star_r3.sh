#!/bin/bash
# bit-star-r3: harvest correct CoTs from bit-star-r2 (0.678), combine with r2harvest (preserve 1-2 tap),
# retrain from base, eval bit_eval500 by tap. The PROVEN bit lever (STaR self-harvest), not a hand grammar.
set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
R2_SAMP="tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/sampler_weights/bit-star-r2-sampler"

echo "=== HARVEST bit-star-r3 from bit-star-r2 (k=12, temp 0.8, short traces) ==="
$PY tinker/star_harvest.py --ckpt "$R2_SAMP" \
    --rows-file pipeline/data/bit_star_src.jsonl --k 12 --temp 0.8 \
    --keep-per-row 2 --max-cot-tokens 4200 --tag bit-star-r3harvest 2>&1 | tee /tmp/bit_star_r3_harvest.log

echo "=== COMBINE r3harvest + r2harvest (dedup by id+cot prefix) -> train CSV ==="
$PY - <<'PYEOF'
import csv, collections
csv.field_size_limit(10**9)
seen=set(); rows=[]
for f in ['pipeline/data/star/bit-star-r3harvest.csv','pipeline/data/star/bit-star-r2harvest.csv']:
    try:
        for r in csv.DictReader(open(f)):
            k=(r['id'], r['raw_output'][:200])
            if k in seen: continue
            seen.add(k); rows.append(r)
    except FileNotFoundError: print("missing",f)
import json
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
tc=collections.Counter(nv(solved[r['id']]['expr']) if (r['id'] in solved and solved[r['id']].get('correct')) else '?' for r in rows)
out='pipeline/data/v16/train_bit_star_r3.csv'
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','prompt','answer','category','raw_output','predicted','correct']); w.writeheader()
    for r in rows: w.writerow({k:r[k] for k in w.fieldnames})
print(f"[combined] {len(rows)} rows -> {out} | tap-comp {dict(tc)}")
PYEOF

echo "=== TRAIN bit-star-r3 (2 epochs, LR 2e-4 from base) ==="
$PY tinker/sft_warmstart.py --data pipeline/data/v16/train_bit_star_r3.csv \
    --epochs 2 --run-name bit-star-r3 2>&1 | tee /tmp/bit_star_r3_train.log
SAMP=$(grep -oP 'sampler path\s*:\s*\K\S+' /tmp/bit_star_r3_train.log | tail -1)

echo "=== EVAL bit-star-r3 on bit_eval500 ==="
$PY tinker/eval_oracle.py --ckpt "$SAMP" --cats bit_manipulation \
    --rows-file pipeline/data/bit_eval500.jsonl --tag bit-star-r3 2>&1 | tee /tmp/bit_star_r3_eval.log

echo "=== tap-count split ==="
$PY - <<'PYEOF'
import json, csv, collections
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
ev=[r for r in csv.DictReader(open('tinker/evals/bit-star-r3.csv'))]
tot=collections.Counter(); ok=collections.Counter()
for r in ev:
    s=solved.get(r['id']); k=nv(s['expr']) if (s and s.get('correct')) else '?'
    tot[k]+=1; ok[k]+=int(r['correct'])
for k in sorted(tot,key=str): print(f"  {k}-tap: {ok[k]}/{tot[k]} = {ok[k]/max(tot[k],1):.3f}")
print(f"  OVERALL: {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/sum(tot.values()):.3f}  (r2 was 0.678)")
PYEOF
