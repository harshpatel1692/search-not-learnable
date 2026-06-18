set -e
cd "$(dirname "$0")/.."
export TINKER_API_KEY="$(grep -oP 'TINKER_API_KEY="\K[^"]+' ~/.bashrc)"
PY=~/.venvs/tinker/bin/python
echo "=== TRAIN bit-2tap (1 ep, 1750 rows: full r2 corpus + 1+2-tap concentration + synth diversity) ==="
$PY tinker/sft_warmstart.py --data pipeline/data/v16/train_bit_2tap_trim.csv --epochs 1 --run-name bit-2tap 2>&1 | tee /tmp/bit_2tap_train.log
SAMP=$(grep -oP 'sampler path : \K\S+' /tmp/bit_2tap_train.log | head -1)
echo "SAMP=[$SAMP]"; [ -z "$SAMP" ] && { echo NO_SAMPLER; exit 1; }
echo "=== EVAL bit-2tap on bit_eval500 (disjoint, honest) ==="
$PY tinker/eval_oracle.py --ckpt "$SAMP" --cats bit_manipulation --rows-file pipeline/data/bit_eval500.jsonl --tag bit-2tap 2>&1 | tee /tmp/bit_2tap_eval.log
$PY - <<'PYEOF'
import json, csv, collections
solved={json.loads(l)['id']:json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl')}
def nv(e): return len([v for v in ('{A}','{B}','{C}') if v in e])
ev=[r for r in csv.DictReader(open('tinker/evals/bit-2tap.csv'))]
tot=collections.Counter(); ok=collections.Counter()
for r in ev:
    s=solved.get(r['id']); k=f"{nv(s['expr'])}-tap" if (s and s.get('correct')) else '?'
    tot[k]+=1; ok[k]+=int(r['correct'])
for k in sorted(tot,key=str): print(f"  {k}: {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}")
print(f"  OVERALL: {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/sum(tot.values()):.3f}")
print("  baseline bit-star-r2: 1t 0.881 | 2t 0.763 | 3t 0.503 | overall 0.678")
PYEOF
echo BIT_2TAP_DONE
