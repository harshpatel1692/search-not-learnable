"""Assemble a fully-synthetic training set (easy-4 + bit_manip), different values from train.
Eval stays on REAL train (val.jsonl) -> zero leakage."""
import sys, json, random
sys.path.insert(0,'pipeline')
from synth import numeral, unit_conversion, gravity, cipher
from synth import bitmanip_perbit   # per-bit token-engineered bit_manip CoT (1/2/3-tap, real sub-dist)
# COUNTS match the REAL train category proportions (the 5 generatable cats); crypt/eq excluded (no solver yet)
COUNTS={'numeral':1576,'unit_conversion':1594,'gravity':1597,'cipher':1576,'bit_manipulation':1602}
GENS={'numeral':numeral.gen,'unit_conversion':unit_conversion.gen,'gravity':gravity.gen,
      'cipher':cipher.gen,'bit_manipulation':bitmanip_perbit.gen}
rng=random.Random(20260607)
rows=[]
for cat,n in COUNTS.items():
    for _ in range(n): rows.append(GENS[cat](rng))
rng.shuffle(rows)
with open('pipeline/data/train_synth.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
print(f"train_synth.jsonl: {len(rows)} rows")
from collections import Counter
print(Counter(r['category'] for r in rows))
# CoT length sanity (brevity matters per the over-thinking finding)
import statistics as st
print("median CoT chars by cat:", {c: int(st.median([len(r['cot']) for r in rows if r['category']==c])) for c in COUNTS})
