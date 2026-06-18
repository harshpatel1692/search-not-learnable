"""v15 training set: REVERT bit_manip to the v13 whole-value CoT (per-bit v14 CoT regressed 0.59->0.54),
ADD equation_numeric (terse, executable, the most learnable hard cat), keep easy-4. Zero leakage (synthetic values).
Grader is vLLM GREEDY 7680/8192 -> CoT must be EXECUTABLE step-by-step, not asserted. Eval on REAL train (val.jsonl).
"""
import sys, json, random
sys.path.insert(0,'pipeline')
from synth import numeral, unit_conversion, gravity, cipher
from synth import bit_manipulation              # v13 whole-value try->reject->confirm CoT (NOT per-bit)
from synth import equation_numeric              # new: pairing x op x format, verify-on-examples, terse
# COUNTS ~ real train proportions; equation_numeric ~ (eq_deduce 598 + eq_guess 133). cryptarithm still excluded
# (no model-reproducible CoT yet; its solve is a 10!-cipher search the base can't do greedily).
COUNTS={'numeral':1576,'unit_conversion':1594,'gravity':1597,'cipher':1576,
        'bit_manipulation':1602,'equation_numeric':730}
GENS={'numeral':numeral.gen,'unit_conversion':unit_conversion.gen,'gravity':gravity.gen,
      'cipher':cipher.gen,'bit_manipulation':bit_manipulation.gen,'equation_numeric':equation_numeric.gen}
rng=random.Random(20260608)
rows=[]
for cat,n in COUNTS.items():
    for _ in range(n): rows.append(GENS[cat](rng))
rng.shuffle(rows)
with open('pipeline/data/train_synth.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
print(f"train_synth.jsonl (v15): {len(rows)} rows")
from collections import Counter
print(Counter(r['category'] for r in rows))
import statistics as st
print("median CoT chars by cat:", {c: int(st.median([len(r['cot']) for r in rows if r['category']==c])) for c in COUNTS})
