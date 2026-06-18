"""Probe the base model's NARROWING ability per bit_manip sub-category, using the global single-rule
method rendered as a narrowing procedure. Measure correct + finish=stop, and save traces for reading."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_subcats")
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))

PROC=("Find the rule, then the query output. The rule makes EACH output bit by ONE fixed formula combining "
 "up to 3 'source bits'. A source for output position p (positions 0..7, left to right) comes from the input "
 "via a fixed transform: rotate-by-k -> input[(p+k) mod 8]; shift-left-k -> input[p+k] if in 0..7 else 0; "
 "shift-right-k -> input[p-k] if in 0..7 else 0.\n"
 "Method (be efficient, stop when verified):\n"
 "1) First hypothesize TWO sources, one operation: out[p] = OP(src1, src2), OP in AND/OR/XOR/NAND/NOR/XNOR or "
 "with a NOT on one source. Find OP and the two transforms by checking output bit 0 across ALL examples, then "
 "confirm bit 1, bit 2 (source positions move +1 each step).\n"
 "2) If no two-source rule fits, use THREE sources (majority / choice / (A op B) op C).\n"
 "3) Verify on 2-3 examples, then apply to the query input and give the 8-bit answer.\n\n")

SUBS=['1tap-IDENTITY','2tap-XOR-shift','2tap-OR-shift','2tap-AND-shift','3tap-XNOR-shift']
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
for sub in SUBS:
    if sub not in ex: print(f"{sub}: MISSING"); continue
    e=ex[sub]
    r=N.ask(PROC+e['prompt'], model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0,
            meta={"sub":sub,"rule":e['expr'],"td":e['td']})
    got=norm(r.get("answer","")); gold=e['answer']
    print(f"{sub:22s} {'CORRECT' if got==gold else 'WRONG':7s} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
