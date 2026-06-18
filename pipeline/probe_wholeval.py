"""Validate a WHOLE-VALUE CoT hypothesis for 2-tap bit_manip (base's proven strength = compute named ops on the
full 8-bit number). Procedure: precompute the transformed copies of input1, then find which two combine (via one
op) to output1 -> lookup over ~28 whole values, not per-bit bookkeeping. Probe base; measure converge + read."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_wholeval")
ex=json.load(open('pipeline/data/bitmanip_subcats.json'))

PROC=("Find the rule, then the query output. The rule is: OUT = OP( T1(input), T2(input) ), where T1 and T2 "
 "are whole-8-bit transforms — rotate-left/right by k, or shift-left/right by k (shifts fill vacated bits with 0) "
 "— and OP is one of AND, OR, XOR, NAND, NOR, XNOR (or one input negated). Work on WHOLE 8-bit values, not bit "
 "positions.\n"
 "Procedure (efficient):\n"
 "1) Take the first example: input X and output Y.\n"
 "2) Compute the candidate copies of X as whole bytes: ROTL1..7, ROTR1..7, SHL1..7, SHR1..7 (list them).\n"
 "3) Find the rule: for XOR, compute Y XOR (each copy) and check if the result equals another copy -> then "
 "Y = copyA XOR copyB. For OR/AND, find two copies whose OR/AND equals Y (use bit counts to guess: OR if Y has "
 "more 1s, AND if fewer, XOR if about half).\n"
 "4) Once OP, T1, T2 match the first example, VERIFY on the second example; if it fails, try the next candidate.\n"
 "5) Apply OP(T1(query), T2(query)) to the query input and give the 8-bit answer.\n\n")

SUBS=['2tap-XOR-shift','2tap-OR-shift','2tap-AND-shift']
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
for sub in SUBS:
    e=ex[sub]
    r=N.ask(PROC+e['prompt'], model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0,
            meta={"sub":sub,"rule":e['expr'],"td":e['td']})
    got=norm(r.get("answer","")); gold=e['answer']
    print(f"{sub:18s} {'CORRECT' if got==gold else 'WRONG':7s} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
