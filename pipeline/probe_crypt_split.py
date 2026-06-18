"""Base few-shot probe (the reviewer's gate): can the base model, handed the
cue->operator->verified-split procedure IN CONTEXT, solve additive deduce rows it
should be able to? If it can't few-shot, SFT won't install it.

Tests on additive-query rows that the engine solves within <=2 splits (the
addressable set). Greedy (temp 0, mirrors the grader). Logs to nvidia_logs/.

  python3 pipeline/probe_crypt_split.py select        # build the row set
  python3 pipeline/probe_crypt_split.py run [n] [model]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solvers'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis', 'crypt_struct'))
import cryptarithm2 as C2
import nvidia_api as NV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWSET = os.path.join(ROOT, 'pipeline', 'data', 'crypt_probe_additive.jsonl')
ADD = {'add', 'add_p1', 'add_m1', 'add_p2', 'add_m2', 'sub_signed', 'rsub_signed'}

# 1-shot exemplar: the reviewer's worked trace (cue -> operator -> verified split
# -> map -> verify -> spaced encode). Self-contained and correct.
EXEMPLAR = """Here is the method on a solved example.

Puzzle:
  @ ! # ! :  =  : < /
  / : ' " "  =  " $ ] /
  ] ! - @ "  =  $
Find: ! $ - ! "

Solution:
Each left side is 5 symbols: two digit-symbols, an operator (middle), two digit-symbols.
Read the operator from the RESULT length, no digit values needed:
  EQ2 result " $ ] / has 4 symbols. Only multiplying two 2-digit numbers reaches 4 digits, so ' = a * b.
  EQ1 result : < / has 3 symbols, reachable by adding, so # = a + b.
  EQ3 result $ has 1 symbol, a small value, so - = a - b.
Decide digit order. Try standard (left = tens): EQ1 sum of two 2-digit numbers is <= 198, so its 3-digit
result starts with 1, i.e. ':' = 1. Then EQ2 first operand '/ :' ends in ':'=1; a number ending in 1 times
'" "' ends in the same digit as '"', but the result ends in '/', forcing '/'='"' -- two symbols can't share a
digit. Contradiction. So little-endian: reverse every number.
Re-read little-endian. EQ1's 3-digit sum still starts with 1, now the LAST symbol of ': < /', so '/' = 1.
Solve the multiplication EQ2 (most constrained): operand1 = ': /' = 10*: + 1; operand2 = '" "' = 11*";
product read little-endian = reverse of '" $ ] /' = 1 ] $ ", in 1000-1999. Scan " = 2..9, products with
distinct digits, none = 1:
  21*66=1386 -> :2 "6 ]3 $8 ;  31*55=1705 -> :3 "5 ]7 $0 ;  51*33=1683 -> :5 "3 ]6 $8 ;
  61*22=1342 -> :6 "2 ]3 $4 ;  71*22=1562 -> :7 "2 ]5 $6
Filter by EQ3 (little-endian (10*!+]) - (10*"+@) = $) and EQ1:
  :2 "6 ]3 $8 -> 10*!-@=65 -> !7 @5 ; EQ1: 77+5+20=102 = 100+10*<+2 -> <0. all distinct. fits.
  the other four each force a digit collision. so the first survives.
Map: / 1, : 2, ] 3, @ 5, " 6, ! 7, $ 8, < 0.
Verify: 75+27=102 ok ; 21*66=1386 ok ; 73-65=8 ok.
Find ! $ - ! " little-endian: '$ !'=87, '" !'=67, 87-67=20; little-endian writes 0 then 2; 0->'<', 2->':'.
\\boxed{<:}
"""

INSTR = ("You solve symbol-cipher puzzles: a hidden map assigns each symbol a digit, the middle symbol of "
         "each 5-symbol left side is an operator, numbers may be little-endian. Use the method shown: read "
         "operators from result length, fix the reading direction by contradiction, then pin digits by "
         "arithmetic (try a value, show the contradiction, keep the survivor). Rewrite symbols spaced. "
         "End with the answer in \\boxed{}.\n\n" + EXEMPLAR)


def select():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'pipeline', 'data',
            'crypt_train_all.jsonl')) if json.loads(l)['category'] == 'cryptarithm_deduce']
    import propagation as P
    FULL = list(C2.OPS) + list(C2.CONCATS)
    keep = []
    for r in rows:
        eqs, qL = C2.parse(r['prompt'])
        # query op additive? (use solver)
        res = C2.solve(r['prompt'], deadline_s=3.0)
        if not res:
            continue
        qop = (res[1] or {}).get('ops', {}).get(qL[2], '')
        if qop not in ADD:
            continue
        oc = {L[2] for L, R in eqs} | {qL[2]}
        try:
            e = P.Eng(eqs, qL, {g: FULL for g in oc}, oc <= set('+-*'), time.time() + 8)
            e.propagate()
            a, _ = e.answer()
            d = 0 if a is not None else None
            if d is None:
                a, _, _ = e.one_split()
                d = 1 if a is not None else None
            if d is None:
                a, _, _ = e.two_split()
                d = 2 if a is not None else None
        except Exception:
            d = None
        if d is not None and a == r['answer']:
            keep.append({'id': r['id'], 'prompt': r['prompt'], 'answer': r['answer'], 'split': d})
        if len(keep) >= 40:
            break
    with open(ROWSET, 'w') as f:
        for k in keep:
            f.write(json.dumps(k) + '\n')
    print(f'selected {len(keep)} additive <=2-split rows -> {ROWSET}')


def run(n=25, model='nvidia/nemotron-3-nano-30b-a3b'):
    NV.set_experiment('crypt_split_probe')
    rows = [json.loads(l) for l in open(ROWSET)][:n]
    ok = 0
    for i, r in enumerate(rows, 1):
        prompt = INSTR + "\n\nNow solve this puzzle.\n\n" + r['prompt']
        out = NV.ask(prompt, model=model, max_tokens=4000, temperature=0.0,
                     meta={'id': r['id'], 'split': r['split']}, add_box=False)
        pred = NV.extract_final_answer(out or '')
        hit = (pred == r['answer'])
        ok += hit
        print(f"[{i}/{len(rows)}] id={r['id']} split={r['split']} gold={r['answer']!r} "
              f"pred={pred!r} {'OK' if hit else 'x'}  (running {ok}/{i} = {ok/i:.2f})", flush=True)
    print(f"\nPROBE {model}: {ok}/{len(rows)} = {ok/len(rows):.2f} on additive <=2-split rows")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'select':
        select()
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 25,
            sys.argv[3] if len(sys.argv) > 3 else 'nvidia/nemotron-3-nano-30b-a3b')
