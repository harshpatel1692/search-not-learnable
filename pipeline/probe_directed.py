"""Conviction battery: can the base do the 2-tap / 3-tap SEARCH when taps are PRE-COMPUTED and the procedure is
DIRECTED (simulates an SFT'd model: terse taps + fixed steps)? Tests reliability across MANY held-out puzzles.
Usage: python3 probe_directed.py <family> <ntests> [maxtok]   family in: xor2 or2 and2 any3"""
import sys,re,json,csv; sys.path.insert(0,'pipeline')
import nvidia_api as N
M=0xFF
def shl(v,k):return (v<<k)&M
def shr(v,k):return v>>k
def rotl(v,k):k&=7;return ((v<<k)|(v>>(8-k)))&M if k else v
def i2b(v):return format(v,'08b')
def alltaps(v):
    t=[(f"SHL{k}",shl(v,k)) for k in range(1,8)]+[(f"SHR{k}",shr(v,k)) for k in range(1,8)]+[(f"ROTL{k}",rotl(v,k)) for k in range(1,8)]
    return t
def table(v): return ", ".join(f"{n}={i2b(x)}" for n,x in alltaps(v))

cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))}
def opof(rec):
    m=re.match(r'^([A-Z_]+)',rec['expr'] or ''); return m.group(1) if m else None
def ntap(rec): return len(rec['td']) if rec['td'] else 0
FAM=sys.argv[1] if len(sys.argv)>1 else 'xor2'
NT=int(sys.argv[2]) if len(sys.argv)>2 else 6
MAXTOK=int(sys.argv[3]) if len(sys.argv)>3 else 3000
pred={'xor2':lambda r: ntap(r)==2 and opof(r) in ('XOR','XNOR'),
      'or2': lambda r: ntap(r)==2 and opof(r)=='OR',
      'and2':lambda r: ntap(r)==2 and opof(r)=='AND',
      'any3':lambda r: ntap(r)==3}[FAM]
cand=[cid for cid,rec in cache.items() if rec['correct'] and pred(rec)][:NT]
N.set_experiment(f"directed_{FAM}")

PROC={'xor2':("The rule is OUT = XOR(tapA, tapB) where tapA,tapB are two of the listed taps. METHOD: for the first "
  "example, for each tap T in its list, compute (OUT XOR T); if that value appears as another tap in the SAME "
  "list, then tapA=T and tapB=that match. Confirm the pair on the 2nd example, then apply to the query (use the "
  "query's listed taps)."),
 'or2':("The rule is OUT = OR(tapA,tapB). METHOD: OUT's 0-bits must be 0 in BOTH taps; each tap must be a subset of "
  "OUT (tap AND OUT == tap). Among taps that are subsets of OUT, find two whose OR == OUT. Confirm on example 2, apply to query."),
 'and2':("The rule is OUT = AND(tapA,tapB). METHOD: OUT's 1-bits must be 1 in BOTH taps; each tap must be a superset of "
  "OUT. Among taps that are supersets of OUT, find two whose AND == OUT. Confirm on example 2, apply to query."),
 'any3':("The rule is a function of THREE taps, e.g. (tapA OP tapB) OP tapC, or MAJORITY/CHOICE/XOR3. METHOD: first "
  "find two taps combined by AND/OR/XOR that ALMOST match OUT, then find the third tap and the second op that fixes "
  "the remaining bits. Confirm on example 2, apply to query.")}[FAM]

def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
ok=bad=0
print(f"FAMILY={FAM} N={len(cand)} maxtok={MAXTOK}",flush=True)
for cid in cand:
    e=rows[cid]; ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',e['prompt']); q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
    (x1,y1),(x2,y2)=ex[0],ex[1]
    p=(f"8-bit bit-rule puzzle. {PROC}\n"
       f"Example1 input {x1} -> output {y1}. Taps of {x1}: {table(int(x1,2))}\n"
       f"Example2 input {x2} -> output {y2}. Taps of {x2}: {table(int(x2,2))}\n"
       f"Query input {q}. Taps of {q}: {table(int(q,2))}\n"
       f"Find the rule and give the query's 8-bit output in \\boxed{{}}.")
    r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=MAXTOK, temperature=0.0, meta={"id":cid,"rule":cache[cid]['expr']})
    got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold)
    ok+=good; bad+=(not good)
    print(f"  {cid} {cache[cid]['expr'][:34]:34s} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
print(f"RESULT {FAM}: {ok}/{ok+bad} correct",flush=True)
