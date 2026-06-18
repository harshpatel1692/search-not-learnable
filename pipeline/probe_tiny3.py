"""Can the base FIND a 3-tap rule when narrowed to a tiny candidate set + adequate budget? Tests feasibility of
Options 2/3 (staged decomposition / structure-template) for the 3-input tail. Candidates = 3 true taps + 1
distractor; values precomputed for ex1, ex2 (verify), query (apply). Budget 3500 (<1min).
Usage: python3 probe_tiny3.py <N> [mode]  mode: find (find rule among candidates) | staged (directed peel)"""
import sys,re,json,csv,hashlib; sys.path.insert(0,'pipeline')
import nvidia_api as N
M=0xFF
def shl(v,k):return (v<<k)&M
def shr(v,k):return v>>k
def rotl(v,k):k&=7;return ((v<<k)|(v>>(8-k)))&M if k else v
def i2b(v):return format(v,'08b')
def tval(v,t):
    op,k=t; return {'shl':shl,'shr':shr,'rot':rotl}[op](v,k)
def tname(t):
    op,k=t; return {'shl':'SHL','shr':'SHR','rot':'ROTL'}[op]+str(k)
ALLT=[('shl',k) for k in range(1,8)]+[('shr',k) for k in range(1,8)]+[('rot',k) for k in range(1,8)]
cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))}
NT=int(sys.argv[1]) if len(sys.argv)>1 else 6
MODE=sys.argv[2] if len(sys.argv)>2 else 'find'
cand=[cid for cid,rec in cache.items() if rec['correct'] and rec['td'] and len(rec['td'])==3][:NT]
N.set_experiment(f"tiny3_{MODE}")
def distract(true_ts,seed):
    h=int(hashlib.md5(seed.encode()).hexdigest(),16); pool=[t for t in ALLT if t not in true_ts]
    return [pool[h%len(pool)]]
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
ok=bad=0; print(f"tiny3 {MODE} N={len(cand)}",flush=True)
for cid in cand:
    e=rows[cid]; rec=cache[cid]; td=rec['td']
    trues=[tuple(v) for v in td.values()]
    cands=sorted(set(trues+distract(trues,cid)), key=tname)
    ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',e['prompt']); q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
    (x1,y1),(x2,y2)=ex[0],ex[1]
    def line(x): return ", ".join(f"{tname(t)}={i2b(tval(int(x,2),t))}" for t in cands)
    if MODE=='struct':
        # GIVE the exact structure; only the tap-assignment must be matched, then apply.
        p=(f"The output is computed by this exact formula: {rec['expr']} where {{A}},{{B}},{{C}} are three of the "
           f"candidate taps below (NOT_A_AND_B(P,Q)=(NOT P)AND Q; NOT_A_OR_B(P,Q)=(NOT P)OR Q). Determine which "
           f"candidate is A, which is B, which is C (so the formula reproduces both examples), then apply to the query.\n"
           f"Ex1 {x1}->{y1}; taps: {line(x1)}\nEx2 {x2}->{y2}; taps: {line(x2)}\nQuery {q}; taps: {line(q)}\n"
           f"Give the query 8-bit output in \\boxed{{}}.")
    else:
        p=(f"A bit-rule's output is a Boolean function of THREE of these candidate taps. The function may be: "
           f"majority(A,B,C); choice (A?B:C); parity A^B^C; or a nested form (A op B) op C with ops AND/OR/XOR and "
           f"possible NOT on a tap. Find the function + which 3 taps reproduce BOTH examples, then apply to the query.\n"
           f"Ex1 {x1}->{y1}; taps: {line(x1)}\n"
           f"Ex2 {x2}->{y2}; taps: {line(x2)}\n"
           f"Query {q}; taps: {line(q)}\n"
           f"Give the query's 8-bit output in \\boxed{{}}.")
    r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=3500, temperature=0.0, meta={"id":cid,"rule":rec['expr']})
    got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold); ok+=good; bad+=(not good)
    print(f"  {cid} {rec['expr'][:40]:40s} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
print(f"RESULT tiny3 {MODE}: {ok}/{ok+bad}",flush=True)
