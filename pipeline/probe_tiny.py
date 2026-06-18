"""Decisive scan-size test. Give a TINY candidate set (2 correct taps + 2 distractors) with values precomputed for
example AND query. Base only: pick the pair whose XOR==OUT (tiny match), then XOR the chosen two query values.
If this converges reliably at scale -> logic is NATIVE and scan-size/verbosity is the ONLY wall (strong conviction).
Usage: python3 probe_tiny.py <N> [ncand]"""
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
def opof(rec):
    m=re.match(r'^([A-Z_]+)',rec['expr'] or ''); return m.group(1) if m else None
NT=int(sys.argv[1]) if len(sys.argv)>1 else 6
NC=int(sys.argv[2]) if len(sys.argv)>2 else 4
cand=[cid for cid,rec in cache.items() if rec['correct'] and rec['td'] and len(rec['td'])==2 and opof(rec)=='XOR'][:NT]
N.set_experiment("tiny_xor")
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
def pick_distractors(true_ts, seed):
    # deterministic distractors disjoint from true taps
    h=int(hashlib.md5(seed.encode()).hexdigest(),16)
    pool=[t for t in ALLT if t not in true_ts]
    out=[]
    for i in range(NC-len(true_ts)):
        out.append(pool[(h+i*7)%len(pool)])
    return out
ok=bad=0; print(f"tiny XOR N={len(cand)} ncand={NC}",flush=True)
for cid in cand:
    e=rows[cid]; rec=cache[cid]
    m=re.match(r'^XOR\((\{[AB]\}), *(\{[AB]\})\)$',rec['expr'])
    ta=tuple(rec['td'][m.group(1)]); tb=tuple(rec['td'][m.group(2)])
    ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',e['prompt']); q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
    x1,y1=ex[0]; X1=int(x1,2); Q=int(q,2)
    cands=[ta,tb]+pick_distractors([ta,tb],cid)
    # stable display order by name
    cands=sorted(set(cands), key=lambda t:tname(t))
    exline=", ".join(f"{tname(t)}={i2b(tval(X1,t))}" for t in cands)
    qline=", ".join(f"{tname(t)}={i2b(tval(Q,t))}" for t in cands)
    p=(f"Two of these candidate values XOR together to give OUT={y1}. Candidates (for input {x1}): {exline}. "
       f"Identify the TWO candidate names whose XOR equals {y1}. Then, for the query input {q} whose same candidates "
       f"are: {qline}, XOR those same two and give the 8-bit result in \\boxed{{}}.")
    r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=2600, temperature=0.0, meta={"id":cid,"rule":rec['expr']})
    got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold); ok+=good; bad+=(not good)
    print(f"  {cid} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
print(f"RESULT tiny XOR (ncand={NC}): {ok}/{ok+bad}",flush=True)
