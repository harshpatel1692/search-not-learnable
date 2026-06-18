"""Generalize tiny-scan to OR/AND (2-tap) and to 3-tap. Tiny candidate set + adequate budget (2600).
Tests whether the FIND+APPLY logic is native across op families and tap-counts (not just XOR).
Usage: python3 probe_tiny2.py <op> <N>   op in: OR AND   (2-tap) | or: TAP3 (3-tap apply-given-taps)"""
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
OP=sys.argv[1] if len(sys.argv)>1 else 'OR'; NT=int(sys.argv[2]) if len(sys.argv)>2 else 6
def distract(true_ts,seed,ncand=4):
    h=int(hashlib.md5(seed.encode()).hexdigest(),16); pool=[t for t in ALLT if t not in true_ts]
    return [pool[(h+i*7)%len(pool)] for i in range(ncand-len(true_ts))]
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
N.set_experiment(f"tiny2_{OP}")
ok=bad=0
if OP in ('OR','AND'):
    cand=[cid for cid,rec in cache.items() if rec['correct'] and rec['td'] and len(rec['td'])==2 and opof(rec)==OP][:NT]
    print(f"tiny2 {OP} N={len(cand)}",flush=True)
    for cid in cand:
        e=rows[cid]; rec=cache[cid]; m=re.match(rf'^{OP}\((\{{[AB]\}}), *(\{{[AB]\}})\)$',rec['expr'])
        ta=tuple(rec['td'][m.group(1)]); tb=tuple(rec['td'][m.group(2)])
        ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',e['prompt']); q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
        x1,y1=ex[0]; X1=int(x1,2); Q=int(q,2)
        cands=sorted(set([ta,tb]+distract([ta,tb],cid)), key=tname)
        exline=", ".join(f"{tname(t)}={i2b(tval(X1,t))}" for t in cands)
        qline=", ".join(f"{tname(t)}={i2b(tval(Q,t))}" for t in cands)
        p=(f"Two of these candidate values, combined with bitwise {OP}, give OUT={y1}. Candidates for input {x1}: "
           f"{exline}. Find the TWO whose {OP} equals {y1}. Then for query {q} (same candidates: {qline}) apply {OP} "
           f"to those same two and give the 8-bit result in \\boxed{{}}.")
        r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=4500, temperature=0.0, meta={"id":cid,"rule":rec['expr']})
        got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold); ok+=good; bad+=(not good)
        print(f"  {cid} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
else:  # TAP3: 3-tap apply given the exact taps+structure (does 3-tap apply scale?)
    cand=[cid for cid,rec in cache.items() if rec['correct'] and rec['td'] and len(rec['td'])==3][:NT]
    print(f"tiny2 TAP3-apply N={len(cand)}",flush=True)
    for cid in cand:
        e=rows[cid]; rec=cache[cid]; q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1); Q=int(q,2)
        expr=rec['expr']; td=rec['td']
        # spell out each variable's tap value for the query
        vals="; ".join(f"{v}={tname(tuple(td[v]))}(query)={i2b(tval(Q,tuple(td[v])))}" for v in td)
        p=(f"For query input {q}, the three taps are: {vals}. The output = {expr} (bitwise on 8-bit values; "
           f"NOT_A_AND_B(P,Q)=(NOT P) AND Q; NOT_A_OR_B(P,Q)=(NOT P) OR Q). Compute the 8-bit output in \\boxed{{}}.")
        r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=4500, temperature=0.0, meta={"id":cid,"rule":expr})
        got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold); ok+=good; bad+=(not good)
        print(f"  {cid} {expr[:38]:38s} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
print(f"RESULT tiny2 {OP}: {ok}/{ok+bad}",flush=True)
