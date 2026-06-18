"""Hardened BFS program search for bit_manipulation -> aim for full coverage.
Richer op set: unary rot/shift/not + self-combine (x OP shifted/rotated(x)) for OP in &,|,^ and all offsets.
Terminal derived const-mask (xor/and/or). BFS in example-image space; shortest program; apply to query."""
import re
from collections import deque
M=0xFF
def rotl(x,k):k&=7;return ((x<<k)|(x>>(8-k)))&M if k else x
def rotr(x,k):return rotl(x,8-(k&7))
def shl(x,k):return (x<<k)&M
def shr(x,k):return x>>k
UN=[("not",lambda v:(~v)&M)]
for _k in range(1,8):
    UN+=[(f"rotl{_k}",lambda v,k=_k:rotl(v,k)),(f"rotr{_k}",lambda v,k=_k:rotr(v,k)),
         (f"shl{_k}",lambda v,k=_k:shl(v,k)),(f"shr{_k}",lambda v,k=_k:shr(v,k))]
SHIFTS=[("rotl",rotl),("rotr",rotr),("shl",shl),("shr",shr)]
SELF=[]
for _nm,_f in SHIFTS:
    for _k in range(1,8):
        SELF+=[(f"x^{_nm}{_k}",lambda v,f=_f,k=_k:v^f(v,k)),
               (f"x&{_nm}{_k}",lambda v,f=_f,k=_k:v&f(v,k)),
               (f"x|{_nm}{_k}",lambda v,f=_f,k=_k:v|f(v,k))]
OPS=UN+SELF

def parse(p):
    ex=[(int(a,2),int(b,2)) for a,b in re.findall(r'([01]{8})\s*->\s*([01]{8})',p)]
    q=re.search(r'output for:\s*([01]{8})',p)
    return ex,(int(q.group(1),2) if q else None)

def _mask(sv,outs):
    mx=sv[0]^outs[0]
    if all((s^mx)==o for s,o in zip(sv,outs)): return ('xor',mx)
    ma=0
    for _,o in zip(sv,outs): ma|=o
    if all((s&ma)==o for s,o in zip(sv,outs)): return ('and',ma)
    mo=M
    for _,o in zip(sv,outs): mo&=o
    if all((s|mo)==o for s,o in zip(sv,outs)): return ('or',mo)
    return None

def solve_val(ex,q,maxdepth=5,cap=150000):
    ins=tuple(i for i,_ in ex); outs=tuple(o for _,o in ex)
    start=ins+(q,); seen={start}; dq=deque([(start,0,())])
    while dq:
        state,d,path=dq.popleft(); sv=state[:-1]
        if sv==outs: return state[-1],path
        m=_mask(sv,outs)
        if m:
            op,mm=m; qv=state[-1]
            return (qv^mm if op=='xor' else qv&mm if op=='and' else qv|mm), path+(f"{op} {format(mm,'08b')}",)
        if d>=maxdepth or len(seen)>cap: continue
        for nm,f in OPS:
            ns=tuple(f(v) for v in state)
            if ns not in seen: seen.add(ns); dq.append((ns,d+1,path+(nm,)))
    return None

def solve(prompt,maxdepth=5,cap=150000):
    ex,q=parse(prompt)
    if len(ex)<2 or q is None: return None
    r=solve_val(ex,q,maxdepth,cap)
    if r is None: return None
    return format(r[0],'08b'), list(r[1])

if __name__=="__main__":
    import csv,sys,time
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
    md=int(sys.argv[2]) if len(sys.argv)>2 else 5
    cap=int(sys.argv[3]) if len(sys.argv)>3 else 150000
    rows=rows[:n]; ok=miss=wrong=0; t0=time.time()
    for idx,r in enumerate(rows):
        res=solve(r['prompt'],md,cap)
        if res is None: miss+=1
        elif res[0]==r['answer'].strip(): ok+=1
        else: wrong+=1
        if (idx+1)%50==0: print(f"  ..{idx+1}/{len(rows)} ok={ok} nofit={miss} wrong={wrong} ({time.time()-t0:.0f}s)",flush=True)
    print(f"BFS2 d<={md} cap={cap}: solved {ok} ({100*ok/len(rows):.1f}%) | NOFIT {miss} | WRONG {wrong} | n={len(rows)} | {time.time()-t0:.0f}s")
