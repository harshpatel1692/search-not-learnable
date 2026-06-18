"""BFS program-search solver for bit_manipulation: chain of bit-ops in example-image space.
The generator builds a short pipeline (rot/shift/not, self-XOR/AND/OR, const mask). BFS over the
(example-inputs, query) tuple until example slots match outputs -> shortest program -> apply to query."""
import re
from collections import deque
M=0xFF
def rotl(x,k):k&=7;return ((x<<k)|(x>>(8-k)))&M if k else x
def rotr(x,k):return rotl(x,8-(k&7))
OPS=[("not",lambda v:(~v)&M)]
for _k in range(1,8):
    OPS+=[(f"rotl{_k}",lambda v,k=_k:rotl(v,k)),(f"rotr{_k}",lambda v,k=_k:rotr(v,k)),
          (f"shl{_k}",lambda v,k=_k:(v<<k)&M),(f"shr{_k}",lambda v,k=_k:v>>k),
          (f"x^rotl{_k}",lambda v,k=_k:v^rotl(v,k)),(f"x&rotl{_k}",lambda v,k=_k:v&rotl(v,k)),
          (f"x|rotl{_k}",lambda v,k=_k:v|rotl(v,k)),(f"x^shr{_k}",lambda v,k=_k:v^(v>>k)),
          (f"x^shl{_k}",lambda v,k=_k:v^((v<<k)&M)),(f"x&shr{_k}",lambda v,k=_k:v&(v>>k)),
          (f"x|shr{_k}",lambda v,k=_k:v|(v>>k))]

def parse(p):
    ex=[]
    for line in p.splitlines():
        m=re.match(r'\s*([01]{8})\s*->\s*([01]{8})\s*$',line)
        if m: ex.append((int(m.group(1),2),int(m.group(2),2)))
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

def solve_val(ex,q,maxdepth=4,cap=20000):
    ins=tuple(i for i,_ in ex); outs=tuple(o for _,o in ex)
    start=ins+(q,); seen={start}; dq=deque([(start,0,[])])
    while dq:
        state,d,path=dq.popleft(); sv=state[:-1]
        if sv==outs: return state[-1],path
        m=_mask(sv,outs)
        if m:
            op,mm=m; qv=state[-1]
            return (qv^mm if op=='xor' else qv&mm if op=='and' else qv|mm), path+[f"{op} {format(mm,'08b')}"]
        if d>=maxdepth or len(seen)>cap: continue
        for nm,f in OPS:
            ns=tuple(f(v) for v in state)
            if ns not in seen: seen.add(ns); dq.append((ns,d+1,path+[nm]))
    return None

def solve(prompt,maxdepth=4,cap=20000):
    ex,q=parse(prompt)
    if len(ex)<2 or q is None: return None
    r=solve_val(ex,q,maxdepth,cap)
    if r is None: return None
    val,path=r
    return format(val,'08b'), path

if __name__=="__main__":
    import csv,sys
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
    md=int(sys.argv[2]) if len(sys.argv)>2 else 4
    cap=int(sys.argv[3]) if len(sys.argv)>3 else 20000
    rows=rows[:n]
    ok=miss=wrong=0
    for idx,r in enumerate(rows):
        res=solve(r['prompt'],md,cap)
        if res is None: miss+=1
        elif res[0]==r['answer'].strip(): ok+=1
        else: wrong+=1
        if (idx+1)%100==0: print(f"  ..{idx+1}/{len(rows)} ok={ok} nofit={miss} wrong={wrong}",flush=True)
    print(f"BFS d<={md} cap={cap}: solved {ok} ({100*ok/len(rows):.1f}%) | NOFIT {miss} | WRONG {wrong} | n={len(rows)}")
