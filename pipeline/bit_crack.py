"""Deep single-row program search to REVERSE-ENGINEER the bit_manip grammar.
Big budget per row; richer ops incl. maj/ch of shifted copies + terminal const-mask. Prints the program."""
import csv,re,sys,time
from collections import deque
M=0xFF
def rotl(x,k):k&=7;return ((x<<k)|(x>>(8-k)))&M if k else x
def rotr(x,k):return rotl(x,8-(k&7))
def shl(x,k):return (x<<k)&M
def shr(x,k):return x>>k
def maj(a,b,c):return (a&b)|(a&c)|(b&c)
def ch(a,b,c):return (a&b)|((~a&M)&c)
SH=[("rotl",rotl),("rotr",rotr),("shl",shl),("shr",shr)]
OPS=[("not",lambda v:(~v)&M)]
for _n,_f in SH:
    for _k in range(1,8):
        OPS.append((f"{_n}{_k}",lambda v,f=_f,k=_k:f(v,k)))
        OPS+=[(f"x^{_n}{_k}",lambda v,f=_f,k=_k:v^f(v,k)),
              (f"x&{_n}{_k}",lambda v,f=_f,k=_k:v&f(v,k)),
              (f"x|{_n}{_k}",lambda v,f=_f,k=_k:v|f(v,k))]
# maj/ch of (x, shift_a, shift_b) — a sample of taps
for _na,_fa in SH:
    for _ka in range(1,8):
        OPS+=[(f"maj(x,{_na}{_ka},rotr{_ka})",lambda v,fa=_fa,ka=_ka:maj(v,fa(v,ka),rotr(v,ka))),
              (f"ch(x,{_na}{_ka},rotr{_ka})",lambda v,fa=_fa,ka=_ka:ch(v,fa(v,ka),rotr(v,ka)))]
def parse(p):
    ex=[(int(a,2),int(b,2)) for a,b in re.findall(r'([01]{8})\s*->\s*([01]{8})',p)]
    q=int(re.search(r'output for:\s*([01]{8})',p).group(1),2); return ex,q
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
def crack(ex,q,maxdepth=6,cap=600000):
    ins=tuple(i for i,_ in ex); outs=tuple(o for _,o in ex)
    start=ins+(q,); seen={start}; dq=deque([(start,0,())])
    while dq:
        state,d,path=dq.popleft(); sv=state[:-1]
        if sv==outs: return list(path),state[-1]
        m=_mask(sv,outs)
        if m: return list(path)+[f"{m[0]} {format(m[1],'08b')}"], (state[-1]^m[1] if m[0]=='xor' else state[-1]&m[1] if m[0]=='and' else state[-1]|m[1])
        if d>=maxdepth or len(seen)>cap: continue
        for nm,f in OPS:
            ns=tuple(f(v) for v in state)
            if ns not in seen: seen.add(ns); dq.append((ns,d+1,path+(nm,)))
    return None
if __name__=="__main__":
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    nrows=int(sys.argv[1]) if len(sys.argv)>1 else 6
    cnt=0
    for r in rows:
        ex,q=parse(r['prompt'])
        t=time.time(); res=crack(ex,q)
        if res:
            prog,pred=res; corr=format(pred,'08b')==r['answer'].strip()
            print(f"[{'OK' if corr else 'BAD'}] prog={'/'.join(prog)} ({time.time()-t:.0f}s, len {len(prog)})",flush=True)
        else:
            print(f"[NOFIT] ({time.time()-t:.0f}s)",flush=True)
        cnt+=1
        if cnt>=nrows: break
