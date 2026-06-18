"""bit_manipulation solver: deterministic search over the rule family.
Strategy escalates: (1) single unary, (2) unary+mask(xor/and/or), (3) combine two unaries,
(4) maj/ch of three rotated args, (5) per-output-bit boolean fit (each out bit = fn of a few in bits).
Returns (answer8, cot) or None if no consistent hypothesis covers ALL examples."""
import re
M=0xFF
def rotl(x,k): k&=7; return ((x<<k)|(x>>(8-k)))&M if k else x
def rotr(x,k): return rotl(x,8-(k&7))
def shl(x,k): return (x<<k)&M
def shr(x,k): return (x>>k)&M
def notx(x): return (~x)&M
def maj(a,b,c): return (a&b)|(a&c)|(b&c)
def ch(a,b,c): return (a&b)|((~a&M)&c)
def b8(x): return format(x,'08b')

def _unaries():
    U=[("input", lambda x:x), ("NOT(input)", notx)]
    for k in range(1,8): U.append((f"ROTL{k}", lambda x,k=k:rotl(x,k)))
    for k in range(1,8): U.append((f"ROTR{k}", lambda x,k=k:rotr(x,k)))
    for k in range(1,8): U.append((f"SHL{k}", lambda x,k=k:shl(x,k)))
    for k in range(1,8): U.append((f"SHR{k}", lambda x,k=k:shr(x,k)))
    return U
U=_unaries()
ROT_SUB=[(n,f) for n,f in U if n.startswith('ROT') or n in ('input','NOT(input)')]

def parse(prompt):
    ex=[]
    for line in prompt.splitlines():
        m=re.match(r'\s*([01]{8})\s*->\s*([01]{8})\s*$', line)
        if m: ex.append((int(m.group(1),2), int(m.group(2),2)))
    qm=re.search(r'determine the output for:\s*([01]{8})', prompt)
    q=int(qm.group(1),2) if qm else None
    return ex,q

def _fit_mask_and(pairs):  # out = src & mask ; derive mask, check
    mask=0
    for s,o in pairs:
        for b in range(8):
            if (o>>b)&1:
                if not (s>>b)&1: return None   # need src bit 1
                mask|=(1<<b)
    # bits where some src=1 but out=0 must be mask=0; bits never constrained -> 0
    for s,o in pairs:
        for b in range(8):
            if (s>>b)&1 and not (o>>b)&1 and (mask>>b)&1: return None
    if all((s&mask)==o for s,o in pairs): return mask
    return None
def _fit_mask_or(pairs):   # out = src | mask
    mask=M
    for s,o in pairs:
        for b in range(8):
            if not (o>>b)&1:
                if (s>>b)&1: return None
                mask&=~(1<<b)
    if all((s|mask)==o for s,o in pairs): return mask
    return None
def _fit_mask_xor(pairs):  # out = src ^ mask
    mask=pairs[0][0]^pairs[0][1]
    if all((s^mask)==o for s,o in pairs): return mask
    return None

def _search(ex):
    """Return (apply_fn, desc) for the first hypothesis consistent with ALL examples, else None."""
    # 1) single unary
    for n,f in U:
        if all(f(i)==o for i,o in ex): return f, n
    # 2) unary + mask
    for n,f in U:
        pairs=[(f(i),o) for i,o in ex]
        mk=_fit_mask_xor(pairs)
        if mk is not None: return (lambda x,f=f,m=mk:f(x)^m), f"{n} XOR {b8(mk)}"
        mk=_fit_mask_and(pairs)
        if mk is not None: return (lambda x,f=f,m=mk:f(x)&m), f"{n} AND {b8(mk)}"
        mk=_fit_mask_or(pairs)
        if mk is not None: return (lambda x,f=f,m=mk:f(x)|m), f"{n} OR {b8(mk)}"
    # 3) combine two unaries
    for n1,f1 in U:
        for n2,f2 in U:
            if all((f1(i)^f2(i))==o for i,o in ex): return (lambda x,f1=f1,f2=f2:f1(x)^f2(x)), f"({n1}) XOR ({n2})"
            if all((f1(i)&f2(i))==o for i,o in ex): return (lambda x,f1=f1,f2=f2:f1(x)&f2(x)), f"({n1}) AND ({n2})"
            if all((f1(i)|f2(i))==o for i,o in ex): return (lambda x,f1=f1,f2=f2:f1(x)|f2(x)), f"({n1}) OR ({n2})"
    # 4) maj / ch of three rotated args
    for n1,f1 in ROT_SUB:
        for n2,f2 in ROT_SUB:
            for n3,f3 in ROT_SUB:
                if all(maj(f1(i),f2(i),f3(i))==o for i,o in ex): return (lambda x,f1=f1,f2=f2,f3=f3:maj(f1(x),f2(x),f3(x))), f"MAJ({n1},{n2},{n3})"
                if all(ch(f1(i),f2(i),f3(i))==o for i,o in ex):  return (lambda x,f1=f1,f2=f2,f3=f3:ch(f1(x),f2(x),f3(x))),  f"CH({n1},{n2},{n3})"
    # 5) per-output-bit: each out bit = single in bit (opt. negated) or constant or XOR of two in bits
    perbit=[None]*8
    for j in range(8):
        ob=[(o>>j)&1 for _,o in ex]
        # constant
        if all(b==ob[0] for b in ob): perbit[j]=('const',ob[0]); continue
        found=False
        for i in range(8):
            ib=[(x>>i)&1 for x,_ in ex]
            if ib==ob: perbit[j]=('copy',i,0); found=True; break
            if [1-b for b in ib]==ob: perbit[j]=('copy',i,1); found=True; break
        if found: continue
        for i in range(8):
            for k in range(8):
                if i>=k: continue
                xb=[((x>>i)&1)^((x>>k)&1) for x,_ in ex]
                if xb==ob: perbit[j]=('xor2',i,k); found=True; break
            if found: break
        if not found: return None
    def apply_pb(x):
        out=0
        for j,rule in enumerate(perbit):
            if rule[0]=='const': bit=rule[1]
            elif rule[0]=='copy': bit=((x>>rule[1])&1)^rule[2]
            else: bit=((x>>rule[1])&1)^((x>>rule[2])&1)
            out|=bit<<j
        return out
    return apply_pb, "per-bit boolean fit"

def solve(prompt):
    ex,q=parse(prompt)
    if not ex or q is None: return None
    res=_search(ex)
    if res is None: return None
    fn,desc=res
    if not all(fn(i)==o for i,o in ex): return None
    ans=b8(fn(q))
    (i1,o1),(i2,o2)=ex[0],ex[1]
    cot=(f"Derive the rule from {b8(i1)} -> {b8(o1)}: the transform is {desc}: {b8(fn(i1))}={b8(o1)} ✓. "
         f"Confirm on {b8(i2)}: {b8(fn(i2))}={b8(o2)} ✓. Apply to {b8(q)}: result = {ans}.")
    return ans, cot
