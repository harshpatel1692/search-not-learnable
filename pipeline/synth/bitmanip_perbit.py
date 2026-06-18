"""Per-bit token-engineered CoT for bit_manipulation (the winner's ~85% method, unified across ops).
Each output bit = a gate of input bits at FIXED positions (sources), which STRIDE +1 down the rule. The CoT shows
the per-bit search (test candidate input-position pairs vs the output column → match), uses the stride to compress
bits 1-7, then applies per-bit to the query. WHOLE columns are read MSB-first (positions 0..7, left to right).
Sources use rotate (wrap) or shift (zero-fill). Covers 1-tap and 2-tap rules here."""
import random
M=0xFF
def b8(x):return format(x,'08b')
def bit(x,p):return (x>>(7-p))&1          # MSB-first position p
def src_pos(p,trans):
    """Return input position feeding output position p for a transform, or None if it shifts in a 0."""
    op,k=trans
    if op=='rot': return (p+k)%8
    if op=='shl':
        s=p+k; return s if s<8 else None
    s=p-k; return s if s>=0 else None     # shr
def tname(t):
    op,k=t; return {'rot':'ROTL','shl':'SHL','shr':'SHR'}[op]+str(k)
G2={'XOR':lambda a,b:a^b,'OR':lambda a,b:a|b,'AND':lambda a,b:a&b,'XNOR':lambda a,b:1-(a^b),
    'NOTA_AND_B':lambda a,b:(1-a)&b,'A_AND_NOTB':lambda a,b:a&(1-b),
    'NOTA_OR_B':lambda a,b:(1-a)|b,'A_OR_NOTB':lambda a,b:a|(1-b)}
# ASCII-only descriptions (verified clean against competition tokenizer; NO rare/byte-fragment tokens)
GSYM={'XOR':'XOR','OR':'OR','AND':'AND','XNOR':'XNOR','NOTA_AND_B':'(NOT A) AND B','A_AND_NOTB':'A AND (NOT B)',
      'NOTA_OR_B':'(NOT A) OR B','A_OR_NOTB':'A OR (NOT B)'}
PROMPT_HDR=("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The "
 "transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
 "choice functions.\n\nHere are some examples of input -> output:\n")

def _gate_bit(op,a,b):
    return G2[op](a,b)
def out_bit(inp,j,tA,tB,op):
    pa=src_pos(j,tA); pb=src_pos(j,tB)
    a=bit(inp,pa) if pa is not None else 0
    b=bit(inp,pb) if pb is not None else 0
    return _gate_bit(op,a,b)
def apply_rule(inp,tA,tB,op):
    return sum(out_bit(inp,j,tA,tB,op)<<(7-j) for j in range(8))

def _rt(rng):
    o=rng.choice(['rot','shl','shr']); return (o, rng.randint(1,7))
def _mk_prompt(ex,q):
    return PROMPT_HDR+"\n".join(f"{b8(i)} -> {b8(o)}" for i,o in ex)+f"\n\nNow, determine the output for: {b8(q)}"
# 2-tap op weights ~ real binary-op mix (XOR/OR/XNOR/AND dominate; asym smaller)
OP2_W=[('XOR',28),('OR',24),('XNOR',20),('AND',13),('NOTA_AND_B',6),('A_AND_NOTB',3),('NOTA_OR_B',4),('A_OR_NOTB',2)]
def _wchoice(rng,wpairs):
    tot=sum(w for _,w in wpairs); r=rng.uniform(0,tot); a=0
    for v,w in wpairs:
        a+=w
        if r<=a: return v
    return wpairs[-1][0]

def gen(rng):
    """Dispatch by real tap-count distribution: 1-tap 9%, 2-tap 56%, 3-tap 34% (const ~1% folded into 1-tap)."""
    r=rng.random()
    if r<0.09: return gen1(rng)
    if r<0.65: return gen2(rng)
    return gen3(rng)

def gen2(rng):
    op=_wchoice(rng,OP2_W)
    tA=_rt(rng); tB=_rt(rng)
    while tB==tA: tB=_rt(rng)
    n=rng.choice([8,9,10]); ins=rng.sample(range(256),n+1)
    ex=[(i,apply_rule(i,tA,tB,op)) for i in ins[:n]]; q=ins[n]; ans=apply_rule(q,tA,tB,op)
    return {"category":"bit_manipulation","prompt":_mk_prompt(ex,q),"answer":b8(ans),"final":b8(ans),
            "cot":render_cot(ex,q,ans,tA,tB,op)}

def render_cot(ex,q,ans,tA,tB,op):
    L=[]
    L.append("Read each 8-bit value as columns, positions 0..7 left-to-right. Each output bit is one Boolean gate "
             "of input bits at fixed positions; as the output position increases by 1, those input positions also "
             "increase by 1 (mod 8), because the rule is built from shifts/rotations.")
    # --- determine bit 0 by a small search over the output column ---
    col0=[bit(o,0) for _,o in ex]
    pa0=src_pos(0,tA); pb0=src_pos(0,tB)
    def colstr(c): return "".join(map(str,c))
    # SYSTEMATIC scan-until-match at bit 0: candidates ordered (op-major, then source-pairs),
    # sources = positions 0..7 and Z (a constant-0 from a shift edge). Reject each until the true gate.
    OP_ORDER=['XOR','OR','AND','XNOR','NOTA_AND_B','A_AND_NOTB','NOTA_OR_B','A_OR_NOTB']
    SRC=[('p',p) for p in range(8)]+[('Z',None)]
    def sbit(i,s): return bit(i,s[1]) if s[0]=='p' else 0
    def slabel(s): return f"pos{s[1]}" if s[0]=='p' else "0"
    trueA=('p',pa0) if pa0 is not None else ('Z',None)
    trueB=('p',pb0) if pb0 is not None else ('Z',None)
    # build the ordered candidate list
    cands=[]
    for cop in OP_ORDER:
        for ia in range(len(SRC)):
            for ib in range(len(SRC)):
                cands.append((cop,SRC[ia],SRC[ib]))   # incl. self-pairs (degenerate same-source gates)
    # locate the true gate, show up to LASTK rejects before it + the hit (keeps it terse but systematic)
    def colof(cop,sa,sb): return [_gate_bit(cop,sbit(i,sa),sbit(i,sb)) for i,_ in ex]
    tidx=next(k for k,(cop,sa,sb) in enumerate(cands) if cop==op and sa==trueA and sb==trueB)
    LASTK=8
    start=max(0,tidx-LASTK)
    L.append(f"Bit 0: output column = {colstr(col0)}. Scan candidate gates op(srcA,srcB) in order (srcs = positions "
             f"0..7 and 0=edge); reject until the column matches" + (f" (skipping the first {start} that miss)" if start>0 else "") + ":")
    for k in range(start,tidx):
        cop,sa,sb=cands[k]; c=colof(cop,sa,sb)
        L.append(f"  {cop}({slabel(sa)},{slabel(sb)}) -> {colstr(c)} != {colstr(col0)}, no.")
    srcA,srcB=slabel(trueA),slabel(trueB)
    L.append(f"  {op}({srcA},{srcB}) -> {colstr(col0)} = {colstr(col0)} ok -> MATCH. So bit0 = {GSYM[op]} of input {srcA},{srcB}.")
    L.append(f"By the +1 stride, bit j uses the same gate {GSYM[op]} on positions shifted by j: "
             f"tapA={tname(tA)}, tapB={tname(tB)} (shifts read 0 past the edge).")
    # verify one more bit
    j=3; col=[bit(o,j) for _,o in ex]; pj=src_pos(j,tA); qj=src_pos(j,tB)
    aj=[bit(i,pj) if pj is not None else 0 for i,_ in ex]; bj=[bit(i,qj) if qj is not None else 0 for i,_ in ex]
    gj=[_gate_bit(op,x,y) for x,y in zip(aj,bj)]
    L.append(f"Check bit {j}: {op}({'pos'+str(pj) if pj is not None else '0'},{'pos'+str(qj) if qj is not None else '0'}) -> {colstr(gj)} = {colstr(col)} ok.")
    # apply per-bit to query — EXPLICIT (this is the learnable step)
    qb=[]; steps=[]
    for j in range(8):
        pj=src_pos(j,tA); qj=src_pos(j,tB)
        a=bit(q,pj) if pj is not None else 0; b=bit(q,qj) if qj is not None else 0
        r=_gate_bit(op,a,b); qb.append(str(r))
        sa=f"q{pj}={a}" if pj is not None else "0"
        sb=f"q{qj}={b}" if qj is not None else "0"
        steps.append(f"b{j}:{op}({sa},{sb})={r}")
    L.append(f"Apply to query {b8(q)} (positions 0..7 = its bits). " + "; ".join(steps) + f". Result: {''.join(qb)}.")
    return " ".join(L)

import sys as _sys; _sys.path.insert(0,'pipeline')
import bit_stride as S   # per-bit solver (gives <=2-input gates, stride-parameterized)

# ---------- 1-tap ----------
def gen1(rng):
    tA=_rt(rng); neg=rng.choice([0,1])
    def ob(inp,j):
        p=src_pos(j,tA); a=bit(inp,p) if p is not None else 0; return a^neg
    def ap(inp): return sum(ob(inp,j)<<(7-j) for j in range(8))
    n=rng.choice([8,9,10]); ins=rng.sample(range(256),n+1)
    ex=[(i,ap(i)) for i in ins[:n]]; q=ins[n]; ans=ap(q)
    return {"category":"bit_manipulation","prompt":_mk_prompt(ex,q),"answer":b8(ans),"final":b8(ans),
            "cot":render_cot1(ex,q,ans,tA,neg)}
def render_cot1(ex,q,ans,tA,neg):
    def colstr(c): return "".join(map(str,c))
    col0=[bit(o,0) for _,o in ex]; pa0=src_pos(0,tA)
    L=["Read each 8-bit value as columns, positions 0..7 left-to-right. Each output bit equals one input bit "
       "(maybe inverted) at a fixed position; that position increases by 1 (mod 8) per output bit (a rotation/shift)."]
    L.append(f"Bit 0: output column = {colstr(col0)}. Scan single-source candidates (pos or NOT pos) until match:")
    SRC=list(range(8))+[None]; tp=pa0
    cands=[('id',s) for s in SRC]+[('not',s) for s in SRC]
    true=('not',pa0) if neg else ('id',pa0)
    tidx=cands.index(true); start=max(0,tidx-6)
    for k in range(start,tidx):
        kind,s=cands[k]; v=[(bit(i,s) if s is not None else 0)^(1 if kind=='not' else 0) for i,_ in ex]
        lab=("NOT " if kind=='not' else "")+("pos%d"%s if s is not None else "0")
        if v!=col0: L.append(f"  {lab} -> {colstr(v)} != {colstr(col0)}, no.")
    lab=("NOT " if neg else "")+("pos%d"%pa0 if pa0 is not None else "0")
    L.append(f"  {lab} -> {colstr(col0)} = {colstr(col0)} ok -> MATCH. So bit0 = {lab}.")
    L.append(f"By the +1 stride, the source is {tname(tA)}{' then NOT' if neg else ''} (reads 0 past the edge).")
    qb=[]; steps=[]
    for j in range(8):
        p=src_pos(j,tA); a=bit(q,p) if p is not None else 0; r=a^neg; qb.append(str(r))
        steps.append(f"b{j}:{('NOT ' if neg else '')}{('q%d=%d'%(p,a)) if p is not None else '0'}={r}")
    L.append(f"Apply to query {b8(q)}: "+"; ".join(steps)+f". Result: {''.join(qb)}.")
    return " ".join(L)

# ---------- 3-tap (rendered via bit_stride's <=2-input per-bit gates; skip genuine 3-input tail) ----------
def _maj(a,b,c): return 1 if (a+b+c)>=2 else 0
def _ch(a,b,c): return b if a else c
def gen3(rng,_tries=0):
    if _tries>40: return gen2(rng)   # fallback if we keep hitting the 3-input tail
    tA=_rt(rng); tB=_rt(rng); tC=_rt(rng)
    form=rng.choice(['nest','nest','maj','ch','par3'])
    o1=rng.choice(['AND','OR','XOR']); o2=rng.choice(['AND','OR','XOR'])
    def F(a,b,c):
        if form=='maj': return _maj(a,b,c)
        if form=='ch':  return _ch(a,b,c)
        if form=='par3':return a^b^c
        return G2[o2](G2[o1](a,b),c)
    def ob(inp,j):
        pa=src_pos(j,tA); pb=src_pos(j,tB); pc=src_pos(j,tC)
        a=bit(inp,pa) if pa is not None else 0; b=bit(inp,pb) if pb is not None else 0; c=bit(inp,pc) if pc is not None else 0
        return F(a,b,c)
    def ap(inp): return sum(ob(inp,j)<<(7-j) for j in range(8))
    n=rng.choice([8,9,10]); ins=rng.sample(range(256),n+1)
    ex=[(i,ap(i)) for i in ins[:n]]; q=ins[n]; ans=ap(q)
    assign,cov=S.solve_bits(ex)
    if cov<8: return gen3(rng,_tries+1)
    pred=S.mk([S.apply_bit(assign[j],q,j) for j in range(8)])
    if format(pred,'08b')!=b8(ans): return gen3(rng,_tries+1)
    return {"category":"bit_manipulation","prompt":_mk_prompt(ex,q),"answer":b8(ans),"final":b8(ans),
            "cot":render_from_assign(ex,q,assign)}

def _applystr(desc,x,j):
    if desc[0]=='C': return f"const{desc[1]}"
    if desc[0]=='U':
        a=(desc[1]+j)%8; return f"{'NOT ' if desc[2] else ''}q{a}={S.bit(x,a)}"
    _,a0,b0,op=desc; a=(a0+j)%8; b=(b0+j)%8; return f"{op}(q{a}={S.bit(x,a)},q{b}={S.bit(x,b)})"
def render_from_assign(ex,q,assign):
    def colstr(c): return "".join(str(S.bit(o,j)) for _,o in []) # placeholder
    cs=lambda c:"".join(map(str,c))
    col0=[S.bit(o,0) for _,o in ex]; d0=assign[0]
    L=["Read each 8-bit value as columns, positions 0..7 left-to-right. Each output bit is a small Boolean gate "
       "of input bits at fixed positions; the positions increase by 1 (mod 8) per output bit (shifts/rotations)."]
    # scan-until-match at bit 0 for d0
    L.append(f"Bit 0: output column = {cs(col0)}. Scan candidate gates until the column matches:")
    if d0[0]=='B':
        _,a0,b0,op=d0; OPO=['XOR','OR','AND','ANDN','ORN','XORN']; SRC=list(range(8))
        order=[(o,i,k) for o in OPO for i in SRC for k in SRC]
        def colof(o,i,k): return [S.OPS[o](S.bit(v,i),S.bit(v,k)) for v,_ in ex]
        ti=next(t for t,(o,i,k) in enumerate(order) if o==op and i==a0 and k==b0)
        for t in range(max(0,ti-6),ti):
            o,i,k=order[t]; c=colof(o,i,k)
            if c!=col0: L.append(f"  {o}(pos{i},pos{k}) -> {cs(c)} != {cs(col0)}, no.")
        L.append(f"  {op}(pos{a0},pos{b0}) -> {cs(col0)} = {cs(col0)} ok -> MATCH. bit0 = {op}(pos{a0},pos{b0}).")
    elif d0[0]=='U':
        a0=d0[1]; lab=("NOT " if d0[2] else "")+f"pos{a0}"; L.append(f"  {lab} -> {cs(col0)} = {cs(col0)} ok -> MATCH. bit0 = {lab}.")
    else:
        L.append(f"  column is constant {d0[1]} across examples -> bit0 = {d0[1]}.")
    # stride / exceptions
    if all(assign[j]==d0 for j in range(8)):
        L.append("Every output bit uses this same gate with positions shifted +j.")
    else:
        ex_bits=[j for j in range(8) if assign[j]!=d0]
        notes=[]
        for j in ex_bits:
            dj=assign[j]
            if dj[0]=='C': notes.append(f"b{j}=const{dj[1]}")
            elif dj[0]=='U': notes.append(f"b{j}={'NOT ' if dj[2] else ''}pos{(dj[1]+j)%8}")
            else: notes.append(f"b{j}={dj[3]}(pos{(dj[1]+j)%8},pos{(dj[2]+j)%8})")
        L.append("Most bits follow the +j stride; edge bits differ: "+"; ".join(notes)+".")
    qb=[S.apply_bit(assign[j],q,j) for j in range(8)]
    steps=[f"b{j}:{_applystr(assign[j],q,j)}={qb[j]}" for j in range(8)]
    L.append(f"Apply to query {b8(q)}: "+"; ".join(steps)+f". Result: {''.join(map(str,qb))}.")
    return " ".join(L)

if __name__=="__main__":
    rng=random.Random(7)
    import statistics as st
    lens=[]; bad=0
    for _ in range(200):
        r=gen(rng)
        # the CoT's stated result must equal the gold answer
        stated=r['cot'].split("Result:")[-1].strip().rstrip('.')
        if stated!=r['answer']: bad+=1
        lens.append(len(r['cot']))
    print(f"200 samples: CoT-answer mismatches={bad} | median CoT chars={int(st.median(lens))} | max={max(lens)}")
    rng=random.Random(1)
    for _ in range(2):
        r=gen(rng); print("="*60); print("PROMPT:\n"+r['prompt']); print("ANSWER:",r['answer']); print("COT:",r['cot'])
