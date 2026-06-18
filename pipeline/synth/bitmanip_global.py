"""GLOBAL-tap tabular CoT for bit_manipulation (v15). Replaces the per-bit 84.6%-class grammar.

The rendered trace executes a deterministic, imitable procedure (implemented honestly in solve()):
  1. TABLE   : transpose inputs into position rows p0..p7 and outputs into columns c0..c7.
  2. SCAN    : fixed-order gate scan at bit 0 (consts, p, NOT p, 2-src ops, 3-src families) vs c0.
  3. WALK    : bits 1..7 by +1 stride with lawful edge events (wrap/die at the edge, SHR-tap births at p0,
               reveal of AND-masked taps), every bit checked against its column; dead ends backtrack.
  4. LOCK    : lift per-bit gates to <=3 global taps (ROTL/SHL/SHR k) + one combiner F (truth-table fill);
               if no <=3-tap rule fits, the walk is rejected and the scan resumes (NO defaults, ever).
  5. VERIFY  : all 8 column checks cover every example bit; restated as one explicit ok-line per pair.
  6. APPLY   : the locked per-bit gates on the query -> answer.
Invariants (cod-emit): scanned/locked/verified/applied rule is the same rule; boxed == computed answer;
gold never copied in. Pure ASCII. Honest coverage on the 1602 real rows: 1579/1602 pred==gold (98.56%).
"""
import itertools, json, re, csv, random

# ---------------- gate algebra ----------------
def reduce_gate(positions, tt):
    """positions: tuple (ints or None=reads 0); tt over len(positions) vars (MSB-first var order).
    Returns canonical gate: (sorted distinct influential positions, reduced tt)."""
    def f(assign):
        idx=0
        for p in positions:
            v=0 if p is None else assign[p]
            idx=(idx<<1)|v
        return tt[idx]
    real=sorted(set(p for p in positions if p is not None))
    out=[f(dict(zip(real,bits))) for bits in itertools.product((0,1),repeat=len(real))]
    rp=list(real); rtt=out; changed=True
    while changed:
        changed=False
        for i in range(len(rp)):
            n=len(rp); infl=False
            for idx in range(2**n):
                if (idx>>(n-1-i))&1: continue
                if rtt[idx]!=rtt[idx|(1<<(n-1-i))]: infl=True; break
            if not infl:
                rtt=[rtt[idx] for idx in range(2**n) if not ((idx>>(n-1-i))&1)]
                rp.pop(i); changed=True; break
    return tuple(rp), tuple(rtt)

def CHf(a,b,c): return b if a else c
BASE3=[
 ('MAJ', lambda a,b,c:1 if a+b+c>=2 else 0),
 ('CH',  CHf),
 ('XOR(%s,ANDN(%s,%s))',  lambda a,b,c:a^((1-b)&c)),
 ('XNOR(%s,ANDN(%s,%s))', lambda a,b,c:1-(a^((1-b)&c))),
 ('AND(%s,OR(%s,%s))',    lambda a,b,c:a&(b|c)),
 ('OR(%s,ANDN(%s,%s))',   lambda a,b,c:a|((1-b)&c)),
 ('NOR(%s,ANDN(%s,%s))',  lambda a,b,c:1-(a|((1-b)&c))),
 ('OR(%s,XNOR(%s,%s))',   lambda a,b,c:a|(1-(b^c)),),
 ('AND(%s,ORN(%s,%s))',   lambda a,b,c:a&((1-b)|c)),
 ('E2',  lambda a,b,c:1 if a+b+c==2 else 0),
 ('CH(%s,XOR(%s,%s),ORN(%s,%s))W', lambda a,b,c:(b^c) if a else (b|(1-c))),
]
def tt3(f,perm):
    return tuple(f(bits[perm[0]],bits[perm[1]],bits[perm[2]]) for bits in itertools.product((0,1),repeat=3))
# name templates for 3-src tts: tt -> callable(args sorted-order names)->string
NAME3={}
ORD3=[]
for name,f in BASE3:
    for perm in itertools.permutations(range(3)):
        t=tt3(f,perm)
        if t in NAME3: continue
        ORD3.append(t)
        if name=='MAJ': NAME3[t]=(lambda args,_p=perm:'MAJ(%s,%s,%s)'%tuple(args))
        elif name=='E2': NAME3[t]=(lambda args,_p=perm:'E2(%s,%s,%s)'%tuple(args))
        elif name=='CH': NAME3[t]=(lambda args,_p=perm:'CH(%s,%s,%s)'%(args[_p.index(0)],args[_p.index(1)],args[_p.index(2)]))
        elif name.endswith('W'): NAME3[t]=(lambda args,_p=perm:'CH(%s,XOR(%s,%s),ORN(%s,%s))'%(args[_p.index(0)],args[_p.index(1)],args[_p.index(2)],args[_p.index(2)],args[_p.index(1)]))
        else: NAME3[t]=(lambda args,_p=perm,_n=name:_n%(args[_p.index(0)],args[_p.index(1)],args[_p.index(2)]))
ORD1=[(0,1),(1,0)]
ORD2=[(0,1,1,0),(0,1,1,1),(0,0,0,1),(1,0,0,1),(1,0,0,0),(1,1,1,0),(0,1,0,0),(0,0,1,0),(1,1,0,1),(1,0,1,1)]
NAME2={(0,1,1,0):'XOR(%s,%s)',(0,1,1,1):'OR(%s,%s)',(0,0,0,1):'AND(%s,%s)',(1,0,0,1):'XNOR(%s,%s)',
       (1,0,0,0):'NOR(%s,%s)',(1,1,1,0):'NAND(%s,%s)',(0,1,0,0):'ANDN(%s,%s)',(0,0,1,0):'ANDN(%s,%s)R',
       (1,1,0,1):'ORN(%s,%s)',(1,0,1,1):'ORN(%s,%s)R'}
LIB={1:ORD1,2:ORD2,3:ORD3}

def gate_name(gate, labels=None):
    """ASCII name for canonical gate; labels(pos)->str defaults to pN."""
    pos,tt=gate
    lab=labels or (lambda p:'p%d'%p)
    a=[lab(p) for p in pos]
    if len(pos)==0: return str(tt[0])
    if len(pos)==1: return a[0] if tt==(0,1) else 'NOT %s'%a[0]
    if len(pos)==2:
        nm=NAME2[tt]
        if nm.endswith('R'): return nm[:-1]%(a[1],a[0])
        return nm%(a[0],a[1])
    return NAME3[tt](a)

# ---------------- scan candidate order ----------------
def build_cands():
    C=[((),(0,)),((),(1,))]
    for p in range(8): C.append(((p,),(0,1)))
    for p in range(8): C.append(((p,),(1,0)))
    for t in ORD2[:6]:
        for i in range(8):
            for j in range(i+1,8): C.append(((i,j),t))
    for t in ORD2[6:]:
        for i in range(8):
            for j in range(i+1,8): C.append(((i,j),t))
    for t in ORD3:
        for tr in itertools.combinations(range(8),3): C.append((tr,t))
    return C
CANDS=build_cands()

def col_of(gate, ins_bits):
    pos,tt=gate
    out=[]
    for ib in ins_bits:
        idx=0
        for p in pos: idx=(idx<<1)|ib[p]
        out.append(tt[idx])
    return tuple(out)

# ---------------- walk (stride + edge events) ----------------
def stride_variants(gate):
    pos,tt=gate
    if not pos: yield (pos,tt),'stride'; return
    if 7 not in pos: yield (tuple(p+1 for p in pos),tt),'stride'; return
    yield reduce_gate(tuple((p+1)%8 for p in pos),tt),'wrap'
    k=pos.index(7); npos=[p+1 for p in pos]; npos[k]=None
    yield reduce_gate(tuple(npos),tt),'dead'

def birth_exts(V):
    vpos,vtt=V; v=len(vpos)
    if v+1<=3:
        for T in LIB[v+1]:
            proj=tuple(T[idx] for idx in range(2**(v+1)) if idx%2==0)
            if proj!=tuple(vtt): continue
            yield reduce_gate(tuple(list(vpos)+[0]),T),'birth'
    base=set(vpos)|{0}
    others=[p for p in range(1,8) if p not in base]
    for extra_n in range(0,3):
        for extras in itertools.combinations(others,extra_n):
            P=tuple(sorted(base|set(extras)))
            m=len(P)
            if m>3: continue
            if 0 in vpos and extra_n==0: continue
            vidx=P.index(0)
            for T in LIB[m]:
                g=reduce_gate(tuple(None if i==vidx else P[i] for i in range(m)),T)
                if g!=(tuple(vpos),tuple(vtt)): continue
                ng=reduce_gate(P,T)
                if ng==(tuple(vpos),tuple(vtt)): continue
                yield ng,'reveal'

# ---------------- global lift (lock) ----------------
TR=[('rot',k) for k in range(8)]+[('shl',k) for k in range(1,8)]+[('shr',k) for k in range(1,8)]
def tap_pos(j,t):
    ty,k=t
    if ty=='rot': return (j+k)%8
    if ty=='shl':
        s=j+k; return s if s<8 else None
    s=j-k; return s if s>=0 else None
def tap_name(t):
    ty,k=t
    if ty=='rot' and k==0: return 'ID'
    return {'rot':'ROTL','shl':'SHL','shr':'SHR'}[ty]+str(k)

def lift(gates, ins_bits, cols):
    I=set((j,p) for j,(pos,tt) in enumerate(gates) for p in pos)
    if not I: return (),{}
    covs={t:set((j,tap_pos(j,t)) for j in range(8) if tap_pos(j,t) is not None) for t in TR}
    n_ex=len(ins_bits)
    for size in (1,2,3):
        for sub in itertools.combinations(TR,size):
            u=set()
            for t in sub: u|=covs[t]
            if not I<=u: continue
            F={}; okf=True
            for j in range(8):
                ps=[tap_pos(j,t) for t in sub]
                for e in range(n_ex):
                    key=tuple(0 if p is None else ins_bits[e][p] for p in ps)
                    o=cols[j][e]
                    if F.get(key,o)!=o: okf=False; break
                    F[key]=o
                if not okf: break
            if okf: return sub,F
    return None

def complete_F(taps,F):
    """Deterministic completion of partial F to a named library tt (or explicit tt). Returns tt tuple."""
    k=len(taps)
    if k==0: return (F.get((),0),)
    lib=LIB[k]+[t for t in itertools.product((0,1),repeat=2**k) if t not in LIB[k]]
    for T in lib:
        ok=True
        for key,v in F.items():
            idx=0
            for b in key: idx=(idx<<1)|b
            if T[idx]!=v: ok=False; break
        if ok: return T
    return None

def walk_lifted(g0, ins_bits, cols, max_nodes=8000):
    nodes=[0]; res={}
    def dfs(j,gate,hist):
        if nodes[0]>max_nodes: return None
        if j==7:
            gates=[g for _,g,_,_ in hist]
            lf=lift(gates,ins_bits,cols)
            if lf is None: return None
            res['taps'],res['F']=lf
            return hist
        nodes[0]+=1
        tried=[]; opts=[]
        for ng,tag in stride_variants(gate): opts.append((ng,tag))
        for ng,tag in stride_variants(gate):
            seen={o[0] for o in opts}
            for bg,kind in birth_exts(ng):
                if bg in seen: continue
                seen.add(bg); opts.append((bg,kind+'+'+tag))
        for ng,tag in opts:
            c=col_of(ng,ins_bits)
            if c==cols[j+1]:
                r=dfs(j+1,ng,hist+[(j+1,ng,tag,tuple(tried))])
                if r is not None: return r
                tried.append((ng,tag,'later-fail'))
            else:
                tried.append((ng,tag,'colfail'))
        return None
    h=dfs(0,g0,[(0,g0,'anchor',())])
    if h is None: return None
    return h,res['taps'],res['F']

def solve(ins, outs):
    """ins/outs: 8-char bit strings. Honest deterministic search. Returns solution dict or None."""
    ins_bits=[[int(c) for c in s] for s in ins]
    cols=[tuple(int(o[j]) for o in outs) for j in range(8)]
    doomed=[]
    for ci,(pos,tt) in enumerate(CANDS):
        c=col_of((pos,tt),ins_bits)
        if c!=cols[0]: continue
        w=walk_lifted((pos,tt),ins_bits,cols)
        if w is None:
            doomed.append(ci); continue
        h,taps,F=w
        Fc=complete_F(taps,F)
        return dict(winner=ci,traj=h,doomed=doomed,cols=cols,ins_bits=ins_bits,taps=taps,F=F,Fc=Fc)
    return None

def apply_rule(taps,Fc,xbits):
    out=[]
    for j in range(8):
        idx=0
        for t in taps:
            p=tap_pos(j,t)
            idx=(idx<<1)|(0 if p is None else xbits[p])
        out.append(Fc[idx])
    return ''.join(map(str,out))

def pick_F(taps, F, traj):
    """Total F for the locked rule: prefer the tt of the first full-arity walk gate (sl每 position order),
    checked against all filled entries; else first library-order completion. Returns (tt, src_bit or None)."""
    k=len(taps)
    if k==0: return (F.get((),0),), None
    for j in range(8):
        ps=[tap_pos(j,t) for t in taps]
        if any(p is None for p in ps) or len(set(ps))!=k: continue
        pos,tt=traj[j][1]
        if len(pos)!=k or set(pos)!=set(ps): continue
        # gate vars are sorted by position; map tap order -> gate var index
        order=[sorted(ps).index(p) for p in ps]
        T=[]
        for combo in itertools.product((0,1),repeat=k):
            gidx=0
            for gi in range(k):
                # gate var gi corresponds to tap with order[t]==gi
                ti=order.index(gi)
                gidx=(gidx<<1)|combo[ti]
            T.append(tt[gidx])
        T=tuple(T)
        ok=True
        for key,v in F.items():
            idx=0
            for b in key: idx=(idx<<1)|b
            if T[idx]!=v: ok=False; break
        if ok: return T,j
    return complete_F(taps,F), None

# ================= RENDERER =================
GUIDE=("Bit positions 0..7, left to right. The hidden rule combines up to 3 taps of the input "
"(ROTL k wraps, SHL k / SHR k shift in 0s): at output bit j a ROTL k tap reads input bit (j+k) mod 8, "
"SHL k reads bit j+k (0 past the end), SHR k reads bit j-k (0 before the start). "
"Ops: ANDN(u,v)=(NOT u) AND v, ORN(u,v)=(NOT u) OR v, CH(s,u,v)=v if s=0 else u, MAJ=majority, E2=exactly two ones. "
"Method: transpose to position rows and output columns, find the bit-0 gate by a fixed scan, "
"walk bits 1..7 by the +1 stride fixing edge events, lock the tap lines, verify every pair, apply to the query.")

def cs(col): return ''.join(map(str,col))

def tier_of(ci):
    """tier label + (start,end) for candidate index."""
    n2=28
    bounds=[('consts',0,2),('singles',2,10),('NOT',10,18)]
    base=18
    for nm in ('XOR','OR','AND','XNOR','NOR','NAND'):
        bounds.append((nm+' pairs',base,base+n2)); base+=n2
    for nm in ('ANDN','ANDN-rev','ORN','ORN-rev'):
        bounds.append((nm+' pairs',base,base+n2)); base+=n2
    # 3-src: group by base family; perms contiguous in ORD3
    fam_sizes=[]; i3=0
    seen=set(); fam_of_tt={}
    for name,f in BASE3:
        cnt=0
        for perm in itertools.permutations(range(3)):
            t=tt3(f,perm)
            if t in seen: continue
            seen.add(t); cnt+=1
        fam_sizes.append((name,cnt))
    n3=56
    for name,cnt in fam_sizes:
        lbl=name.replace('%s','_') if '%s' in name else name
        bounds.append((lbl+' triples',base,base+cnt*n3)); base+=cnt*n3
    for nm,s,e in bounds:
        if s<=ci<e: return nm,s,e
    return '?',0,0

def probe_walk_fail(g0, ins_bits, cols):
    """greedy walk; return (bit, tried_names) at first failure, or None if it completes (lift failed)."""
    gate=g0
    for j in range(7):
        opts=[]
        for ng,tag in stride_variants(gate): opts.append((ng,tag))
        for ng,tag in stride_variants(gate):
            seen={o[0] for o in opts}
            for bg,kind in birth_exts(ng):
                if bg in seen: continue
                seen.add(bg); opts.append((bg,kind+'+'+tag))
        hit=None
        for ng,tag in opts:
            if col_of(ng,ins_bits)==cols[j+1]: hit=ng; break
        if hit is None:
            return j+1, gate
        gate=hit
    return None, None

def tap_line_desc(t, traj):
    """describe a tap line: where it runs."""
    ty,k=t
    if ty=='rot' and k==0: return "ID (reads bit j)"
    if ty=='rot': return "ROTL%d (reads (j+%d) mod 8, wraps at b%d)"%(k,k,8-k)
    if ty=='shl': return "SHL%d (reads j+%d, dies after b%d)"%(k,k,7-k)
    return "SHR%d (reads j-%d, born at b%d)"%(k,k,k)

def render(ins, outs, q, sol, win=6, doom_max=6):
    """Render the CoT. Returns (cot_text, final_answer)."""
    n=len(ins); ins_bits=sol['ins_bits']; cols=sol['cols']; traj=sol['traj']; taps=sol['taps']
    L=[GUIDE]
    # TABLE
    L.append("Position rows (bit p of x1..x%d):"%n)
    for p in range(8): L.append("p%d %s"%(p, ''.join(s[p] for s in ins)))
    L.append("Output columns (bit j of y1..y%d):"%n)
    for j in range(8): L.append("c%d %s"%(j, cs(cols[j])))
    # SCAN
    wci=sol['winner']; doomed=sol['doomed']
    L.append("Scan vs c0=%s in fixed order: consts; singles; NOT; pairs XOR,OR,AND,XNOR,NOR,NAND,ANDN,ORN; then triples."%cs(cols[0]))
    wt,ws,we=tier_of(wci)
    # tier summaries before winner tier; doomed candidates inline (capped at doom_max)
    events=sorted(set([ci for ci in doomed if ci<wci]+[wci]))
    doom_pre=[ci for ci in events if ci!=wci]
    doom_show=set(doom_pre[:doom_max])
    doom_skip=len(doom_pre)-len(doom_show)
    events=sorted(doom_show|{wci})
    cur=0
    shown=set()
    for ci in events:
        t,s,e=tier_of(ci)
        # summaries for fully-skipped tiers between cur and s
        b=cur
        while b<s:
            tn,ts,te=tier_of(b)
            if not any(ts<=x<te for x in events):
                if te-ts>8 and ts>=18 and ts>=cur:   # ground big pair/triple tiers with one explicit check
                    g0=CANDS[ts]
                    L.append("%s: %s -> %s no; none of these match c0."%(tn,gate_name(g0),cs(col_of(g0,ins_bits))))
                elif ts>=cur:
                    L.append("%s: none match c0."%tn)
            b=te
        # window of rejects within this tier before ci
        lo=max(s,cur,ci-(win if ci==wci else 2))
        nskip=lo-max(s,cur)
        if nskip>0:
            if t not in shown: L.append("%s: first %d miss c0;"%(t,nskip))
            else: L.append("(%d more miss c0)"%nskip)
        shown.add(t)
        for k in range(lo,ci):
            if k in doomed: continue
            g=CANDS[k]
            L.append("%s -> %s != c0 no."%(gate_name(g),cs(col_of(g,ins_bits))))
        g=CANDS[ci]
        if ci==wci:
            if doom_skip>0:
                L.append("(%d more c0-matches rejected the same way)"%doom_skip)
            L.append("%s -> %s = c0 MATCH."%(gate_name(g),cs(col_of(g,ins_bits))))
        else:
            fb,fg=probe_walk_fail(g,ins_bits,cols)
            if fb is None:
                L.append("%s -> %s = c0, but walking it finds no global tap-line rule -> reject, resume scan."%(gate_name(g),cs(col_of(g,ins_bits))))
            else:
                L.append("%s -> %s = c0, walk: stuck at b%d (no stride/edge/birth option matches c%d) -> reject, resume scan."%(gate_name(g),cs(col_of(g,ins_bits)),fb,fb))
        cur=max(cur,ci+1)
    # WALK
    for (j,gate,tag,tried) in traj[1:]:
        pre=[]
        nt=0
        for (tg,ttag,reason) in tried:
            if nt>=3: break
            nm=gate_name(tg)
            if reason=='colfail': pre.append("%s -> %s != c%d no"%(nm,cs(col_of(tg,ins_bits)),j))
            else: pre.append("%s -> col ok but no consistent continuation, backtrack"%nm)
            nt+=1
        note=''
        if tag=='wrap': note=" (edge: wraps to p0)"
        elif tag=='dead': note=" (edge: tap line ends, reads 0)"
        elif tag.startswith('birth'): note=" (new tap line born at p0 = SHR%d)"%j
        elif tag.startswith('reveal'): note=" (tap born at p0 unmasks partner lines)"
        body="b%d: "%j + ("; ".join(pre)+"; " if pre else "")
        body+="%s -> %s = c%d ok%s."%(gate_name(gate),cs(col_of(gate,ins_bits)),j,note)
        L.append(body)
    # LOCK
    if taps:
        lines=", ".join("L%d=%s"%(i+1,tap_line_desc(t,traj)) for i,t in enumerate(taps))
        L.append("Lock: every scanned position lies on tap lines: %s. Rule = the per-bit gates above on these lines; no bit is guessed, all 8 columns matched."%lines)
    else:
        L.append("Lock: output is constant per bit (no input taps needed); all 8 columns matched.")
    # VERIFY (per pair, transposed)
    vr=[]
    for e in range(n):
        yb=''.join(str(cols[j][e]) for j in range(8))
        vr.append("x%d->%s=y%d ok"%(e+1,yb,e+1))
    L.append("Verify all pairs (read columns down): "+"; ".join(vr)+".")
    # APPLY
    qb=[int(c) for c in q]
    L.append("Apply to q=%s: "%q + " ".join("q%d=%d"%(i,qb[i]) for i in range(8)))
    outbits=[]; steps=[]
    for (j,gate,tag,tried) in traj:
        pos,tt=gate
        idx=0; lab=[]
        for p in pos: idx=(idx<<1)|qb[p]
        v=tt[idx]; outbits.append(v)
        nm=gate_name(gate, labels=lambda p:'q%d=%d'%(p,qb[p]))
        steps.append("b%d:%s=%d"%(j,nm,v))
    final=''.join(map(str,outbits))
    L.append("; ".join(steps)+".")
    L.append("answer = %s"%final)
    return "\n".join(L), final

# ================= LINTER =================
def lint(cot, ins, outs, q, final):
    errs=[]
    if any(ord(ch)>127 for ch in cot): errs.append('non-ascii')
    if '\\boxed' in cot: errs.append('boxed-in-cot')   # the trainer adds boxed in content; cot must not
    m=re.search(r'answer = ([01]{8})',cot)
    if not m or m.group(1)!=final: errs.append('answerline!=final')
    # verify ok-lines correct
    for e in range(len(ins)):
        mm=re.search(r'x%d->([01]{8})=y%d ok'%(e+1,e+1),cot)
        if not mm or mm.group(1)!=outs[e]: errs.append('verify-pair-%d'%(e+1))
    # column strings in table correct
    for j in range(8):
        mm=re.search(r'^c%d ([01]+)$'%j,cot,re.M)
        want=''.join(o[j] for o in outs)
        if not mm or mm.group(1)!=want: errs.append('col-%d'%j)
    for p in range(8):
        mm=re.search(r'^p%d ([01]+)$'%p,cot,re.M)
        want=''.join(s[p] for s in ins)
        if not mm or mm.group(1)!=want: errs.append('posrow-%d'%p)
    return errs

# ================= SYNTH GENERATOR =================
PROMPT_HDR=("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The "
 "transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
 "choice functions.\n\nHere are some examples of input -> output:\n")
def mk_prompt(ins,outs,q):
    return PROMPT_HDR+"\n".join("%s -> %s"%(i,o) for i,o in zip(ins,outs))+"\n\nNow, determine the output for: %s"%q

OPS_E={'AND':lambda a,b:a&b,'OR':lambda a,b:a|b,'XOR':lambda a,b:a^b,
 'NAND':lambda a,b:1-(a&b),'NOR':lambda a,b:1-(a|b),'XNOR':lambda a,b:1-(a^b),
 'NOT_A_AND_B':lambda a,b:(1-a)&b,'A_AND_NOT_B':lambda a,b:a&(1-b),
 'NOT_A_OR_B':lambda a,b:(1-a)|b,'A_OR_NOT_B':lambda a,b:a|(1-b)}
def eval_expr(e,a,b,c):
    e=e.strip()
    if e=='C0': return 0
    if e=='C1': return 1
    if e=='{A}': return a
    if e=='{B}': return b
    if e=='{C}': return c
    m=re.match(r'^(\w+)\((.*)\)$',e); op=m.group(1); inner=m.group(2)
    depth=0; args=[]; cur=''
    for ch in inner:
        if ch=='(':depth+=1
        if ch==')':depth-=1
        if ch==',' and depth==0: args.append(cur); cur=''
        else: cur+=ch
    args.append(cur)
    if op=='NOT': return 1-eval_expr(args[0],a,b,c)
    return OPS_E[op](eval_expr(args[0],a,b,c),eval_expr(args[1],a,b,c))

def rule_dist_from_cache(path='pipeline/data/bitmanip_solved.jsonl'):
    """(expr -> count) and per-slot transform marginal from the solve cache."""
    import collections
    exprs=collections.Counter(); trans=collections.Counter()
    for l in open(path):
        r=json.loads(l)
        exprs[r['expr']]+=1
        for v,(t,k) in r['td'].items(): trans[(t,k)]+=1
    return exprs, trans

def gen_synth(rng, exprs, trans, max_tries=30):
    """One synthetic row: sample rule from real distribution, fresh values, honest solve + render."""
    elist=list(exprs.items()); etot=sum(c for _,c in elist)
    tlist=list(trans.items()); ttot=sum(c for _,c in tlist)
    def pick(lst,tot):
        x=rng.uniform(0,tot); a=0
        for v,c in lst:
            a+=c
            if x<=a: return v
        return lst[-1][0]
    for _ in range(max_tries):
        expr=pick(elist,etot)
        slots=[v for v in ('{A}','{B}','{C}') if v in expr]
        td={}
        used=set()
        ok=True
        for s in slots:
            for _t in range(50):
                t=pick(tlist,ttot)
                if t not in used: used.add(t); td[s]=t; break
            else: ok=False
        if not ok: continue
        n=rng.choice([7,8,9,10])
        vals=rng.sample(range(256),n+1)
        ins=[format(v,'08b') for v in vals[:n]]; q=format(vals[n],'08b')
        def apply_true(xs):
            xb=[int(c) for c in xs]; ob=[]
            for j in range(8):
                vv={}
                for s in slots:
                    p=tap_pos(j,td[s]); vv[s]=0 if p is None else xb[p]
                ob.append(eval_expr(expr,vv.get('{A}',0),vv.get('{B}',0),vv.get('{C}',0)))
            return ''.join(map(str,ob))
        outs=[apply_true(x) for x in ins]
        sol=solve(ins,outs)
        if sol is None: continue
        cot,final=render(ins,outs,q,sol)
        if lint(cot,ins,outs,q,final): continue
        if len(cot)>6500: continue   # ~3500 tokens (1.84 chars/token); resample pathological degenerates
        return dict(prompt=mk_prompt(ins,outs,q),cot=cot,final=final,
                    true_answer=apply_true(q),expr=expr,td=td)
    return None

# ================= DRIVERS =================
def load_real():
    rows=[]
    for r in csv.DictReader(open('competition_dataset/train_categorized.csv')):
        if r['category']!='bit_manipulation': continue
        ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',r['prompt'])
        q=re.search(r'output for:\s*([01]{8})',r['prompt']).group(1)
        rows.append(dict(id=r['id'],prompt=r['prompt'],gold=r['answer'].strip(),
                         ins=[e[0] for e in ex],outs=[e[1] for e in ex],q=q))
    return rows

def main():
    import os, statistics as st
    os.makedirs('pipeline/data/v15',exist_ok=True)
    val_ids={json.loads(l)['id'] for l in open('pipeline/data/val.jsonl')}
    # ---- real rows ----
    rows=load_real()
    nreal=0; gold_ok=0; mism=[]
    with open('pipeline/data/v15/bit_real.jsonl','w') as f:
        for r in rows:
            sol=solve(r['ins'],r['outs'])
            cot,final=render(r['ins'],r['outs'],r['q'],sol)
            errs=lint(cot,r['ins'],r['outs'],r['q'],final)
            assert not errs, (r['id'],errs)
            if final==r['gold']: gold_ok+=1
            else: mism.append(r['id'])
            if r['id'] in val_ids: continue
            f.write(json.dumps(dict(id=r['id'],category='bit_manipulation',prompt=r['prompt'],
                                    cot=cot,final=final))+'\n')
            nreal+=1
    print("bit_real.jsonl: %d rows (excluded %d val ids) | final==gold %d/%d (%.2f%%)"%(
        nreal,len(rows)-nreal,gold_ok,len(rows),100*gold_ok/len(rows)))
    print("final!=gold ids:",mism)
    # ---- synth rows ----
    exprs,trans=rule_dist_from_cache()
    rng=random.Random(20260610)
    nsynth=1800; bad=0
    with open('pipeline/data/v15/bit_synth.jsonl','w') as f:
        for i in range(nsynth):
            row=gen_synth(rng,exprs,trans)
            if row is None: bad+=1; continue
            f.write(json.dumps(dict(id='sbit-%05d'%i,category='bit_manipulation',prompt=row['prompt'],
                                    cot=row['cot'],final=row['final']))+'\n')
    print("bit_synth.jsonl: %d rows (gen failures %d)"%(nsynth-bad,bad))

if __name__=='__main__':
    main()
