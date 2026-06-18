"""z3-backed cryptarithm/equation-symbolic solver. Same model as cryptarithm.py (operator=index2, per-op
operation, digit cipher, radix, reversed-digit mode, sign-prefix) but the DIGIT CSP is solved by z3 (instant) ->
no timeouts. Enumerate the few sign/length-pruned op candidates per operator; z3 solves the bijection per combo.
Run with the WINDOWS venv python (has z3): ./.venv/Scripts/python.exe pipeline/solvers/cryptarithm_z3.py <cat> <n> [gold]
"""
import z3, math, itertools

# z3-expressible ops (gcd/lcm/bitwise excluded -> rare; handled by the Python fallback solver if needed)
def _opexpr(name, L, R):
    if name=='add': return L+R
    if name=='add_p1': return L+R+1
    if name=='add_m1': return L+R-1
    if name=='add_p2': return L+R+2
    if name=='add_m2': return L+R-2
    if name=='sub_signed': return L-R
    if name=='rsub_signed': return R-L
    if name=='absdiff': return z3.If(L>=R, L-R, R-L)
    if name=='neg_absdiff': return z3.If(L>=R, L-R, R-L)
    if name=='absdiff_p1': return z3.If(L>=R, L-R, R-L)+1
    if name=='absdiff_m1': return z3.If(L>=R, L-R, R-L)-1
    if name=='absdiff_p2': return z3.If(L>=R, L-R, R-L)+2
    if name=='absdiff_m2': return z3.If(L>=R, L-R, R-L)-2
    if name=='sub_p1': return L-R+1
    if name=='sub_m1': return L-R-1
    if name=='rsub_p1': return R-L+1
    if name=='rsub_m1': return R-L-1
    if name=='mul': return L*R
    if name=='mul_p1': return L*R+1
    if name=='mul_m1': return L*R-1
    if name=='mul_p2': return L*R+2
    if name=='mul_m2': return L*R-2
    if name=='mul_double': return L*R*2
    if name=='mul_half': return (L*R)/2          # z3 int division
    if name=='mul_plus_a': return L*R+L
    if name=='mul_plus_b': return L*R+R
    if name=='mul_minus_a': return L*R-L
    if name=='mul_minus_b': return L*R-R
    if name=='a2_plus_b': return L*L+R
    if name=='a_plus_b2': return L+R*R
    if name=='sq_diff': return (L-R)*(L-R)
    if name=='sq_sum': return (L+R)*(L+R)
    if name=='fdiv': return L/R
    if name=='rdiv': return R/L
    if name=='mod': return L%R
    if name=='rmod': return R%L
    if name=='min': return z3.If(L<=R,L,R)
    if name=='max': return z3.If(L>=R,L,R)
    return None
# LINEAR ops: no product of two digit-vars -> z3 linear-int = fast & complete. (mul/sq/div/mod = nonlinear -> slow.)
LINEAR={'add','add_p1','add_m1','add_p2','add_m2','sub_signed','rsub_signed','absdiff','neg_absdiff',
 'absdiff_p1','absdiff_m1','absdiff_p2','absdiff_m2','sub_p1','sub_m1','rsub_p1','rsub_m1','min','max'}
SIGNED={'sub_signed','rsub_signed'}
ALWAYS_PREFIX={'neg_absdiff'}
NEEDS_NONZERO_R={'fdiv','mod'}
NEEDS_NONZERO_L={'rdiv','rmod'}
PRIORITY=['add','sub_signed','rsub_signed','absdiff','mul','neg_absdiff','add_m1','add_p1','mul_m1','mul_p1',
 'absdiff_m1','absdiff_p1','sub_m1','sub_p1','rsub_m1','rsub_p1','add_m2','add_p2','mul_m2','mul_p2',
 'absdiff_m2','absdiff_p2','mul_double','mul_half','mul_plus_a','mul_plus_b','mul_minus_a','mul_minus_b',
 'a2_plus_b','a_plus_b2','sq_diff','sq_sum','fdiv','rdiv','mod','rmod','min','max']
_OPMAX={o:4 for o in ('mul','mul_p1','mul_m1','mul_p2','mul_m2','mul_double','mul_half',
 'a2_plus_b','a_plus_b2','mul_plus_a','mul_plus_b','mul_minus_a','mul_minus_b','sq_diff')}
_OPMAX['sq_sum']=5
for o in ('add','add_p1','add_m1','add_p2','add_m2','sub_signed','rsub_signed'): _OPMAX[o]=3

def parse(prompt):
    body=prompt.split('examples:\n',1)[1]; body,q=body.split('\nNow, determine the result for:',1)
    eqs=[]
    for line in body.splitlines():
        if '=' not in line: continue
        L,R=line.split('=',1); L=L.strip(); R=R.strip()
        if len(L)!=5: return None
        eqs.append((L[0],L[1],L[2],L[3],L[4],R))
    qL=q.strip()
    if len(qL)!=5: return None
    return eqs,(qL[0],qL[1],qL[2],qL[3],qL[4])

def _cands(rows, op, tier='full'):
    signs={hs for *_,hs,_ in rows}; lens={rl for *_,rl in rows}
    c=[t for t in PRIORITY]
    if signs=={True}: c=[t for t in c if t in ('neg_absdiff','sub_signed','rsub_signed')]
    elif signs=={False}: c=[t for t in c if t!='neg_absdiff']
    else: c=[t for t in c if t in ('sub_signed','rsub_signed')]
    c=[t for t in c if all(_OPMAX.get(t,2)>=L for L in lens)]
    if tier=='linear': c=[t for t in c if t in LINEAR]
    return c

def _num(dvars, syms, radix, rev):
    ds=[dvars[s] for s in syms]
    if rev: ds=ds[::-1]
    v=0
    for d in ds: v=v*radix+d
    return v

def solve(prompt, gold=None):
    pr=parse(prompt)
    if pr is None: return None
    eqs,(qa,qb,qop,qra,qrb)=pr
    opchars={op for _,_,op,_,_,_ in eqs}|{qop}
    work=list(eqs)
    if gold: work=work+[(qa,qb,qop,qra,qrb,gold.strip())]
    content=set()
    for la,lb,op,ra,rb,rhs in work:
        content|={la,lb,ra,rb}; content|=set(rhs[1:] if (len(rhs)>1 and rhs[0]==op) else rhs)
    content|={qa,qb,qra,qrb}; content=sorted(content-opchars)
    if len(content)>10: return None
    # rows per operator (with sign + magnitude)
    rowsby={}
    for la,lb,op,ra,rb,rhs in work:
        hs=len(rhs)>1 and rhs[0]==op; mag=rhs[1:] if hs else rhs
        if any(c not in content for c in (la,lb,ra,rb)) or any(c not in content for c in mag): continue
        rowsby.setdefault(op,[]).append((la,lb,ra,rb,mag,hs,len(mag)))
    if not rowsby: return None
    radixes=[len(content)]+([10] if len(content)!=10 else [])
    # TIER pass: linear ops first (fast/complete), then add nonlinear (mul/sq/div/mod) only if needed.
    for tier in ('linear','full'):
        for radix in radixes:
            for rev in (False,True):
                opcands={op:_cands(rows,op,tier) for op,rows in rowsby.items()}
                if any(not c for c in opcands.values()): continue
                m,opmap=_z3solve(rowsby,opcands,content,radix,rev,qop)
                if m is None: continue
                ans=_apply(qa,qb,qop,qra,qrb,opmap.get(qop),m,radix,rev,content)
                if ans is None: continue
                if gold is not None and ans!=gold.strip(): continue
                return ans,{"radix":radix,"rev":rev,"ops":opmap,"mapping":m}
    return None

def _rowcons(name, d, la,lb,ra,rb,mag,hs, radix, rev):
    """z3 BoolRef: this row holds under candidate `name` (sign-aware). None if structurally impossible."""
    L=_num(d,(la,lb),radix,rev); R=_num(d,(ra,rb),radix,rev)
    e=_opexpr(name,L,R)
    if e is None: return None
    resv=_num(d,tuple(mag),radix,rev)
    cons=[]
    if name in NEEDS_NONZERO_R: cons.append(R!=0)
    if name in NEEDS_NONZERO_L: cons.append(L!=0)
    if name in SIGNED:
        if hs: cons+=[e==-resv, e<0]
        else: cons+=[e==resv, e>=0]
    elif name in ALWAYS_PREFIX:
        if not hs: return None
        cons.append(e==resv)
    else:
        if hs: return None
        cons+=[e==resv, e>=0]
    return z3.And(cons)

def _z3solve(rowsby, opcands, content, radix, rev, qop):
    """ONE z3 solve: digits + per-operator op selection (disjunction). Returns (mapping, opmap)."""
    s=z3.Solver(); s.set("timeout",3000)
    d={c:z3.Int('d_'+str(i)) for i,c in enumerate(content)}
    for c in content: s.add(d[c]>=0, d[c]<radix)
    s.add(z3.Distinct(list(d.values())))
    sel={}  # op -> z3 Int selecting candidate index
    for op,rows in rowsby.items():
        cands=opcands[op]; si=z3.Int('s_'+op if op.isalnum() else 's_'+str(ord(op))); sel[op]=(si,cands)
        disj=[]
        for k,name in enumerate(cands):
            rc=[_rowcons(name,d,la,lb,ra,rb,mag,hs,radix,rev) for la,lb,ra,rb,mag,hs,rl in rows]
            if any(x is None for x in rc): continue
            disj.append(z3.And(si==k, *rc))
        if not disj: return None,None
        s.add(z3.Or(disj))
    if s.check()!=z3.sat: return None,None
    mod=s.model()
    mp={c:mod[d[c]].as_long() for c in content}
    opmap={op:cands[mod[si].as_long()] for op,(si,cands) in sel.items()}
    return mp,opmap

def _apply(qa,qb,qop,qra,qrb,name,m,radix,rev,content):
    if name is None: return None
    inv={m[c]:c for c in content}
    def val(s2):
        ds=[m[s2[0]],m[s2[1]]]
        if rev: ds=ds[::-1]
        return ds[0]*radix+ds[1]
    L=val((qa,qb)); R=val((qra,qrb))
    import math as _m
    fns={'add':L+R,'sub_signed':L-R,'rsub_signed':R-L,'absdiff':abs(L-R),'neg_absdiff':abs(L-R),
     'mul':L*R,'mul_p1':L*R+1,'mul_m1':L*R-1,'mul_p2':L*R+2,'mul_m2':L*R-2,'add_p1':L+R+1,'add_m1':L+R-1,
     'add_p2':L+R+2,'add_m2':L+R-2,'absdiff_p1':abs(L-R)+1,'absdiff_m1':abs(L-R)-1,'absdiff_p2':abs(L-R)+2,
     'absdiff_m2':abs(L-R)-2,'sub_p1':L-R+1,'sub_m1':L-R-1,'rsub_p1':R-L+1,'rsub_m1':R-L-1,
     'mul_double':L*R*2,'mul_half':(L*R)//2,'mul_plus_a':L*R+L,'mul_plus_b':L*R+R,'mul_minus_a':L*R-L,
     'mul_minus_b':L*R-R,'a2_plus_b':L*L+R,'a_plus_b2':L+R*R,'sq_diff':(L-R)**2,'sq_sum':(L+R)**2,
     'fdiv':(L//R if R else None),'rdiv':(R//L if L else None),'mod':(L%R if R else None),
     'rmod':(R%L if L else None),'min':min(L,R),'max':max(L,R)}
    v=fns.get(name)
    if v is None: return None
    pre=''
    if name in SIGNED:
        if v<0: pre=qop; v=-v
    elif name in ALWAYS_PREFIX: pre=qop; v=abs(v)
    elif v<0: return None
    digs=[]; x=v
    if v==0: digs=[0]
    while x: digs.append(x%radix); x//=radix
    digs=digs[::-1]
    if rev: digs=digs[::-1]
    if any(dd not in inv for dd in digs): return None
    return pre+''.join(inv[dd] for dd in digs)

if __name__=="__main__":
    import csv,sys,time
    cat=sys.argv[1] if len(sys.argv)>1 else 'cryptarithm_deduce'
    n=int(sys.argv[2]) if len(sys.argv)>2 else 60
    usegold=(len(sys.argv)>3 and sys.argv[3]=='gold')
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']==cat][:n]
    ok=wrong=none=0; t0=time.time()
    for i,r in enumerate(rows):
        try: res=solve(r['prompt'], gold=(r['answer'].strip() if usegold else None))
        except Exception as e: res=None
        p=res[0] if res else None
        if p is None: none+=1
        elif p==r['answer'].strip(): ok+=1
        else: wrong+=1
        if (i+1)%50==0: print(f"  ..{i+1}/{len(rows)} ok={ok} wrong={wrong} none={none} ({time.time()-t0:.0f}s)",flush=True)
    print(f"[z3] {cat} gold={usegold}: CORRECT {ok} ({100*ok/len(rows):.1f}%) | WRONG {wrong} | NONE {none} | n={len(rows)} {time.time()-t0:.0f}s")
