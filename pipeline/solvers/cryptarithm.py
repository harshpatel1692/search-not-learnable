"""cryptarithm / equation-symbolic solver. Algorithm ported from the AliceEquationSolver Rust accelerator:
ONE digit-DFS over symbol->digit, with inline per-operator candidate pruning (track ALL candidate ops per
operator; prune when any operator has zero surviving candidate for the fully-assigned equations); symbols
ordered most-constrained-first. Operator = char at index 2 (arbitrary per-puzzle symbol). RHS may be SIGN-PREFIXED
(rhs[0]==op => negative; magnitude = rhs[1:]). For DATA-GEN, pass gold (gold-conditioning) -> fast + 0 wrong.
"""
import math, time
def _ss(a,b): return a-b
def _rs(a,b): return b-a
def _fd(a,b): return a//b if b else None
def _rd(a,b): return b//a if a else None
def _mo(a,b): return a%b if b else None
def _rm(a,b): return b%a if a else None
def _lcm(a,b): return a*b//math.gcd(a,b) if (a and b) else 0
def _off(fn,d):
    def g(a,b):
        v=fn(a,b); return None if v is None else v+d
    return g
OPS={'add':lambda a,b:a+b,'sub_signed':_ss,'rsub_signed':_rs,'absdiff':lambda a,b:abs(a-b),
 'neg_absdiff':lambda a,b:abs(a-b),'mul':lambda a,b:a*b,'gcd':math.gcd,'lcm':_lcm,'fdiv':_fd,'rdiv':_rd,
 'mod':_mo,'rmod':_rm,'min':lambda a,b:min(a,b),'max':lambda a,b:max(a,b),
 'add_m1':_off(lambda a,b:a+b,-1),'add_p1':_off(lambda a,b:a+b,1),
 'add_m2':_off(lambda a,b:a+b,-2),'add_p2':_off(lambda a,b:a+b,2),
 'mul_m1':_off(lambda a,b:a*b,-1),'mul_p1':_off(lambda a,b:a*b,1),
 'mul_m2':_off(lambda a,b:a*b,-2),'mul_p2':_off(lambda a,b:a*b,2),
 'absdiff_m1':_off(lambda a,b:abs(a-b),-1),'absdiff_p1':_off(lambda a,b:abs(a-b),1),
 'absdiff_m2':_off(lambda a,b:abs(a-b),-2),'absdiff_p2':_off(lambda a,b:abs(a-b),2),
 'sub_m1':_off(_ss,-1),'sub_p1':_off(_ss,1),'rsub_m1':_off(_rs,-1),'rsub_p1':_off(_rs,1),
 'mul_double':lambda a,b:a*b*2,'mul_half':lambda a,b:(a*b)//2,
 'mul_plus_a':lambda a,b:a*b+a,'mul_plus_b':lambda a,b:a*b+b,
 'mul_minus_a':_off(lambda a,b:a*b-a,0),'mul_minus_b':_off(lambda a,b:a*b-b,0),
 'a2_plus_b':lambda a,b:a*a+b,'a_plus_b2':lambda a,b:a+b*b,
 'sq_diff':lambda a,b:(a-b)**2,'sq_sum':lambda a,b:(a+b)**2,
 'xor':lambda a,b:a^b,'band':lambda a,b:a&b,'bor':lambda a,b:a|b}
SIGNED={'sub_signed','rsub_signed'}
ALWAYS_PREFIX={'neg_absdiff'}

def _units(op, lu, ru, base, hs):
    """SOUND units-digit constraint: allowed set of (result units digit) given operand units lu,ru.
    Uses (X∘Y) mod base = (xu∘yu) mod base. Returns None when not derivable (-> no pruning). hs=sign-prefixed."""
    b=base
    if op=='add': return {(lu+ru)%b}
    if op=='add_p1': return {(lu+ru+1)%b}
    if op=='add_m1': return {(lu+ru-1)%b}
    if op=='add_p2': return {(lu+ru+2)%b}
    if op=='add_m2': return {(lu+ru-2)%b}
    if op=='mul': return {(lu*ru)%b}
    if op=='mul_p1': return {(lu*ru+1)%b}
    if op=='mul_m1': return {(lu*ru-1)%b}
    if op=='mul_p2': return {(lu*ru+2)%b}
    if op=='mul_m2': return {(lu*ru-2)%b}
    if op=='mul_double': return {(lu*ru*2)%b}
    if op=='mul_plus_a': return {(lu*ru+lu)%b}
    if op=='mul_plus_b': return {(lu*ru+ru)%b}
    if op=='mul_minus_a': return {(lu*ru-lu)%b}
    if op=='mul_minus_b': return {(lu*ru-ru)%b}
    if op=='a2_plus_b': return {(lu*lu+ru)%b}
    if op=='a_plus_b2': return {(lu+ru*ru)%b}
    if op=='sq_diff': return {((lu-ru)*(lu-ru))%b}
    if op=='sq_sum': return {((lu+ru)*(lu+ru))%b}
    if op=='sub_signed': return {((ru-lu) if hs else (lu-ru))%b}
    if op=='rsub_signed': return {((lu-ru) if hs else (ru-lu))%b}
    if op in ('sub_p1','sub_m1','rsub_p1','rsub_m1'): return None  # offset+sign subtlety: conservative
    if op in ('absdiff','neg_absdiff'): return {(lu-ru)%b,(ru-lu)%b}
    if op=='absdiff_p1': return {(lu-ru+1)%b,(ru-lu+1)%b}
    if op=='absdiff_m1': return {(lu-ru-1)%b,(ru-lu-1)%b}
    if op=='absdiff_p2': return {(lu-ru+2)%b,(ru-lu+2)%b}
    if op=='absdiff_m2': return {(lu-ru-2)%b,(ru-lu-2)%b}
    return None  # gcd/lcm/fdiv/rdiv/mod/rmod/min/max/mul_half/xor/band/bor/concat: units not derivable
# priority order (simple/common first -> canonical answer & early hit)
PRIORITY=['add','sub_signed','rsub_signed','absdiff','mul','gcd','lcm','neg_absdiff',
          'fdiv','rdiv','mod','rmod','min','max','add_m1','add_p1','mul_m1','mul_p1','absdiff_m1','absdiff_p1',
          'sub_m1','sub_p1','rsub_m1','rsub_p1','add_m2','add_p2','mul_m2','mul_p2','absdiff_m2','absdiff_p2',
          'mul_double','mul_half','mul_plus_a','mul_plus_b','mul_minus_a','mul_minus_b',
          'a2_plus_b','a_plus_b2','sq_diff','sq_sum','xor','band','bor']
# max result length (2-digit operands) for length-pruning
_OPMAX={o:4 for o in ('mul','mul_p1','mul_m1','mul_p2','mul_m2','mul_double','mul_half','lcm',
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

def _matches(opname, eq, perm, rev, base):
    """eq=(l0,l1,r0,r1,res_idx_tuple,has_sign,res_len) with symbol INDICES; perm[idx]=digit."""
    l0,l1,r0,r1,res,hs,rl=eq
    if opname=='concat_fwd': return (not hs) and rl==4 and res==(l0,l1,r0,r1)
    if opname=='concat_rev': return (not hs) and rl==4 and res==(r0,r1,l0,l1)
    L=perm[l0]*base+perm[l1]; R=perm[r0]*base+perm[r1]
    if rev: L=(L%base)*base+L//base; R=(R%base)*base+R//base
    fn=OPS[opname]; v=fn(L,R)
    if v is None: return False
    if opname in SIGNED:
        if (v<0)!=hs: return False
        v=abs(v)
    elif opname in ALWAYS_PREFIX:
        if not hs: return False
        v=abs(v)
    else:
        if hs or v<0: return False
    if v>=base**rl: return False
    digs=[]; x=v
    for _ in range(rl): digs.append(x%base); x//=base
    digs=digs[::-1]                     # MSB-first, zero-padded to rl
    if rev: digs=digs[::-1]
    return all(perm[res[k]]==digs[k] for k in range(rl))

def _opcands(rows, tier_set):
    """rows for one operator: candidate ops filtered by sign-pattern + result-length, within tier_set."""
    signs={hs for *_,hs,_ in rows}; lens={rl for *_,rl in rows}
    cand=[t for t in PRIORITY if t in tier_set]
    if signs=={True}: cand=[t for t in cand if t in ('neg_absdiff','sub_signed','rsub_signed')]
    elif signs=={False}: cand=[t for t in cand if t!='neg_absdiff']
    else: cand=[t for t in cand if t in ('sub_signed','rsub_signed')]
    cand=[t for t in cand if all(_OPMAX.get(t,2)>=L for L in lens)]
    if all((not hs) and rl==4 for *_,hs,rl in rows):
        cand+=[t for t in ('concat_fwd','concat_rev') if t not in cand]
    return cand

def _search(group_keys, groups, n, rev, base, deadline):
    """groups: opchar -> (candidates, list of index-eqs). Single DFS over n symbols; prune via per-group survival."""
    # variable order: UNITS-column symbols first (so the sound units-digit prune fires at shallow depth),
    # then most-constrained-first.
    score=[0]*n; isunit=[False]*n
    for op in group_keys:
        cands,eqs=groups[op]; w=1+len(cands)
        for l0,l1,r0,r1,res,hs,rl in eqs:
            for s in (l0,l1,r0,r1): score[s]+=w
            for s in res: score[s]+=w
            ul=l0 if rev else l1; ur=r0 if rev else r1; ures=res[0] if rev else res[rl-1]
            for s in (ul,ur,ures): isunit[s]=True
    order=sorted(range(n),key=lambda i:(not isunit[i],-score[i],i))
    perm=[0]*n; used=[False]*base; asg=[False]*n; sol=[None]; cnt=[0]
    def group_ok(op):
        cands,eqs=groups[op]
        for c in cands:
            ok=True
            for eq in eqs:
                l0,l1,r0,r1,res,hs,rl=eq
                if asg[l0] and asg[l1] and asg[r0] and asg[r1] and all(asg[s] for s in res):
                    if not _matches(c,eq,perm,rev,base): ok=False;break
                else:   # partial: SOUND units-digit prune (fires once the 3 units symbols are placed)
                    ul=l0 if rev else l1; ur=r0 if rev else r1; ures=res[0] if rev else res[rl-1]
                    if asg[ul] and asg[ur] and asg[ures]:
                        al=_units(c,perm[ul],perm[ur],base,hs)
                        if al is not None and perm[ures] not in al: ok=False;break
            if ok: return True
        return False
    def leaf_ops():
        out={}
        for op in group_keys:
            cands,eqs=groups[op]
            valid=[c for c in cands if all(_matches(c,eq,perm,rev,base) for eq in eqs)]
            if not valid: return None
            out[op]=valid
        return out
    def dfs(depth):
        if sol[0] is not None: return
        cnt[0]+=1
        if cnt[0]&2047==0 and time.time()>deadline: return
        if depth==n:
            ov=leaf_ops()
            if ov is not None: sol[0]=(perm[:],ov)
            return
        si=order[depth]
        for d in range(base):
            if used[d]: continue
            used[d]=True; asg[si]=True; perm[si]=d
            if all(group_ok(op) for op in group_keys): dfs(depth+1)
            used[d]=False; asg[si]=False
            if sol[0] is not None: return
    dfs(0)
    return sol[0]

def solve(prompt, gold=None, deadline_s=3.0):
    pr=parse(prompt)
    if pr is None: return None
    eqs,(qa,qb,qop,qra,qrb)=pr
    opchars={op for _,_,op,_,_,_ in eqs}|{qop}
    work=list(eqs)
    if gold: work=work+[(qa,qb,qop,qra,qrb,gold.strip())]   # gold-conditioning
    content=set()
    for la,lb,op,ra,rb,rhs in work:
        content|={la,lb,ra,rb}; content|=set(rhs[1:] if (len(rhs)>1 and rhs[0]==op) else rhs)
    content|={qa,qb,qra,qrb}
    content=sorted(content-opchars)
    if len(content)>10: return None
    idx={c:i for i,c in enumerate(content)}; n=len(content)
    # group rows per operator (index form)
    rowsby={}
    for la,lb,op,ra,rb,rhs in work:
        hs=len(rhs)>1 and rhs[0]==op; mag=rhs[1:] if hs else rhs
        if any(c not in idx for c in (la,lb,ra,rb)) or any(c not in idx for c in mag): continue
        rowsby.setdefault(op,[]).append((idx[la],idx[lb],idx[ra],idx[rb],tuple(idx[c] for c in mag),hs,len(mag)))
    if not rowsby: return None
    TIER0={'add','sub_signed','rsub_signed','absdiff','neg_absdiff','mul','gcd','lcm','concat_fwd','concat_rev'}
    TIER1=TIER0|{'fdiv','rdiv','mod','rmod','min','max'}
    TIER2=TIER1|{'add_m1','add_p1','mul_m1','mul_p1','absdiff_m1','absdiff_p1'}
    TIERF=set(OPS)|{'concat_fwd','concat_rev'}
    deadline=time.time()+deadline_s
    radixes=[n]+([10] if n!=10 else [])
    # gold-conditioning disambiguates -> use full op set directly (no costly tier exhaustion).
    tier_seq=(TIERF,) if gold else (TIER0,TIER1,TIER2,TIERF)
    for tier in tier_seq:
        for radix in radixes:
            for rev in (False,True):
                if time.time()>deadline: return None
                groups={}
                ok=True
                for op,rows in rowsby.items():
                    c=_opcands(rows,tier)
                    if not c: ok=False;break
                    groups[op]=(c,rows)
                if not ok: continue
                res=_search(list(groups),groups,n,rev,radix,deadline)
                if res is None: continue
                perm,ov=res; inv={perm[i]:content[i] for i in range(n)}
                # apply query op (prefer its valid ops by priority)
                for qt in ov.get(qop,[]):
                    if qt=='concat_fwd': ans=qa+qb+qra+qrb
                    elif qt=='concat_rev': ans=qra+qrb+qa+qb
                    else:
                        L=perm[idx[qa]]*radix+perm[idx[qb]]; R=perm[idx[qra]]*radix+perm[idx[qrb]]
                        if rev: L=(L%radix)*radix+L//radix; R=(R%radix)*radix+R//radix
                        v=OPS[qt](L,R)
                        if v is None: continue
                        pre=''
                        if qt in SIGNED:
                            if v<0: pre=qop; v=-v
                        elif qt in ALWAYS_PREFIX: pre=qop; v=abs(v)
                        elif v<0: continue
                        digs=[]; x=v
                        if v==0: digs=[0]
                        while x: digs.append(x%radix); x//=radix
                        digs=digs[::-1]
                        if rev: digs=digs[::-1]
                        if any(d not in inv for d in digs): continue
                        ans=pre+''.join(inv[d] for d in digs)
                    if gold is not None and ans!=gold.strip(): continue
                    return ans,{"radix":radix,"rev":rev,"ops":{o:vs[0] for o,vs in ov.items()},
                                "qop":qt,"mapping":{content[i]:perm[i] for i in range(n)},"tier":len(tier)}
    return None

if __name__=="__main__":
    import csv,sys
    cat=sys.argv[1] if len(sys.argv)>1 else 'cryptarithm_deduce'
    n=int(sys.argv[2]) if len(sys.argv)>2 else 100
    usegold=(len(sys.argv)>3 and sys.argv[3]=='gold')
    ded=float(sys.argv[4]) if len(sys.argv)>4 else 3.0
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']==cat][:n]
    ok=wrong=none=0; t0=time.time()
    for i,r in enumerate(rows):
        try: res=solve(r['prompt'], gold=(r['answer'].strip() if usegold else None), deadline_s=ded)
        except Exception: res=None
        p=res[0] if res else None
        if p is None: none+=1
        elif p==r['answer'].strip(): ok+=1
        else: wrong+=1
        if (i+1)%50==0: print(f"  ..{i+1}/{len(rows)} ok={ok} wrong={wrong} none={none} ({time.time()-t0:.0f}s)",flush=True)
    print(f"{cat} gold={usegold} ded={ded}s: CORRECT {ok} ({100*ok/len(rows):.1f}%) | WRONG {wrong} | NONE {none} | n={len(rows)} {time.time()-t0:.0f}s")
