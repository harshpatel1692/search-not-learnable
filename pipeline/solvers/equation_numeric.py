"""equation_numeric solver (SYMBOL-DIGIT = digits VISIBLE, NO cipher). The generator is AB[op]CD where op is an
arbitrary symbol at index 2; per operator deduce: PAIRING (each 2-digit operand read fwd/rev), OPERATION, and
OUTPUT FORMAT (the result string transform). Because digits are visible there is NO permutation search -- just a
tiny enumerate pairing x op x format, fit to the operator's examples, verify, apply. Instant + near-complete.
Discovered empirically (2026-06-08) over the eq_numeric_deduce NONE set: dominant rule = ('rr','add','rev')
(reverse both operands, add, reverse the result), plus rr+mul/sub/add±1/mul±1/mod variants. Run:
  python3 pipeline/solvers/equation_numeric.py <cat> <n> [gold]
"""
import math

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
 'concat':lambda a,b:a*100+b,'rconcat':lambda a,b:b*100+a,   # juxtapose the two 2-digit operands (very common)
 'mul':lambda a,b:a*b,'gcd':math.gcd,'lcm':_lcm,'fdiv':_fd,'rdiv':_rd,'mod':_mo,'rmod':_rm,
 'min':lambda a,b:min(a,b),'max':lambda a,b:max(a,b),
 'add_m1':_off(lambda a,b:a+b,-1),'add_p1':_off(lambda a,b:a+b,1),
 'add_m2':_off(lambda a,b:a+b,-2),'add_p2':_off(lambda a,b:a+b,2),
 'mul_m1':_off(lambda a,b:a*b,-1),'mul_p1':_off(lambda a,b:a*b,1),
 'mul_m2':_off(lambda a,b:a*b,-2),'mul_p2':_off(lambda a,b:a*b,2),
 'absdiff_m1':_off(lambda a,b:abs(a-b),-1),'absdiff_p1':_off(lambda a,b:abs(a-b),1),
 'sub_p1':_off(_ss,1),'sub_m1':_off(_ss,-1),'rsub_p1':_off(_rs,1),'rsub_m1':_off(_rs,-1),
 'mul_double':lambda a,b:a*b*2,'mul_half':lambda a,b:(a*b)//2,
 'mul_plus_a':lambda a,b:a*b+a,'mul_plus_b':lambda a,b:a*b+b,
 'mul_minus_a':lambda a,b:a*b-a,'mul_minus_b':lambda a,b:a*b-b,
 'a2_plus_b':lambda a,b:a*a+b,'a_plus_b2':lambda a,b:a+b*b,
 'sq_diff':lambda a,b:(a-b)**2,'sq_sum':lambda a,b:(a+b)**2,
 'xor':lambda a,b:a^b,'band':lambda a,b:a&b,'bor':lambda a,b:a|b}
# priority: common ops first -> canonical fit when several combos tie on the examples
OP_PRIORITY=['add','sub_signed','rsub_signed','absdiff','concat','rconcat','mul','gcd','mod','rmod','fdiv','rdiv','min','max','lcm',
 'add_m1','add_p1','mul_m1','mul_p1','absdiff_m1','absdiff_p1','sub_m1','sub_p1','rsub_m1','rsub_p1',
 'add_m2','add_p2','mul_m2','mul_p2','mul_double','mul_half','mul_plus_a','mul_plus_b','mul_minus_a','mul_minus_b',
 'a2_plus_b','a_plus_b2','sq_diff','sq_sum','xor','band','bor']
# GUESS prior: when the query operator is UNSEEN (eq_guess), order by the TRUE marginal of operations measured on
# train eq_guess (sub_signed 25% > add 15% > rsub 11% > mul 9% ...). Best single guess ~ sub_signed (~20% ceiling).
GUESS_PRIORITY=['sub_signed','add','rsub_signed','mul','add_p1','concat','mul_m1','mul_p1','add_m1','rmod','mod',
 'rconcat','absdiff','gcd','fdiv','rdiv','min','max','lcm']+[o for o in OP_PRIORITY if o not in (
 'sub_signed','add','rsub_signed','mul','add_p1','concat','mul_m1','mul_p1','add_m1','rmod','mod','rconcat','absdiff',
 'gcd','fdiv','rdiv','min','max','lcm')]
PAIRINGS=['ff','rr','rf','fr']
FMTS=['id','abs','rev','revopsuf','oppre','opsuf','revoppre','dsum','opneg','zf2','zf3','zf4','zf2rev','zf3rev','zf4rev']

def _dsum(v):
    v=abs(v); s=0
    while v: s+=v%10; v//=10
    return s
def _render(v, fmt, op):
    """render raw value v under output-format fmt (op = operator char). Returns str or None."""
    if v is None: return None
    s=str(abs(v))
    sg='-' if v<0 else ''            # sign, preserved across reversing formats
    if fmt=='id': return str(v)
    if fmt=='abs': return s
    if fmt=='rev': return sg+s[::-1]
    if fmt=='revopsuf': return s[::-1]+op
    if fmt=='opsuf': return s+op
    if fmt=='oppre': return op+s
    if fmt=='revoppre': return op+s[::-1]
    if fmt=='dsum': return str(_dsum(v))
    if fmt=='opneg': return (op+s) if v<0 else str(v)   # negative shown with the operator char instead of '-'
    if fmt=='zf2': return s.zfill(2)
    if fmt=='zf3': return s.zfill(3)
    if fmt=='zf4': return s.zfill(4)
    if fmt=='zf2rev': return sg+s.zfill(2)[::-1]
    if fmt=='zf3rev': return sg+s.zfill(3)[::-1]
    if fmt=='zf4rev': return sg+s.zfill(4)[::-1]
    return None
def _pair(L, pr):
    a0,a1,b0,b1=int(L[0]),int(L[1]),int(L[3]),int(L[4])
    if pr=='ff': return a0*10+a1, b0*10+b1
    if pr=='rr': return a1*10+a0, b1*10+b0
    if pr=='rf': return a1*10+a0, b0*10+b1
    if pr=='fr': return a0*10+a1, b1*10+b0

def parse(prompt):
    if 'examples:\n' in prompt: body=prompt.split('examples:\n',1)[1]
    else: body=prompt
    body,q=body.split('Now, determine the result for:',1)
    byop={}
    for line in body.splitlines():
        if '=' not in line: continue
        L,R=line.split('=',1); L=L.strip(); R=R.strip()
        if len(L)!=5 or not (L[0:2].isdigit() and L[3:5].isdigit()): continue
        byop.setdefault(L[2],[]).append((L,R))
    qL=q.strip().split('\n')[0].strip()
    if len(qL)!=5: return None
    return byop, qL

def _fit_op(rows, pairing, fmt, opch):
    """ops (priority order) consistent with ALL example rows of one operator under (pairing,fmt)."""
    out=[]
    for opn in OP_PRIORITY:
        fn=OPS[opn]; good=True
        for (L,R) in rows:
            if _render(fn(*_pair(L,pairing)),fmt,opch)!=R: good=False; break
        if good: out.append(opn)
    return out

def _n_global(byop, pairing, fmt):
    """how many operators are explained by SOME op under this (pairing,fmt) (0 = combo invalid for the puzzle)."""
    n=0
    for opch,rows in byop.items():
        if not _fit_op(rows,pairing,fmt,opch): return -1
        n+=1
    return n

def solve(prompt, gold=None):
    pr=parse(prompt)
    if pr is None: return None
    byop,qL=pr; qop=qL[2]
    qexs=byop.get(qop,[])
    g=None if gold is None else gold.strip()
    # PASS 1: GLOBAL fit — (pairing,fmt) that explains EVERY operator. Most reliable (shared rule). Prefer
    # combos explaining more operators; within that, OP_PRIORITY / PAIRINGS / FMTS order picks the canonical one.
    cands=[]
    for pairing in PAIRINGS:
        for fmt in FMTS:
            ng=_n_global(byop,pairing,fmt)
            if ng>0: cands.append((ng,pairing,fmt))
    cands.sort(key=lambda t:(-t[0], PAIRINGS.index(t[1]), FMTS.index(t[2])))
    for ng,pairing,fmt in cands:
        qa,qb=_pair(qL,pairing)
        qops=_fit_op(qexs,pairing,fmt,qop) if qexs else GUESS_PRIORITY  # unseen op (guess) -> marginal prior order
        for op in qops:
            ans=_render(OPS[op](qa,qb),fmt,qop)
            if ans is None: continue
            if g is not None and ans!=g: continue
            return ans,{"pairing":pairing,"fmt":fmt,"qop":op,"mode":"global"}
    # PASS 2: fallback — fit (pairing,fmt,op) to the QUERY operator's examples only (looser; gold filters wrong).
    if qexs:
        for pairing in PAIRINGS:
            for fmt in FMTS:
                for op in _fit_op(qexs,pairing,fmt,qop):
                    ans=_render(OPS[op](*_pair(qL,pairing)),fmt,qop)
                    if ans is None: continue
                    if g is not None and ans!=g: continue
                    return ans,{"pairing":pairing,"fmt":fmt,"qop":op,"mode":"qonly"}
    return None

if __name__=="__main__":
    import csv,sys,time
    cat=sys.argv[1] if len(sys.argv)>1 else 'equation_numeric_deduce'
    n=int(sys.argv[2]) if len(sys.argv)>2 else 50
    usegold=(len(sys.argv)>3 and sys.argv[3]=='gold')
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']==cat][:n]
    ok=wrong=none=0; t0=time.time()
    for r in rows:
        try: res=solve(r['prompt'], gold=(r['answer'].strip() if usegold else None))
        except Exception: res=None
        p=res[0] if res else None
        if p is None: none+=1
        elif p==r['answer'].strip(): ok+=1
        else: wrong+=1
    print(f"[eqn] {cat} gold={usegold}: CORRECT {ok} ({100*ok/len(rows):.1f}%) | WRONG {wrong} | NONE {none} | n={len(rows)} {time.time()-t0:.1f}s")
