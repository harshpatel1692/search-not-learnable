"""Synthetic equation_numeric CoT (SYMBOL-DIGIT: digits VISIBLE, no cipher). Zero leakage: random operands/ops,
different from train. The model-reproducible CoT shows the SMALL search the base can actually do in-CoT:
hypothesize PAIRING (operands fwd/rev) + OUTPUT-FORMAT, VERIFY on an example (target-verification, no teleport),
reject a wrong guess, lock the rule, apply to the query, box. ASCII-only (ops as words, ->, no unicode).
Mirrors the real generator (AB[op]CD, operator=arbitrary symbol at index 2; (pairing,fmt) global, op per-operator).
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from solvers import equation_numeric as E

H="In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"
OPSYM=list("/|\\{}<>?@$%&*#~^!")              # arbitrary operator symbols (NOT restricted to + - *)
# op -> (english phrase, infix used in the CoT arithmetic line)
OPTXT={'add':('add','+'),'sub_signed':('subtract (left-right)','-'),'rsub_signed':('subtract (right-left)','-'),
 'absdiff':('absolute difference','absdiff'),'mul':('multiply','*'),'add_p1':('add then +1','+'),
 'add_m1':('add then -1','+'),'mul_p1':('multiply then +1','*'),'mul_m1':('multiply then -1','*'),
 'mod':('left mod right','mod'),'absdiff_p1':('absolute difference then +1','absdiff')}
# weighted op pool (common first; reflects the discovered rr+rev distribution)
OP_POOL=(['add']*7+['mul']*4+['sub_signed']*3+['absdiff']*3+['rsub_signed']*1
         +['add_m1','add_p1','mul_m1','mul_p1','absdiff_p1','mod'])
PAIR_POOL=['rr']*6+['ff']*5+['rf']*1+['fr']*1
FMT_POOL=['rev']*6+['id']*5+['abs']*2+['revopsuf']*1
PAIR_TXT={'ff':'read both operands as written','rr':'reverse the digits of BOTH operands',
          'rf':'reverse only the left operand','fr':'reverse only the right operand'}
FMT_TXT={'id':'write the result as is','abs':'write the magnitude (drop any sign)',
         'rev':'reverse the digits of the result','revopsuf':'reverse the result digits then append the operator',
         'oppre':'prepend the operator to the result'}

def _L(a,b,opch): return f"{a:02d}{opch}{b:02d}"
def _ok(op,pr,fmt,a,b,opch,want):
    v=E.OPS[op](*E._pair(_L(a,b,opch),pr)); return E._render(v,fmt,opch)==want

def gen(rng):
    pairing=rng.choice(PAIR_POOL); fmt=rng.choice(FMT_POOL)
    nops=rng.choice([2,2,3]); syms=rng.sample(OPSYM,nops); qop=syms[0]
    opmap={}
    for s in syms:
        # pick an op whose render is well-defined (nonneg unless signed/absdiff) on random operands
        for _ in range(40):
            op=rng.choice(OP_POOL)
            a,b=rng.randint(0,99),rng.randint(0,99)
            if E._render(E.OPS[op](*E._pair(_L(a,b,s),pairing)),fmt,s) is not None: opmap[s]=op; break
        opmap.setdefault(s,'add')
    # build example lines: each operator appears >=1; query op appears >=2 (so it is deducible)
    lines=[]; want_per={}
    def mk(opch):
        op=opmap[opch]
        for _ in range(60):
            a,b=rng.randint(0,99),rng.randint(0,99)
            r=E._render(E.OPS[op](*E._pair(_L(a,b,opch),pairing)),fmt,opch)
            if r is not None and r!='': return a,b,r
        return None
    plan=[qop,qop]+[s for s in syms[1:]]+[rng.choice(syms)]
    rng.shuffle(plan)
    exrows=[]
    for opch in plan:
        m=mk(opch)
        if not m: continue
        a,b,r=m; exrows.append((opch,a,b,r)); lines.append(f"{_L(a,b,opch)} = {r}")
    # query
    qm=mk(qop)
    if qm is None: return gen(rng)
    qa,qb,qans=qm
    prompt=H+"\n"+"\n".join(lines)+f"\nNow, determine the result for: {_L(qa,qb,qop)}"
    # ---- CoT ----
    op=opmap[qop]; phrase,infix=OPTXT.get(op,(op,'?'))
    qexs=[(a,b,r) for (s,a,b,r) in exrows if s==qop][:2]
    def paired(a,b):
        L=_L(a,b,qop); x,y=E._pair(L,pairing); return x,y
    c=[]
    c.append(f"Operator symbol '{qop}' is at index 2; the operands are the two-digit numbers around it.")
    # show a WRONG simple guess first (plain ff+id) if that is not the real rule -> teaches verify/reject
    if not (pairing=='ff' and fmt=='id') and qexs:
        a,b,r=qexs[0]
        plainv=E.OPS[op](a,b); plain=E._render(plainv,'id',qop)
        if plain!=r:
            c.append(f"Guess: just {phrase} as written. {a} {infix} {b} -> {plain}, but example says {r}. Reject.")
    # state + verify the real rule on the example(s)
    c.append(f"Try: {PAIR_TXT[pairing]}, {phrase}, then {FMT_TXT[fmt]}.")
    for a,b,r in qexs:
        x,y=paired(a,b); raw=E.OPS[op](x,y)
        step=f"  {_L(a,b,qop)}: {PAIR_TXT[pairing].split()[0]} -> {x} {infix} {y}"
        step+=f" = {raw} -> render -> {E._render(raw,fmt,qop)} (given {r}) ok"
        c.append(step)
    c.append(f"Rule confirmed: pairing={pairing}, op={op}, format={fmt}.")
    x,y=paired(qa,qb); raw=E.OPS[op](x,y)
    c.append(f"Apply to {_L(qa,qb,qop)}: -> {x} {infix} {y} = {raw} -> render -> {qans}.")
    cot="\n".join(c)
    return {"category":"equation_numeric","prompt":prompt,"answer":qans,"final":qans,"cot":cot}

if __name__=="__main__":
    import random
    rng=random.Random(1)
    okc=0; lens=[]
    for i in range(300):
        r=gen(rng)
        # self-check: the rendered answer must be reproducible by the solver under gold
        s=E.solve(r['prompt'], gold=r['answer'])
        okc+= (s is not None and s[0]==r['answer']); lens.append(len(r['cot']))
        if i<3:
            print("="*70); print(r['prompt']); print("--- CoT ---"); print(r['cot']); print("ANSWER:",r['answer'])
    import statistics as st
    print(f"\nself-consistent (solver re-derives w/ gold): {okc}/300 | median CoT chars {int(st.median(lens))}")
