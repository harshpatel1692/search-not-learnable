#!/usr/bin/env python3
"""crypt_mechanical_render.py -- user-provided forward no-guess CoT renderer."""
from itertools import product
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple

@dataclass(frozen=True)
class Op:
    glyph: str; name: str; fn: Callable[[int,int],int]; is_mul: bool=False; signed: bool=False

_OP_LIB: Dict[str, Callable[[str],Op]] = {
    "add":        lambda g: Op(g,"a+b",   lambda a,b:a+b),
    "add_p1":     lambda g: Op(g,"a+b+1", lambda a,b:a+b+1),
    "add_m1":     lambda g: Op(g,"a+b-1", lambda a,b:a+b-1),
    "mul":        lambda g: Op(g,"a*b",   lambda a,b:a*b, is_mul=True),
    "mul_p1":     lambda g: Op(g,"a*b+1", lambda a,b:a*b+1, is_mul=True),
    "mul_m1":     lambda g: Op(g,"a*b-1", lambda a,b:a*b-1, is_mul=True),
    "absdiff":    lambda g: Op(g,"|a-b|", lambda a,b:abs(a-b)),
    "neg_absdiff":lambda g: Op(g,"-|a-b|",lambda a,b:-abs(a-b), signed=True),
    "sub":        lambda g: Op(g,"a-b",   lambda a,b:a-b, signed=True),
}
def make_op(glyph,name): return _OP_LIB[name](glyph)

@dataclass
class Equation:
    a:str; op:Op; b:str; rhs:str
def parse_eq(lhs,rhs,ops): return Equation(lhs[0:2],ops[lhs[2]],lhs[3:5],rhs)
def split_sign(eq):
    r=eq.rhs
    if eq.op.signed and r and r[0]==eq.op.glyph: return True,r[1:]
    return False,r
def eq_symbols(eq): return list(eq.a)+list(eq.b)+list(split_sign(eq)[1])
def read_num(glyphs,order,m):
    ds=[m[s] for s in glyphs]
    if order=="reversed": ds=ds[::-1]
    return ds[0]*10+ds[1]
def encode_answer(val,order,inv,sign_glyph):
    neg=val<0; mag=abs(val); digits=[int(c) for c in str(mag)]
    written=digits[::-1] if order=="reversed" else digits
    try: glyphs="".join(inv[d] for d in written)
    except KeyError: return None,"free query digit"
    if neg and sign_glyph: glyphs=sign_glyph+glyphs
    return glyphs,"ok"

@dataclass
class State:
    syms:List[str]; order:str; dom:Dict[str,set]=field(default_factory=dict)
    log:List[Tuple[str,int,str,Optional[int]]]=field(default_factory=list); logged:set=field(default_factory=set)
    def solved(self): return all(len(self.dom[s])==1 for s in self.syms)
    def pinned_map(self): return {s:next(iter(self.dom[s])) for s in self.syms if len(self.dom[s])==1}
    def record_new_pins(self,routine,eqi):
        for s in self.syms:
            if s not in self.logged and len(self.dom[s])==1:
                self.log.append((s,next(iter(self.dom[s])),routine,eqi)); self.logged.add(s)
def seed(state,eqs,query):
    for s in state.syms: state.dom[s]=set(range(10))
    for eq in [*eqs,query]:
        state.dom[eq.a[0]].discard(0); state.dom[eq.b[0]].discard(0)
        mag=split_sign(eq)[1]
        if len(mag)>=2: state.dom[mag[0]].discard(0)
def rule_alldiff(state):
    changed=False; pv={next(iter(state.dom[s])) for s in state.syms if len(state.dom[s])==1}
    for s in state.syms:
        if len(state.dom[s])>1:
            nd=state.dom[s]-pv
            if nd!=state.dom[s]: state.dom[s]=nd; changed=True
    for v in range(10):
        homes=[s for s in state.syms if v in state.dom[s]]
        if len(homes)==1 and len(state.dom[homes[0]])>1: state.dom[homes[0]]={v}; changed=True
    if changed: state.record_new_pins("alldiff",None)
    return changed
def rule_gac(state,eqs,max_free=5):
    changed=False
    for ei,eq in enumerate(eqs):
        base=state.pinned_map(); neg_exp,mag_syms=split_sign(eq)
        local=list(dict.fromkeys(eq_symbols(eq))); free=[s for s in local if len(state.dom[s])>1]
        if not free or len(free)>max_free: continue
        feas={s:set() for s in free}; doms=[sorted(state.dom[s]) for s in free]
        for combo in product(*doms):
            m=dict(base); m.update(dict(zip(free,combo))); vals=list(m.values())
            if len(set(vals))!=len(vals): continue
            a=read_num(eq.a,state.order,m); b=read_num(eq.b,state.order,m); val=eq.op.fn(a,b)
            if (val<0)!=neg_exp: continue
            digits=[int(c) for c in str(abs(val))]; written=digits[::-1] if state.order=="reversed" else digits
            if len(written)!=len(mag_syms): continue
            if any(m[mag_syms[i]]!=written[i] for i in range(len(mag_syms))): continue
            for s in free: feas[s].add(m[s])
        for s in free:
            nd=feas[s]&state.dom[s]
            if nd and nd!=state.dom[s]: state.dom[s]=nd; changed=True; state.record_new_pins("gac",ei)
    return changed
def propagate(state,eqs):
    while True:
        c=rule_alldiff(state); c=rule_gac(state,eqs) or c; c=rule_alldiff(state) or c
        if not c: break
def explain(s,v,routine,eqi,eqs,m,order):
    sp=lambda t:" ".join(t)
    if routine=="alldiff": return f"digit {v} is unused and only '{sp(s)}' can take it -> '{sp(s)}'={v}"
    if routine=="gac" and eqi is not None:
        eq=eqs[eqi]; a=read_num(eq.a,order,m); b=read_num(eq.b,order,m)
        ui=0 if order=="reversed" else 1; ti=1-ui
        a0,b0,a1,b1=m[eq.a[ui]],m[eq.b[ui]],m[eq.a[ti]],m[eq.b[ti]]
        Pn=abs(eq.op.fn(a,b)); r0=Pn%10; mag=split_sign(eq)[1]
        is_au=(s==eq.a[ui]); is_bu=(s==eq.b[ui]); is_at=(s==eq.a[ti]); is_bt=(s==eq.b[ti])
        if eq.op.name in ("a+b","a+b-1","a+b+1"):
            off={'a+b':0,'a+b+1':1,'a+b-1':-1}[eq.op.name]
            if is_au or is_bu:
                return f"EQ{eqi+1} ({eq.op.name}): result units {r0}; {a0}+{b0}{off:+d} ends in {r0} -> '{sp(s)}'={v}"
            return f"EQ{eqi+1} ({eq.op.name}): tens column {a1}+{b1}+carry matches the result tens -> '{sp(s)}'={v}"
        if eq.op.is_mul:
            hi=Pn//100
            if is_bt:
                return (f"EQ{eqi+1} (a*b): product P={Pn}, floor(P/100)={hi}. Single-digit recall: which b1 give "
                        f"{a1}*b1 <= {hi} < {a1+1}*(b1+1)? -> '{sp(s)}'={v}")
            if is_at:
                return (f"EQ{eqi+1} (a*b): product P={Pn}, floor(P/100)={hi}. Recall: a1*{b1} <= {hi} < "
                        f"(a1+1)*{b1+1} -> '{sp(s)}'={v}")
            if is_bu:
                return f"EQ{eqi+1} (a*b): P={Pn} ends in {r0}; {a0}*b0 ends in {r0} (times table) -> '{sp(s)}'={v}"
            if is_au:
                return f"EQ{eqi+1} (a*b): P={Pn} ends in {r0}; a0*{b0} ends in {r0} (times table) -> '{sp(s)}'={v}"
            try:
                pos=list(mag).index(s); pd=[int(c) for c in str(Pn)]; pd=pd[::-1] if order=='reversed' else pd
                return f"EQ{eqi+1} (a*b): product P={Pn}; its digit at that place is {pd[pos]} -> '{sp(s)}'={v}"
            except Exception:
                return f"EQ{eqi+1} (a*b): product P={Pn} fixes '{sp(s)}'={v}"
        if eq.op.name in ("|a-b|","-|a-b|","a-b"):
            return f"EQ{eqi+1} ({eq.op.name}): {a} {eq.op.name} {b} = {eq.op.fn(a,b)}; column matches -> '{sp(s)}'={v}"
        return f"EQ{eqi+1} ({eq.op.name}): {a} {eq.op.name} {b} admits only '{sp(s)}'={v}"
    return f"forced by consistency -> '{sp(s)}'={v}"
def render(eq_pairs,query_lhs,query_rhs_unused,glyph_to_op,order):
    sp=lambda t:" ".join(t)
    ops={g:make_op(g,name) for g,name in glyph_to_op.items()}
    eqs=[parse_eq(l,r,ops) for (l,r) in eq_pairs]
    q=Equation(query_lhs[0:2],ops[query_lhs[2]],query_lhs[3:5],"")
    syms=sorted({s for eq in eqs for s in eq_symbols(eq)}|set(q.a)|set(q.b))
    state=State(syms=syms,order=order); seed(state,eqs,q); propagate(state,eqs)
    if not state.solved():
        return None,f"STALLED (needs search): {''.join(s for s in syms if len(state.dom[s])!=1)}"
    m=state.pinned_map(); inv={v:k for k,v in m.items()}
    out=[]; op_desc="; ".join(f"'{g}'={ops[g].name}" for g in glyph_to_op)
    out.append(f"Operators (by result length): {op_desc}. Order: {order}. Each symbol is a distinct digit 0-9. Solve forward (units mod-10, magnitude bracket, all-different); no guessing.")
    out.append(""); out.append("Derivation (every digit forced before it is used):")
    for (s,v,routine,eqi) in state.log: out.append(f"  {explain(s,v,routine,eqi,eqs,m,order)}")
    out.append(""); out.append("Map (summary of the above): "+", ".join(f"{sp(s)}={m[s]}" for s in sorted(syms)))
    out.append(""); out.append("Verify all examples:")
    for i,eq in enumerate(eqs):
        a,b=read_num(eq.a,order,m),read_num(eq.b,order,m); val=eq.op.fn(a,b)
        out.append(f"  EQ{i+1}: {a} {eq.op.name} {b} = {val}  ({sp(eq.a)} {eq.op.glyph} {sp(eq.b)} = {sp(eq.rhs)})")
    qa,qb=read_num(q.a,order,m),read_num(q.b,order,m); qval=q.op.fn(qa,qb)
    ans,reason=encode_answer(qval,order,inv,q.op.glyph if q.op.signed else None)
    if ans is None: return None,reason
    out.append(""); out.append("Encode the query:")
    out.append(f"  {sp(query_lhs)}: {qa} {q.op.name} {qb} = {qval}")
    return "\n".join(out)+f"\n\\boxed{{{ans}}}","ok"
