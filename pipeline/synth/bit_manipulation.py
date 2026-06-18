"""Synthetic bit_manipulation generator: sample a rule from the real program family,
emit (prompt, answer, bounded hypothesize->verify->apply CoT) with NEW random values.
Bounded CoT teaches a DECISIVE procedure (base model fails by over-thinking/truncating)."""
import random
M=0xFF
def rotl(x,k): k&=7; return ((x<<k)|(x>>(8-k)))&M if k else x
def rotr(x,k): return rotl(x,8-(k&7))
def shl(x,k): return (x<<k)&M
def shr(x,k): return (x>>k)&M
def notx(x): return (~x)&M
def maj(a,b,c): return (a&b)|(a&c)|(b&c)
def ch(a,b,c): return (a&b)|((~a&M)&c)
def b8(x): return format(x,'08b')

# unary transforms with human-readable names
def _unaries():
    U=[("the input itself", lambda x:x), ("NOT(input)", notx)]
    for k in range(1,8): U.append((f"a left-rotation by {k} (ROTL{k})", lambda x,k=k:rotl(x,k)))
    for k in range(1,8): U.append((f"a left-shift by {k} (SHL{k})", lambda x,k=k:shl(x,k)))
    for k in range(1,8): U.append((f"a right-shift by {k} (SHR{k})", lambda x,k=k:shr(x,k)))
    return U
U=_unaries()

def sample_rule(rng):
    """Return (describe, fn) sampled to roughly match real mix: mostly single-op(+mask), some combine/maj/ch."""
    r=rng.random()
    if r<0.30:   # single unary + optional mask
        name,f=rng.choice(U); op=rng.choice(['none','xor','and','or'])
        if op=='none': return (f"the output is {name}", f)
        mask=rng.randint(0,255)
        if op=='xor': return (f"the output is {name}, then XOR with {b8(mask)}", lambda x,f=f,m=mask:f(x)^m)
        if op=='and': return (f"the output is {name}, then AND with {b8(mask)}", lambda x,f=f,m=mask:f(x)&m)
        return (f"the output is {name}, then OR with {b8(mask)}", lambda x,f=f,m=mask:f(x)|m)
    elif r<0.55:  # combine two unaries
        (n1,f1),(n2,f2)=rng.choice(U),rng.choice(U)
        op=rng.choice([('XOR',lambda a,b:a^b),('AND',lambda a,b:a&b),('OR',lambda a,b:a|b)])
        return (f"the output is ({n1}) {op[0]} ({n2})", lambda x,f1=f1,f2=f2,o=op[1]:o(f1(x),f2(x)))
    else:        # maj/ch of three rot/not args
        sub=[(n,f) for n,f in U if 'rotation' in n or 'itself' in n or 'NOT' in n]
        (n1,f1),(n2,f2),(n3,f3)=rng.choice(sub),rng.choice(sub),rng.choice(sub)
        if rng.random()<0.5:
            return (f"the output is the bitwise MAJORITY of ({n1}), ({n2}), ({n3})", lambda x,f1=f1,f2=f2,f3=f3:maj(f1(x),f2(x),f3(x)))
        return (f"the output is the bitwise CHOICE: where ({n1}) is 1 take ({n2}) else ({n3})", lambda x,f1=f1,f2=f2,f3=f3:ch(f1(x),f2(x),f3(x)))

PROMPT_HDR=("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
            "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
            "and possibly majority or choice functions.\n\nHere are some examples of input -> output:\n")

def _reject_menu(i1, o1, rng):
    """Up to 2 simple primitive guesses that FAIL to reproduce o1 — teaches the model to TRY then rule out."""
    pool=[("input unchanged", i1), ("NOT(input)", notx(i1)),
          ("ROTL1", rotl(i1,1)), ("ROTL2", rotl(i1,2)),
          ("SHL1", shl(i1,1)), ("SHR1", shr(i1,1))]
    wrong=[(lab,val) for lab,val in pool if val!=o1]
    rng.shuffle(wrong)
    return wrong[:2]

def gen(rng):
    desc,fn=sample_rule(rng)
    rule=desc.replace("the output is ","")   # noun phrase for the converge clause
    n=rng.choice([7,8,9,10])
    ins=rng.sample(range(256), n+1)
    ex=[(i, fn(i)) for i in ins[:n]]; q=ins[n]; ans=fn(q)
    prompt=PROMPT_HDR + "\n".join(f"{b8(i)} -> {b8(o)}" for i,o in ex) + f"\n\nNow, determine the output for: {b8(q)}"
    # SEARCH-CoT: try simple transforms on ex1 -> reject the ones that miss -> converge on the rule
    # -> verify on ex2 -> apply. Teaches CONVERGENCE, not memorization of a fixed phrasing.
    (i1,o1),(i2,o2)=ex[0],ex[1]
    rej=_reject_menu(i1,o1,rng)
    rej_txt=("; ".join(f"{lab}? {b8(val)}≠{b8(o1)}, no" for lab,val in rej)+"; ") if rej else ""
    cot=(f"Derive the rule from the first example {b8(i1)} -> {b8(o1)}: test simple transforms. "
         f"{rej_txt}"
         f"try {rule}: {b8(fn(i1))}={b8(o1)} ✓. "
         f"Confirm on {b8(i2)}: {b8(fn(i2))}={b8(o2)} ✓. "
         f"So the rule is {rule}. Apply to {b8(q)}: result = {b8(ans)}.")
    return {"category":"bit_manipulation","prompt":prompt,"answer":b8(ans),"final":b8(ans),"cot":cot}

if __name__=="__main__":
    rng=random.Random(1)
    for _ in range(3):
        r=gen(rng); print("PROMPT:\n"+r['prompt']); print("ANSWER:",r['answer']); print("COT:",r['cot']); print("="*60)
