"""bit_manipulation solver — GLOBAL single-rule search (port of MAJ0RT0M's replication of the 99.94% method).
One boolean function over up to 3 variables {A},{B},{C}; each variable = a transformed copy of the input
(rot k / shl k / shr k, zero-fill at shift boundaries). SAME (function, transform-assignment) for all 8 bits.
Grammar built by composition (NOT + 10 two-input ops) up to depth 3. Backtracking with fail-fast on bit 0."""
import re, time, itertools
OPS = {
 'AND': lambda a,b:a&b, 'OR': lambda a,b:a|b, 'XOR': lambda a,b:a^b,
 'NAND':lambda a,b:~(a&b),'NOR':lambda a,b:~(a|b),'XNOR':lambda a,b:~(a^b),
 'NOT_A_AND_B':lambda a,b:(~a)&b,'A_AND_NOT_B':lambda a,b:a&(~b),
 'NOT_A_OR_B':lambda a,b:(~a)|b,'A_OR_NOT_B':lambda a,b:a|(~b),
}
TRANSFORMATIONS=[('rot',0)]
for _k in range(1,8): TRANSFORMATIONS += [('rot',_k),('shl',_k),('shr',_k)]

def get_source_bit(in_bits,out_idx,trans):
    t,sv=trans
    if t=='rot': return in_bits[(out_idx+sv)%8]
    if t=='shl':
        s=out_idx+sv; return in_bits[s] if 0<=s<8 else 0
    s=out_idx-sv; return in_bits[s] if 0<=s<8 else 0     # shr

def get_used_vars(expr):
    return [v for v in ('{A}','{B}','{C}') if v in expr]

def evaluate_bit(ev,td,bi,ins,outs,n):
    for ex in range(n):
        ib=ins[ex]; exp=outs[ex][bi]
        a=get_source_bit(ib,bi,td.get('{A}',('rot',0)))
        b=get_source_bit(ib,bi,td.get('{B}',('rot',0)))
        c=get_source_bit(ib,bi,td.get('{C}',('rot',0)))
        if ev(a,b,c,1)!=exp: return False
    return True

def generate_grammar():
    mask=255
    l0={0:("C0",lambda a,b,c,m:0),255:("C1",lambda a,b,c,m:m),
        0b11110000:("{A}",lambda a,b,c,m:a),0b11001100:("{B}",lambda a,b,c,m:b),
        0b10101010:("{C}",lambda a,b,c,m:c)}
    visited=set(l0); levels=[l0]
    for tt,(e,f) in l0.items(): yield tt,e,f
    for depth in range(1,4):
        nxt={}
        for v,(e,f) in levels[-1].items():
            nv=(~v)&mask
            if nv not in visited:
                ne=f"NOT({e})"; nf=lambda a,b,c,m,ff=f:(~ff(a,b,c,m))&m
                visited.add(nv); nxt[nv]=(ne,nf); yield nv,ne,nf
        for i in range(depth):
            j=depth-1
            for v1,(e1,f1) in levels[i].items():
                for v2,(e2,f2) in levels[j].items():
                    for on,of in OPS.items():
                        if i==j and v1>v2 and on in ('AND','OR','XOR','NAND','NOR','XNOR'): continue
                        val=of(v1,v2)&mask
                        if val not in visited:
                            ne=f"{on}({e1}, {e2})"; nf=lambda a,b,c,m,a1=f1,a2=f2,op=of:op(a1(a,b,c,m),a2(a,b,c,m))&m
                            visited.add(val); nxt[val]=(ne,nf); yield val,ne,nf
                        if i!=j:
                            val2=of(v2,v1)&mask
                            if val2 not in visited:
                                ne=f"{on}({e2}, {e1})"; nf=lambda a,b,c,m,a1=f1,a2=f2,op=of:op(a2(a,b,c,m),a1(a,b,c,m))&m
                                visited.add(val2); nxt[val2]=(ne,nf); yield val2,ne,nf
        levels.append(nxt)

def solve(prompt, time_limit=5.0):
    ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',prompt)
    q=re.search(r'output for:\s*([01]{8})',prompt)
    if not ex or not q: return None
    n=len(ex)
    ins=[[int(ex[i][0][j]) for j in range(8)] for i in range(n)]
    outs=[[int(ex[i][1][j]) for j in range(8)] for i in range(n)]
    qin=[int(b) for b in q.group(1)]
    t0=time.time()
    for tt,expr,ev in generate_grammar():
        if time.time()-t0>time_limit: return None
        used=get_used_vars(expr)
        if len(used)==0: combos=[{}]
        elif len(used)==1: combos=({used[0]:t1} for t1 in TRANSFORMATIONS)
        elif len(used)==2: combos=({used[0]:t1,used[1]:t2} for t1 in TRANSFORMATIONS for t2 in TRANSFORMATIONS if t1!=t2)
        else: combos=({used[0]:t1,used[1]:t2,used[2]:t3} for t1,t2,t3 in itertools.permutations(TRANSFORMATIONS,3))
        for td in combos:
            if time.time()-t0>time_limit: return None
            if not evaluate_bit(ev,td,0,ins,outs,n): continue
            if all(evaluate_bit(ev,td,b,ins,outs,n) for b in range(1,8)):
                res=[str(ev(get_source_bit(qin,bi,td.get('{A}',('rot',0))),
                            get_source_bit(qin,bi,td.get('{B}',('rot',0))),
                            get_source_bit(qin,bi,td.get('{C}',('rot',0))),1)) for bi in range(8)]
                return "".join(res), expr, td
    return None

if __name__=="__main__":
    import csv,sys
    rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
    n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
    tl=float(sys.argv[2]) if len(sys.argv)>2 else 5.0
    rows=rows[:n]; ok=found=miss=wrong=0; t0=time.time()
    for idx,r in enumerate(rows):
        res=solve(r['prompt'],tl)
        if res is None: miss+=1
        else:
            found+=1
            if res[0]==r['answer'].strip(): ok+=1
            else: wrong+=1
        if (idx+1)%200==0: print(f"  ..{idx+1}/{len(rows)} ok={ok} nofit={miss} wrong={wrong} ({time.time()-t0:.0f}s)",flush=True)
    print(f"GLOBAL solver: correct {ok} ({100*ok/len(rows):.2f}%) | found {found} | NOFIT {miss} | WRONG {wrong} | n={len(rows)} | {time.time()-t0:.0f}s")
