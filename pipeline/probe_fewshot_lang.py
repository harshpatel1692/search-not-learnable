"""Hunt for the base's native 'language' for bit_manip. Give 3 FULLY-worked mechanical solves that show the
search (rejections -> hit) in DECIMAL-native shift arithmetic (the base's fluent representation), plus extreme
hints, then test on NEW puzzles. If the base adopts the language and converges -> that's our CoT language."""
import sys,re,json; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_fewshot_lang")
M=0xFF
def b2i(s): return int(s,2)
def i2b(v): return format(v&M,'08b')
def shl(v,k): return (v<<k)&M
def shr(v,k): return v>>k
def rotl(v,k): k&=7; return ((v<<k)|(v>>(8-k)))&M if k else v
def rotr(v,k): return rotl(v,8-(k&7))
def tval(v,t):
    op,k=t
    return {'shl':shl,'shr':shr,'rot':rotl}[op](v,k)
def tname(t):
    op,k=t; return {'shl':'SHL','shr':'SHR','rot':'ROTL'}[op]+str(k) if k else "IDENTITY"
def tdecimal(v,t):
    op,k=t
    if op=='shl': return f"{v}*{2**k} mod 256 = {shl(v,k)}"
    if op=='shr': return f"{v}>>{k} = {shr(v,k)}"
    return f"rotl({v},{k}) = {rotl(v,k)}"
OPF={'XOR':lambda a,b:a^b,'OR':lambda a,b:a|b,'AND':lambda a,b:a&b,'XNOR':lambda a,b:(~(a^b))&M}

def render(prompt, expr, td):
    m=re.match(r'^([A-Z]+)\((\{[AB]\}), *(\{[AB]\})\)$', expr)
    if not m or m.group(1) not in OPF: return None
    op=m.group(1); ta=td[m.group(2)]; tb=td[m.group(3)]
    rows=re.findall(r'([01]{8})\s*->\s*([01]{8})',prompt)
    q=re.search(r'output for:\s*([01]{8})',prompt).group(1)
    x1,y1=rows[0]; X1=b2i(x1); Y1=b2i(y1)
    nones=bin(Y1).count('1')
    fam={'XOR':'about half 1s','OR':'many 1s','AND':'few 1s','XNOR':'about half 1s'}[op]
    A1=tval(X1,ta); B1=tval(X1,tb)
    L=[f"PUZZLE: examples {'; '.join(a+'->'+b for a,b in rows[:4])} ; query {q}",
       f"Step1 op-family: output {y1} has {nones} ones ({fam}) -> try {op}.",
       f"Step2 find the two taps (work in decimal, convert to 8-bit). First example X={X1}({x1}), Y={Y1}({y1})."]
    if op in ('XOR','XNOR'):
        wrong=('shl',1) if ta!=('shl',1) and tb!=('shl',1) else ('shr',1)
        wv=tval(X1,wrong)
        L.append(f"  {op} is invertible: Y {op} tapA should equal tapB. Try tapA={tname(wrong)}: {tdecimal(X1,wrong)}={i2b(wv)}; Y^{tname(wrong)} = {i2b(Y1^wv)} - not a clean shift of X, reject.")
        L.append(f"  Try tapA={tname(ta)}: {tdecimal(X1,ta)}={i2b(A1)}; Y^{tname(ta)} = {i2b(Y1^A1)} = {tname(tb)}({tdecimal(X1,tb)}={i2b(B1)}) MATCH -> rule = {op}({tname(ta)},{tname(tb)}).")
    else:
        L.append(f"  {op} of two shifts. Try {tname(ta)}={i2b(A1)} and {tname(tb)}={i2b(B1)}: {op} = {i2b(OPF[op](A1,B1))} = {y1} MATCH -> rule = {op}({tname(ta)},{tname(tb)}).")
    x2,y2=rows[1]; X2=b2i(x2)
    L.append(f"Step3 verify on X={x2}: {tname(ta)}={i2b(tval(X2,ta))}, {tname(tb)}={i2b(tval(X2,tb))}, {op}={i2b(OPF[op](tval(X2,ta),tval(X2,tb)))} = {y2} OK.")
    Q=b2i(q); ans=i2b(OPF[op](tval(Q,ta),tval(Q,tb)))
    L.append(f"Step4 apply to query {q}: {tname(ta)}={i2b(tval(Q,ta))}, {tname(tb)}={i2b(tval(Q,tb))}, {op} = {ans}. Answer: \\boxed{{{ans}}}")
    return "\n".join(L)

cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in __import__('csv').DictReader(open('competition_dataset/train_categorized.csv'))}
def is2(rec):
    if not rec['expr']: return False
    mm=re.match(r'^([A-Z]+)\(\{[AB]\}, *\{[AB]\}\)$',rec['expr']); return mm and mm.group(1) in OPF and len(rec['td'])==2
cand=[cid for cid,rec in cache.items() if is2(cache[cid])]
# build demos: one XOR, one OR, one AND
def opof(cid): return re.match(r'^([A-Z]+)',cache[cid]['expr']).group(1)
demos=[]
for want in ['XOR','OR','AND']:
    for cid in cand:
        if opof(cid)==want: demos.append(cid); break
demo_ids=set(demos)
demo_txt="\n\n".join(render(rows[cid]['prompt'],cache[cid]['expr'],cache[cid]['td']) for cid in demos)
HINTS=("You solve 'bit manipulation' puzzles. The rule is OP(T1(input),T2(input)) where T1,T2 are SHL/SHR/ROTL by "
 "k and OP in XOR/OR/AND/XNOR. WORK IN DECIMAL: SHLk = v*2^k mod 256, SHRk = v>>k, ROTLk = ((v<<k)|(v>>(8-k)))&255. "
 "EXTREME HINTS: (1) guess OP from the count of 1s in the output (XOR~half, OR~many, AND~few). (2) XOR is "
 "invertible: Y XOR T1 must equal T2, so test taps fast. (3) Be terse: one line per step, no re-derivation, STOP "
 "when verified. Here are 3 fully-worked solutions in this exact style:\n\n"+demo_txt+
 "\n\nNow solve this new puzzle the SAME terse way:\n\n")
test=[c for c in cand if c not in demo_ids][:4]
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
print("demos:",demos,"-> ops",[opof(c) for c in demos]); print("="*50)
for tid in test:
    e=rows[tid]
    r=N.ask(HINTS+e['prompt']+"\n\nGive the 8-bit output in \\boxed{}.", model="nvidia/nemotron-3-nano-30b-a3b",
            max_tokens=7680, temperature=0.0, meta={"id":tid,"rule":cache[tid]['expr']})
    got=norm(r.get('answer','')); gold=e['answer'].strip()
    print(f"{tid} {cache[tid]['expr']:18s} {'CORRECT' if got==gold else 'WRONG'} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
