"""The base ADOPTS the decimal language but drowns in re-derivation. Test: demos that show the FULL TERSE
tap-table -> mechanical invertibility search (no teleport), + a hard 'compute each tap once, never recompute'
clamp. If it converges now, the gap is purely verbosity (SFT-fixable) and decimal mechanical CoT is the answer."""
import sys,re,json,csv; sys.path.insert(0,'pipeline')
import nvidia_api as N
N.set_experiment("probe_terse")
M=0xFF
def b2i(s):return int(s,2)
def i2b(v):return format(v&M,'08b')
def shl(v,k):return (v<<k)&M
def shr(v,k):return v>>k
def rotl(v,k):k&=7;return ((v<<k)|(v>>(8-k)))&M if k else v
def tval(v,t):
    op,k=t;return {'shl':shl,'shr':shr,'rot':rotl}[op](v,k)
def tname(t):
    op,k=t;return {'shl':'SHL','shr':'SHR','rot':'ROTL'}[op]+str(k)
OPF={'XOR':lambda a,b:a^b,'OR':lambda a,b:a|b,'AND':lambda a,b:a&b,'XNOR':lambda a,b:(~(a^b))&M}
def alltaps(v):
    out=[]
    for k in range(1,8): out.append((f"SHL{k}",shl(v,k)))
    for k in range(1,8): out.append((f"SHR{k}",shr(v,k)))
    for k in range(1,8): out.append((f"ROTL{k}",rotl(v,k)))
    return out
def taptable(v):
    return ", ".join(f"{n}={i2b(x)}" for n,x in alltaps(v))

def render_full(prompt, expr, td):
    m=re.match(r'^([A-Z]+)\((\{[AB]\}), *(\{[AB]\})\)$', expr)
    if not m or m.group(1) not in OPF: return None
    op=m.group(1); ta=tuple(td[m.group(2)]); tb=tuple(td[m.group(3)])
    rows=re.findall(r'([01]{8})\s*->\s*([01]{8})',prompt)
    q=re.search(r'output for:\s*([01]{8})',prompt).group(1)
    x1,y1=rows[0]; X1=b2i(x1); Y1=b2i(y1)
    nones=bin(Y1).count('1')
    fam={'XOR':'~half','OR':'many','AND':'few','XNOR':'~half'}[op]
    tapd=dict(alltaps(X1)); A1=tval(X1,ta); B1=tval(X1,tb)
    L=[f"PUZZLE examples: {'; '.join(a+'->'+b for a,b in rows[:4])}  query {q}",
       f"Taps of X={x1} (decimal-once): {taptable(X1)}",
       f"Output {y1} has {nones} ones ({fam}) -> {op}."]
    if op in ('XOR','XNOR'):
        # show a couple Y^tap that aren't taps, then the hit
        rej=[t for t in alltaps(X1) if t[1]!=A1 and t[1]!=B1][:2]
        rl="; ".join(f"{y1}^{n}={i2b(Y1^x)}(not a tap)" for n,x in rej)
        L.append(f"{op}: Y^tapA must be tapB. {rl}; {y1}^{tname(ta)}={i2b(Y1^A1)}={tname(tb)} HIT -> rule {op}({tname(ta)},{tname(tb)}).")
    else:
        L.append(f"{op}: scan pairs -> {tname(ta)}={i2b(A1)} {op} {tname(tb)}={i2b(B1)} = {i2b(OPF[op](A1,B1))}={y1} HIT -> rule {op}({tname(ta)},{tname(tb)}).")
    x2,y2=rows[1]; X2=b2i(x2)
    L.append(f"verify {x2}: {op}({i2b(tval(X2,ta))},{i2b(tval(X2,tb))})={i2b(OPF[op](tval(X2,ta),tval(X2,tb)))}={y2} ok.")
    Q=b2i(q); ans=i2b(OPF[op](tval(Q,ta),tval(Q,tb)))
    L.append(f"query {q}: {op}({i2b(tval(Q,ta))},{i2b(tval(Q,tb))}) = \\boxed{{{ans}}}")
    return "\n".join(L)

cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))}
def is2(rec):
    if not rec['expr']:return False
    mm=re.match(r'^([A-Z]+)\(\{[AB]\}, *\{[AB]\}\)$',rec['expr']);return mm and mm.group(1) in OPF and len(rec['td'])==2
cand=[cid for cid,rec in cache.items() if is2(cache[cid])]
def opof(cid):return re.match(r'^([A-Z]+)',cache[cid]['expr']).group(1)
demos=[]
for want in ['XOR','OR']:
    for cid in cand:
        if opof(cid)==want: demos.append(cid);break
demo_txt="\n\n".join(render_full(rows[c]['prompt'],cache[c]['expr'],cache[c]['td']) for c in demos)
HINTS=("Solve bit puzzles: rule = OP(T1,T2), T1/T2 = SHL/SHR/ROTL by k, OP in XOR/OR/AND/XNOR. Compute shifts in "
 "decimal: SHLk=v*2^k mod256, SHRk=v>>k, ROTLk=((v<<k)|(v>>(8-k)))&255.\n"
 "RULES: list every tap of the first input ONCE on one line (never recompute or double-check a shift). Guess OP "
 "from output 1-count (XOR ~half, OR many, AND few). For XOR use invertibility (Y^tapA=tapB). Then verify on a "
 "2nd example and apply to the query. Keep it to ~6 lines like the examples. Two worked examples:\n\n"+demo_txt+
 "\n\nNow solve this the SAME terse way (do NOT recompute shifts, do NOT second-guess):\n\n")
test=[c for c in cand if c not in set(demos)][:4]
def norm(s):
    m=re.findall(r'[01]{8}', s or '');return m[-1] if m else (s or '').strip()
print("demos:",demos,[opof(c) for c in demos]);print("="*50)
for tid in test:
    e=rows[tid]
    r=N.ask(HINTS+e['prompt']+"\n\nAnswer in \\boxed{}.", model="nvidia/nemotron-3-nano-30b-a3b",
            max_tokens=7680, temperature=0.0, meta={"id":tid,"rule":cache[tid]['expr']})
    got=norm(r.get('answer',''));gold=e['answer'].strip()
    print(f"{tid} {cache[tid]['expr']:16s} {'CORRECT' if got==gold else 'WRONG'} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
