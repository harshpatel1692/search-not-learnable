"""Decisive learnability test: prepend ONE fully-worked whole-value trace (generated from the known rule),
then ask the base to solve a NEW 2-tap puzzle the same way. If the demo makes it EXECUTE (not deliberate) and
converge, the whole-value CoT style is SFT-learnable. Tests on held-out 2-tap rows from the cache."""
import sys,re,json,csv; sys.path.insert(0,'pipeline')
import nvidia_api as N
import bit_global as G
N.set_experiment("probe_oneshot")

def transform_str(in_bits, t):
    return "".join(str(G.get_source_bit(in_bits,p,tuple(t))) for p in range(8))
def parse_rows(prompt):
    ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',prompt)
    q=re.search(r'output for:\s*([01]{8})',prompt).group(1)
    return ex,q
TNAME={'rot':'ROTL','shl':'SHL','shr':'SHR'}
def tname(t):
    k=t[1];
    return f"{TNAME[t[0]]}{k}" if k else "IDENTITY"
def bitop(a,b,op):
    A=int(a,2);B=int(b,2);M=0xFF
    f={'XOR':A^B,'OR':A|B,'AND':A&B,'XNOR':(~(A^B))&M,'NAND':(~(A&B))&M,'NOR':(~(A|B))&M,
       'NOT_A_AND_B':((~A)&B)&M,'A_AND_NOT_B':(A&(~B))&M,'NOT_A_OR_B':((~A)|B)&M,'A_OR_NOT_B':(A|(~B))&M}[op]
    return format(f,'08b')
def render_demo(prompt, expr, td):
    """Whole-value worked CoT for a 2-tap rule root op(varX, varY)."""
    m=re.match(r'^([A-Z_]+)\((\{[ABC]\}), *(\{[ABC]\})\)$', expr)
    if not m: return None
    op,v1,v2=m.group(1),m.group(2),m.group(3)
    t1=td[v1]; t2=td[v2]
    ex,q=parse_rows(prompt)
    x1,y1=ex[0]; ib1=[int(c) for c in x1]
    c1=transform_str(ib1,t1); c2=transform_str(ib1,t2)
    x2,y2=ex[1]; ib2=[int(c) for c in x2]; d1=transform_str(ib2,t1); d2=transform_str(ib2,t2)
    qb=[int(c) for c in q]; qc1=transform_str(qb,t1); qc2=transform_str(qb,t2)
    ans=bitop(qc1,qc2,op)
    s=(f"Work on whole 8-bit values. First example X={x1}, Y={y1}. The rule is OUT = {op}(T1(X), T2(X)) for some "
       f"transforms. Compute candidate copies of X and match:\n"
       f"  {tname(t1)}(X) = {c1}\n  {tname(t2)}(X) = {c2}\n"
       f"Test {op}: {op}({c1}, {c2}) = {bitop(c1,c2,op)} = {y1} ✓. So rule = {op}({tname(t1)}(X), {tname(t2)}(X)).\n"
       f"Verify on X={x2}: {tname(t1)}={d1}, {tname(t2)}={d2}, {op} = {bitop(d1,d2,op)} = {y2} ✓.\n"
       f"Apply to query {q}: {tname(t1)}={qc1}, {tname(t2)}={qc2}, {op} = {ans}. Answer: {ans}.")
    return s

# load cache, pick 2tap rows
cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))}
def is_2tap_op(rec,ops):
    if not rec['expr']: return False
    mm=re.match(r'^([A-Z_]+)\(\{[ABC]\}, *\{[ABC]\}\)$',rec['expr'])
    return mm and mm.group(1) in ops and len(rec['td'])==2
cand=[cid for cid,rec in cache.items() if is_2tap_op(rec,{'XOR','OR','AND'})]
demo_id=cand[0]; test_ids=cand[1:4]
demo=render_demo(rows[demo_id]['prompt'], cache[demo_id]['expr'], cache[demo_id]['td'])
print("DEMO rule:",cache[demo_id]['expr'],cache[demo_id]['td']); print(demo); print("="*60)
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
PRE=("Here is a worked example of finding a bit-rule by computing WHOLE 8-bit transformed copies and matching "
     "(do it the same decisive way — actually COMPUTE the copies, do not just plan):\n\n"+demo+
     "\n\nNow solve this new puzzle the same way:\n\n")
for tid in test_ids:
    e=rows[tid]
    r=N.ask(PRE+e['prompt'], model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0,
            meta={"id":tid,"rule":cache[tid]['expr'],"td":cache[tid]['td']})
    got=norm(r.get("answer","")); gold=e['answer'].strip()
    print(f"test {tid} rule={cache[tid]['expr']}: {'CORRECT' if got==gold else 'WRONG'} got={got} gold={gold} think={len(r.get('reasoning',''))}c finish={r.get('finish')}",flush=True)
