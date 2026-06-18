"""Isolate the LOGIC from the computation. Pre-compute EVERYTHING (taps + OUT^tap values); the base only has to
MATCH (its cheap native skill, 399c) to identify the rule, then APPLY via the query's listed taps. If this solves
2-tap reliably at scale, then 2-tap is SOLVED modulo terse computation (the SFT job) — strong conviction.
Usage: python3 probe_matchonly.py <fam> <N>   fam: xor or and  (2-tap only)"""
import sys,re,json,csv; sys.path.insert(0,'pipeline')
import nvidia_api as N
M=0xFF
def shl(v,k):return (v<<k)&M
def shr(v,k):return v>>k
def rotl(v,k):k&=7;return ((v<<k)|(v>>(8-k)))&M if k else v
def i2b(v):return format(v,'08b')
def alltaps(v):
    return [(f"SHL{k}",shl(v,k)) for k in range(1,8)]+[(f"SHR{k}",shr(v,k)) for k in range(1,8)]+[(f"ROTL{k}",rotl(v,k)) for k in range(1,8)]
def tbl(v): return ", ".join(f"{n}={i2b(x)}" for n,x in alltaps(v))
cache={r['id']:r for r in (json.loads(l) for l in open('pipeline/data/bitmanip_solved.jsonl'))}
rows={r['id']:r for r in csv.DictReader(open('competition_dataset/train_categorized.csv'))}
def opof(rec):
    m=re.match(r'^([A-Z_]+)',rec['expr'] or ''); return m.group(1) if m else None
FAM=sys.argv[1] if len(sys.argv)>1 else 'xor'; NT=int(sys.argv[2]) if len(sys.argv)>2 else 8
OPU=FAM.upper()
cand=[cid for cid,rec in cache.items() if rec['correct'] and rec['td'] and len(rec['td'])==2 and opof(rec)==OPU][:NT]
N.set_experiment(f"matchonly_{FAM}")
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
ok=bad=0; print(f"FAM={FAM} N={len(cand)}",flush=True)
for cid in cand:
    e=rows[cid]; ex=re.findall(r'([01]{8})\s*->\s*([01]{8})',e['prompt']); q=re.search(r'output for:\s*([01]{8})',e['prompt']).group(1)
    x1,y1=ex[0]; X1=int(x1,2); Y1=int(y1,2); taps1=alltaps(X1)
    if FAM=='xor':
        combo=", ".join(f"OUT^{n}={i2b(Y1^x)}" for n,x in taps1)
        method=(f"OUT={y1}. Below 'OUT^TAP' values are precomputed. Find the ONE whose value matches a tap name in the "
                f"tap list (i.e. OUT^TAP_i == TAP_j). Then the rule is XOR(TAP_i, TAP_j).")
    elif FAM=='and':
        combo=""  # for AND, just say find two taps whose AND==OUT among given
        method=(f"OUT={y1}. Find two taps in the list whose bitwise AND equals OUT (a tap qualifies only if it is a "
                f"superset of OUT: tap AND OUT == OUT). The rule is AND of those two.")
    else:
        combo=""; method=(f"OUT={y1}. Find two taps whose bitwise OR equals OUT (each must be a subset of OUT). Rule = OR of those two.")
    p=(f"Taps of {x1}: {tbl(X1)}\n"+ (f"Precomputed: {combo}\n" if combo else "")+
       f"{method}\nThen APPLY that rule to the query using the query's taps. Taps of query {q}: {tbl(int(q,2))}\n"
       f"Give the query 8-bit output in \\boxed{{}}.")
    r=N.ask(p, model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=2000, temperature=0.0, meta={"id":cid,"rule":cache[cid]['expr']})
    got=norm(r.get('answer','')); gold=e['answer'].strip(); good=(got==gold); ok+=good; bad+=(not good)
    print(f"  {cid} {cache[cid]['expr'][:30]:30s} {'OK ' if good else 'BAD'} got={got} gold={gold} {len(r.get('reasoning',''))}c {r.get('finish')}",flush=True)
print(f"RESULT match-only {FAM}: {ok}/{ok+bad}",flush=True)
