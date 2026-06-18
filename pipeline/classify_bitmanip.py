"""Classify each bit_manip row by the rule structure bit_global discovered -> real sub-category distribution.
Buckets by: #taps (vars used), function family (root op), and whether any transform is a shift (zero-fill)."""
import csv,sys,json,collections,re; sys.path.insert(0,'pipeline')
import bit_global as G

def family(expr):
    # root operation of the discovered boolean expression
    if expr in ('C0','C1'): return 'CONST'
    if expr in ('{A}','{B}','{C}'): return 'IDENTITY'
    m=re.match(r'^([A-Z_]+)\(',expr)
    root=m.group(1) if m else 'OTHER'
    return root

def ntaps(td): return len(td)  # number of distinct variables (taps) used

rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
n=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
rows=rows[:n]
dist=collections.Counter(); fam=collections.Counter(); tapc=collections.Counter(); shiftc=collections.Counter()
examples={}   # subcat -> example prompt+answer (first seen)
unsolved=0
for i,r in enumerate(rows):
    res=G.solve(r['prompt'],3.0)
    if res is None: unsolved+=1; continue
    pred,expr,td=res
    nt=ntaps(td); f=family(expr)
    has_shift=any(t[0] in ('shl','shr') for t in td.values())
    sub=f"{nt}tap-{f}{'-shift' if has_shift else ''}"
    dist[sub]+=1; fam[f]+=1; tapc[nt]+=1; shiftc['has_shift' if has_shift else 'rot_only']+=1
    if sub not in examples:
        examples[sub]={'id':r['id'],'expr':expr,'td':{k:list(v) for k,v in td.items()},
                       'prompt':r['prompt'],'answer':r['answer'].strip()}
    if (i+1)%300==0: print(f"  ..{i+1}/{len(rows)}",flush=True)
print("\n=== TAP COUNT ===");
for k,v in sorted(tapc.items()): print(f"  {k}-tap: {v} ({100*v/len(rows):.1f}%)")
print("=== TRANSFORM ===")
for k,v in shiftc.items(): print(f"  {k}: {v} ({100*v/len(rows):.1f}%)")
print("=== FUNCTION FAMILY ===")
for k,v in fam.most_common(): print(f"  {k}: {v} ({100*v/len(rows):.1f}%)")
print("=== SUB-CATEGORY (tap x family x shift) ===")
for k,v in dist.most_common(): print(f"  {k}: {v} ({100*v/len(rows):.1f}%)")
print(f"unsolved: {unsolved}")
json.dump(examples, open('pipeline/data/bitmanip_subcats.json','w'), indent=1)
print(f"\nsaved {len(examples)} sub-category example prompts -> pipeline/data/bitmanip_subcats.json")
