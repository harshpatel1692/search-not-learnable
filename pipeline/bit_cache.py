"""Solve every bit_manip row with bit_global ONCE and persist to a JSONL cache so we never re-run the
~7-min solve. Usage:
  python3 pipeline/bit_cache.py build      # solve all rows -> pipeline/data/bitmanip_solved.jsonl
  python3 pipeline/bit_cache.py stats       # load cache -> distribution + sub-category example prompts
Other scripts: `from bit_cache import load` -> {id: {pred,expr,td,gold,correct,...}} (instant)."""
import csv,sys,json,os,collections,re
CACHE='pipeline/data/bitmanip_solved.jsonl'

def _rows():
    return [r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']

def build(time_limit=5.0):
    sys.path.insert(0,'pipeline'); import bit_global as G
    rows=_rows()
    with open(CACHE,'w') as f:
        ok=0
        for i,r in enumerate(rows):
            res=G.solve(r['prompt'],time_limit)
            rec={'id':r['id'],'gold':r['answer'].strip()}
            if res is None:
                rec.update(pred=None,expr=None,td=None,correct=False)
            else:
                pred,expr,td=res
                rec.update(pred=pred,expr=expr,td={k:list(v) for k,v in td.items()},correct=(pred==r['answer'].strip()))
                ok+=rec['correct']
            f.write(json.dumps(rec)+'\n')
            if (i+1)%200==0: print(f"  ..{i+1}/{len(rows)} correct={ok}",flush=True)
    print(f"built cache: {len(rows)} rows, correct={ok} ({100*ok/len(rows):.2f}%) -> {CACHE}")

def load():
    if not os.path.exists(CACHE): raise FileNotFoundError(f"{CACHE} missing — run: python3 pipeline/bit_cache.py build")
    return {r['id']:r for r in (json.loads(l) for l in open(CACHE))}

def _family(expr):
    if expr in ('C0','C1'): return 'CONST'
    if expr in ('{A}','{B}','{C}'): return 'IDENTITY'
    m=re.match(r'^([A-Z_]+)\(',expr); return m.group(1) if m else 'OTHER'

def stats():
    cache=load(); rows=_rows(); pid={r['id']:r for r in rows}
    fam=collections.Counter(); tapc=collections.Counter(); shiftc=collections.Counter(); sub=collections.Counter()
    examples={}; unsolved=0; n=len(cache)
    for cid,rec in cache.items():
        if rec['expr'] is None: unsolved+=1; continue
        td=rec['td']; nt=len(td); f=_family(rec['expr'])
        has_shift=any(v[0] in ('shl','shr') for v in td.values())
        s=f"{nt}tap-{f}{'-shift' if has_shift else ''}"
        fam[f]+=1; tapc[nt]+=1; shiftc['has_shift' if has_shift else 'rot_only']+=1; sub[s]+=1
        if s not in examples:
            examples[s]={'id':cid,'expr':rec['expr'],'td':td,'prompt':pid[cid]['prompt'],'answer':rec['gold']}
    print("=== TAP COUNT ===")
    for k,v in sorted(tapc.items()): print(f"  {k}-tap: {v} ({100*v/n:.1f}%)")
    print("=== TRANSFORM ===")
    for k,v in shiftc.items(): print(f"  {k}: {v} ({100*v/n:.1f}%)")
    print("=== FUNCTION FAMILY ===")
    for k,v in fam.most_common(): print(f"  {k}: {v} ({100*v/n:.1f}%)")
    print("=== SUB-CATEGORY (tap x family x shift) ===")
    for k,v in sub.most_common(): print(f"  {k}: {v} ({100*v/n:.1f}%)")
    print(f"unsolved: {unsolved}")
    json.dump(examples, open('pipeline/data/bitmanip_subcats.json','w'), indent=1)
    print(f"saved {len(examples)} sub-category examples -> pipeline/data/bitmanip_subcats.json")

if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else 'build'
    (build if cmd=='build' else stats)()
