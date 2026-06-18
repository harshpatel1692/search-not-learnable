"""Build SFT data from the per-category solvers (pipeline/solvers/) and a permanent
stratified held-out val set across all 9 categories. Easy-4 solvers reproduce 100% of train."""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); csv.field_size_limit(10**7)
from solvers import numeral, unit_conversion, gravity, cipher, verify

SRC=Path('competition_dataset/train_categorized.csv')
OUT=Path('pipeline/data'); OUT.mkdir(parents=True, exist_ok=True)
VAL_PER_CAT=100; SEED=20260606
rows=list(csv.DictReader(SRC.open()))
byc=defaultdict(list)
for r in rows: byc[r['category']].append(r)
voc=cipher.build_vocab(byc['cipher'])
def solve(cat, prompt):
    return {'numeral':numeral.solve,'unit_conversion':unit_conversion.solve,
            'gravity':gravity.solve}.get(cat, lambda p: cipher.solve(p, voc))(prompt) \
           if cat!='cipher' else cipher.solve(prompt, voc)

rng=random.Random(SEED); val_ids=set(); val=[]
for cat in sorted(byc):
    pick=rng.sample(sorted(byc[cat], key=lambda r:r['id']), min(VAL_PER_CAT,len(byc[cat])))
    for r in pick: val_ids.add(r['id']); val.append({'id':r['id'],'category':cat,'prompt':r['prompt'],'answer':r['answer']})

train=[]; stats=defaultdict(lambda:[0,0])
for cat in ['numeral','unit_conversion','gravity','cipher']:
    for r in byc[cat]:
        if r['id'] in val_ids: continue
        stats[cat][1]+=1
        try: ans,cot=solve(cat, r['prompt'])
        except Exception: continue
        if not verify(r['answer'], ans): continue
        stats[cat][0]+=1
        train.append({'id':r['id'],'category':cat,'prompt':r['prompt'],'answer':r['answer'],'cot':cot,'final':ans})

(OUT/'val.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in val))
(OUT/'train_easy.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in train))
print(f"val.jsonl: {len(val)} | train_easy.jsonl: {len(train)}")
for cat in sorted(stats): ok,tot=stats[cat]; print(f"  {cat:16s} kept {ok}/{tot} = {100*ok/tot:.1f}%")
