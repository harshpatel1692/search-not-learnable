"""In-context narrowing test: re-run the SAME 3 bit_manip tasks the base failed, but prefix a short
narrowing procedure (popcount prune + COLUMN-WISE position mapping + decisive reject + stop).
If guided base converges within budget AND gets them right -> the narrowing lever is real."""
import sys,re,csv; sys.path.insert(0,'pipeline')
import nvidia_api as N
import bit_bfs as BFS
N.set_experiment("base_probe_narrow")

PRE=("Find the rule by TESTING NAMED OPERATIONS on the FIRST example only (do not analyze bit columns). "
 "Take the first input and compute these candidates, comparing each to the first output:\n"
 "  ROTL1..ROTL7, ROTR1..ROTR7 (rotations), SHL1..SHR7 (shifts), NOT.\n"
 "Find the FIRST candidate that equals the first output. If none matches exactly, the rule combines two "
 "of these or adds a constant: compute (first_output XOR first_input) as a candidate constant, or try "
 "(input XOR rotated-input). \n"
 "Once a candidate matches the first example, VERIFY it on the second example. If it fails, drop it in "
 "one line and try the next candidate. Once one matches two examples, apply it to the query and STOP. "
 "Keep each computation short — you already know how to rotate/shift/XOR an 8-bit value.\n\n")

rows=[r for r in csv.DictReader(open('competition_dataset/train_categorized.csv')) if r['category']=='bit_manipulation']
picks=[]
for r in rows:
    res=BFS.solve(r['prompt'],3,8000); picks.append((len(res[1]) if res else 99, r))
picks.sort(key=lambda x:x[0])
sample=[picks[0][1], picks[len(picks)//2][1], picks[-1][1]]
def norm(s):
    m=re.findall(r'[01]{8}', s or ''); return m[-1] if m else (s or '').strip()
for i,r in enumerate(sample):
    res=N.ask(PRE+r['prompt'], model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=7680, temperature=0.0, meta={"tier":"narrow","i":i})
    got=norm(res.get("answer","")); gold=r['answer'].strip()
    print(f"task {i}: {'CORRECT' if got==gold else 'WRONG'} got={got} gold={gold} think={len(res.get('reasoning',''))}c finish={res.get('finish')}", flush=True)
