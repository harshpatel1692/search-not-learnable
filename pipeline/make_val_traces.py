"""Build val_traces.jsonl — target CoT traces for the 900-row holdout (val.jsonl).

Used by the eval kernel's argmax-match diagnostic: teacher-force each trace
through (a) the Ali warm-start adapter and (b) the v15 adapter and check, at
every assistant-span token, whether that token is the greedy argmax. A trace
exists only where our procedure is correct (retention rows use Ali's own
correct trace), so coverage per category = the procedure ceiling:
  numeral/gravity/unit : Ali's correct traces (byte-exact, ~99.7-100%)
  cipher               : cipher_v15 renderer, trace-correct rows (~100%)
  bit_manipulation     : bitmanip_global, ALL rows (procedure answer; field
                         gold_match marks the ~1.4% ambiguity rows)
  equation_numeric_*   : equation_cot, policy-hit rows (~95% / ~20%)
  cryptarithm_*        : cryptarithm_cot, solver-vote-correct rows (~56% / ~20%)

Record: {id, category, prompt, cot, final, gold, gold_match, src}
  cot is the TRAINING cot (exactly what went into train_v15.csv for that
  family); final = the trace's answer; gold = train.csv answer.

Usage: python3 pipeline/make_val_traces.py   (from repo root; ~15-30 min, crypt solver-bound)
"""
import csv, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                       # bitmanip_global.load_real uses relative paths
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
csv.field_size_limit(10 ** 8)

OUT = f"{ROOT}/pipeline/data/v15/val_traces.jsonl"
RETAIN = {"gravity", "numeral", "unit_conversion"}

def main():
    val = [json.loads(l) for l in open(f"{ROOT}/pipeline/data/val.jsonl")]
    val_ids = {v['id'] for v in val}
    by_cat = {}
    for v in val:
        by_cat.setdefault(v['category'], set()).add(v['id'])
    gold = {r['id']: r['answer'].strip()
            for r in csv.DictReader(open(f"{ROOT}/competition_dataset/train_categorized.csv"))}
    out = []

    def emit(rid, cat, prompt, cot, final, src):
        out.append(dict(id=rid, category=cat, prompt=prompt, cot=cot, final=final,
                        gold=gold[rid], gold_match=(final == gold[rid]), src=src))

    # ---- retention: Ali's own correct traces (same extraction as the train kernel) ----
    n = 0
    for r in csv.DictReader(open(f"{ROOT}/analysis/ali_validation/all_inferences.csv")):
        if r['category'] in RETAIN and r['id'] in val_ids and r['correct'] == 'True':
            cot = r['raw_output'].split('</think>')[0].rstrip()
            emit(r['id'], r['category'], r['prompt'], cot, r['answer'].strip(), 'ali')
            n += 1
    print(f"[retention] {n} traces")

    # ---- cipher ----
    from synth import cipher_v15 as CV
    rows = CV.load_rows()
    vocab = CV.build_vocab(rows)
    n = 0
    for r in rows:
        if r['id'] not in val_ids:
            continue
        cot, ans = CV.render(r['prompt'], vocab)
        if ans == r['answer'].strip():
            emit(r['id'], 'cipher', r['prompt'], cot, ans, 'cipher_v15')
            n += 1
    print(f"[cipher] {n}/{len(by_cat.get('cipher', []))} traces")

    # ---- bit_manipulation: keep ALL (procedure answer; ambiguity rows flagged) ----
    from synth import bitmanip_global as BG
    n = nm = 0
    for r in BG.load_real():
        if r['id'] not in val_ids:
            continue
        sol = BG.solve(r['ins'], r['outs'])
        cot, final = BG.render(r['ins'], r['outs'], r['q'], sol)
        errs = BG.lint(cot, r['ins'], r['outs'], r['q'], final)
        assert not errs, (r['id'], errs)
        emit(r['id'], 'bit_manipulation', r['prompt'], cot, final, 'bit_global')
        n += 1
        nm += (final != r['gold'])
    print(f"[bit] {n} traces ({nm} ambiguity rows, final!=gold)")

    # ---- equation_numeric (deduce + guess): policy-hit rows only ----
    from synth import equation_cot as EC
    n = {'equation_numeric_deduce': 0, 'equation_numeric_guess': 0}
    for row in EC.E.load_rows():
        if row['id'] not in val_ids or row['cat'] not in n:
            continue
        ans, _ = EC.E.policy_seq(row)
        if ans != row['gold']:
            continue
        cot, final, locked = EC.emit_trace(row['byop'], row['q'])
        EC.lint(cot, final, locked, row['q'][1])
        emit(row['id'], row['cat'], row['prompt'], cot, final, 'equation_cot')
        n[row['cat']] += 1
    print(f"[eq] deduce {n['equation_numeric_deduce']}, guess {n['equation_numeric_guess']} traces")

    # ---- cryptarithm (deduce + guess): solver-correct rows only ----
    from synth import cryptarithm_cot as CC
    from solvers import cryptarithm2 as C2
    tk = CC.tokenizer()
    for cat, policy in [('cryptarithm_deduce', None), ('cryptarithm_guess', ['sub_signed'])]:
        kept, drop, tl = [], {}, {1: [], 2: [], 3: []}
        t0 = time.time()
        for r in CC.load_real(cat):
            if r['id'] not in val_ids:
                continue
            kw = dict(deadline_s=10.0)
            if policy:
                kw['guess_policy'] = policy
            res = C2.solve(r['prompt'], **kw)
            if res is None or res[0] != r['answer'].strip():
                continue
            ans, meta = res
            CC._emit(kept, drop, tl, r, r['prompt'], ans, meta, tk, r['id'], cat)
        for k in kept:
            emit(k['id'], cat, k['prompt'], k['cot'], k['final'], 'cryptarithm_cot')
        print(f"[{cat}] {len(kept)} traces (drops: {drop}) in {time.time()-t0:.0f}s")

    # ---- gates ----
    assert all(o['id'] in val_ids for o in out)
    assert all('\\boxed' not in o['cot'] or o['category'] in
               RETAIN | {'cipher', 'cryptarithm_deduce', 'cryptarithm_guess'} for o in out)
    # ascii gate — ali traces and cipher (「」 brackets are Ali grammar) exempt
    bad_ascii = [o['id'] for o in out
                 if o['src'] not in ('ali', 'cipher_v15')
                 and not (o['cot'] + o['prompt']).isascii()]
    assert not bad_ascii, bad_ascii[:5]
    with open(OUT, 'w') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    from collections import Counter
    print(f"wrote {len(out)} traces -> {OUT}")
    print("per-cat:", dict(Counter(o['category'] for o in out)))

if __name__ == '__main__':
    main()
