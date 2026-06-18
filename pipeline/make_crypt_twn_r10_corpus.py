"""Build the crypt-twn-r9 SFT corpus (trace_we_need format):
  - value-row synth (concat-query capped to ~12% to match test weight)
  - real cryptarithm_deduce rows the generator covers (VAL HELD OUT)
  - existing crypt_guess renders (val held out)
Output: pipeline/data/v16/train_crypt_twn_r10.csv (train_v16 schema) + a jsonl mirror.
Asserts 0 val leak (by id AND by prompt). Run:
  python3 pipeline/make_crypt_twn_corpus.py [n_value] [seed]
"""
import os, sys, json, csv, gzip, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
import crypt_twn as T
import cryptarithm_cot as CC
import cryptarithm2 as C2

N_VALUE = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 950
CONCAT_FRAC = 0.12
BAIL_FRAC = 0.30          # ~30% bail traces (teach STOP, not give-up); rest solve
csv.field_size_limit(10 ** 9)

val = [json.loads(l) for l in open(os.path.join(ROOT, 'pipeline/data/val.jsonl'))]
val_ids = {r['id'] for r in val}
val_prompts = {r['prompt'] for r in val}
tok = CC.tokenizer()


def row(rid, prompt, answer, cat, cot):
    return {'id': rid, 'prompt': prompt, 'answer': str(answer), 'category': cat,
            'raw_output': cot, 'predicted': str(answer), 'correct': 'True'}


_VALP = None
_TOK = None


def _syn_init(valp):
    global _VALP, _TOK
    _VALP = valp
    _TOK = CC.tokenizer()


def _syn_work(args):
    i, seed = args
    rng = random.Random(seed * 1_000_000 + i)
    p = CC.gen_puzzle_r5(rng)
    if p is None or p['prompt'] in _VALP:
        return None
    try:
        cot, ans, kind = T.render(p['prompt'], p['ops'], p['rev'])
    except Exception:
        return None
    if cot is None:
        return None
    if kind != 'bail' and ans != p['answer']:     # solve/concat must be correct
        return None
    if len(_TOK.encode(cot).ids) > T.TOK_CAP:
        return None
    return (kind, p['prompt'], ans, cot)


def main():
    import multiprocessing as mp
    rows = []
    seen_prompts = set()
    concat_cap = round(N_VALUE / (1 - CONCAT_FRAC) * CONCAT_FRAC)
    bail_cap = round(N_VALUE * BAIL_FRAC)
    nsolve = nbail = ncon = 0
    i0, batch = 0, 4000
    with mp.Pool(7, initializer=_syn_init, initargs=(val_prompts,)) as pool:
        while nsolve < N_VALUE and i0 < N_VALUE * 200:
            res = pool.map(_syn_work, [(i, SEED) for i in range(i0, i0 + batch)], chunksize=20)
            i0 += batch
            for r in res:
                if r is None:
                    continue
                kind, prompt, ans, cot = r
                if prompt in seen_prompts:
                    continue
                if kind == 'concat' and ncon >= concat_cap:
                    continue
                if kind == 'bail' and nbail >= bail_cap:
                    continue
                if kind == 'solve' and nsolve >= N_VALUE:
                    continue
                seen_prompts.add(prompt)
                rows.append(row(f'twn-s{SEED}-{len(rows):05d}', prompt, ans, 'cryptarithm_deduce', cot))
                ncon += kind == 'concat'
                nbail += kind == 'bail'
                nsolve += kind == 'solve'
            print(f'  synth: scanned {i0}, solve {nsolve}/{N_VALUE}, bail {nbail}/{bail_cap}, '
                  f'concat {ncon}/{concat_cap}', flush=True)
    print(f'[synth] solve={nsolve} bail={nbail} concat={ncon} (scanned {i0})')

    # ---- 2. real deduce rows the generator covers (val held out) ----
    acc = {}
    for line in gzip.open(os.path.join(ROOT, 'analysis/crypt_struct/hitk_detail_fix.jsonl.gz'), 'rt'):
        d = json.loads(line)
        if d.get('status') != 'ok':
            continue
        fc = d.get('first_consistent')
        if not fc:
            continue
        for e in d.get('detail', []):
            if e['rank'] == fc:
                acc[d['id']] = (e['ops'], e['rev'])
                break
    real = [json.loads(l) for l in open(os.path.join(ROOT, 'pipeline/data/crypt_train_all.jsonl'))
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    nreal = 0
    for r in real:
        if r['id'] in val_ids or r['id'] not in acc:
            continue
        ops, rev = acc[r['id']]
        try:
            cot, ans, kind = T.render(r['prompt'], ops, rev)
        except Exception:
            cot = None
        # real rows only as correct solve/concat (no real-row bail -> would box wrong gold)
        if cot is None or kind == 'bail' or ans != r['answer'] or len(tok.encode(cot).ids) > T.TOK_CAP:
            continue
        rows.append(row(r['id'], r['prompt'], ans, 'cryptarithm_deduce', cot))
        nreal += 1
    print(f'[real deduce] {nreal} (val held out)')

    # ---- 3. existing crypt_guess renders (val held out) ----
    nguess = 0
    for l in open(os.path.join(ROOT, 'pipeline/data/crypt_r8/crypt_guess.jsonl')):
        d = json.loads(l)
        if d['id'] in val_ids:
            continue
        rows.append(row(d['id'], d['prompt'], d['final'], d['category'], d['cot']))
        nguess += 1
    print(f'[guess] {nguess} (val held out)')

    # ---- val-leak assertions ----
    assert not [r for r in rows if r['id'] in val_ids], 'VAL LEAK by id'
    assert not [r for r in rows if r['prompt'] in val_prompts], 'VAL LEAK by prompt'

    random.Random(SEED).shuffle(rows)
    os.makedirs(os.path.join(ROOT, 'pipeline/data/v16'), exist_ok=True)
    out_csv = os.path.join(ROOT, 'pipeline/data/v16/train_crypt_twn_r10.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'category',
                                          'raw_output', 'predicted', 'correct'])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ROOT, 'pipeline/data/crypt_r9/crypt_twn_r10_corpus.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    import collections, statistics
    cat = collections.Counter(r['category'] for r in rows)
    nt = [len(tok.encode(r['raw_output']).ids) for r in rows]
    print(f'\n[out] {out_csv}: {len(rows)} rows, 0 val leak')
    print(f'  categories: {dict(cat)}')
    print(f'  synth mix: solve {nsolve} / bail {nbail} / concat {ncon}')
    print(f'  raw_output ntok median {int(statistics.median(nt))} '
          f'p95 {int(sorted(nt)[int(len(nt)*.95)])} max {max(nt)}')


if __name__ == '__main__':
    main()
