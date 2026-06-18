# DEAD END (2026-06-14): hand-written 3-tap grammar -> bit 0.062 (REGRESSION).
# Teleports the tap-search (model hallucinates rejects + locks wrong rule) and grammar-mixing
# misroutes the tap-count. 3-tap was NEVER 0.07 (that was a confound) -- it's 0.503 in bit-star-r2.
# Bit lever is STaR self-harvest (run_bit_star_r3.sh), NOT a hand grammar. See LOG.md 2026-06-14.
"""Build a bit_manipulation SFT corpus that TEACHES the genuine 3-input (3-tap) tail.

The proven perbit renderer covers 1-tap/2-tap ~100% but SKIPS genuine 3-input rules (gen3 falls back to gen2),
so the model never trains on them -> 3-tap eval = 0.07. This corpus = perbit gen1/gen2 (preserve 1-2 tap) +
NEW bitmanip_3tap.gen3tap (genuine 3-input, 11 real truth tables x 22 taps), 3-tap OVER-sampled for signal.

All synthetic (random inputs/taps) => zero leakage vs the real bit_eval500 / train rows.
Output: pipeline/data/v16/train_bit_3tap.csv (train_v16 schema).
  python3 pipeline/make_bit_3tap_corpus.py [seed]
"""
import os, sys, csv, json, random, collections, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
import bitmanip_perbit as PB
import bitmanip_3tap as T3
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
from cryptarithm_cot import tokenizer as _tok

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 314
N_3TAP = 2400      # genuine 3-input (the lever)
N_2TAP = 1800      # preserve 2-tap
N_1TAP = 400       # preserve 1-tap
TOK_CAP = 1700
csv.field_size_limit(10 ** 9)


def row(rid, p, ans, cot):
    return {'id': rid, 'prompt': p, 'answer': str(ans), 'category': 'bit_manipulation',
            'raw_output': cot, 'predicted': str(ans), 'correct': 'True'}


def main():
    rng = random.Random(SEED)
    tok = _tok()
    rows = []
    seen = set()

    def add(gen_fn, want, tag):
        got = 0; tries = 0
        while got < want and tries < want * 40:
            tries += 1
            r = gen_fn(rng)
            if r is None:
                continue
            p = r['prompt']
            if p in seen:
                continue
            cot = r['cot']
            # sanity: CoT's stated Result must equal the gold answer
            stated = cot.split('Result:')[-1].strip().rstrip('.')
            if stated != r['answer']:
                continue
            if len(tok.encode(cot).ids) > TOK_CAP:
                continue
            seen.add(p)
            rows.append(row(f'{tag}-{SEED}-{len(rows):05d}', p, r['answer'], cot))
            got += 1
        print(f'  [{tag}] {got}/{want} (scanned {tries})', flush=True)
        return got

    add(T3.gen3tap, N_3TAP, 'bit3')
    add(PB.gen2, N_2TAP, 'bit2')
    add(PB.gen1, N_1TAP, 'bit1')

    rng.shuffle(rows)
    out = os.path.join(ROOT, 'pipeline/data/v16/train_bit_3tap.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'category',
                                          'raw_output', 'predicted', 'correct'])
        w.writeheader(); w.writerows(rows)
    nt = [len(tok.encode(r['raw_output']).ids) for r in rows]
    print(f'\n[out] {out}: {len(rows)} rows')
    print(f'  ntok median {int(statistics.median(nt))} p95 {int(sorted(nt)[int(len(nt)*.95)])} max {max(nt)}')


if __name__ == '__main__':
    main()
