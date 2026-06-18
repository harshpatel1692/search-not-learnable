"""make_crypt_honest.py -- generate the HONEST crypt_deduce corpus from crypt_column.build_trace.

Sources (zero leakage vs the eval set):
  * synthetic crypt puzzles (pipeline/data/crypt_r8/crypt_deduce_synth.jsonl) -- different ciphers
  * REAL crypt_deduce train rows (competition_dataset/train_categorized.csv) EXCLUDING val.jsonl ids
Each row is rendered by the two-pass honest engine; only fully-showable cracks (no opaque line,
verified answer==gold) are emitted. Output schema matches sft_warmstart.py (raw_output = CoT body).

Run: python3 pipeline/synth/make_crypt_honest.py
Out: pipeline/data/crypt_honest.csv  (+ stdout: counts, leak=0 assert, opaque=0 assert)
"""
import os, sys, csv, json, time, signal
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
import cryptarithm2 as C2
import crypt_column as CC
csv.field_size_limit(10 ** 9)


class _TO(Exception):
    pass


def _alarm(s, f):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)

OUT = os.path.join(ROOT, 'pipeline', 'data', 'crypt_honest.csv')
SYNTH = os.path.join(ROOT, 'pipeline', 'data', 'crypt_r8', 'crypt_deduce_synth.jsonl')
REAL = os.path.join(ROOT, 'competition_dataset', 'train_categorized.csv')
VAL = os.path.join(ROOT, 'pipeline', 'data', 'val.jsonl')


def crack(prompt, gold):
    signal.alarm(8)                              # per-puzzle cap; skip branch-heavy stalls
    try:
        res = C2.solve(prompt, deadline_s=2.0)
        if not res:
            return None
        ops = (res[1] or {}).get('ops'); rev = (res[1] or {}).get('rev')
        if ops is None:
            return None
        t, reason = CC.build_trace(prompt, ops, rev, gold)
        if t is None or 'residue+exact' in t:    # opaque guard
            return None
        return t.rsplit('\\boxed{', 1)[0].rstrip()
    except Exception:
        return None
    finally:
        signal.alarm(0)


def main():
    val_ids = {json.loads(l)['id'] for l in open(VAL)}
    rows = []; leak = 0; t0 = time.time()
    # synthetic
    n_syn_try = n_syn_ok = 0
    for l in open(SYNTH, encoding='utf-8'):
        d = json.loads(l); n_syn_try += 1
        if n_syn_try % 100 == 0:
            print('  synth %d tried, %d cracked (%.0fs)' % (n_syn_try, n_syn_ok, time.time() - t0), flush=True)
        if d['id'] in val_ids:
            leak += 1; continue
        body = crack(d['prompt'], str(d['final']).strip())
        if body:
            n_syn_ok += 1
            rows.append({'id': d['id'], 'prompt': d['prompt'], 'answer': str(d['final']).strip(),
                         'category': 'cryptarithm_deduce', 'raw_output': body,
                         'predicted': str(d['final']).strip(), 'correct': 'True'})
    # real (exclude val)
    n_real_try = n_real_ok = 0
    for r in csv.DictReader(open(REAL)):
        if r['category'] != 'cryptarithm_deduce':
            continue
        if r['id'] in val_ids:
            continue
        n_real_try += 1
        if n_real_try % 100 == 0:
            print('  real %d tried, %d cracked (%.0fs)' % (n_real_try, n_real_ok, time.time() - t0), flush=True)
        body = crack(r['prompt'], r['answer'].strip())
        if body:
            n_real_ok += 1
            rows.append({'id': r['id'], 'prompt': r['prompt'], 'answer': r['answer'].strip(),
                         'category': 'cryptarithm_deduce', 'raw_output': body,
                         'predicted': r['answer'].strip(), 'correct': 'True'})
    assert leak == 0, 'VAL LEAK %d' % leak
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'category',
                                          'raw_output', 'predicted', 'correct'])
        w.writeheader(); w.writerows(rows)
    print('[out] %s : %d honest crypt_deduce rows (leak 0)' % (OUT, len(rows)))
    print('  synthetic: %d/%d cracked (%.0f%%)' % (n_syn_ok, n_syn_try, 100 * n_syn_ok / max(n_syn_try, 1)))
    print('  real(non-val): %d/%d cracked (%.0f%%)' % (n_real_ok, n_real_try, 100 * n_real_ok / max(n_real_try, 1)))
    print('  elapsed %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
