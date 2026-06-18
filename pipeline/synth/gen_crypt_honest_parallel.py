"""Mass-generate honest crypt_deduce training traces, in PARALLEL.

Uses cryptarithm_cot.gen_puzzle (the real-distribution generator) -> gives prompt/answer/ops/rev
directly, so we SKIP the slow C2.solve. Each worker generates + renders with the honest two-pass
engine (crypt_column.build_trace) under a per-puzzle timeout, writing cracked rows to a shard.
Combine -> pipeline/data/crypt_honest_big.csv.

Run: python3 pipeline/synth/gen_crypt_honest_parallel.py [workers] [seconds] [timeout_s]
"""
import os, sys, csv, json, time, signal, random
import multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'solvers'))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 'crypt_struct'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
csv.field_size_limit(10 ** 9)
SHARD_DIR = os.path.join(ROOT, 'pipeline', 'data', 'crypt_honest_shards')
OUT = os.path.join(ROOT, 'pipeline', 'data', 'crypt_honest_big.csv')
COLS = ["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"]


class _TO(Exception):
    pass


def _alarm(s, f):
    raise _TO()


def worker(wid, seed, budget, timeout_s, val_prompts):
    import cryptarithm_cot as CCOT
    import crypt_column as CC
    signal.signal(signal.SIGALRM, _alarm)
    rng = random.Random(seed)
    shard = os.path.join(SHARD_DIR, 'shard_%d.jsonl' % wid)
    f = open(shard, 'w', encoding='utf-8')
    t0 = time.time(); tried = ok = 0
    while time.time() - t0 < budget:
        tried += 1
        try:
            p = CCOT.gen_puzzle(rng)
        except Exception:
            continue
        if not p or p.get('category') != 'cryptarithm_deduce':
            continue
        if p['prompt'] in val_prompts:
            continue
        signal.alarm(timeout_s)
        try:
            t, reason = CC.build_trace(p['prompt'], p['ops'], p['rev'], str(p['answer']).strip())
        except Exception:
            t = None
        finally:
            signal.alarm(0)
        if t is None or 'residue+exact' in t:
            continue
        body = t.rsplit('\\boxed{', 1)[0].rstrip()
        ok += 1
        f.write(json.dumps({'id': 'synh-%d-%d' % (wid, ok), 'prompt': p['prompt'],
                            'answer': str(p['answer']).strip(), 'raw_output': body}) + '\n')
        if ok % 25 == 0:
            f.flush()
            print('  [w%d] %d cracked / %d tried (%.0fs)' % (wid, ok, tried, time.time() - t0), flush=True)
    f.close()
    print('[w%d] DONE %d cracked / %d tried' % (wid, ok, tried), flush=True)


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    os.makedirs(SHARD_DIR, exist_ok=True)
    for fn in os.listdir(SHARD_DIR):
        os.remove(os.path.join(SHARD_DIR, fn))
    val_prompts = {json.loads(l).get('prompt') for l in open(os.path.join(ROOT, 'pipeline', 'data', 'val.jsonl'))}
    print('[gen] %d workers x %ds, timeout %ds/puzzle' % (W, budget, timeout_s), flush=True)
    procs = []
    for wid in range(W):
        pr = mp.Process(target=worker, args=(wid, 1000 + wid * 7919, budget, timeout_s, val_prompts))
        pr.start(); procs.append(pr)
    for pr in procs:
        pr.join()
    # combine + dedup by prompt
    rows = []; seen = set()
    for fn in sorted(os.listdir(SHARD_DIR)):
        for l in open(os.path.join(SHARD_DIR, fn), encoding='utf-8'):
            d = json.loads(l)
            if d['prompt'] in seen:
                continue
            seen.add(d['prompt'])
            rows.append({'id': d['id'], 'prompt': d['prompt'], 'answer': d['answer'],
                         'category': 'cryptarithm_deduce', 'raw_output': d['raw_output'],
                         'predicted': d['answer'], 'correct': 'True'})
    with open(OUT, 'w', newline='', encoding='utf-8') as fo:
        w = csv.DictWriter(fo, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print('[out] %s : %d distinct honest crypt_deduce rows' % (OUT, len(rows)))


if __name__ == '__main__':
    main()
