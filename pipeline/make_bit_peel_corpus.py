"""Peel-only SFT corpus: peel->2-tap CoT for peelable (XNOR/XOR-headed) 3-tap problems. All synth (fresh
taps+inputs) => zero leakage vs bit_eval500. Output pipeline/data/v16/train_bit_peel.csv.
  python3 pipeline/make_bit_peel_corpus.py [n] [seed]"""
import os, sys, csv, random, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'synth'))
import bitmanip_peel as P
from cryptarithm_cot import tokenizer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 41
csv.field_size_limit(10 ** 9)

def main():
    rng = random.Random(SEED); tok = tokenizer(); rows = []; seen = set()
    ev_prompts = set()
    import json
    for l in open(os.path.join(ROOT, 'pipeline/data/bit_eval500.jsonl')):
        ev_prompts.add(json.loads(l)['prompt'])
    tries = 0
    while len(rows) < N and tries < N * 40:
        tries += 1
        r = P.gen_peel(rng)
        if r['prompt'] in seen or r['prompt'] in ev_prompts: continue
        if r['cot'].split('Result:')[-1].strip().rstrip('.') != r['answer']: continue
        seen.add(r['prompt'])
        rows.append({'id': f'peel-{SEED}-{len(rows):05d}', 'prompt': r['prompt'], 'answer': r['answer'],
                     'category': 'bit_manipulation', 'raw_output': r['cot'], 'predicted': r['answer'], 'correct': 'True'})
    rng.shuffle(rows)
    out = os.path.join(ROOT, 'pipeline/data/v16/train_bit_peel.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'category', 'raw_output', 'predicted', 'correct'])
        w.writeheader(); w.writerows(rows)
    nt = [len(tok.encode(r['raw_output']).ids) for r in rows]
    print(f"[out] {out}: {len(rows)} rows | ntok median {int(statistics.median(nt))} p95 {sorted(nt)[int(len(nt)*.95)]} max {max(nt)}")

if __name__ == '__main__':
    main()
