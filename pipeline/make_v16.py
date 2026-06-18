"""v16 corpus assembler — OPTION B: repair the executable grammars.

Lesson from v15 + the v16 probe (LOG 2026-06-11): one SFT epoch teaches a new
grammar's SURFACE but not its computations; half-installed grammars interfere
destructively. v16 therefore trains only on what the warm start already
EXECUTES:

  ALI-OWN   — every category: Ali's correct traces byte-exact (her grammar,
              her computations; training on own outputs ~= protective no-op).
  BIT-FIX   — pipeline/data/v16/bit_alifix.jsonl: HER bit grammar, but with
              correct continuations at her punt/extrapolation decision points
              (rendered by bit_stride, her own method class). Upsampled — it
              is the only behavioral delta in the corpus.

Warm-start: asalhi (0.86), NOT v15 (cross-grammar contamination).
Output: pipeline/data/v16/train_v16.csv (all_inferences.csv schema).
Val ids (900) excluded everywhere.

Usage: python3 pipeline/make_v16.py [BITFIX_UPSAMPLE]   (default 3)
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V16 = f"{ROOT}/pipeline/data/v16"
csv.field_size_limit(10 ** 9)

def main(upsample=3):
    val_ids = {json.loads(l)['id'] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}
    out_rows, stats = [], {}

    # --- ALI-OWN: every correct trace byte-exact, EXCEPT bit (re-rendered in
    #     the compressed dialect below — never mix two dialects of one grammar) ---
    for r in csv.DictReader(open(f"{ROOT}/analysis/ali_validation/all_inferences.csv")):
        if r['correct'] != 'True' or r['id'] in val_ids:
            continue
        if r['category'] == 'bit_manipulation':
            continue
        out_rows.append(r)
        stats[r['category']] = stats.get(r['category'], 0) + 1

    # --- BIT: single compressed dialect — alipass (her wins re-rendered, 1x)
    #     + alifix (her fails repaired, upsampled: the behavioral delta) ---
    def _load(name):
        p = f"{V16}/{name}.jsonl"
        if not os.path.exists(p):
            print(f"[WAIT] {p} missing — run pipeline/synth/bitmanip_alifix.py first")
            sys.exit(1)
        return [json.loads(l) for l in open(p) if json.loads(l)['id'] not in val_ids]

    for j in _load('bit_alipass'):
        out_rows.append({"id": j['id'], "prompt": j['prompt'], "answer": j['final'],
                         "category": j['category'], "raw_output": j['cot'] + "\n</think>",
                         "predicted": j['final'], "correct": "True"})
        stats['bit_alipass'] = stats.get('bit_alipass', 0) + 1

    n_fix = 0
    fixes = _load('bit_alifix')
    for k in range(upsample):
        for j in fixes:
            out_rows.append({
                "id": f"{j['id']}-u{k}" if k else j['id'],
                "prompt": j['prompt'], "answer": j['final'],
                "category": j['category'],
                "raw_output": j['cot'] + "\n</think>",
                "predicted": j['final'], "correct": "True",
            })
            n_fix += 1
    stats['bit_alifix(xups)'] = n_fix

    # --- gates ---
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/competition_dataset/tokenizer.json")
    too_long = []
    for r in out_rows:
        n = len(tok.encode(r['prompt'] + r['raw_output']).ids) + 90
        if n > 8100:
            too_long.append((r['id'], r['category'], n))
    leaked = [r['id'] for r in out_rows
              if r['id'].split('-u')[0] in val_ids or r['id'] in val_ids]

    print(f"=== v16 corpus: {len(out_rows)} rows (bitfix upsample x{upsample}) ===")
    for c in sorted(stats):
        print(f"  {c:26s} {stats[c]:5d}")
    print(f"holdout leak: {len(leaked)} (must be 0) | >8100tok: {len(too_long)}: {too_long[:3]}")
    if leaked:
        sys.exit(1)

    os.makedirs(V16, exist_ok=True)
    with open(f"{V16}/train_v16.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category",
                                          "raw_output", "predicted", "correct"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {V16}/train_v16.csv  (steps at eff_batch 32 = {len(out_rows)//32})")

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
