"""v15 corpus assembler.

Combines:
  RETENTION  — Ali 0.86 adapter's correct CoTs, BYTE-EXACT, for gravity/numeral/unit_conversion
               (these grammars are proven at 99.7-100% and the warm-start already speaks them)
  REPLACED   — our solver-rendered traces from pipeline/data/v15/*.jsonl for
               cipher (vocab-snap + final-redecode fix), bit (global-tap grammar),
               cryptarithm (tiered arithmetic crack), equation_numeric (early-stop + fixed prior)

Output: pipeline/data/v15/train_v15.csv in the exact all_inferences.csv schema the
v6_train notebook consumes (id,prompt,answer,category,raw_output,predicted,correct).
raw_output = cot + "\n</think>" (notebook splits on </think> and re-wraps).

All real-train rows in pipeline/data/val.jsonl (900 = 100/cat holdout) are EXCLUDED.
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V15 = f"{ROOT}/pipeline/data/v15"
csv.field_size_limit(10**8)

RETAIN_CATS = {"gravity", "numeral", "unit_conversion"}
REPLACE_FILES = [  # (file, expected category prefix)
    "cipher_real.jsonl", "cipher_synth.jsonl",
    "bit_real.jsonl", "bit_synth.jsonl",
    "crypt_deduce_real.jsonl", "crypt_deduce_synth.jsonl", "crypt_guess.jsonl",
    "eq_deduce_real.jsonl", "eq_deduce_synth.jsonl", "eq_guess.jsonl",
]

def main():
    val_ids = {json.loads(l)['id'] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}
    out_rows, stats = [], {}

    # --- retention slice (byte-exact Ali traces) ---
    for r in csv.DictReader(open(f"{ROOT}/analysis/ali_validation/all_inferences.csv")):
        if r['category'] not in RETAIN_CATS or r['correct'] != 'True' or r['id'] in val_ids:
            continue
        out_rows.append(r)  # untouched
        stats[r['category']] = stats.get(r['category'], 0) + 1

    # --- replaced slices ---
    missing = []
    for fn in REPLACE_FILES:
        path = f"{V15}/{fn}"
        if not os.path.exists(path):
            missing.append(fn); continue
        for line in open(path):
            j = json.loads(line)
            if j['id'] in val_ids:
                continue
            out_rows.append({
                "id": j['id'], "prompt": j['prompt'], "answer": j['final'],
                "category": j['category'],
                "raw_output": j['cot'] + "\n</think>",
                "predicted": j['final'], "correct": "True",
            })
            stats[j['category']] = stats.get(j['category'], 0) + 1

    # --- validation gates ---
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/competition_dataset/tokenizer.json")
    too_long, n_checked = [], 0
    for r in out_rows:
        # full training seq ≈ prompt + cot + template overhead (~80 tok)
        n = len(tok.encode(r['prompt'] + r['raw_output']).ids) + 90
        n_checked += 1
        if n > 8100:
            too_long.append((r['id'], r['category'], n))
    leaked = [r['id'] for r in out_rows if r['id'] in val_ids]

    print(f"=== v15 corpus: {len(out_rows)} rows ===")
    for c in sorted(stats):
        print(f"  {c:26s} {stats[c]:5d}")
    if missing:
        print(f"[WAIT] missing renderer outputs: {missing}")
    print(f"holdout leak check: {len(leaked)} (must be 0)")
    print(f"seq-len >8100tok: {len(too_long)}: {too_long[:5]}")
    if leaked or (not missing and not out_rows):
        sys.exit(1)

    with open(f"{V15}/train_v15.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category",
                                          "raw_output", "predicted", "correct"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {V15}/train_v15.csv  (NUM_STEPS at eff_batch 32 = {len(out_rows)//32})")

if __name__ == '__main__':
    main()
