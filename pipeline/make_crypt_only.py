"""Crypt-ONLY training corpus for the Tinker honest-crypt probe (rebalanced).
 = honest crypt_deduce (crypt_honest_big.csv, fixed engine) with concat CAPPED so it doesn't
   dominate, all add/sub/mul kept, + crypt_guess from the prior corpus. val-excluded.
Out: pipeline/data/crypt_honest_only.csv
"""
import csv, json, os, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
BIG = f"{ROOT}/pipeline/data/crypt_honest_big.csv"
GUESS_SRC = f"{ROOT}/pipeline/data/v16/train_crypt_twn_r11.csv"
VAL = f"{ROOT}/pipeline/data/val.jsonl"
OUT = f"{ROOT}/pipeline/data/crypt_honest_only.csv"
COLS = ["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"]
CONCAT_CAP = 350         # keep concat from flooding; arith kept in full


def fam(b):
    # classify by the QUERY operation (Step 2 apply for concat, Encode line for arithmetic)
    if 'is concatenation, so copy' in b:
        return 'concat'
    m = re.search(r'Encode the query.*?;\s*(a\*b[+-]?1?|a\+b[+-]?1?|\|a-b\||a-b|b-a)\s*=', b, re.S)
    ph = m.group(1) if m else '?'
    return 'mul' if 'a*b' in ph else 'sub' if ('a-b' in ph or 'b-a' in ph or '|' in ph) \
        else 'add' if 'a+b' in ph else 'other'


def main():
    val_ids = {json.loads(l)["id"] for l in open(VAL)}
    # bucket by family
    buckets = {'concat': [], 'add': [], 'sub': [], 'mul': []}
    for r in csv.DictReader(open(BIG)):
        if r["id"] in val_ids:
            continue
        f = fam(r["raw_output"])
        if f in buckets:
            buckets[f].append({c: r.get(c, "") for c in COLS})
    # EQUAL per sub-category = min count across the four families
    target = min(len(buckets[f]) for f in buckets)
    rows = []
    for f in buckets:
        rows.extend(buckets[f][:target])
    g = 0
    for r in csv.DictReader(open(GUESS_SRC)):
        if r["category"] == "cryptarithm_guess" and r["id"] not in val_ids and g < target:
            rows.append({c: r.get(c, "") for c in COLS}); g += 1
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"[out] {OUT}: {len(rows)} rows  BALANCED to {target}/sub-category "
          f"(available: {dict((k, len(v)) for k, v in buckets.items())}) +{g} crypt_guess")


if __name__ == "__main__":
    main()
