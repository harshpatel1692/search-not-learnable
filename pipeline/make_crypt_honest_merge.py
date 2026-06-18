"""Build the crypt-honest training corpus for the Tinker probe.

= v17 corpus (all categories) with crypt_deduce UPGRADED to honest traces:
  - keep v17's non-crypt_deduce rows  (preserves easy-4 / bit / eq / crypt_guess incl. the concat slice)
  - keep v17's crypt_deduce rows EXCEPT where we have an honest trace for that id (prefer honest)
  - ADD all honest crypt_deduce rows (synthetic + real-non-val) from crypt_honest.csv
  - EXCLUDE every val.jsonl id everywhere (clean held-out eval, zero leakage)

Out: pipeline/data/crypt_honest_train.csv   (sft_warmstart.py --data)
"""
import csv, json, os
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
V17 = f"{ROOT}/kaggle_datasets/nemotron-v17-data/all_inferences.csv"
HONEST = f"{ROOT}/pipeline/data/crypt_honest.csv"
VAL = f"{ROOT}/pipeline/data/val.jsonl"
OUT = f"{ROOT}/pipeline/data/crypt_honest_train.csv"
COLS = ["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"]


def main():
    val_ids = {json.loads(l)["id"] for l in open(VAL)}
    honest = list(csv.DictReader(open(HONEST)))
    honest_ids = {r["id"] for r in honest}
    rows = []; leak = 0
    for r in csv.DictReader(open(V17)):
        if r["id"] in val_ids:
            leak += 1; continue
        if r["category"] == "cryptarithm_deduce" and r["id"] in honest_ids:
            continue                       # replaced by honest version below
        rows.append({c: r.get(c, "") for c in COLS})
    for r in honest:
        if r["id"] in val_ids:
            leak += 1; continue
        rows.append({c: r.get(c, "") for c in COLS})
    assert leak == 0 or True  # report below
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    c = Counter(r["category"] for r in rows)
    print(f"[out] {OUT}: {len(rows)} rows (val-excluded; {leak} val rows dropped)")
    print(f"  crypt_deduce now: {c['cryptarithm_deduce']} ({len(honest)} honest)")
    print(f"  full dist: {dict(c)}")


if __name__ == "__main__":
    main()
