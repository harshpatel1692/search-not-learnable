"""v18 merge corpus = v17 corpus + ADDITIVE bit STaR traces.

Change vs v17 (isolated, for clean LB attribution of the bit lever):
  - keep ALL v17 rows as-is (easy-4 + v16-dialect bit 1554 + v15 eq + crypt concat/guess harvest)
  - ADD the deduped bit STaR traces (model's own correct rollouts, same v16 dialect, short).
    ids suffixed '-star' so they coexist with the v16 teacher trace for the same row (additive).
NOT in v18 (deferred to v19, need renderer work): crypt_guess family-bet fix, eq_guess prior/encoding fix.

Output: kaggle_datasets/nemotron-v18-data/all_inferences.csv
Usage: python3 pipeline/make_v18_merge.py
"""
import csv, json, os
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
V17 = f"{ROOT}/kaggle_datasets/nemotron-v17-data/all_inferences.csv"
STAR_FILES = [f"{ROOT}/pipeline/data/star/bit-star-probe.csv",
              f"{ROOT}/pipeline/data/star/bit-star-rest.csv",
              f"{ROOT}/pipeline/data/star/bit-star-r2harvest.csv"]
OUT_DIR = f"{ROOT}/kaggle_datasets/nemotron-v18-data"
COLS = ["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"]


def v16ify(raw):
    """Match the v16 bit format the kernel expects: '<reasoning ... \\boxed{X}>\n</think>'
    (boxed inside the think block, nothing after). STaR raw ends '...</think>\n\\boxed{X}'."""
    head = raw.split("</think>")[0].rstrip()
    return head + "\n</think>"


def main():
    v17 = list(csv.DictReader(open(V17)))
    # dedup STaR traces by (id, first 200 chars), reformat to v16 bit schema
    seen, star = set(), []
    for f in STAR_FILES:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            k = (r["id"], r["raw_output"][:200])
            if k in seen:
                continue
            seen.add(k)
            star.append({"id": r["id"] + "-star", "prompt": r["prompt"], "answer": r["answer"],
                         "category": "bit_manipulation", "raw_output": v16ify(r["raw_output"]),
                         "predicted": r["answer"], "correct": "True"})
    rows = v17 + star

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/all_inferences.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    json.dump({"title": "nemotron-v18-data", "id": "harshpatel1692/nemotron-v18-data",
               "licenses": [{"name": "CC0-1.0"}]},
              open(f"{OUT_DIR}/dataset-metadata.json", "w"), indent=2)

    # distribution
    v17c = Counter(r["category"] for r in v17)
    allc = Counter(r["category"] for r in rows)
    print(f"v18 corpus: {len(rows)} rows  (v17 {len(v17)} + STaR {len(star)})")
    print(f"\n{'category':<26}{'v17':>8}{'+STaR':>8}{'v18':>8}")
    for cat in sorted(allc):
        print(f"{cat:<26}{v17c.get(cat,0):>8}{allc[cat]-v17c.get(cat,0):>8}{allc[cat]:>8}")
    print(f"{'TOTAL':<26}{len(v17):>8}{len(star):>8}{len(rows):>8}")
    print(f"\n-> {OUT_DIR}/all_inferences.csv")


if __name__ == "__main__":
    main()
