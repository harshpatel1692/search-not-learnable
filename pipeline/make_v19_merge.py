"""v19 Kaggle corpus = v17 (proven 0.85, ALL categories) with cryptarithm_deduce REPLACED by the new
HONEST crypt synth (crypt_column.py engine: concat-decision + borrow/cast/factor + refute operators,
terse). Everything else (easy-4, bit, eq, crypt_guess) kept byte-for-byte from v17.

Honest caveat (proven on Tinker, 4 configs): the honest crypt arithmetic does NOT learn on this base
(crypt_deduce ~0.05 = concat floor; 76 finished-arithmetic, 0 correct). So v19 ~= v17 on the LB. It is a
free from-base Kaggle run; select the better of v17/v19. bit kept as v17's (bit-STaR was neutral/regressive
in v18=0.84, so NOT added).

raw_output format matched to the kernel: '{cot body}\n</think>' (no \\boxed in body; answer in `answer`).
Out: kaggle_datasets/nemotron-v19-data/all_inferences.csv
"""
import csv, json, os, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
V17 = f"{ROOT}/kaggle_datasets/nemotron-v17-data/all_inferences.csv"
HONEST = f"{ROOT}/pipeline/data/crypt_honest_big.csv"
VAL = f"{ROOT}/pipeline/data/val.jsonl"
OUT_DIR = f"{ROOT}/kaggle_datasets/nemotron-v19-data"
COLS = ["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"]
CONCAT_CAP = 200          # keep concat from flooding; all add/sub/mul kept


def fam(b):
    if 'is concatenation, so copy' in b:
        return 'concat'
    m = re.search(r'Encode the query.*?;\s*(a\*b[+-]?1?|a\+b[+-]?1?|\|a-b\||a-b|b-a)\s*=', b, re.S)
    ph = m.group(1) if m else '?'
    return 'mul' if 'a*b' in ph else 'sub' if ('a-b' in ph or 'b-a' in ph or '|' in ph) \
        else 'add' if 'a+b' in ph else 'other'


EQ_GUESS = f"{ROOT}/pipeline/data/v15/eq_guess.jsonl"
CRYPT_GUESS = f"{ROOT}/pipeline/data/crypt_r5/crypt_guess.jsonl"


def _strip_boxed(cot):
    return re.split(r'\n*\\boxed\{', cot, 1)[0].rstrip()


def _jsonl_rows(path, cat, idpfx, val_ids):
    out = []
    for l in open(path, encoding='utf-8'):
        d = json.loads(l)
        if d.get('id') in val_ids:
            continue
        body = _strip_boxed(str(d['cot']))
        out.append({"id": idpfx + str(d.get('id', len(out))), "prompt": d['prompt'], "answer": str(d['final']),
                    "category": cat, "raw_output": body + "\n</think>", "predicted": str(d['final']),
                    "correct": "True"})
    return out


def main():
    val_ids = {json.loads(l)["id"] for l in open(VAL)}
    v17 = list(csv.DictReader(open(V17)))
    # keep v17 BIG categories; the small/crypt cats are replaced with scaled/honest synth below
    drop = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}
    rows = [{c: r.get(c, "") for c in COLS} for r in v17 if r["category"] not in drop]
    kept = Counter(r["category"] for r in rows)
    # scaled small categories
    eqg = _jsonl_rows(EQ_GUESS, "equation_numeric_guess", "eqg-", val_ids)
    cg = _jsonl_rows(CRYPT_GUESS, "cryptarithm_guess", "cg-", val_ids)
    rows += eqg + cg
    print(f"  scaled small cats: eq_guess {len(eqg)} (was 99), crypt_guess {len(cg)} (was 139)")
    # honest crypt_deduce (synthetic -> no val leakage; cap concat, keep all arithmetic)
    concat = 0; added = Counter(); boxed_warn = 0
    for r in csv.DictReader(open(HONEST)):
        if r["id"] in val_ids:
            continue
        f = fam(r["raw_output"])
        if f == 'concat':
            if concat >= CONCAT_CAP:
                continue
            concat += 1
        if "\\boxed" in r["raw_output"]:
            boxed_warn += 1; continue
        ro = r["raw_output"].rstrip() + "\n</think>"          # match v17 kernel format
        rows.append({"id": "h-" + r["id"], "prompt": r["prompt"], "answer": r["answer"],
                     "category": "cryptarithm_deduce", "raw_output": ro,
                     "predicted": r["answer"], "correct": "True"})
        added[f] += 1
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/all_inferences.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    json.dump({"title": "nemotron-v19-data", "id": "harshpatel1692/nemotron-v19-data",
               "licenses": [{"name": "CC0-1.0"}]}, open(f"{OUT_DIR}/dataset-metadata.json", "w"), indent=2)
    print(f"[v19] {len(rows)} rows -> {OUT_DIR}/all_inferences.csv")
    print(f"  kept from v17 (non-crypt_deduce): {dict(kept)}")
    print(f"  NEW honest cryptarithm_deduce: {dict(added)} (total {sum(added.values())}, concat capped {CONCAT_CAP})")
    if boxed_warn:
        print(f"  [warn] dropped {boxed_warn} honest rows with stray \\boxed in body")


if __name__ == "__main__":
    main()
