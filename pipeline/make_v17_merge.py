"""v17 merge corpus — the no-crypt-wall BANK train (from-base, Kaggle, no SVD loss).

Composition (every component has a measured number behind it; LOG 2026-06-12):
  EASY-4 + BIT — train_v16.csv minus eq/crypt rows. easy-4 proven ~.99 from base
                 (v16-warmstart oracle); bit = alipass+alifix compressed dialect,
                 measured .61 from base @1ep (0 truncations -> execution fidelity;
                 2ep is this run's controlled variable for bit).
  EQ           — v15 policy renders (95.1% first-hit policy): eq_deduce_real +
                 eq_deduce_synth + eq_guess. Replaces Ali's 458 eq rows (never mix
                 two dialects of one grammar — v15 lesson). v16 measured .80 on
                 Ali-dialect; renders target ~.90+.
  CRYPT        — ONLY the strata with evidence of transfer: r8 concat stratum
                 (transcription — the one crypt behavior the model executes) +
                 r8 guess renders (family-aware bet). NO chain/split/bail: r1-r8
                 proved value-DFS content does not transfer by SFT (deduce .03-.07).
                 Trailing \\boxed{} stripped from cot (kernel appends its own after
                 </think>; lint guaranteed boxed == final).

Output: kaggle_datasets/nemotron-v17-data/all_inferences.csv (kernel schema).
Val ids (900) excluded everywhere.

Usage: python3 pipeline/make_v17_merge.py
"""
import csv, json, os, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
OUT_DIR = f"{ROOT}/kaggle_datasets/nemotron-v17-data"
OUT = f"{OUT_DIR}/all_inferences.csv"

DROP_CATS = {"equation_numeric_deduce", "equation_numeric_guess",
             "cryptarithm_deduce", "cryptarithm_guess"}
BOX_RE = re.compile(r"\s*\\boxed\{[^}]*\}\s*$")


def main():
    val_ids = {json.loads(l)["id"] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}
    rows, stats = [], Counter()

    # --- spine: easy-4 + bit from train_v16.csv (already val-filtered at build) ---
    for r in csv.DictReader(open(f"{ROOT}/pipeline/data/v16/train_v16.csv")):
        if r["category"] in DROP_CATS:
            stats[f"dropped:{r['category']}"] += 1
            continue
        rows.append(r)
        stats[r["category"]] += 1

    def add(path, kinds=None, strip_box=False, tag=""):
        n = 0
        for l in open(path, encoding="utf-8"):
            d = json.loads(l)
            if d["id"] in val_ids:
                continue
            if kinds is not None and d.get("kind") not in kinds:
                continue
            cot = d["cot"]
            if strip_box:
                # answers are glyph strings that may contain '}' -> regex unusable;
                # the lint guarantees exactly one boxed, terminal, == final
                i = cot.rfind("\\boxed{")
                assert i != -1, d["id"]
                cot = cot[:i].rstrip()
                assert "\\boxed{" not in cot, f"inner boxed left: {d['id']}"
            rows.append({"id": d["id"], "prompt": d["prompt"],
                         "answer": str(d["final"]), "category": d["category"],
                         "raw_output": cot + "\n</think>",
                         "predicted": str(d["final"]), "correct": "True"})
            n += 1
        stats[tag or os.path.basename(path)] += n

    # --- EQ: v15 policy renders ---
    add(f"{ROOT}/pipeline/data/v15/eq_deduce_real.jsonl",  tag="eq_v15_real")
    add(f"{ROOT}/pipeline/data/v15/eq_deduce_synth.jsonl", tag="eq_v15_synth")
    add(f"{ROOT}/pipeline/data/v15/eq_guess.jsonl",        tag="eq_v15_guess")

    # --- CRYPT: r8 concat stratum + guess bets only ---
    add(f"{ROOT}/pipeline/data/crypt_r8/crypt_deduce_real.jsonl",
        kinds={"concat", "concat2"}, strip_box=True, tag="crypt_concat_real")
    add(f"{ROOT}/pipeline/data/crypt_r8/crypt_deduce_synth.jsonl",
        kinds={"concat"}, strip_box=True, tag="crypt_concat_synth")
    add(f"{ROOT}/pipeline/data/crypt_r8/crypt_guess.jsonl",
        kinds=None, strip_box=True, tag="crypt_guess_r8")

    # --- gates ---
    leaked = [r["id"] for r in rows if r["id"].split("-u")[0] in val_ids or r["id"] in val_ids]
    assert not leaked, f"VAL LEAK: {leaked[:5]}"
    dup = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
    # alifix upsample duplicates ids by design (-u suffix added); anything else is a bug
    assert not dup, f"DUP IDS: {dup[:5]}"

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/competition_dataset/tokenizer.json")
    too_long = [(r["id"], r["category"],
                 len(tok.encode(r["prompt"] + r["raw_output"]).ids) + 90)
                for r in rows]
    too_long = [t for t in too_long if t[2] > 8100]
    assert not too_long, f"OVERLEN: {too_long[:5]}"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category",
                                          "raw_output", "predicted", "correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {OUT}: {len(rows)} rows, leak 0, overlen 0")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
