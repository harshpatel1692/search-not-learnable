"""Crypt-ONLY corpus for the Tinker isolation loop (user directive 2026-06-12:
"train and figure out cryptarithm only; hammer it on tinker until we succeed").

Sources = the v15 crypt renders (solver-verified, lint-clean, attractor-basin
openers): crypt_deduce_real 311 + crypt_deduce_synth 700 + crypt_guess 127.
Output schema matches train_v16.csv so sft_warmstart.py consumes it directly.

python3 pipeline/make_crypt_corpus.py            # -> pipeline/data/v16/train_crypt_r8.csv
"""
import csv
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
OUT = f"{ROOT}/pipeline/data/v16/train_crypt_r8.csv"
SRC = [
    f"{ROOT}/pipeline/data/crypt_r8/crypt_deduce_real.jsonl",
    f"{ROOT}/pipeline/data/crypt_r8/crypt_deduce_synth.jsonl",
    f"{ROOT}/pipeline/data/crypt_r8/crypt_guess.jsonl",
]

val_ids = {json.loads(l)["id"] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}
rows, leak = [], 0
for path in SRC:
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        if d["id"] in val_ids:
            leak += 1
            continue
        rows.append({
            "id": d["id"], "prompt": d["prompt"], "answer": str(d["final"]),
            "category": d["category"], "raw_output": d["cot"],
            "predicted": str(d["final"]), "correct": "True",
        })
assert leak == 0, f"VAL LEAK: {leak}"
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category",
                                      "raw_output", "predicted", "correct"])
    w.writeheader()
    w.writerows(rows)
print(f"[out] {OUT}: {len(rows)} rows, leak 0")
print(Counter(r["category"] for r in rows))
