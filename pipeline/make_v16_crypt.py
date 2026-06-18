"""Build train_v16_crypt.csv = the v16 corpus + the v15 crypt slice (x2 deduce
upsample) for the Tinker crypt-bootstrap retrain.

Rationale (LOG 2026-06-12): the Tinker from-base 2e-4 recipe installs
EXECUTABLE grammars (bit dialect .61 proves it), while v15's crypt failure was
an LR/warm-start artifact (5e-6 on Ali = surface-only). Crypt traces all open
in the attractor basin. One variable changed vs the proven v16 run: + crypt.

Output schema matches train_v16.csv (id,prompt,answer,category,raw_output,
predicted,correct); raw_output = cot (no </think> inside -> sft_warmstart's
load_rows keeps it whole). Gates: val-leak 0 (re-checked here), boxed-in-cot
allowed (grader takes LAST boxed; Ali retention/cipher slices do the same).
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)

V16 = f"{ROOT}/pipeline/data/v16/train_v16.csv"
OUT = f"{ROOT}/pipeline/data/v16/train_v16_crypt.csv"
CRYPT = [
    (f"{ROOT}/pipeline/data/v15/crypt_deduce_real.jsonl", 2),
    (f"{ROOT}/pipeline/data/v15/crypt_deduce_synth.jsonl", 2),
    (f"{ROOT}/pipeline/data/v15/crypt_guess.jsonl", 1),
]

val_ids = {json.loads(l)["id"] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}

rows = list(csv.DictReader(open(V16, encoding="utf-8")))
print(f"[v16] {len(rows)} rows")

n_added, n_leak = 0, 0
for path, times in CRYPT:
    added = 0
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        if d["id"] in val_ids:
            n_leak += 1
            continue
        for rep in range(times):
            rows.append({
                "id": f"{d['id']}-r{rep}" if rep else d["id"],
                "prompt": d["prompt"],
                "answer": str(d["final"]),
                "category": d["category"],
                "raw_output": d["cot"],
                "predicted": str(d["final"]),
                "correct": "True",
            })
            added += 1
    print(f"[crypt] {os.path.basename(path)} x{times} -> +{added}")
    n_added += added

assert n_leak == 0, f"VAL LEAK: {n_leak} crypt rows hit val ids"
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category",
                                      "raw_output", "predicted", "correct"])
    w.writeheader()
    w.writerows(rows)
from collections import Counter
print(f"[out] {OUT}: {len(rows)} rows (+{n_added} crypt), leak 0")
print(Counter(r["category"] for r in rows))
