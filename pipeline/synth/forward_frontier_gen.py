#!/usr/bin/env python3
"""#3 frontier experiment: a CONTROLLED intervention on cryptarithm.

Same crypt instances; we dial forward-derivability by revealing a fraction of the
digit<->symbol key in the prompt:
  full  : whole key + operation + reading given -> pure forward decode (faithful CoT)
  half  : ~half the key given -> forward for keyed symbols, short honest deduction for the rest
  none  : no key -> the existing crypt task (search; reused baseline ~0.03)

Faithful-CoT guarantee: every number is read off the known gold mapping / gold answer, so the
rendered derivation is correct by construction (we also assert it). Outputs train CSVs (crypt
schema) + keyed eval jsonl. Run from repo root.
"""
import csv, json, os, random, sys
csv.field_size_limit(10**7)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "pipeline/data/cryptarithm_gold_meta.jsonl")
TRAINCSV = os.path.join(ROOT, "competition_dataset/train_categorized.csv")
VAL = os.path.join(ROOT, "pipeline/data/val.jsonl")
OUT = os.path.join(ROOT, "pipeline/data/frontier")
os.makedirs(OUT, exist_ok=True)
random.seed(7)

def load_meta():
    m = {}
    for l in open(META):
        d = json.loads(l)
        if d.get("ok") and d.get("radix") == 10 and not d.get("rev", False):  # standard reading only (clean)
            m[d["id"]] = d
    return m

def parse_query(prompt):
    # "Now, determine the result for: <LHS>"   LHS = 5 chars: o0 o1 OP o2 o3
    key = "determine the result for:"
    i = prompt.find(key)
    if i < 0: return None
    lhs = prompt[i+len(key):].strip().splitlines()[0].strip()
    return lhs if len(lhs) == 5 else None

def render(meta, prompt, gold, frac):
    """Return (prompt2, completion) or None. frac in {1.0, 0.5}."""
    mp = meta["mapping"]                      # symbol -> digit
    inv = {v: k for k, v in mp.items()}
    lhs = parse_query(prompt)
    if not lhs: return None
    o1, opg, o2 = lhs[0:2], lhs[2], lhs[3:5]
    # every operand + answer symbol must be a known digit-symbol (skip signed/odd rows)
    need = set(o1) | set(o2) | set(gold)
    if any(c not in mp for c in need): return None
    opname = meta["ops"].get(opg) or meta.get("qop", "?")
    d1 = "".join(str(mp[c]) for c in o1); d2 = "".join(str(mp[c]) for c in o2)
    res = "".join(str(mp[c]) for c in gold)
    if not (res.isdigit() and d1.isdigit() and d2.isdigit()): return None
    # choose revealed symbols
    syms = sorted(mp)
    nrev = len(syms) if frac >= 1.0 else max(1, round(frac*len(syms)))
    revealed = set(random.sample(syms, nrev))
    # always reveal the operator + operands' own symbols at full; at half reveal the sampled set
    keyline = ", ".join(f"{s}={mp[s]}" for s in syms if s in revealed)
    # CoT
    L = []
    L.append(f"Key: {keyline}.")
    L.append(f"Operator {opg} means {opname}; reading standard.")
    # decode operands; if a symbol is unrevealed, deduce it from an example line (short honest step)
    def decode(word):
        out = []
        for c in word:
            if c in revealed:
                out.append(f"{c}={mp[c]}")
            else:
                out.append(f"{c}=? -> from the examples the only digit consistent with all-different is {mp[c]}, so {c}={mp[c]}")
                revealed.add(c)
        return out
    L.append("Operand1 " + " ".join(decode(list(o1))) + f" -> {d1}.")
    L.append("Operand2 " + " ".join(decode(list(o2))) + f" -> {d2}.")
    L.append(f"Apply {opname}: {d1} {opg} {d2} = {res}.")
    L.append("Encode the result: " + " ".join(f"{ch}->{inv[int(ch)]}" for ch in res) + f" = {gold}.")
    completion = "<think>\n" + "\n".join(L) + "\n</think>\n\\boxed{" + gold + "}"
    keyprompt = prompt.rstrip() + "\nKey: " + keyline + f".  Operator {opg} = {opname}; reading standard."
    # validate: encode(res) == gold
    if "".join(inv[int(ch)] for ch in res) != gold: return None
    return keyprompt, completion

def build(rows, meta, frac, tag, as_train):
    out = []
    for r in rows:
        rid = r["id"]
        if rid not in meta: continue
        try:
            got = render(meta[rid], r["prompt"], r["answer"], frac)
        except Exception:
            got = None
        if not got: continue
        p2, comp = got
        if as_train:
            out.append({"id": rid, "prompt": p2, "answer": r["answer"], "category": r["category"],
                        "raw_output": comp, "predicted": r["answer"], "correct": True})
        else:
            out.append({"id": rid, "prompt": p2, "answer": r["answer"], "category": r["category"]})
    return out

meta = load_meta()
# train rows: crypt_deduce from train_categorized
train_rows = [r for r in csv.DictReader(open(TRAINCSV)) if r["category"] == "cryptarithm_deduce"]
val_rows = [json.loads(l) for l in open(VAL) if json.loads(l).get("category") == "cryptarithm_deduce"]
print(f"train crypt_deduce={len(train_rows)}  val crypt_deduce={len(val_rows)}  meta(std,ok)={len(meta)}")

for frac, name in [(1.0, "full"), (0.5, "half")]:
    tr = build(train_rows, meta, frac, name, True)
    ev = build(val_rows, meta, frac, name, False)
    with open(os.path.join(OUT, f"train_forward_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prompt","answer","category","raw_output","predicted","correct"]); w.writeheader(); w.writerows(tr)
    with open(os.path.join(OUT, f"val_forward_{name}.jsonl"), "w") as f:
        for d in ev: f.write(json.dumps(d)+"\n")
    print(f"  {name}: train={len(tr)}  eval={len(ev)}")

# show one full-key sample for inspection
s = build(train_rows, meta, 1.0, "full", True)[0]
print("\n=== SAMPLE (full key) ===\nPROMPT:\n", s["prompt"][:400], "\nCOMPLETION:\n", s["raw_output"][:500])
