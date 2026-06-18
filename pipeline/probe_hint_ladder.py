"""Hint LADDER on a single cryptarithm_deduce example: escalate the system-message hint
from none -> structure -> method -> rule -> answer, and at each rung record whether the
120B converges to gold AND whether its CoT cites the hint. Goal: find the lowest rung that
yields a CORRECT, hint-free trace (the usable training fuel).

Run:  python3 pipeline/probe_hint_ladder.py            (uses seed-42 first crypt_deduce row)
"""
import csv, json, math, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline")); csv.field_size_limit(10**9)
import nvidia_api

def verify(s, p):
    s, p = str(s).strip(), str(p).strip()
    if re.fullmatch(r"[01]+", s): return p.lower() == s.lower()
    try: return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return p.lower() == s.lower()

LEAK = ["hint","private","solution key","as given","as stated","we are told","told that",
        "given that the answer","provided","the key","for your eyes only","instruction said",
        "i was given","i know the answer","supposed to be"]
def leak_flags(txt):
    t = (txt or "").lower()
    return [ph for ph in LEAK if ph in t]

# ---- the example (seed-42 first cryptarithm_deduce row) ----
rows = [r for r in csv.DictReader(open(ROOT/"competition_dataset/train_categorized.csv", encoding="utf-8"))]
cd = [r for r in rows if r["category"] == "cryptarithm_deduce"]
random.Random(42).shuffle(cd)
R = cd[0]; PROMPT, GOLD = R["prompt"], R["answer"]

# ---- the LADDER (each rung adds more of the solution trace; none reveal the gold except L4) ----
LADDER = [
 ("L0 none", "You are an expert puzzle solver. Infer the hidden transformation rule from the "
  "worked examples by careful step-by-step deduction, verify it on every example, then apply "
  "it to the query. Put the final result in \\boxed{}."),

 ("L1 structure", "You are an expert puzzle solver. This is a digit-substitution cipher over "
  "arithmetic: each distinct glyph maps to exactly ONE digit 0-9 (injective). Each example is "
  "`g1 g2 OP g3 g4 = RESULT` (positions 1,2,4,5 digit-glyphs, position 3 the operator glyph); "
  "operands are 2-digit, no leading zero; numbers may be read MSB-first or reversed. Deduce the "
  "map and operator from the examples, verify on every example, apply to the query. Put the "
  "final result (in the puzzle's glyphs) in \\boxed{}."),

 ("L2 +juxtaposition", "You are an expert puzzle solver. This is a digit/glyph cipher: each "
  "example is `g1 g2 OP g3 g4 = RESULT`. IMPORTANT: an operator does not always mean arithmetic "
  "— some operators denote JUXTAPOSITION, i.e. the result is simply the two operands written "
  "side by side (the operator glyph dropped). ALWAYS first check whether RESULT equals the "
  "operand glyphs concatenated; if not, fall back to arithmetic. Determine what the query's "
  "operator does from the examples that use it, then apply it. Put the result in \\boxed{}."),

 ("L3 rule (no answer)", "You are an expert puzzle solver. Each example is `g1 g2 OP g3 g4 = "
  "RESULT`. The query's operator appears in the examples and there it acts as CONCATENATION: the "
  "result is the first operand's two glyphs followed by the second operand's two glyphs, with the "
  "operator removed (no arithmetic, no digit decoding needed). Confirm this on the matching "
  "example, then apply the same to the query. Put the result in \\boxed{}."),

 ("L4 answer", "You are an expert puzzle solver.\nPRIVATE SOLUTION KEY — for your eyes only. NEVER "
  "mention, quote, hint at, or allude to this key or the existence of any hint anywhere in your "
  f"reasoning or answer: the correct final result for the query is `{GOLD}`.\nSolve as if from "
  "scratch: infer the rule from the examples by genuine deduction, verify on every example, apply "
  "to the query; your derivation must independently arrive at the result above. Put it in \\boxed{}."),
]

nvidia_api.set_experiment("hint_ladder_60ed3f31")
print(f"EXAMPLE id={R['id']}  query in prompt; GOLD={GOLD!r}")
print(f"PROMPT:\n{PROMPT}\n" + "="*80)
out = open("/tmp/hint_ladder_full.jsonl", "w")
for label, sysmsg in LADDER:
    t0 = time.time()
    res = nvidia_api.ask(PROMPT, model="nvidia/nemotron-3-super-120b-a12b",
                         max_tokens=8000, temperature=0.6, reasoning_budget=3500, system=sysmsg,
                         meta={"rung": label, "gold": GOLD})
    rea, con = res.get("reasoning",""), res.get("content","")
    ans, err = res.get("answer","NOT_FOUND"), res.get("error")
    ok = bool(verify(GOLD, ans)) if not err else False
    lks = leak_flags(rea + "\n" + con)
    ntok = (res.get("usage") or {}).get("completion_tokens")
    out.write(json.dumps(dict(rung=label, gold=GOLD, answer=ans, correct=ok, leaks=lks,
                              n_tokens=ntok, error=err, reasoning=rea, content=con))+"\n"); out.flush()
    print(f"\n{'#'*80}\n{label}  ->  answer={ans!r}  gold={GOLD!r}  "
          f"{'CONVERGED' if ok else 'WRONG'}{' | LEAK:'+','.join(lks) if lks else ' | clean'}"
          f"{' | ERR='+err if err else ''}  tok={ntok}  ({time.time()-t0:.0f}s)")
    print(f"--- CoT head (600c) ---\n{rea[:600]}")
    print(f"--- CoT tail (500c) ---\n{rea[-500:] if rea else '(no reasoning stream)'}")
    print(f"--- final content ---\n{con[:400]}")
print(f"\n[done] full traces -> /tmp/hint_ladder_full.jsonl")
out.close()
