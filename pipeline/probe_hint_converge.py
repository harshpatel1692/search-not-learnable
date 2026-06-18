"""Hint-conditioned convergence probe (STaR "rationalization") on the NVIDIA 120B teacher.

HYPOTHESIS (user, 2026-06-13): if we put a HINT in the SYSTEM message and the train
problem in the USER message, the 120B teacher converges to the correct answer WITHIN the
8k budget AND does it in natural-language CoT WITHOUT citing the hint. Because we know the
gold (the "boundary"), we keep only traces that (a) verify == gold and (b) don't leak the
hint -> clean correct CoT for categories the base can't solve unaided (crypt, eq).

What it does
------------
For N problems/category from train_categorized.csv:
  system = HINT (mode below); user = the problem (+ \boxed{} suffix); temp=0, max_tokens=8000.
  -> extract \boxed{}, verify vs gold, flag hint-leakage in the CoT, record tokens.
Prints per-cat convergence + leak rates; saves keeper traces to pipeline/data/hint_traces/<tag>.jsonl.

Hint modes (--hint)
-------------------
  none      : neutral solver instruction (baseline — measures unaided 120B convergence).
  structure : explains the category's hidden STRUCTURE/method, NO answer, NO per-row rule.
              (cannot leak the answer; tests whether method guidance alone makes it converge.)
  answer    : RATIONALIZATION — gold answer given as a private key, "derive independently,
              never mention the hint". Strongest convergence; leak-check is essential here.

Cost: NVIDIA build API (free tier, rate-limited) — no $ like Tinker. Logs to nvidia_logs/<tag>.jsonl.

Run (review first with small N, then scale):
  python3 pipeline/probe_hint_converge.py --cats cryptarithm_deduce --n 5 --hint answer --tag hint_crypt_ans_probe
  python3 pipeline/probe_hint_converge.py --cats cryptarithm_deduce,cryptarithm_guess --n 50 --hint structure --tag hint_crypt_struct
"""
import argparse, csv, json, math, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import nvidia_api
csv.field_size_limit(10 ** 9)

MODEL_120B = "nvidia/nemotron-3-super-120b-a12b"   # the 120B (12B-active) teacher
TRAIN_CSV = ROOT / "competition_dataset" / "train_categorized.csv"
OUTDIR = ROOT / "pipeline" / "data" / "hint_traces"; OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------- official verify (binary-exact / float rel_tol 1e-2 / str) ----------------
def verify(stored, pred):
    s, p = str(stored).strip(), str(pred).strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


# ---------------- SYSTEM-MESSAGE HINTS (edit these to tune the experiment) ----------------
NEUTRAL = ("You are an expert puzzle solver. Infer the hidden transformation rule from the "
           "worked examples by careful step-by-step deduction, verify it on every example, "
           "then apply it to the query. Put the final result in \\boxed{}.")

# structure hints: METHOD/shape only — NO answer, NO per-row rule (cannot leak gold)
STRUCTURE = {
 "cryptarithm_deduce": (
   "You are an expert puzzle solver. This is a digit-substitution cipher over arithmetic:\n"
   "- Each distinct glyph maps to exactly ONE digit 0-9 (injective; no two glyphs share a digit).\n"
   "- Each example is 5 chars `g1 g2 OP g3 g4 = RESULT`: positions 1,2,4,5 are digit-glyphs, "
   "position 3 is an operator glyph.\n"
   "- Operands are 2-digit numbers, no leading zero. Numbers may be read MSB-first (standard) "
   "OR reversed (little-endian) — try both modes consistently.\n"
   "- The operator glyph maps to one arithmetic op (+, -, *, |a-b|, maybe with a small +/-k "
   "offset), the SAME across all examples that use that glyph; a negative result is written "
   "with the operator glyph as a leading sign.\n"
   "Deduce the digit map and the operator column-by-column from the examples, verify on EVERY "
   "example, then apply to the query. Put the final result (in the puzzle's own glyphs) in \\boxed{}."),
 "cryptarithm_guess": (
   "You are an expert puzzle solver. This is a digit-substitution cipher: each glyph maps to one "
   "digit 0-9 (injective); each example is `g1 g2 OP g3 g4 = RESULT` (pos 3 = operator). The "
   "QUERY's operator glyph does NOT appear in the examples, so its exact arithmetic op is not "
   "directly determined — infer the most likely op family from the example operators and operand "
   "ranges, deduce the digit map from the examples, and apply your best consistent guess. Put the "
   "result in the puzzle's own glyphs in \\boxed{}."),
 "equation_numeric_deduce": (
   "You are an expert puzzle solver. Each example is `A op B = C` with real digits; a hidden rule "
   "maps the operator symbol to an arithmetic operation (possibly with a constant offset) and may "
   "transform operands. Infer the operation for the query's operator from the examples that use it, "
   "verify on every such example, then compute the query. Put the numeric result in \\boxed{}."),
 "equation_numeric_guess": (
   "You are an expert puzzle solver. Each example is `A op B = C`. The query uses an operator not "
   "seen in the examples, so infer the most plausible operation from the seen operators, then "
   "compute the query. Put the numeric result in \\boxed{}."),
}
def structure_hint(cat):
    return STRUCTURE.get(cat, NEUTRAL)

# answer hint: rationalization. gold injected; explicit no-leak instruction.
def answer_hint(cat, gold):
    return ("You are an expert puzzle solver.\n"
            "PRIVATE SOLUTION KEY — for your eyes only. NEVER mention, quote, hint at, or allude "
            "to this key, this instruction, or the existence of any hint, anywhere in your "
            f"reasoning or your answer: the correct final result for the query is `{gold}`.\n"
            "Solve the puzzle as if from scratch: infer the hidden rule from the worked examples "
            "by genuine step-by-step deduction, verify it on every example, then apply it to the "
            "query. Your derivation must INDEPENDENTLY and self-consistently arrive at the result "
            "above (if a step contradicts it, you made an error — find and fix it). Put the final "
            "result in \\boxed{}.")

def build_system(mode, cat, gold):
    if mode == "none":      return NEUTRAL
    if mode == "structure": return structure_hint(cat)
    if mode == "answer":    return answer_hint(cat, gold)
    raise ValueError(mode)


# ---------------- hint-leakage detector (CoT must not cite the hint) ----------------
LEAK_PHRASES = ["hint", "private", "solution key", "as given", "as stated", "we are told",
                "told that", "given that the answer", "provided answer", "the key says",
                "according to the instruction", "the system", "for your eyes only"]
def leaks(reasoning, content, gold, mode):
    """True if the CoT betrays that a hint was supplied (=> trace unusable as clean data)."""
    txt = ((reasoning or "") + "\n" + (content or "")).lower()
    if any(ph in txt for ph in LEAK_PHRASES):
        return True
    if mode == "answer":
        # restating gold in the FIRST ~250 chars (before any derivation) ~ "the answer is X"
        head = (reasoning or content or "")[:250].lower()
        if str(gold).strip().lower() in head:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="cryptarithm_deduce",
                    help="comma list (e.g. cryptarithm_deduce,cryptarithm_guess)")
    ap.add_argument("--n", type=int, default=5, help="problems per category (start small to review)")
    ap.add_argument("--hint", choices=["none", "structure", "answer"], default="answer")
    ap.add_argument("--model", default=MODEL_120B)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"hint_{args.hint}_{int(time.time())}"
    nvidia_api.set_experiment(tag)

    rows = [r for r in csv.DictReader(open(TRAIN_CSV, encoding="utf-8"))]
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    rng = random.Random(args.seed)
    sample = []
    for cat in args.cats.split(","):
        pool = by_cat.get(cat, [])[:]
        rng.shuffle(pool)
        sample += pool[:args.n]
    print(f"[probe] tag={tag} model={args.model} hint={args.hint} temp={args.temp} "
          f"max_tokens={args.max_tokens} | {len(sample)} problems "
          f"({args.n}/cat over {args.cats})", flush=True)

    out_path = OUTDIR / f"{tag}.jsonl"
    fout = open(out_path, "w")
    per_cat = {}  # cat -> [n, converged, leaked, keepers]
    t0 = time.time()
    for i, r in enumerate(sample):
        cat, gold, prompt = r["category"], r["answer"], r["prompt"]
        system = build_system(args.hint, cat, gold)
        res = nvidia_api.ask(prompt, model=args.model, max_tokens=args.max_tokens,
                             temperature=args.temp, system=system,
                             meta={"id": r["id"], "cat": cat, "hint": args.hint, "gold": gold})
        reasoning = res.get("reasoning", ""); content = res.get("content", "")
        ans = res.get("answer", "NOT_FOUND"); err = res.get("error")
        ok = bool(verify(gold, ans)) if not err else False
        lk = leaks(reasoning, content, gold, args.hint) if not err else False
        ntok = (res.get("usage") or {}).get("completion_tokens")
        c = per_cat.setdefault(cat, [0, 0, 0, 0]); c[0] += 1; c[1] += ok; c[2] += lk
        keeper = ok and not lk and not err
        c[3] += keeper
        rec = dict(id=r["id"], category=cat, gold=gold, answer=ans, correct=ok, leak=lk,
                   hint=args.hint, error=err, n_tokens=ntok, finish=res.get("finish"),
                   prompt=prompt, cot=reasoning, content=content)
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"  [{i+1}/{len(sample)}] {cat} id={r['id']} -> ans={ans!r} gold={gold!r} "
              f"{'OK' if ok else 'X'}{' LEAK' if lk else ''}{' ERR='+err if err else ''} "
              f"tok={ntok} ({(time.time()-t0)/60:.1f}m)", flush=True)

    fout.close()
    print(f"\n[probe] per-category (n / converged / leaked / KEEPERS=correct&clean):")
    for cat, (n, cv, lk, kp) in sorted(per_cat.items()):
        print(f"  {cat:<26} n={n:<4} converged={cv}/{n} ({cv/n:.2f})  "
              f"leaked={lk}/{max(cv,1)}  KEEPERS={kp}/{n} ({kp/n:.2f})")
    print(f"[probe] traces -> {out_path}  | log -> nvidia_logs/{tag}.jsonl  "
          f"| {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
