"""900-row LB-oracle eval via Tinker sampling (tinker 0.22.3).

Greedy (temp 0, max_tokens 7680) over pipeline/data/val.jsonl prompts
(+ EVAL_SUFFIX, grader template — Phase-0 proved HF template == Tinker
renderer), scored with the OFFICIAL extract+verify (NOT the strict reward —
this estimates the real LB), aggregated per category with train-distribution
weights (train ≡ test).

Cost ~ $0.70/run list. Outputs: per-cat table + weighted-LB estimate,
tinker/evals/<tag>.csv, RUNS.md row.

Run:
  ~/.venvs/tinker/bin/python tinker/eval_oracle.py --ckpt tinker://...   # adapter
  ~/.venvs/tinker/bin/python tinker/eval_oracle.py --base               # base model

NOTE: near checkpoint-selection time, ALSO eval the CONVERTED adapter on
Kaggle (SVD cost included) — this script alone is directional, not final.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tinker"))
from prompts import EVAL_SUFFIX  # byte-identical to grader

RUNS = os.path.join(ROOT, "tinker", "RUNS.md")
EVALS_DIR = os.path.join(ROOT, "tinker", "evals")
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MAX_TOKENS = 7680
csv.field_size_limit(10 ** 9)


# ---- OFFICIAL metric, verbatim from kaggle_kernels/v15_eval cell 4 ----
def extract_final_answer(text):
    if text is None:
        return "NOT_FOUND"
    starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, m in enumerate(starts):
        s = m.end()
        e = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[s:e]
        lb = seg.rfind("}")
        matches.append(seg[:lb] if lb != -1 else seg)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    for pattern in [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]:
        m = re.findall(pattern, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer, predicted):
    stored = str(stored_answer).strip()
    pred = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == stored.lower()
# ----------------------------------------------------------------------


def category_weights():
    """Test-distribution weights = train distribution (train ≡ test)."""
    counts = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "competition_dataset/train_categorized.csv"))):
        counts[r["category"]] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="tinker:// sampler-weights path")
    ap.add_argument("--base", action="store_true", help="eval the raw base model")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--tag", default=None, help="output tag (default: derived)")
    ap.add_argument("--max-rows", type=int, default=None, help="debug subsample")
    ap.add_argument("--cats", default=None,
                    help="comma-list category filter (e.g. cryptarithm_deduce,"
                         "cryptarithm_guess); weighted estimate covers only these")
    ap.add_argument("--rows-file", default=None,
                    help="alternate rows jsonl (schema like val.jsonl: "
                         "id/category/prompt/answer); default pipeline/data/val.jsonl")
    args = ap.parse_args()
    if not args.ckpt and not args.base:
        ap.error("need --ckpt tinker://... or --base")

    rows_path = args.rows_file or os.path.join(ROOT, "pipeline/data/val.jsonl")
    rows = [json.loads(l) for l in open(rows_path)]
    if args.cats:
        keep = set(args.cats.split(","))
        rows = [r for r in rows if r["category"] in keep]
    if args.max_rows:
        rows = rows[:args.max_rows]
    tag = args.tag or (("base" if args.base else
                        re.sub(r"[^A-Za-z0-9_-]+", "_", args.ckpt.split("/")[-1]))
                       + time.strftime("-%m%d%H%M"))

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(args.base_model)
    sc = tinker.ServiceClient()
    sampler = (sc.create_sampling_client(base_model=args.base_model) if args.base
               else sc.create_sampling_client(model_path=args.ckpt))
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0)

    print(f"[eval] {len(rows)} rows, greedy, max_tokens {MAX_TOKENS}, tag {tag}",
          flush=True)
    futs = []
    prefill_tok = 0
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefill_tok += len(ids)
        futs.append(sampler.sample(tinker.ModelInput.from_ints(ids),
                                   num_samples=1, sampling_params=params))

    os.makedirs(EVALS_DIR, exist_ok=True)
    out_csv = os.path.join(EVALS_DIR, f"{tag}.csv")
    per_cat = defaultdict(lambda: [0, 0, 0])  # n, correct, truncated
    sample_tok = 0
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "gold", "pred", "correct", "truncated", "n_tokens"])
        for i, (r, fut) in enumerate(zip(rows, futs)):
            seq = fut.result().sequences[0]
            sample_tok += len(seq.tokens)
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            truncated = seq.stop_reason == "length"
            pred = extract_final_answer(text)
            ok = bool(verify(r["answer"], pred))
            c = per_cat[r["category"]]
            c[0] += 1; c[1] += ok; c[2] += truncated
            w.writerow([r["id"], r["category"], r["answer"], pred, int(ok),
                        int(truncated), len(seq.tokens)])
            if (i + 1) % 50 == 0:
                done = sum(v[0] for v in per_cat.values())
                acc = sum(v[1] for v in per_cat.values()) / done
                print(f"[eval] {done}/{len(rows)} acc-so-far {acc:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    weights = category_weights()
    print(f"\n{'category':<26}{'n':>5}{'acc':>8}{'trunc':>7}{'weight':>8}")
    weighted = 0.0
    for cat in sorted(per_cat):
        n, ok, tr = per_cat[cat]
        acc = ok / n
        weighted += acc * weights.get(cat, 0)
        print(f"{cat:<26}{n:>5}{acc:>8.3f}{tr/n:>7.2f}{weights.get(cat,0):>8.3f}")
    cost = (prefill_tok * 0.13 + sample_tok * 0.33) / 1e6
    print(f"\nWEIGHTED-LB ESTIMATE: {weighted:.4f}")
    print(f"[cost] prefill {prefill_tok/1e6:.2f}M + sample {sample_tok/1e6:.2f}M "
          f"~= ${cost:.2f} list / ${cost/2:.2f} discounted")
    print(f"[out] {out_csv}")
    with open(RUNS, "a") as f:
        f.write(f"| eval {tag} | {time.strftime('%F %T')} | "
                f"ckpt={'BASE' if args.base else args.ckpt} | "
                f"weighted {weighted:.4f} | ${cost:.2f} | {out_csv} |\n")


if __name__ == "__main__":
    main()
