"""STaR / expert-iteration rollout harvester (tinker 0.22.3).

Samples K completions per prompt at temperature from a trained ckpt, keeps the
ones whose extracted answer VERIFIES against gold, and writes a train-ready CSV
(id,prompt,answer,category,raw_output,predicted,correct) that sft_warmstart.py
can consume directly (raw_output = the model's OWN correct CoT -> retraining on
it closes the exposure-bias gap that caps imitation SFT).

Only works where pass@K is healthy (bit: yes; crypt value-rules: no, pass@16~.05).

Rows-file schema = val.jsonl (id/category/prompt/answer). MUST be DISJOINT from
the eval set (bit_eval500.jsonl ids) or you leak eval into train.

Run:
  ~/.venvs/tinker/bin/python tinker/star_harvest.py --ckpt tinker://... \
     --rows-file pipeline/data/bit_star_src.jsonl --k 8 --temp 0.8 \
     --keep-per-row 2 --max-cot-tokens 4200 --tag bit-star-r1
"""
import argparse, csv, json, math, os, re, sys, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tinker"))
from prompts import EVAL_SUFFIX
from eval_oracle import extract_final_answer, verify  # reuse OFFICIAL metric

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
csv.field_size_limit(10 ** 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="tinker:// sampler-weights path")
    ap.add_argument("--rows-file", required=True, help="src prompts jsonl (DISJOINT from eval)")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=7680, help="generation cap")
    ap.add_argument("--keep-per-row", type=int, default=2, help="max distinct correct CoTs kept/row")
    ap.add_argument("--max-cot-tokens", type=int, default=None,
                    help="reject correct rollouts longer than this (favor short, executable traces)")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample src rows (cost control)")
    ap.add_argument("--cats", default=None, help="comma category filter")
    ap.add_argument("--tag", required=True, help="output tag -> pipeline/data/star/<tag>.csv")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rows_file)]
    if args.cats:
        keep = set(args.cats.split(",")); rows = [r for r in rows if r["category"] in keep]
    if args.max_rows:
        rows = rows[:args.max_rows]

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(args.base_model)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=1.0)

    print(f"[star] {len(rows)} src rows x K={args.k} temp={args.temp} max_cot={args.max_cot_tokens}", flush=True)
    futs = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok.encode(text, add_special_tokens=False)
        futs.append(sampler.sample(tinker.ModelInput.from_ints(ids), num_samples=args.k, sampling_params=params))

    out_dir = os.path.join(ROOT, "pipeline/data/star"); os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.tag}.csv")
    n_kept = 0; rows_with_hit = 0; sample_tok = 0
    n_correct_total = 0  # for pass@K stats
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "prompt", "answer", "category", "raw_output", "predicted", "correct"])
        for i, (r, fut) in enumerate(zip(rows, futs)):
            seqs = fut.result().sequences
            kept_here = 0; seen = set()
            row_hit = False
            for seq in seqs:
                sample_tok += len(seq.tokens)
                text = tok.decode(seq.tokens, skip_special_tokens=True)
                pred = extract_final_answer(text)
                if not verify(r["answer"], pred):
                    continue
                n_correct_total += 1; row_hit = True
                if args.max_cot_tokens and len(seq.tokens) > args.max_cot_tokens:
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                if kept_here >= args.keep_per_row:
                    continue
                w.writerow([r["id"], r["prompt"], r["answer"], r["category"], text, pred, "True"])
                kept_here += 1; n_kept += 1
            rows_with_hit += int(row_hit)
            if (i + 1) % 25 == 0:
                print(f"[star] {i+1}/{len(rows)} kept={n_kept} rows_hit={rows_with_hit} "
                      f"passK~{rows_with_hit/(i+1):.2f} ({time.time()-t0:.0f}s)", flush=True)

    cost = sample_tok * 0.33 / 1e6
    print(f"\n[star] DONE kept {n_kept} traces from {rows_with_hit}/{len(rows)} rows "
          f"(pass@{args.k}={rows_with_hit/len(rows):.3f}); {n_correct_total} correct samples total")
    print(f"[cost] sample {sample_tok/1e6:.2f}M ~= ${cost:.2f} list / ${cost/2:.2f} discounted")
    print(f"[out] {out_csv}")


if __name__ == "__main__":
    main()
