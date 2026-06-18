"""Starting-signal probe for the GRPO curriculum: sample a checkpoint on N
puzzles from one tier of pipeline/data/crypt_curriculum.jsonl and report the
mean reward (reward.py strict-boxed + official verify) + per-puzzle hit rates.
Gates the GRPO campaign: tier-1 mean must be >= ~0.15 or tier 1 gets eased.

  ~/.venvs/tinker/bin/python tinker/probe_curriculum.py \
      --ckpt 'tinker://...sampler' --tier 1 --n 30 --k 8 \
      --temp 0.8 --max-tokens 4000

Progress prints per group so a 402 retry-forever hang is VISIBLE (budget
exhausted -> the print stream stalls; kill and report BUDGET EXHAUSTED).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tinker"))
from prompts import EVAL_SUFFIX
from reward import reward

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="tinker:// sampler-weights path")
    ap.add_argument("--prompts-file",
                    default=os.path.join(ROOT, "pipeline/data/crypt_curriculum.jsonl"))
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts_file)]
    rows = [r for r in rows if int(r["tier"]) == args.tier][:args.n]
    print(f"[probe] {len(rows)} tier-{args.tier} puzzles, K={args.k}, "
          f"temp {args.temp}, max_tokens {args.max_tokens}", flush=True)

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    params = tinker.SamplingParams(max_tokens=args.max_tokens,
                                   temperature=args.temp, top_p=1.0)

    futs, prefill_tok = [], 0
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefill_tok += len(ids)
        futs.append(sampler.sample(tinker.ModelInput.from_ints(ids),
                                   num_samples=args.k, sampling_params=params))

    t0 = time.time()
    hits, total, sample_tok, n_trunc, pass_any = 0, 0, 0, 0, 0
    per = []
    for i, (r, fut) in enumerate(zip(rows, futs)):
        seqs = fut.result().sequences
        rs = []
        for s in seqs:
            sample_tok += len(s.tokens)
            n_trunc += s.stop_reason == "length"
            text = tokenizer.decode(s.tokens, skip_special_tokens=True)
            rs.append(reward(text, r["answer"], truncated=(s.stop_reason == "length")))
        hits += sum(rs); total += len(rs); pass_any += any(rs)
        per.append(sum(rs) / len(rs))
        print(f"  [{i+1}/{len(rows)}] {r['id']} hit {sum(rs):.0f}/{len(rs)} "
              f"| running mean {hits/total:.3f} ({time.time()-t0:.0f}s)", flush=True)

    cost = (prefill_tok * 0.13 + sample_tok * 0.33) / 1e6
    print(f"\n[probe] tier-{args.tier} MEAN REWARD {hits/total:.3f} "
          f"({hits:.0f}/{total}) | pass@{args.k} {pass_any}/{len(rows)} "
          f"| trunc {n_trunc/total:.2f}")
    print(f"[probe] non-degenerate groups (0<mean<1): "
          f"{sum(1 for p in per if 0 < p < 1)}/{len(per)}")
    print(f"[cost] ~${cost:.2f} list (prefill {prefill_tok/1e6:.2f}M "
          f"+ sample {sample_tok/1e6:.2f}M)")


if __name__ == "__main__":
    main()
