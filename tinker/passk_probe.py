"""pass@k probe on val deduce rows for a Tinker sampler ckpt (GRPO-signal gate).
Usage: passk_probe.py --ckpt tinker://... [--n-rows 20 --k 16 --temp 0.8 --max-tokens 2500]
"""
import argparse, json, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tinker"))
from prompts import EVAL_SUFFIX
from eval_oracle import extract_final_answer, verify

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-rows", type=int, default=20)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--cat", default="cryptarithm_deduce")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "pipeline/data/val.jsonl"))]
    rows = [r for r in rows if r["category"] == args.cat]
    rows = random.Random(7).sample(rows, args.n_rows)

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    params = tinker.SamplingParams(max_tokens=args.max_tokens,
                                   temperature=args.temp, top_p=1.0)

    futs = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        futs.append(sampler.sample(tinker.ModelInput.from_ints(ids),
                                   num_samples=args.k, sampling_params=params))

    hit_rows = 0
    tot_hits = 0
    tot_samples = 0
    sample_tok = 0
    for r, fut in zip(rows, futs):
        seqs = fut.result().sequences
        hits = 0
        for s in seqs:
            sample_tok += len(s.tokens)
            pred = extract_final_answer(tokenizer.decode(s.tokens, skip_special_tokens=True))
            hits += bool(verify(r["answer"], pred))
        tot_hits += hits
        tot_samples += len(seqs)
        hit_rows += hits > 0
        print(f"{r['id']} hits {hits}/{len(seqs)}", flush=True)

    print(f"\npass@{args.k}: {hit_rows}/{len(rows)} = {hit_rows/len(rows):.3f}")
    print(f"mean hit rate: {tot_hits}/{tot_samples} = {tot_hits/tot_samples:.4f}")
    print(f"[cost] sample {sample_tok/1e6:.2f}M ~= ${sample_tok*0.33/1e6:.2f} list")


if __name__ == "__main__":
    main()
