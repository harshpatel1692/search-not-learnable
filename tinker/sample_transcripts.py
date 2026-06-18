"""Dump full greedy transcripts for specific val rows from a Tinker sampler ckpt.
Usage: sample_transcripts.py --ckpt tinker://... --ids id1,id2,... [--out DIR]
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tinker"))
from prompts import EVAL_SUFFIX

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated row id prefixes")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis/crypt_struct/r7_transcripts"))
    ap.add_argument("--max-tokens", type=int, default=7680)
    args = ap.parse_args()

    want = args.ids.split(",")
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "pipeline/data/val.jsonl"))]
    rows = [r for r in rows if any(str(r["id"]).startswith(w) for w in want)]
    assert rows, "no matching ids"

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    futs = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        futs.append(sampler.sample(tinker.ModelInput.from_ints(ids),
                                   num_samples=1, sampling_params=params))

    os.makedirs(args.out, exist_ok=True)
    for r, fut in zip(rows, futs):
        seq = fut.result().sequences[0]
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        p = os.path.join(args.out, f"{r['id']}.txt")
        with open(p, "w") as f:
            f.write(f"# id={r['id']} gold={r['answer']} stop={seq.stop_reason} "
                    f"ntok={len(seq.tokens)}\n## PROMPT\n{r['prompt']}\n## OUTPUT\n{text}\n")
        print(f"{r['id']} gold={r['answer']!r} ntok={len(seq.tokens)} -> {p}", flush=True)


if __name__ == "__main__":
    main()
