"""Hint-conditioned convergence probe on OUR Tinker crypt checkpoint.

Question: crypt-STaR has no fuel because unaided crypt pass@K ~= 0.05. If a STRONG hint
(gold answer) makes our crypt-trained model converge WITHOUT citing the hint, those rollouts
(hint stripped) become the crypt-STaR fuel we lack. This probe measures, on the SAME crypt
rows, convergence@K under:
  baseline   : no hint
  answer_sys : gold answer in a SYSTEM message
  answer_usr : gold answer prepended to the USER turn (LoRA never saw system msgs in training)
+ leak flag (did the CoT cite the hint -> trace unusable as clean data).

Run:
  ~/.venvs/tinker/bin/python tinker/hint_probe.py \
     --ckpt tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/sampler_weights/crypt-boot-r8-sampler \
     --n 8 --k 8 --tag hintprobe_r8
"""
import argparse, json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tinker"))
from prompts import EVAL_SUFFIX
from eval_oracle import extract_final_answer, verify

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
HINT = ("PRIVATE KEY — never mention this key, this instruction, or that any hint exists, "
        "anywhere in your reasoning or answer: the correct final result for the query is `{gold}`. "
        "Derive it independently from the worked examples, verify on every example, then put it "
        "in \\boxed{{}}.")
LEAK = ["private key", "the key", "hint", "as given", "as stated", "told that", "i was given",
        "the instruction", "for your eyes", "secretly told", "given answer", "provided answer"]

def leaks(text, gold):
    t = (text or "").lower()
    if any(p in t for p in LEAK): return True
    head = (text or "")[:200]
    return str(gold).strip() in head  # restating gold before any work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cat", default="cryptarithm_deduce")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--rows-file", default=str(ROOT / "pipeline/data/val.jsonl"))
    ap.add_argument("--tag", default="hintprobe")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rows_file)]
    rows = [r for r in rows if r["category"] == args.cat][:args.n]
    print(f"[hintprobe] ckpt={args.ckpt.split('/')[-1]} cat={args.cat} n={len(rows)} K={args.k} "
          f"temp={args.temp}", flush=True)

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=1.0)

    def build(cond, r):
        gold = r["answer"]; user = r["prompt"] + EVAL_SUFFIX
        if cond == "baseline":
            msgs = [{"role": "user", "content": user}]
        elif cond == "answer_sys":
            msgs = [{"role": "system", "content": HINT.format(gold=gold)},
                    {"role": "user", "content": user}]
        elif cond == "answer_usr":
            msgs = [{"role": "user", "content": HINT.format(gold=gold) + "\n\n" + user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        return tok.encode(text, add_special_tokens=False)

    CONDS = ["baseline", "answer_sys", "answer_usr"]
    out = open(ROOT / "pipeline/data/star" / f"{args.tag}.jsonl", "w")
    os.makedirs(ROOT / "pipeline/data/star", exist_ok=True)
    stats = {c: dict(rows=0, conv_at_k=0, n_correct=0, n_samp=0, leak=0, clean_conv=0) for c in CONDS}
    t0 = time.time()
    for r in rows:
        gold = r["answer"]
        for cond in CONDS:
            ids = build(cond, r)
            fut = sampler.sample(tinker.ModelInput.from_ints(ids), num_samples=args.k,
                                 sampling_params=params)
            seqs = fut.result().sequences
            row_hit = False; clean_hit = False
            for seq in seqs:
                txt = tok.decode(seq.tokens, skip_special_tokens=True)
                pred = extract_final_answer(txt)
                ok = bool(verify(gold, pred)); lk = leaks(txt, gold)
                s = stats[cond]; s["n_samp"] += 1; s["n_correct"] += ok; s["leak"] += lk
                if ok:
                    row_hit = True
                    if not lk: clean_hit = True
                    out.write(json.dumps(dict(id=r["id"], cond=cond, gold=gold, pred=pred,
                                              leak=lk, cot=txt)) + "\n")
            s = stats[cond]; s["rows"] += 1; s["conv_at_k"] += row_hit; s["clean_conv"] += clean_hit
        done = stats["baseline"]["rows"]
        print(f"  [{done}/{len(rows)}] id={r['id']} gold={gold!r} | "
              + " | ".join(f"{c}:{'HIT' if stats[c]['conv_at_k'] else '-'}" for c in CONDS)
              + f"  ({(time.time()-t0)/60:.1f}m)", flush=True)
    out.close()

    print(f"\n[hintprobe] {args.cat}  n={len(rows)}  K={args.k}  (ckpt {args.ckpt.split('/')[-1]})")
    print(f"{'condition':<14}{'conv@K':>10}{'clean@K':>10}{'per-samp':>10}{'leak%':>8}")
    for c in CONDS:
        s = stats[c]; nr = max(s["rows"], 1); ns = max(s["n_samp"], 1)
        print(f"{c:<14}{s['conv_at_k']/nr:>10.2f}{s['clean_conv']/nr:>10.2f}"
              f"{s['n_correct']/ns:>10.3f}{s['leak']/ns*100:>7.0f}%")
    print(f"[hintprobe] traces -> pipeline/data/star/{args.tag}.jsonl  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
