"""Behavioral-clone warm start on Tinker (tinker 0.22.3 / cookbook 0.4.2).

Recreates the v16 adapter IN Tinker-native format (no external import exists):
supervised SFT over pipeline/data/v16/train_v16.csv from the BASE model,
rank-32 LoRA incl. lm_head (train_unembed=True).

Replicates kaggle_kernels/v16_train byte-for-byte where it matters:
  - same data filter (correct == True), same CoT body extraction
  - build_example masking VERBATIM (prefix/full apply_chat_template split;
    Phase-0 proved HF template == Tinker renderer, 259/259 ids equal)
  - stratified round-robin order, seed 42, eff batch 32
  - LR 5e-6 linear decay, AdamW(0.9, 0.95, 1e-8, wd 0)
What CANNOT replicate: the Kaggle kernel warm-starts from the Ali adapter;
Tinker cannot import external weights, so this trains from the BASE. Expect
to need the full corpus (and possibly --epochs 2) to approach v16 quality.

Run (venv has the SDK):
  ~/.venvs/tinker/bin/python tinker/sft_warmstart.py --dry-run        # no key needed
  ~/.venvs/tinker/bin/python tinker/sft_warmstart.py --run-name v16-warmstart-e1

Output: tinker:// state path (resume/RL) + sampler path (eval_oracle.py).
Gate (CHECKLIST Phase 1): converted adapter oracle >= ~0.83 before RL.
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "tinker", "RUNS.md")
csv.field_size_limit(10 ** 9)

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MAX_SEQ_LEN = 8192
BATCH_SIZE = 32
# 5e-6 was the Kaggle kernel's REFINEMENT LR on top of the Ali warm start.
# Tinker trains a FRESH LoRA from base -> fresh-train regime (repo lineage:
# 2e-4, same rank-32/alpha-32/batch-32). Cookbook has no Nemotron calibration.
LR_FRESH = 2e-4
LR_RESUME = 5e-6
SEED = 42
EVAL_USER_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                    "For example: `\\boxed{your answer}`")
TRAIN_PRICE_PER_MTOK = 0.40  # list; 50%-discount flag may halve it


def ledger(line):
    with open(RUNS, "a") as f:
        f.write(line.rstrip() + "\n")


def extract_cot_body(raw_output):
    s = str(raw_output)
    if "</think>" in s:
        s = s.split("</think>")[0]
    return s.rstrip()


def load_rows(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if str(r.get("correct", "True")) != "True":
            continue
        rows.append(dict(id=r["id"], prompt=r["prompt"], answer=str(r["answer"]),
                         type=r["category"], generated_cot=extract_cot_body(r["raw_output"])))
    return rows


def build_example(tokenizer, row):
    """VERBATIM port of kaggle_kernels/v16_train cell-10 build_example."""
    user_content = row["prompt"] + EVAL_USER_SUFFIX
    assistant_content = (f"<think>\n{row['generated_cot']}\n</think>\n"
                         f"\\boxed{{{row['answer']}}}")
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content},
         {"role": "assistant", "content": assistant_content}],
        tokenize=False, add_generation_prompt=False, enable_thinking=True)
    if not full_text.startswith(prefix_text):
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) < 4:
            return None
        n = min(len(full_ids) - 1, MAX_SEQ_LEN - 1)
        return {"tokens": full_ids[:-1][:n], "targets": full_ids[1:][:n],
                "weights": [1.0] * n, "id": row["id"], "type": row["type"]}
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[:MAX_SEQ_LEN]
    tokens = full_ids[:-1]
    targets = full_ids[1:]
    weights = [0.0] * (n_prefix - 1) + [1.0] * (len(targets) - (n_prefix - 1))
    weights = weights[:len(targets)]
    if sum(weights) < 1.0:
        return None
    return {"tokens": tokens, "targets": targets, "weights": weights,
            "id": row["id"], "type": row["type"]}


def build_stratified_order(labels, seed=SEED):
    """VERBATIM kernel cell 11: round-robin one row per type."""
    buckets = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[lab].append(i)
    rng = random.Random(seed)
    for k in buckets:
        rng.shuffle(buckets[k])
    keys = sorted(buckets.keys())
    if len(keys) <= 1:
        order = list(range(len(labels)))
        rng.shuffle(order)
        return order
    order = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop())
    return order


def example_to_datum(ex):
    import tinker
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(ex["tokens"]),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=[float(w) for w in ex["weights"]],
                                         dtype="float32", shape=[len(ex["weights"])]),
            "target_tokens": tinker.TensorData(data=list(ex["targets"]),
                                               dtype="int64", shape=[len(ex["targets"])]),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "pipeline/data/v16/train_v16.csv"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-4 fresh / 5e-6 when --from-ckpt")
    ap.add_argument("--run-name", default="v16-warmstart-e1")
    ap.add_argument("--ckpt-every", type=int, default=100, help="save_state every N steps")
    ap.add_argument("--from-ckpt", default=None,
                    help="tinker:// state to resume (e.g. 2nd epoch); else fresh LoRA")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + report datums only; no API key needed")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="continue an interrupted run: start the step counter "
                         "(and LR schedule) here and skip the first N batches. "
                         "Use with --from-ckpt <state> --lr <original-fresh-lr>.")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = LR_RESUME if args.from_ckpt else LR_FRESH

    print(f"[data] loading {args.data} (lr {args.lr})", flush=True)
    rows = load_rows(args.data)
    print(f"[data] {len(rows)} correct rows", flush=True)

    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(args.base_model)

    examples, dropped = [], 0
    for row in rows:
        ex = build_example(tokenizer, row)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    order = build_stratified_order([e["type"] for e in examples])
    examples = [examples[i] for i in order]
    n_steps_per_epoch = len(examples) // BATCH_SIZE
    train_tok = sum(len(e["tokens"]) for e in examples)
    unmasked = sum(sum(e["weights"]) for e in examples)
    est = train_tok / 1e6 * TRAIN_PRICE_PER_MTOK * args.epochs
    print(f"[data] {len(examples)} examples ({dropped} dropped) | "
          f"{n_steps_per_epoch} steps/epoch @ batch {BATCH_SIZE} | "
          f"{train_tok/1e6:.1f}M input tok ({unmasked/1e6:.1f}M unmasked) | "
          f"est ${est:.2f} list / ${est/2:.2f} discounted", flush=True)
    if args.dry_run:
        print("[dry-run] datum build OK; exiting before any API call")
        return

    import numpy as np
    import tinker
    sc = tinker.ServiceClient()
    if args.from_ckpt:
        tc = sc.create_training_client_from_state(args.from_ckpt)
        print(f"[client] resumed from {args.from_ckpt}", flush=True)
    else:
        tc = sc.create_lora_training_client(
            base_model=args.base_model, rank=args.rank, seed=SEED,
            train_mlp=True, train_attn=True, train_unembed=True)
        print(f"[client] fresh rank-{args.rank} LoRA on {args.base_model}", flush=True)

    ledger(f"| {args.run_name} | START {time.strftime('%F %T')} | sft warmstart "
           f"{len(examples)} ex x {args.epochs} ep, lr {args.lr}, est ${est:.2f} |")

    total_steps = n_steps_per_epoch * args.epochs
    step = args.resume_step
    t0 = time.time()
    last_path = None
    for epoch in range(args.epochs):
        bs_start = args.resume_step * BATCH_SIZE if epoch == 0 else 0
        for bs in range(bs_start, n_steps_per_epoch * BATCH_SIZE, BATCH_SIZE):
            batch = examples[bs:bs + BATCH_SIZE]
            lr_now = args.lr * (1 - step / total_steps)
            datums = [example_to_datum(e) for e in batch]
            fb = tc.forward_backward(datums, "cross_entropy")
            opt = tc.optim_step(tinker.AdamParams(
                learning_rate=lr_now, beta1=0.9, beta2=0.95, eps=1e-8,
                weight_decay=0.0))
            out = fb.result()
            opt.result()
            # token-mean NLL over unmasked positions
            lp, w = 0.0, 0.0
            for d, o in zip(datums, out.loss_fn_outputs):
                logprobs = o["logprobs"].to_numpy()
                weights = np.asarray(d.loss_fn_inputs["weights"].data, dtype="float64")
                lp += float((logprobs * weights).sum())
                w += float(weights.sum())
            step += 1
            print(f"[train] step {step}/{total_steps} ep{epoch} lr {lr_now:.2e} "
                  f"nll {-lp/max(w,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step % args.ckpt_every == 0 or step == total_steps:
                r = tc.save_state(name=f"{args.run_name}-step{step}").result()
                last_path = r.path
                ledger(f"| {args.run_name} | ckpt step {step} | {r.path} |")
                print(f"[ckpt] {r.path}", flush=True)

    samp = tc.save_weights_for_sampler(name=f"{args.run_name}-sampler").result()
    ledger(f"| {args.run_name} | DONE {time.strftime('%F %T')} | state={last_path} "
           f"sampler={samp.path} | {(time.time()-t0)/60:.0f} min |")
    print(f"[done] state path   : {last_path}")
    print(f"[done] sampler path : {samp.path}")
    print(f"[next] ~/.venvs/tinker/bin/python tinker/eval_oracle.py --ckpt '{samp.path}'")


if __name__ == "__main__":
    main()
