"""GRPO campaign loop on Tinker (tinker 0.22.3). Local reward/prompts are
DONE and tested; this file wires them to the real SDK.

Usage:
  ~/.venvs/tinker/bin/python tinker/rl_loop.py --arms bit_manipulation=1.0 \
      --iters 50 --from-ckpt tinker://... --run-name pilot-bit
  ~/.venvs/tinker/bin/python tinker/rl_loop.py --arms bit_manipulation=.45,\
cryptarithm_deduce=.25,equation_numeric_deduce=.15,retention=.15 \
      --iters 200 --from-ckpt tinker://... --run-name campaign

Design (PLAN_OPTION_C §3–4):
  batch 128 prompts/iter, K=16 rollouts each, temp 0.9, max_tokens 7680
  reward: tinker/reward.py (strict-boxed, truncation=0)
  advantage: per-group (r - mean)/(std + 1e-6); skip all-0/all-1 groups
  optional per-token KL penalty to the warm-start reference (--kl-coef;
  costs ~$0.7/iter extra prefill — 0 disables)
  checkpoint every 25 iters; oracle eval (eval_oracle.py) every 50
  hard $ cap: --max-dollars, enforced on REALIZED token spend
  RUNS.md ledger row at start + every checkpoint + completion

STaR fallback (--star): same rollouts, keep verified-correct samples
(<= --star-keep per group), supervised cross_entropy on them instead of
importance sampling. ~5-10x cheaper per learning signal.

CURRICULUM MODE (crypt hammer, post-r6; mutually exclusive with --arms):
  --prompts-file pipeline/data/crypt_curriculum.jsonl   rows {id,prompt,answer,tier}
  --adaptive-tiers   start at tier 1; PROMOTE when the current tier's rolling
                     mean reward >= 0.5, DEMOTE when < 0.1 (window 256 samples,
                     min 128 before any move); every batch mixes ~20% of the
                     next tier. Without the flag the loop stays on --start-tier.
  --eval-every N + --eval-cats cryptarithm_deduce   greedy eval_oracle.py on the
                     100 REAL val deduce rows (the LB oracle) every N iters.
  Reward is unchanged: reward.py (strict boxed + official verify).
  ~/.venvs/tinker/bin/python tinker/rl_loop.py \
      --prompts-file pipeline/data/crypt_curriculum.jsonl --adaptive-tiers \
      --iters 50 --batch 64 --k 8 --max-tokens 4000 --eval-every 10 \
      --eval-cats cryptarithm_deduce --from-ckpt tinker://... --run-name crypt-grpo
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import EVAL_SUFFIX, PromptPool
from reward import reward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "tinker", "RUNS.md")

K = 16
BATCH = 128
TEMP = 0.9
MAX_TOKENS = 7680
CKPT_EVERY = 25
EVAL_EVERY = 50
PRICE = {"prefill": 0.13, "sample": 0.33, "train": 0.40}  # $/Mtok list


def ledger(line):
    with open(RUNS, "a") as f:
        f.write(line.rstrip() + "\n")


def pct(v, q):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


class CurriculumPool:
    """tiered synthetic-prompt pool (rows {id,prompt,answer,tier}) with optional
    adaptive scheduling: stay on cur_tier, mix ~20% of the next tier, promote
    when cur_tier's rolling mean reward >= 0.5, demote when < 0.1."""
    WINDOW = 256          # rolling rewards kept per tier
    MIN_OBS = 128         # observations required before promote/demote
    NEXT_MIX = 0.20
    PROMOTE_AT = 0.5
    DEMOTE_AT = 0.1

    def __init__(self, path, adaptive=False, start_tier=1):
        self.by_tier = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            self.by_tier[int(r["tier"])].append(
                dict(category=f"tier{r['tier']}", prompt=r["prompt"],
                     gold=str(r["answer"]).strip(), source=f"cur:{r['id']}"))
        self.tiers = sorted(self.by_tier)
        assert start_tier in self.by_tier, f"tier {start_tier} not in {path}"
        self.cur = start_tier
        self.adaptive = adaptive
        self.rolling = {t: deque(maxlen=self.WINDOW) for t in self.tiers}

    def _next_tier(self):
        up = [t for t in self.tiers if t > self.cur]
        return up[0] if up else None

    def sample(self, rng):
        t = self.cur
        nxt = self._next_tier()
        if self.adaptive and nxt is not None and rng.random() < self.NEXT_MIX:
            t = nxt
        return rng.choice(self.by_tier[t])

    def user_content(self, item):
        return item["prompt"] + EVAL_SUFFIX

    def observe(self, item, rewards):
        t = int(item["category"][4:])
        self.rolling[t].extend(rewards)

    def maybe_move(self):
        """returns a log string when the tier changes, else None."""
        if not self.adaptive:
            return None
        win = self.rolling[self.cur]
        if len(win) < self.MIN_OBS:
            return None
        m = sum(win) / len(win)
        nxt = self._next_tier()
        if m >= self.PROMOTE_AT and nxt is not None:
            old, self.cur = self.cur, nxt
            self.rolling[old].clear()
            return f"PROMOTE tier{old}->tier{self.cur} (rolling {m:.3f})"
        dn = [t for t in self.tiers if t < self.cur]
        if m < self.DEMOTE_AT and dn:
            old, self.cur = self.cur, dn[-1]
            self.rolling[old].clear()
            return f"DEMOTE tier{old}->tier{self.cur} (rolling {m:.3f})"
        return None

    def tier_line(self):
        return " ".join(f"t{t}:{(sum(w)/len(w)):.2f}/{len(w)}"
                        for t, w in sorted(self.rolling.items()) if w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None)           # "bit_manipulation=1.0" or csv
    ap.add_argument("--prompts-file", default=None,
                    help="curriculum jsonl {id,prompt,answer,tier}; excludes --arms")
    ap.add_argument("--adaptive-tiers", action="store_true",
                    help="curriculum: promote >=0.5 / demote <0.1 rolling mean, 20%% next-tier mix")
    ap.add_argument("--start-tier", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                    help="greedy eval_oracle.py every N iters")
    ap.add_argument("--eval-cats", default=None,
                    help="--cats passed to eval_oracle.py (e.g. cryptarithm_deduce)")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--from-ckpt", required=True)      # tinker:// STATE path (warm start)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--max-dollars", type=float, default=900.0)
    ap.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="per-token KL-to-ref penalty folded into advantages")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--star", action="store_true", help="STaR fallback mode")
    ap.add_argument("--star-keep", type=int, default=4,
                    help="STaR: max correct samples kept per group")
    args = ap.parse_args()

    if bool(args.arms) == bool(args.prompts_file):
        ap.error("need exactly one of --arms / --prompts-file")
    if args.prompts_file:
        pool = CurriculumPool(args.prompts_file, adaptive=args.adaptive_tiers,
                              start_tier=args.start_tier)
        print(f"[pool] curriculum {args.prompts_file}: "
              + " ".join(f"tier{t}={len(v)}" for t, v in sorted(pool.by_tier.items()))
              + f" | start tier {pool.cur} adaptive={args.adaptive_tiers}", flush=True)
    else:
        weights = {k: float(v) for k, v in
                   (kv.split("=") for kv in args.arms.split(","))}
        pool = PromptPool(weights=weights)
    rng = random.Random(20260612)

    import numpy as np
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(args.base_model)
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(args.from_ckpt)
    ref_sampler = (sc.create_sampling_client(model_path=args.from_ckpt)
                   if args.kl_coef > 0 else None)
    params = tinker.SamplingParams(max_tokens=args.max_tokens,
                                   temperature=args.temp, top_p=1.0)

    src = (f"arms={args.arms}" if args.arms else
           f"curriculum={os.path.basename(args.prompts_file)}"
           f" adaptive={args.adaptive_tiers}")
    ledger(f"| {args.run_name} | START {time.strftime('%F %T')} | {src} "
           f"| iters={args.iters} batch={args.batch} k={args.k} "
           f"maxtok={args.max_tokens} temp={args.temp} | from={args.from_ckpt} | "
           f"{'STaR' if args.star else 'GRPO'} lr={args.lr} kl={args.kl_coef} |")

    spent = 0.0  # realized $, list price
    last_sampler_path = None
    for it in range(args.iters):
        if spent >= args.max_dollars:
            ledger(f"| {args.run_name} | DOLLAR-CAP ${spent:.0f} at iter {it} |")
            print(f"[cap] ${spent:.0f} >= {args.max_dollars}; stopping", flush=True)
            break
        t0 = time.time()
        batch = [pool.sample(rng) for _ in range(args.batch)]

        # ---------------- rollouts ----------------
        sampler = tc.save_weights_and_get_sampling_client(
            name=f"{args.run_name}-it{it:04d}")
        prompt_ids, futs = [], []
        prefill_tok = 0
        for b in batch:
            msgs = [{"role": "user", "content": pool.user_content(b)}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompt_ids.append(ids)
            prefill_tok += len(ids)
            futs.append(sampler.sample(tinker.ModelInput.from_ints(ids),
                                       num_samples=args.k, sampling_params=params))

        # ---------------- rewards + advantages ----------------
        groups = []          # (i, sequences, rewards)
        sample_tok = 0
        lengths, arm_stats = [], defaultdict(lambda: [0, 0])
        for i, fut in enumerate(futs):
            seqs = fut.result().sequences
            if (i + 1) % 8 == 0 or (i + 1) == len(futs):
                print(f"  [rollout] {i+1}/{len(futs)} groups resolved "
                      f"({time.time()-t0:.0f}s)", flush=True)
            rs = []
            for s in seqs:
                sample_tok += len(s.tokens)
                lengths.append(len(s.tokens))
                text = tokenizer.decode(s.tokens, skip_special_tokens=True)
                r = reward(text, batch[i]["gold"],
                           truncated=(s.stop_reason == "length"))
                rs.append(r)
            a = arm_stats[batch[i]["category"]]
            a[0] += len(rs); a[1] += sum(rs)
            if isinstance(pool, CurriculumPool):
                pool.observe(batch[i], rs)
            groups.append((i, seqs, rs))

        datums = []
        n_kept_groups = 0
        if args.star:
            for i, seqs, rs in groups:
                kept = 0
                for s, r in zip(seqs, rs):
                    if r != 1.0 or kept >= args.star_keep:
                        continue
                    kept += 1
                    full = prompt_ids[i] + list(s.tokens)
                    P = len(prompt_ids[i])
                    tgt = full[1:]
                    w = [0.0] * (P - 1) + [1.0] * (len(tgt) - (P - 1))
                    datums.append(tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(full[:-1]),
                        loss_fn_inputs={
                            "weights": tinker.TensorData(
                                data=w, dtype="float32", shape=[len(w)]),
                            "target_tokens": tinker.TensorData(
                                data=tgt, dtype="int64", shape=[len(tgt)]),
                        }))
                n_kept_groups += kept > 0
        else:
            for i, seqs, rs in groups:
                if sum(rs) in (0.0, float(len(rs))):
                    continue  # all-fail / all-pass: no gradient signal
                n_kept_groups += 1
                mean = sum(rs) / len(rs)
                std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
                P = len(prompt_ids[i])
                for s, r in zip(seqs, rs):
                    adv = (r - mean) / (std + 1e-6)
                    full = prompt_ids[i] + list(s.tokens)
                    C = len(s.tokens)
                    tgt = full[1:]
                    lps = [0.0] * (P - 1) + list(s.logprobs)
                    advs = [0.0] * (P - 1) + [adv] * C
                    # NOTE: no "mask" field — the server's ImportanceSampling
                    # loss rejects it (smoke 2026-06-12); prompt positions are
                    # excluded via advantage 0.0 instead.
                    if ref_sampler is not None:
                        ref_lp = ref_sampler.compute_logprobs(
                            tinker.ModelInput.from_ints(full)).result()
                        prefill_tok += len(full)
                        for j in range(C):
                            rl = ref_lp[P + j]
                            if rl is not None:
                                advs[P - 1 + j] -= args.kl_coef * (
                                    s.logprobs[j] - rl)
                    datums.append(tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(full[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tgt, dtype="int64", shape=[len(tgt)]),
                            "logprobs": tinker.TensorData(
                                data=lps, dtype="float32", shape=[len(lps)]),
                            "advantages": tinker.TensorData(
                                data=advs, dtype="float32", shape=[len(advs)]),
                        }))

        # ---------------- policy update ----------------
        train_tok = sum(d.model_input.length for d in datums)
        if datums:
            fb = tc.forward_backward(
                datums, "cross_entropy" if args.star else "importance_sampling")
            opt = tc.optim_step(tinker.AdamParams(
                learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8,
                weight_decay=0.0))
            fb.result()
            opt.result()

        # ---------------- bookkeeping ----------------
        iter_cost = (prefill_tok * PRICE["prefill"] + sample_tok * PRICE["sample"]
                     + train_tok * PRICE["train"]) / 1e6
        spent += iter_cost
        succ = (sum(sum(rs) for _, _, rs in groups)
                / max(1, sum(len(rs) for _, _, rs in groups)))
        arms = " ".join(f"{c}={s/max(1,n):.2f}" for c, (n, s) in sorted(arm_stats.items()))
        print(f"[it {it+1}/{args.iters}] succ {succ:.3f} | {arms} | "
              f"groups kept {n_kept_groups}/{len(groups)} datums {len(datums)} | "
              f"len p50 {pct(lengths,.5)} p95 {pct(lengths,.95)} | "
              f"${iter_cost:.2f} (total ${spent:.2f}) | {time.time()-t0:.0f}s",
              flush=True)
        if isinstance(pool, CurriculumPool):
            print(f"  [tiers] cur=tier{pool.cur} | rolling {pool.tier_line()}",
                  flush=True)
            mv = pool.maybe_move()
            if mv:
                ledger(f"| {args.run_name} | it{it+1} | {mv} |")
                print(f"  [tiers] {mv}", flush=True)

        if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.iters:
            st = tc.save_state(name=f"{args.run_name}-it{it+1:04d}").result()
            sp = tc.save_weights_for_sampler(
                name=f"{args.run_name}-it{it+1:04d}-sampler").result()
            last_sampler_path = sp.path
            ledger(f"| {args.run_name} | ckpt it{it+1} | state={st.path} "
                   f"sampler={sp.path} | succ {succ:.3f} | ${spent:.0f} |")
            print(f"[ckpt] state {st.path}\n[ckpt] sampler {sp.path}", flush=True)
        if (it + 1) % args.eval_every == 0 and (it + 1) < args.iters:
            sp = tc.save_weights_for_sampler(
                name=f"{args.run_name}-eval-it{it+1:04d}").result()
            last_sampler_path = sp.path
            cmd = [sys.executable, os.path.join(ROOT, "tinker", "eval_oracle.py"),
                   "--ckpt", sp.path, "--tag", f"{args.run_name}-it{it+1}"]
            if args.eval_cats:
                cmd += ["--cats", args.eval_cats]
            print(f"[eval] launching oracle on {sp.path}"
                  + (f" (cats {args.eval_cats})" if args.eval_cats else ""), flush=True)
            subprocess.Popen(cmd)

    if args.eval_cats and last_sampler_path:
        cmd = [sys.executable, os.path.join(ROOT, "tinker", "eval_oracle.py"),
               "--ckpt", last_sampler_path, "--tag", f"{args.run_name}-final",
               "--cats", args.eval_cats]
        print(f"[eval] final oracle on {last_sampler_path}", flush=True)
        subprocess.call(cmd)
    ledger(f"| {args.run_name} | DONE {time.strftime('%F %T')} | ${spent:.2f} |")
    print(f"[done] total ${spent:.2f} list; last sampler {last_sampler_path}")


if __name__ == "__main__":
    main()
