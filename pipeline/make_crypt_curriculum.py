"""Synthetic difficulty curriculum for cryptarithm GRPO (crypt hammer, post-r6).

4 tiers x ~500 verified puzzles -> pipeline/data/crypt_curriculum.jsonl
(rows {id, prompt, answer, tier}). ALL synthetic (zero real/val ids); every
answer is independently confirmed by the solver (C2.solve(prompt) == gold) so
GRPO's reward is noise-free (no under-determined puzzles).

Tier knobs (difficulty = constraint level; see analysis/reports/crypt_grpo_setup.md):
  tier1  canonical regime ('+','-','*' literal glyphs), 2 ops, PLAIN family
         rules only (no offsets/concat/mod), 6-8 examples, standard digit
         order (rev=False), sign prefix. Maximally constrained.
  tier1e EASED tier1 (auto-fallback if the r6 ckpt scores < .15 on tier1):
         1 op, family forced to the identity rule ('+'->add,'*'->mul,
         '-'->sub_signed), 8 examples.
  tier2  50% canonical 3-ops plain families / 50% scrambled glyphs with
         plain ops only (no offsets), measured example counts (3-5), rev
         per measured mode priors.
  tier3  full two-regime mix incl. offsets = gen_puzzle_r5 (measured real
         distribution, EM priors).
  tier4  gen_puzzle defaults (r4 measured distribution, scrambled glyphs,
         full op table) = real-train-distribution synth.

Usage:
  python3 pipeline/make_crypt_curriculum.py [--per-tier 500] [--seed 461]
          [--tiers 1,2,3,4] [--ease-tier1]    # --ease-tier1 swaps tier1 -> tier1e knobs
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from solvers import cryptarithm2 as C2
from synth import cryptarithm_cot as CC

OUT_DEFAULT = os.path.join(HERE, "data", "crypt_curriculum.jsonl")

PLAIN_FAM = dict(CC.FAM_PLAIN5)              # '+':[add] '*':[mul] '-':[sub_signed,absdiff,neg_absdiff]
IDENTITY_FAM = {'+': ['add'], '*': ['mul'], '-': ['sub_signed']}
CONCAT_FAM = {'+': ['concat_fwd'], '*': ['concat_rev']}   # transcription-only rules
PLAIN_SCRAMBLED = list(CC.PLAIN_VOCAB)       # add, sub_signed, mul, rsub_signed, absdiff, neg_absdiff
FAMS = {'plain': PLAIN_FAM, 'identity': IDENTITY_FAM, 'concat': CONCAT_FAM}


def gen_custom(rng, *, canonical, n_ops, fams=None, vocab=None, nex_lo, nex_hi,
               rev=None, sign_end='pre'):
    """parametrized deduce-puzzle generator (reuses CC.render_eq / pools).
    canonical: glyphs are literal '+','-','*' chars, op drawn from fams[char].
    scrambled: glyphs arbitrary symbols, ops drawn w/o replacement from vocab."""
    if rev is None:
        rev = rng.random() < (CC.P_REV_A5 if canonical else CC.P_REV_B5)
    nex = rng.randint(nex_lo, nex_hi)
    digs = rng.sample(CC.DIGIT_POOL, 10)
    smap = {d: digs[d] for d in range(10)}
    if canonical:
        glyphs = rng.sample(sorted(fams), n_ops)
        gops = {gl: rng.choice(fams[gl]) for gl in glyphs}
    else:
        pool = [c for c in CC.DIGIT_POOL + CC.OP_EXTRA if c not in digs]
        glyphs = rng.sample(pool, n_ops)
        ops = rng.sample(vocab, n_ops)       # without replacement (measured)
        gops = dict(zip(glyphs, ops))

    def one_eq(gl, tries):
        for _ in range(tries):
            a = rng.randint(10, 99); b = rng.randint(10, 99)
            op = gops[gl]
            if op in ('concat_fwd', 'concat_rev'):   # gen_puzzle's concat branch
                da = [a // 10, a % 10]; db = [b // 10, b % 10]
                if rev:
                    da = da[::-1]; db = db[::-1]
                lhs = smap[da[0]] + smap[da[1]] + gl + smap[db[0]] + smap[db[1]]
                rhs = (lhs[0] + lhs[1] + lhs[3] + lhs[4]) if op == 'concat_fwd' \
                    else (lhs[3] + lhs[4] + lhs[0] + lhs[1])
                return lhs, rhs
            r = CC.render_eq(a, b, op, smap, gl, rev, sign_end)
            if r is not None:
                return r
        return None

    qglyph = rng.choice(glyphs)
    asg = list(glyphs) + [rng.choice(glyphs) for _ in range(nex - len(glyphs))]
    rng.shuffle(asg)
    lines = []
    for gl in asg:
        r = one_eq(gl, 40)
        if r is None:
            return None
        lines.append(r)
    r = one_eq(qglyph, 60)
    if r is None:
        return None
    qL, ans = r
    prompt = CC.HDR + "\n" + "\n".join(f"{l} = {r}" for l, r in lines) + \
             "\n" + CC.QSTR + qL
    return {'prompt': prompt, 'answer': ans}


def gen_tier(rng, tier, eased=False, t1=None):
    if tier == 1:
        if t1:                                # explicit override knobs
            return gen_custom(rng, canonical=True, n_ops=t1['ops'],
                              fams=FAMS[t1['fam']],
                              nex_lo=t1['nex_lo'], nex_hi=t1['nex_hi'], rev=False)
        if eased:
            return gen_custom(rng, canonical=True, n_ops=1, fams=IDENTITY_FAM,
                              nex_lo=8, nex_hi=8, rev=False)
        # SHIPPED tier1 (probe-calibrated 2026-06-12): the r6 ckpt scores 0.000
        # on plain-value canonical and 0.004 on 1-op identity-value, but 0.742
        # on concat-only -> 50% concat (dense reward) / 50% eased value
        # (the skill to grow). Expected starting mean ~0.37.
        if rng.random() < 0.5:
            return gen_custom(rng, canonical=True, n_ops=rng.randint(1, 2),
                              fams=CONCAT_FAM, nex_lo=5, nex_hi=8, rev=False)
        return gen_custom(rng, canonical=True, n_ops=1, fams=IDENTITY_FAM,
                          nex_lo=8, nex_hi=8, rev=False)
    if tier == 2:
        if rng.random() < 0.5:
            return gen_custom(rng, canonical=True, n_ops=3, fams=PLAIN_FAM,
                              nex_lo=3, nex_hi=5)
        return gen_custom(rng, canonical=False, n_ops=min(CC.w_choice(rng, CC.N_GLYPH_W), 3),
                          vocab=PLAIN_SCRAMBLED, nex_lo=3, nex_hi=5)
    if tier == 3:
        p = CC.gen_puzzle_r5(rng)
        return p and {'prompt': p['prompt'], 'answer': p['answer']}
    if tier == 4:
        p = CC.gen_puzzle(rng)
        return p and {'prompt': p['prompt'], 'answer': p['answer']}
    raise ValueError(tier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=500)
    ap.add_argument("--seed", type=int, default=461)
    ap.add_argument("--tiers", default="1,2,3,4")
    ap.add_argument("--ease-tier1", action="store_true")
    ap.add_argument("--t1-spec", default=None,
                    help="tier1 override 'ops=1,fam=identity,nex=4-5' (beats --ease-tier1)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--deadline", type=float, default=8.0, help="solver verify deadline/row")
    args = ap.parse_args()

    t1 = None
    if args.t1_spec:
        kv = dict(p.split("=") for p in args.t1_spec.split(","))
        lo, hi = (kv.get("nex", "6-8").split("-") + [None])[:2]
        t1 = dict(ops=int(kv.get("ops", 2)), fam=kv.get("fam", "plain"),
                  nex_lo=int(lo), nex_hi=int(hi or lo))

    tiers = [int(t) for t in args.tiers.split(",")]
    rows, stats = [], {}
    for tier in tiers:
        rng = random.Random(args.seed + 1000 * tier)
        kept, tried, t0 = 0, 0, time.time()
        while kept < args.per_tier:
            tried += 1
            if tried > args.per_tier * 40:
                print(f"[tier{tier}] giving up after {tried} attempts", flush=True)
                break
            p = gen_tier(rng, tier, eased=args.ease_tier1, t1=t1)
            if p is None:
                continue
            sv = C2.solve(p['prompt'], deadline_s=args.deadline)
            if sv is None or sv[0] != p['answer']:
                continue                      # ambiguous / solver-mismatch: drop
            kept += 1
            rows.append({"id": f"cur_t{tier}_{kept:04d}", "prompt": p['prompt'],
                         "answer": p['answer'], "tier": tier})
            if kept % 100 == 0:
                print(f"[tier{tier}] {kept}/{args.per_tier} "
                      f"(verify-keep {kept/tried:.2f}, {time.time()-t0:.0f}s)", flush=True)
        stats[tier] = (kept, tried, time.time() - t0)
        print(f"[tier{tier}] DONE {kept} rows / {tried} tried "
              f"(keep {kept/max(1,tried):.2f}) in {stats[tier][2]:.0f}s", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[out] {len(rows)} rows -> {args.out}")
    for tier, (k, t, s) in stats.items():
        print(f"  tier{tier}: {k} rows, solver-keep {k/max(1,t):.2f}")


if __name__ == "__main__":
    main()
