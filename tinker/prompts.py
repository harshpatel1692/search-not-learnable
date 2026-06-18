"""Prompt sampler for the Tinker RL campaign. FULLY LOCAL — no Tinker dependency.

PromptPool.sample(rng) -> dict(category, prompt, gold, source)

Sources, per category:
  real  — train.csv rows (NEVER the 900 val.jsonl ids — they are the LB oracle)
  synth — fresh problems from our reverse-engineered generators (zero leakage;
          gold computed by the generator itself)

Arm weights default to the campaign allocation (PLAN_OPTION_C §4); override
via PromptPool(weights={...}). 'retention' is a pseudo-arm pooling
numeral/gravity/unit_conversion/cipher real rows (reward pins them).

User content for rollouts = prompt + EVAL_SUFFIX (byte-identical to grader).

Self-test: python3 tinker/prompts.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
csv.field_size_limit(10 ** 9)

EVAL_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
               "For example: `\\boxed{your answer}`")

DEFAULT_WEIGHTS = {
    "bit_manipulation": 0.45,
    "cryptarithm_deduce": 0.25,
    "equation_numeric_deduce": 0.15,
    "retention": 0.15,
}
RETENTION_CATS = ("numeral", "gravity", "unit_conversion", "cipher")
SYNTH_SHARE = 0.5   # within an arm: half real rows, half fresh synthetic


class PromptPool:
    def __init__(self, weights=None, synth_share=SYNTH_SHARE):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.synth_share = synth_share
        val_ids = {json.loads(l)["id"] for l in open(f"{ROOT}/pipeline/data/val.jsonl")}
        self.real = {}
        for r in csv.DictReader(open(f"{ROOT}/competition_dataset/train_categorized.csv")):
            if r["id"] in val_ids:
                continue
            self.real.setdefault(r["category"], []).append(
                dict(category=r["category"], prompt=r["prompt"],
                     gold=r["answer"].strip(), source=f"real:{r['id']}"))
        self._synth_ready = {}

    # ---- lazy synthetic generators (import cost only on first use) ----
    def _synth(self, cat, rng):
        if cat == "bit_manipulation":
            if "bit" not in self._synth_ready:
                from synth import bitmanip_global as BG
                exprs, trans = BG.rule_dist_from_cache(
                    os.path.join(ROOT, "pipeline/data/bitmanip_solved.jsonl"))
                self._synth_ready["bit"] = (BG, exprs, trans)
            BG, exprs, trans = self._synth_ready["bit"]
            row = BG.gen_synth(rng, exprs, trans)
            if row is None:
                return None
            return dict(category=cat, prompt=row["prompt"], gold=row["final"],
                        source="synth:bit")
        if cat == "cryptarithm_deduce":
            if "crypt" not in self._synth_ready:
                from synth import cryptarithm_cot as CC
                from solvers import cryptarithm2 as C2
                self._synth_ready["crypt"] = (CC, C2)
            CC, C2 = self._synth_ready["crypt"]
            p = CC.gen_puzzle(rng)
            if not p:
                return None
            return dict(category=cat, prompt=p["prompt"], gold=p["answer"],
                        source="synth:crypt")
        if cat == "equation_numeric_deduce":
            if "eq" not in self._synth_ready:
                from synth import equation_cot as EC
                self._synth_ready["eq"] = EC
            EC = self._synth_ready["eq"]
            byop, q, gold = EC.gen_row(rng, "deduce")
            return dict(category=cat, prompt=EC.make_prompt(byop, q, rng),
                        gold=str(gold), source="synth:eq")
        return None  # retention: real rows only

    def sample(self, rng):
        arms, w = zip(*self.weights.items())
        arm = rng.choices(arms, weights=w, k=1)[0]
        if arm == "retention":
            cat = rng.choice(RETENTION_CATS)
            return rng.choice(self.real[cat])
        if rng.random() < self.synth_share:
            s = self._synth(arm, rng)
            if s is not None:
                return s
        return rng.choice(self.real[arm])

    def user_content(self, item):
        return item["prompt"] + EVAL_SUFFIX


if __name__ == "__main__":
    import random
    pool = PromptPool()
    rng = random.Random(7)
    from collections import Counter
    seen = Counter()
    n_synth = 0
    for _ in range(300):
        it = pool.sample(rng)
        assert it["prompt"] and it["gold"], it
        seen[it["category"]] += 1
        n_synth += it["source"].startswith("synth")
    print("arm draw counts:", dict(seen))
    print(f"synth share among drawn: {n_synth}/300")
    # verify synthetic golds round-trip through the strict reward
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reward import reward
    it = next(pool._synth(c, rng) for c in ["bit_manipulation"])
    assert reward(f"\\boxed{{{it['gold']}}}", it["gold"]) == 1.0
    print("prompts.py self-test pass")
