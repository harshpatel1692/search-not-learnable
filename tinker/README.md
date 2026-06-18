# tinker/ — Option C workstream (RLVR on Tinker). FULL STATE + RESUME GUIDE

Last updated: 2026-06-12 (~01:00). Fresh session: read `CRYPT_LOOP.md` FIRST
(the active workstream + resume procedure), then this file, then LOG.md
entries dated 2026-06-12. Plans:
`../TINKER_PLAN.md` (readable), `../PLAN_OPTION_C.md` (technical).

## Why this workstream exists (3 sentences)

v15 proved SFT teaches our model grammar SURFACE but not the computations
inside (probe: forced-entry bit accuracy 0/30; argmax fidelity .95 yet free-run
collapse). RLVR rewards only verified-correct end-to-end solutions the model
sampled itself, which trains execution directly and cannot be satisfied by
format mimicry. Tinker is the only rentable platform with proven LoRA support
for this exact model (the 0.86 reference adapter came from Tinker).

## Current state of EVERYTHING (2026-06-11)

| track | state |
|---|---|
| LB best (banked) | **0.85** (Jun 4, Ali). v15 = 0.82. **v16 = 0.72 REGRESSION** -> Ali+patches line is DEAD (new grammars don't install at 5e-6 on Ali; from-base 2e-4 installs everything — 3-run pattern). |
| public LB (06-12) | leader 0.90; 12-team wall 0.87-0.88 (= no-crypt ceiling). Win = 0.92-0.93 = crypt_d .55-.62 + bit .90-.93. |
| Tinker | **FULLY OPERATIONAL + E2E CERTIFIED.** Phase 0+1 done: warm start oracle 0.8165 / LB 0.79 -> SVD calibration LB ~= oracle - 0.025. ~$15 list spent of $20. |
| **CRYPT HAMMER LOOP** | **THE active workstream (user directive). START AT `CRYPT_LOOP.md`** — r1 .03 -> r2 .07, r3 in flight (truthful conditionals). |
| deadline | 2026-06-15. ~2.5 days runway. |

## What is in this folder

| file | status | what |
|---|---|---|
| `README.md` | — | this file |
| `REQUIREMENTS.md` | — | exhaustive list of what the USER must provide/decide |
| `CHECKLIST.md` | — | phase-by-phase execution plan w/ status boxes, commands, gates, costs |
| `reward.py` | **DONE, tested locally** | strict-boxed extraction + grader-verbatim verify() = the RL reward |
| `prompts.py` | **DONE, tested locally** | category-weighted prompt sampler: real train rows (never the 900 val) + fresh synthetic from our generators |
| `sft_warmstart.py` | **DONE (real SDK), --dry-run tested** | behavioral-clone v16 corpus on Tinker; kernel-verbatim masking; 7987 ex / 249 steps / ~$9 |
| `rl_loop.py` | **DONE (real SDK), imports tested** | GRPO loop: K=16, group adv, optional KL-to-ref, ckpt/25, eval/50, realized-$ cap; --star fallback |
| `eval_oracle.py` | **DONE (real SDK), imports tested** | greedy 900-row holdout via Tinker sampling -> per-cat table + weighted-LB estimate (~$0.70) |
| `phase0_smoke.py` | **DONE — run when key lands** | ONE command covering all key-required Phase-0 checks (~$1): models/client/tokenizer/base-sample/dummy-ckpt-download |
| `NOTES_API.md` | — | everything verified about Tinker's API/pricing + the export/conversion chain |

## The non-negotiable facts a fresh session must know

1. **Tinker checkpoints are NOT directly submittable.** Conversion chain (user-
   provided, in `../analysis/notebooks/`): `tinker-adapter-to-ready-to-submit-
   adapter.ipynb` (SVD-compress fused projections to rank 32, FORCED_FUSED_RANK)
   then `tinker-submission-notebook.ipynb` (config target_modules alignment,
   key rename model->backbone, gate_proj+x_proj -> in_proj merge, key-diff vs
   reference, vLLM test-gen, zip). SELECT CHECKPOINTS ON THE CONVERTED ADAPTER.
2. **No external-weight import into Tinker** (load paths take tinker:// only).
   Warm start = re-SFT the v16 corpus on Tinker (~25M train tok ~= $10-15).
3. **Reward must be STRICTER than the grader**: boxed-only extraction (the
   grader's last-number fallback is a reward-hacking surface).
4. **Never train or eval on the 900 val ids** (`pipeline/data/val.jsonl`) —
   they are the LB oracle.
5. Pricing (confirmed 2026-06-11, nemotron-3-nano, 64K ctx, 50%-discount flag):
   prefill $0.13 / sample $0.33 / train $0.40 per Mtok; ~$4.50 per GRPO iter at
   128 prompts x K=16 x ~2.5k tok; oracle eval ~$0.70.
6. Arm allocation + kill rules: bit 45%, crypt_deduce 25% (kill after 2 flat
   eval windows), eq_deduce 15%, retention-pin 15%, guess cats 0% (Bayes-capped).

## Resume procedure for a fresh session

1. `git log --oneline -5` + `tail -150 LOG.md` — what happened since this file.
2. Check Kaggle state: `kaggle kernels status harshpatel1692/nemotron-v16-train`
   ; `kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge | head -5`.
3. Open `CHECKLIST.md`, find the first unchecked box, continue from there.
4. If Tinker work already started: the run ledger is `tinker/RUNS.md` (created
   by the scripts; every run id, tinker:// path, cost, eval result gets a row).
