# What is needed FROM THE USER to start Tinker — exhaustive

## 1. Account + key (blocking, ~10 min)

- [ ] Create account at https://tinker.thinkingmachines.ai (Thinking Machines).
- [ ] Add billing / fund it (see budget below).
- [ ] Create an API key from the console.
- [ ] Put the key in `~/.bashrc` (NOT the repo — same convention as the Kaggle
      creds): `export TINKER_API_KEY="..."` then `source ~/.bashrc`.
- [ ] Confirm in the console that `Nemotron-3-Nano-30B-A3B-BF16` appears in the
      model list and note the exact model id string Tinker uses for it (the
      scripts take it as `--base-model`; expected something like
      `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
- [ ] Note whether the "limited-time 50% discount" applies to the listed
      prices (prefill $0.13 / sample $0.33 / train $0.40 per Mtok) or halves
      them — changes the iteration budget by 2x.

## 2. Budget decisions (blocking)

- [ ] HARD ceiling in dollars. Planning anchors:
      - Phase 0+1 (smoke + plumbing + $15 warm-start SFT): ~$25
      - Phase 2 pilot (50 GRPO iters, bit-only): ~$60–80
      - Phase 3 campaign: ~$4.50/iter -> 200 iters ~$900 list / ~$450 disc.
      The scripts enforce the cap via --max-iters and refuse to exceed it.
- [ ] Confirm auto-submit policy: any converted checkpoint that beats the
      current LB best gets submitted to Kaggle without asking (recommended:
      YES — submissions are free and best-of is kept).

## 3. Decision gates you will be asked for (non-blocking now)

- [ ] GATE 1 (after Phase 2 pilot, ~$80 spent): GO full campaign / switch to
      STaR / stop. Input: the pilot curves (group success rate, oracle bit
      accuracy at iters 0/25/50, rollout-length trend, retention spot-check).
- [ ] GATE 2 (daily during Phase 3): continue / reallocate arms / stop.
      Input: per-category oracle table per ~50 iters.

## 4. Things you do NOT need to provide

- Training data: in-repo (v16 corpus, Ali harvest, generators).
- Verifier/grader: in-repo, grader-verbatim (tinker/reward.py).
- Eval oracle: the same 900-row holdout used all along.
- Export/conversion: your two notebooks in analysis/notebooks/ are the flow;
  they run as a Kaggle notebook (internet ON) or locally.
- Kaggle credentials: already in ~/.bashrc.

## 5. Open questions an early Phase-0 run must answer (no user input needed,
      listed so they are not lost)

- Does `service_client.create_lora_training_client(base_model=..., rank=32)`
  accept rank 32 for this model? (Ali's was rank 32 -> expected yes.)
- Is lm_head included in Tinker's LoRA targeting for this model? (Ali's
  adapter has lm_head LoRA -> expected yes.)
- Exact chat-template rendering: Tinker-side renderer vs our
  `apply_chat_template(..., enable_thinking=True)` must produce byte-identical
  prompts (Phase 0 check #1).
- Throughput reality: measured sec/iter on the pilot determines whether
  Phase 3 is 200 or 400 iters within the wall-clock budget.
