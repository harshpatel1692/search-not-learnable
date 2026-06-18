# Tinker run ledger — every run/checkpoint/eval/cost gets a row. Append-only.

base model id (cookbook-confirmed 2026-06-11; server-confirm at phase0_smoke): nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
discount applies (check console billing page vs NOTES_API prices): ? — user funded $20 on 2026-06-11

| run / event | when | details |
|---|---|---|
| key-check | 2026-06-11 | key VALID (auth passes); 402 billing-blocked — account unfunded. NOTE: SDK retries 402 forever ("job is paused"), scripts HANG (not fail) on zero balance. |
| phase0-smoke | 2026-06-11 | PASSED (funded $20). Server confirms nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (41 models). rank-32 LoRA client OK (is_lora=True, lora_rank=32). Tokenizer parity client==competition TRUE. 3 greedy base bit samples sane ("We need to infer transformation rule..."). dummy state=tinker://775f069c-a53a-579b-8c67-158db64b9524:train:0/weights/phase0-dummy sampler=.../sampler_weights/phase0-dummy-sampler |
| phase0-export | 2026-06-11 | archive endpoint serves ONLY sampler_weights/ ckpts (400 on weights/). Downloaded sampler archive 1.54GB -> /tmp/tinker_dummy_ckpt.tar = adapter_config.json (PEFT r=32 alpha=32 all-linear) + adapter_model.safetensors. NEXT: conversion-chain notebooks on it. |
| v16-warmstart-e1 | START 2026-06-11 21:30:09 | sft warmstart 7987 ex x 1 ep, lr 0.0002, est $8.92 |
| v16-warmstart-e1 | ckpt step 100 | tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/weights/v16-warmstart-e1-step100 |
| dummy-convert | 2026-06-11 | convert_local.py (CPU, no 60GB base: remote header reads) on the 0-step dummy -> key-diff vs reference CLEAN (12010/12010 keys+shapes+dtypes), config identical, all 6005 lora_B == 0 (no-op proven). |
| dummy-submit | 2026-06-11 | SUBMITTED to Kaggle (e2e proof). Expect ~base 0.46. GOTCHA: file MUST be named submission.zip (400 otherwise). |
| v16-warmstart-e1 | ckpt step 200 | tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/weights/v16-warmstart-e1-step200 |
| v16-warmstart-e1 | ckpt step 249 | tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/weights/v16-warmstart-e1-step249 |
| v16-warmstart-e1 | DONE 2026-06-11 21:52:18 | state=tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/weights/v16-warmstart-e1-step249 sampler=tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/sampler_weights/v16-warmstart-e1-sampler | 22 min |
| v16-warmstart-e1 | 2026-06-11 | TRAINED: 249 steps / 22 min, lr 2e-4 linear, final nll ~0.001. state=tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/weights/v16-warmstart-e1-step249 sampler=.../sampler_weights/v16-warmstart-e1-sampler. ~$9 list est. |
| eval v16-warmstart-e1 | 2026-06-11 21:55:26 | ckpt=tinker://09002ee4-9aaa-50fa-b60a-1343fa6c558f:train:0/sampler_weights/v16-warmstart-e1-sampler | weighted 0.8165 | $1.01 | tinker/evals/v16-warmstart-e1.csv |
| dummy-submit SCORED | 2026-06-12 | LB 0.50 == base (no-op adapter; base ~.466 train, binomial noise on 500). E2E CHAIN CERTIFIED: tinker ckpt -> download -> convert_local.py -> submission.zip -> vLLM loads -> scores. PHASE 0 COMPLETE. |
| crypt-boot-r1 | START 2026-06-11 23:01:07 | sft warmstart 1138 ex x 3 ep, lr 0.0002, est $1.38 |
| crypt-boot-r1 | ckpt step 35 | tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/weights/crypt-boot-r1-step35 |
| warmstart-submit | 2026-06-12 | v16-warmstart-e1 CONVERTED adapter submitted to Kaggle (SVD-on-real-weights calibration: oracle 0.8165 -> LB ?). |
| crypt-boot-r1 | 2026-06-12 | LAUNCHED: crypt-ONLY isolation (user directive). 1138 traces (deduce 1011 + guess 127, v15 renders) from BASE @2e-4 x3ep = 105 steps ~$1.4. Eval auto-chained (crypt-only 200 rows). epoch-1 nll 0.146@35. |
| crypt-boot-r1 | ckpt step 70 | tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/weights/crypt-boot-r1-step70 |
| crypt-boot-r1 | ckpt step 105 | tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/weights/crypt-boot-r1-step105 |
| crypt-boot-r1 | DONE 2026-06-11 23:10:21 | state=tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/weights/crypt-boot-r1-step105 sampler=tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/sampler_weights/crypt-boot-r1-sampler | 9 min |
| eval crypt-boot-r1 | 2026-06-11 23:11:52 | ckpt=tinker://74c392a0-fee5-5b62-b5de-3daa7a566519:train:0/sampler_weights/crypt-boot-r1-sampler | weighted 0.0023 | $0.07 | tinker/evals/crypt-boot-r1.csv |
| warmstart LB SCORED | 2026-06-12 | 53588517 = 0.79 vs oracle 0.8165 -> SVD conversion cost ~2.6pts (<=1.5sd of 500-row noise, plan on ~2-3 real). CALIBRATION RULE: converted LB ~= oracle - 0.025; RL ckpt needs oracle >=~0.876 to beat banked 0.85. |
| crypt-boot-r2 | START 2026-06-11 23:58:49 | sft warmstart 3438 ex x 2 ep, lr 0.0002, est $3.33 |
| crypt-boot-r2 | ckpt step 107 | tinker://6b748402-70a7-56e2-888c-33cad98f6c9a:train:0/weights/crypt-boot-r2-step107 |
| crypt-boot-r2 | ckpt step 214 | tinker://6b748402-70a7-56e2-888c-33cad98f6c9a:train:0/weights/crypt-boot-r2-step214 |
| crypt-boot-r2 | DONE 2026-06-12 00:15:49 | state=tinker://6b748402-70a7-56e2-888c-33cad98f6c9a:train:0/weights/crypt-boot-r2-step214 sampler=tinker://6b748402-70a7-56e2-888c-33cad98f6c9a:train:0/sampler_weights/crypt-boot-r2-sampler | 17 min |
| eval crypt-boot-r2 | 2026-06-12 00:17:49 | ckpt=tinker://6b748402-70a7-56e2-888c-33cad98f6c9a:train:0/sampler_weights/crypt-boot-r2-sampler | weighted 0.0050 | $0.11 | tinker/evals/crypt-boot-r2.csv |
| crypt-boot-r2 | 2026-06-12 | TRAINED (3438 r2 traces, 2ep, $3.3) + eval: crypt_deduce 0.07 (r1 0.03), guess 0.01, trunc 2%. DIAGNOSIS: derivation now real on both sides AND mode-backtracking works, but match/ok learned as CONSTANT (writes match over real mismatches; distinct-ok over duplicates) - training data never varied the condition. r3 = wrong-branch episodes + truthful-comparison lint. |
| crypt-boot-r3 | START 2026-06-12 01:19:49 | sft warmstart 3437 ex x 2 ep, lr 0.0002, est $3.64 |
| crypt-boot-r3 | ckpt step 107 | tinker://244a956d-f9ab-5fea-bd72-49fc8e452da9:train:0/weights/crypt-boot-r3-step107 |
| crypt-boot-r3 | ckpt step 214 | tinker://244a956d-f9ab-5fea-bd72-49fc8e452da9:train:0/weights/crypt-boot-r3-step214 |
| crypt-boot-r3 | DONE 2026-06-12 01:36:34 | state=tinker://244a956d-f9ab-5fea-bd72-49fc8e452da9:train:0/weights/crypt-boot-r3-step214 sampler=tinker://244a956d-f9ab-5fea-bd72-49fc8e452da9:train:0/sampler_weights/crypt-boot-r3-sampler | 17 min |
| eval crypt-boot-r3 | 2026-06-12 01:38:48 | ckpt=tinker://244a956d-f9ab-5fea-bd72-49fc8e452da9:train:0/sampler_weights/crypt-boot-r3-sampler | weighted 0.0036 | $0.40 | tinker/evals/crypt-boot-r3.csv |
| crypt-boot-r5 | START 2026-06-12 03:53:43 | sft warmstart 3515 ex x 2 ep, lr 0.0002, est $4.20 |
| crypt-boot-r5 | ckpt step 107 | tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/weights/crypt-boot-r5-step107 |
| crypt-boot-r5 | ckpt step 214 | tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/weights/crypt-boot-r5-step214 |
| crypt-boot-r5 | ckpt step 218 | tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/weights/crypt-boot-r5-step218 |
| crypt-boot-r5 | DONE 2026-06-12 04:11:10 | state=tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/weights/crypt-boot-r5-step218 sampler=tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/sampler_weights/crypt-boot-r5-sampler | 17 min |
| eval crypt-boot-r5 | 2026-06-12 04:13:39 | ckpt=tinker://0e95f309-65e6-5724-a6d9-4cedb0c72578:train:0/sampler_weights/crypt-boot-r5-sampler | weighted 0.0028 | $0.40 | tinker/evals/crypt-boot-r5.csv |
| crypt-boot-r6 | START 2026-06-12 04:54:03 | sft warmstart 3515 ex x 2 ep, lr 0.0002, est $4.93 |
| crypt-boot-r6 | ckpt step 107 | tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step107 |
| crypt-boot-r6 | ckpt step 214 | tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step214 |
| crypt-boot-r6 | ckpt step 218 | tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step218 |
| crypt-boot-r6 | DONE 2026-06-12 05:10:00 | state=tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step218 sampler=tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/sampler_weights/crypt-boot-r6-sampler | 16 min |
| eval crypt-boot-r6 | 2026-06-12 05:13:20 | ckpt=tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/sampler_weights/crypt-boot-r6-sampler | weighted 0.0035 | $0.47 | tinker/evals/crypt-boot-r6.csv |
| crypt-grpo-smoke | START 2026-06-12 05:33:25 | curriculum=cur_tier1_new.jsonl adaptive=False | iters=2 batch=16 k=8 maxtok=4000 temp=0.8 | from=tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step218 | GRPO lr=1e-05 kl=0.0 |
| crypt-grpo-smoke2 | START 2026-06-12 05:36:26 | curriculum=cur_tier1_new.jsonl adaptive=False | iters=2 batch=16 k=8 maxtok=4000 temp=0.8 | from=tinker://44542fac-6f18-5b66-95a0-4703989e5837:train:0/weights/crypt-boot-r6-step218 | GRPO lr=1e-05 kl=0.0 |
| crypt-grpo-smoke2 | ckpt it2 | state=tinker://8171d113-d219-58e9-ac2e-0f3c0a6690d7:train:0/weights/crypt-grpo-smoke2-it0002 sampler=tinker://8171d113-d219-58e9-ac2e-0f3c0a6690d7:train:0/sampler_weights/crypt-grpo-smoke2-it0002-sampler | succ 0.445 | $0 |
| crypt-grpo-smoke2 | DONE 2026-06-12 05:39:35 | $0.21 |
| crypt-grpo-smoke | NOTE | attempt 1 crashed in forward_backward: server ImportanceSampling loss rejects "mask" loss_fn_input -> rl_loop.py fixed (advantage-0 masking); ~$0.10 sampling spent |
| crypt-grpo-smoke2 | SUMMARY 2026-06-12 | GRPO smoke PASS: 2 it, batch 16 K=8 tier-1 (50% concat / 50% eased-value), succ .453/.445, 7/16 non-degen groups + 56 datums/iter, ckpt OK, $0.21 | starting-signal probes: plain-value t1 0.000, eased-value 0.004, concat 0.742 ($0.65) | full run gated on budget: see analysis/reports/crypt_grpo_setup.md |
| crypt-boot-r7 | START 2026-06-12 11:15:03 | sft warmstart 3695 ex x 2 ep, lr 0.0002, est $2.95 |
| crypt-boot-r7 | ckpt step 107 | tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/weights/crypt-boot-r7-step107 |
| crypt-boot-r7 | ckpt step 214 | tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/weights/crypt-boot-r7-step214 |
| crypt-boot-r7 | ckpt step 230 | tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/weights/crypt-boot-r7-step230 |
| crypt-boot-r7 | DONE 2026-06-12 11:30:48 | state=tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/weights/crypt-boot-r7-step230 sampler=tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/sampler_weights/crypt-boot-r7-sampler | 16 min |
| eval crypt-boot-r7 | 2026-06-12 11:32:28 | ckpt=tinker://a92961d0-5884-561c-9614-12d6f80af4bb:train:0/sampler_weights/crypt-boot-r7-sampler | weighted 0.0049 | $0.07 | tinker/evals/crypt-boot-r7.csv |
| crypt-boot-r8 | START 2026-06-12 15:33:50 | sft warmstart 3693 ex x 2 ep, lr 0.0002, est $6.74 |
| crypt-boot-r8 | ckpt step 107 | tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step107 |
| crypt-boot-r8 | ckpt step 214 | tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step214 |
| crypt-boot-r8 | ckpt step 230 | tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step230 |
| crypt-boot-r8 | DONE 2026-06-12 15:54:24 | state=tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step230 sampler=tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/sampler_weights/crypt-boot-r8-sampler | 21 min |
| eval crypt-boot-r8 | 2026-06-12 15:56:40 | ckpt=tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/sampler_weights/crypt-boot-r8-sampler | weighted 0.0036 | $0.17 | tinker/evals/crypt-boot-r8.csv |
| bit-global-r1 | START 2026-06-12 22:03:44 | sft warmstart 3302 ex x 2 ep, lr 0.0002, est $4.13 |
| bit-global-r1 | ckpt step 103 | tinker://94d16a17-0f52-55f0-837b-036d1899b960:train:0/weights/bit-global-r1-step103 |
| bit-global-r1 | START 2026-06-13 ~00:40 | sft FROM BASE 2e-4 2ep, train_bitglobal.csv (3302 = bit_real 1502 + bit_synth 1800), the v15 GLOBAL single-rule CoT (98.56% solver) — NEVER trained from base before (only warm-start v15=LB.82). eval = 100 val bit. log /tmp/bitglobal_train.log. Tests if bit .61(per-bit)->.85-.92(global). |
| bit-global-r1 | ckpt step 206 | tinker://94d16a17-0f52-55f0-837b-036d1899b960:train:0/weights/bit-global-r1-step206 |
| bit-global-r1 | DONE 2026-06-12 22:18:55 | state=tinker://94d16a17-0f52-55f0-837b-036d1899b960:train:0/weights/bit-global-r1-step206 sampler=tinker://94d16a17-0f52-55f0-837b-036d1899b960:train:0/sampler_weights/bit-global-r1-sampler | 15 min |
| eval bit-global-r1 | 2026-06-12 22:21:15 | ckpt=tinker://94d16a17-0f52-55f0-837b-036d1899b960:train:0/sampler_weights/bit-global-r1-sampler | weighted 0.0874 | $0.50 | tinker/evals/bit-global-r1.csv |
| bit-global-r2-synth | START 2026-06-12 22:21:43 | sft warmstart 1800 ex x 2 ep, lr 0.0002, est $2.25 |
| bit-global-r2-synth | ckpt step 56 | tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/weights/bit-global-r2-synth-step56 |
| bit-global-r2-synth | ckpt step 112 | tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/weights/bit-global-r2-synth-step112 |
| bit-global-r2-synth | DONE 2026-06-12 22:31:00 | state=tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/weights/bit-global-r2-synth-step112 sampler=tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/sampler_weights/bit-global-r2-synth-sampler | 9 min |
| bit-global-r2-synth | 2026-06-13 ~01:20 | sft FROM BASE 2e-4 2ep, train_bitglobal_synth.csv (1800 SYNTH-ONLY, zero real rows). CLEAN eval = 500 real bit (bit_eval500.jsonl, true gold, 0 leakage). The honest global-format number. sampler tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/sampler_weights/bit-global-r2-synth-sampler |
| eval bit-global-r2-synth | 2026-06-12 22:33:28 | ckpt=tinker://4c690f9c-3bdb-5c1e-8e55-cd7917738ab6:train:0/sampler_weights/bit-global-r2-synth-sampler | weighted 0.0256 | $0.51 | tinker/evals/bit-global-r2-synth.csv |
| bit-r3 | START 2026-06-12 22:58:38 | sft warmstart 0 ex x 2 ep, lr 0.0002, est $0.00 |
| bit-r3 | DONE 2026-06-12 22:59:00 | state=None sampler=tinker://bec73adf-3759-5726-87d0-794723d98845:train:0/sampler_weights/bit-r3-sampler | 0 min |
| bit-r3 | START 2026-06-12 23:00:26 | sft warmstart 1673 ex x 2 ep, lr 0.0002, est $3.56 |
| bit-r3 | 2026-06-13 ~02:00 | sft FROM BASE 2e-4 2ep, train_bit_r3.csv (1673 SYNTH-only, the SCAN-CAP fix: terse train==inference prior-ordered scan + real re-apply verify; fixes the v1 imitation-gap truncation + constant-ok verify). CLEAN eval 500 real bit. Tests if global-format-FIXED beats r2 .15 and per-bit .80. |
| bit-r3 | ckpt step 53 | tinker://f19eb50a-ea9a-5561-b371-6c8018db5c21:train:0/weights/bit-r3-step53 |
| bit-r3 | ckpt step 104 | tinker://f19eb50a-ea9a-5561-b371-6c8018db5c21:train:0/weights/bit-r3-step104 |
| bit-r3 | DONE 2026-06-12 23:08:22 | state=tinker://f19eb50a-ea9a-5561-b371-6c8018db5c21:train:0/weights/bit-r3-step104 sampler=tinker://f19eb50a-ea9a-5561-b371-6c8018db5c21:train:0/sampler_weights/bit-r3-sampler | 8 min |
| eval bit-r3 | 2026-06-12 23:10:45 | ckpt=tinker://f19eb50a-ea9a-5561-b371-6c8018db5c21:train:0/sampler_weights/bit-r3-sampler | weighted 0.0887 | $0.57 | tinker/evals/bit-r3.csv |
| bit-dialect2-r1 | START 2026-06-12 23:54:32 | sft warmstart 1101 ex x 2 ep, lr 0.0002, est $3.77 |
| bit-dialect2-r1 | ckpt step 34 | tinker://eebd4db5-f87e-5a76-bed1-d5062d723031:train:0/weights/bit-dialect2-r1-step34 |
| bit-dialect2-r1 | ckpt step 68 | tinker://eebd4db5-f87e-5a76-bed1-d5062d723031:train:0/weights/bit-dialect2-r1-step68 |
| bit-dialect2-r1 | DONE 2026-06-12 23:59:53 | state=tinker://eebd4db5-f87e-5a76-bed1-d5062d723031:train:0/weights/bit-dialect2-r1-step68 sampler=tinker://eebd4db5-f87e-5a76-bed1-d5062d723031:train:0/sampler_weights/bit-dialect2-r1-sampler | 5 min |
| eval bit-dialect2-r1 | 2026-06-13 00:02:34 | ckpt=tinker://eebd4db5-f87e-5a76-bed1-d5062d723031:train:0/sampler_weights/bit-dialect2-r1-sampler | weighted 0.0078 | $1.03 | tinker/evals/bit-dialect2-r1.csv |
| bit-star-r1 | START 2026-06-13 00:11:28 | sft warmstart 1984 ex x 2 ep, lr 0.0002, est $6.39 |
| bit-star-r1 | ckpt step 62 | tinker://1270095b-fed5-5e8e-b431-943b405fa3fc:train:0/weights/bit-star-r1-step62 |
| bit-star-r1 | ckpt step 124 | tinker://1270095b-fed5-5e8e-b431-943b405fa3fc:train:0/weights/bit-star-r1-step124 |
| bit-star-r1 | DONE 2026-06-13 00:20:03 | state=tinker://1270095b-fed5-5e8e-b431-943b405fa3fc:train:0/weights/bit-star-r1-step124 sampler=tinker://1270095b-fed5-5e8e-b431-943b405fa3fc:train:0/sampler_weights/bit-star-r1-sampler | 9 min |
| eval bit-star-r1 | 2026-06-13 00:21:45 | ckpt=tinker://1270095b-fed5-5e8e-b431-943b405fa3fc:train:0/sampler_weights/bit-star-r1-sampler | weighted 0.1106 | $0.65 | tinker/evals/bit-star-r1.csv |
| bit-star-r2 | START 2026-06-13 00:28:01 | sft warmstart 2163 ex x 2 ep, lr 0.0002, est $6.99 |
| bit-star-r2 | ckpt step 68 | tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/weights/bit-star-r2-step68 |
| bit-star-r2 | ckpt step 134 | tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/weights/bit-star-r2-step134 |
| bit-star-r2 | DONE 2026-06-13 00:38:57 | state=tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/weights/bit-star-r2-step134 sampler=tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/sampler_weights/bit-star-r2-sampler | 11 min |
| eval bit-star-r2 | 2026-06-13 00:41:04 | ckpt=tinker://b99d43c6-1d41-5d0c-a989-744f5f09fc01:train:0/sampler_weights/bit-star-r2-sampler | weighted 0.1143 | $0.65 | tinker/evals/bit-star-r2.csv |
| crypt-twn-r9 | START 2026-06-14 02:52:48 | sft warmstart 3180 ex x 2 ep, lr 0.0002, est $2.21 |
| crypt-twn-r9 | ckpt step 100 | tinker://9b8f6686-bf5d-5493-8db0-3ab98800fc81:train:0/weights/crypt-twn-r9-step100 |
| crypt-twn-r9 | ckpt step 198 | tinker://9b8f6686-bf5d-5493-8db0-3ab98800fc81:train:0/weights/crypt-twn-r9-step198 |
| crypt-twn-r9 | DONE 2026-06-14 03:06:52 | state=tinker://9b8f6686-bf5d-5493-8db0-3ab98800fc81:train:0/weights/crypt-twn-r9-step198 sampler=tinker://9b8f6686-bf5d-5493-8db0-3ab98800fc81:train:0/sampler_weights/crypt-twn-r9-sampler | 14 min |
| eval crypt-twn-r9 | 2026-06-14 03:09:34 | ckpt=tinker://9b8f6686-bf5d-5493-8db0-3ab98800fc81:train:0/sampler_weights/crypt-twn-r9-sampler | weighted 0.0007 | $0.38 | tinker/evals/crypt-twn-r9.csv |
| crypt-twn-r10 | START 2026-06-14 03:54:33 | sft warmstart 2094 ex x 2 ep, lr 0.0002, est $1.31 |
| crypt-twn-r10 | ckpt step 100 | tinker://37e504df-b45c-5a20-a082-8ed3bcad2258:train:0/weights/crypt-twn-r10-step100 |
| crypt-twn-r10 | ckpt step 130 | tinker://37e504df-b45c-5a20-a082-8ed3bcad2258:train:0/weights/crypt-twn-r10-step130 |
| crypt-twn-r10 | DONE 2026-06-14 04:03:15 | state=tinker://37e504df-b45c-5a20-a082-8ed3bcad2258:train:0/weights/crypt-twn-r10-step130 sampler=tinker://37e504df-b45c-5a20-a082-8ed3bcad2258:train:0/sampler_weights/crypt-twn-r10-sampler | 9 min |
| eval crypt-twn-r10 | 2026-06-14 04:05:35 | ckpt=tinker://37e504df-b45c-5a20-a082-8ed3bcad2258:train:0/sampler_weights/crypt-twn-r10-sampler | weighted 0.0007 | $0.05 | tinker/evals/crypt-twn-r10.csv |
| crypt-twn-r11 | START 2026-06-14 04:22:17 | sft warmstart 2085 ex x 2 ep, lr 0.0002, est $1.40 |
| crypt-twn-r11 | ckpt step 100 | tinker://d183ea9a-08b5-52f0-a53c-c8467f0cec01:train:0/weights/crypt-twn-r11-step100 |
| crypt-twn-r11 | ckpt step 130 | tinker://d183ea9a-08b5-52f0-a53c-c8467f0cec01:train:0/weights/crypt-twn-r11-step130 |
| crypt-twn-r11 | DONE 2026-06-14 04:28:30 | state=tinker://d183ea9a-08b5-52f0-a53c-c8467f0cec01:train:0/weights/crypt-twn-r11-step130 sampler=tinker://d183ea9a-08b5-52f0-a53c-c8467f0cec01:train:0/sampler_weights/crypt-twn-r11-sampler | 6 min |
| eval crypt-twn-r11 | 2026-06-14 04:30:18 | ckpt=tinker://d183ea9a-08b5-52f0-a53c-c8467f0cec01:train:0/sampler_weights/crypt-twn-r11-sampler | weighted 0.0007 | $0.09 | tinker/evals/crypt-twn-r11.csv |
| bit-3tap | START 2026-06-14 05:13:13 | sft warmstart 4600 ex x 2 ep, lr 0.0002, est $3.45 |
| bit-3tap | ckpt step 100 | tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/weights/bit-3tap-step100 |
| bit-3tap | ckpt step 200 | tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/weights/bit-3tap-step200 |
| bit-3tap | ckpt step 286 | tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/weights/bit-3tap-step286 |
| bit-3tap | DONE 2026-06-14 05:34:31 | state=tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/weights/bit-3tap-step286 sampler=tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/sampler_weights/bit-3tap-sampler | 21 min |
| eval bit-3tap | 2026-06-14 05:36:24 | ckpt=tinker://e6178dcc-baeb-5bb3-8880-8dd09630b1c5:train:0/sampler_weights/bit-3tap-sampler | weighted 0.0105 | $0.35 | tinker/evals/bit-3tap.csv |
| bit-star-r3 | START 2026-06-14 06:07:04 | sft warmstart 1067 ex x 2 ep, lr 0.0002, est $3.46 |
| bit-star-r3 | BLOCKED 2026-06-14 06:10 | TRAIN crashed step 47/66 -- Tinker 402 BILLING (balance exhausted). Harvest DONE (998 traces, pass@12=0.90); train corpus pipeline/data/v16/train_bit_star_r3.csv (1067 rows, 3t=301) READY. Retrain is ~$3.46/5min once billing restored: bash tinker/run_bit_star_r3.sh (or just the TRAIN+EVAL steps; harvest can be skipped). |
| bit-star-r3 | START 2026-06-14 08:31:44 | sft warmstart 1067 ex x 2 ep, lr 0.0002, est $3.46 |
| bit-star-r3 | ckpt step 66 | tinker://607d9c42-6d2b-5188-88ac-433d9ed552ab:train:0/weights/bit-star-r3-step66 |
| bit-star-r3 | DONE 2026-06-14 08:35:53 | state=tinker://607d9c42-6d2b-5188-88ac-433d9ed552ab:train:0/weights/bit-star-r3-step66 sampler=tinker://607d9c42-6d2b-5188-88ac-433d9ed552ab:train:0/sampler_weights/bit-star-r3-sampler | 4 min |
| eval bit-star-r3 | 2026-06-14 08:39:25 | ckpt=tinker://607d9c42-6d2b-5188-88ac-433d9ed552ab:train:0/sampler_weights/bit-star-r3-sampler | weighted 0.0067 | $0.94 | tinker/evals/bit-star-r3.csv |
| bit-synth3 | START 2026-06-14 08:46:59 | sft warmstart 1836 ex x 2 ep, lr 0.0002, est $6.05 |
| bit-synth3 | BLOCKED 2026-06-14 ~06:45 | harvest DONE (753 synth-3tap CoTs, pass@10=0.58); corpus train_bit_synth3.csv READY (1836 rows, 1619 distinct prompts, 1069 3-tap CoTs = 309 real + 760 synth). TRAIN crashed step 19/114 -- Tinker 402 BILLING again. Retrain+eval is ~$4-5: bash -c '<TRAIN+EVAL steps of tinker/run_bit_synth3.sh>' once topped up. Eval = bit_eval500 by tap (3-tap slice = 169 real train rows). |
| bit-synth3 | START 2026-06-14 09:18:00 | sft warmstart 1836 ex x 2 ep, lr 0.0002, est $6.05 |
| bit-synth3 | ckpt step 100 | tinker://3101310e-4365-548d-bce8-7de5c4f49eb1:train:0/weights/bit-synth3-step100 |
| bit-synth3 | ckpt step 114 | tinker://3101310e-4365-548d-bce8-7de5c4f49eb1:train:0/weights/bit-synth3-step114 |
| bit-synth3 | ckpt step 100 | tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/weights/bit-synth3-step100 |
| bit-synth3 | DONE 2026-06-14 09:25:55 | state=tinker://3101310e-4365-548d-bce8-7de5c4f49eb1:train:0/weights/bit-synth3-step114 sampler=tinker://3101310e-4365-548d-bce8-7de5c4f49eb1:train:0/sampler_weights/bit-synth3-sampler | 39 min |
| bit-synth3 | ckpt step 114 | tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/weights/bit-synth3-step114 |
| bit-synth3 | DONE 2026-06-14 09:26:43 | state=tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/weights/bit-synth3-step114 sampler=tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/sampler_weights/bit-synth3-sampler | 9 min |
| eval bit-synth3 | 2026-06-14 09:28:46 | ckpt=tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/sampler_weights/bit-synth3-sampler | weighted 0.1015 | $0.65 | tinker/evals/bit-synth3.csv |
| eval bit-synth3 | 2026-06-14 ~09:30 | ckpt=tinker://d84aa386-d027-5bb4-9dfe-f88400d326c9:train:0/sampler_weights/bit-synth3-sampler | bit 0.602 (1t.905/2t.654/3t.456) | WORSE than r2 0.678. Synth 3-tap STaR (760 synth CoTs + diversity-restored corpus) did NOT help: no collapse (vs r3 0.04) but 2-tap diluted .76->.65 and 3-tap .50->.46. Conclusion: r2 0.678 is the bit ceiling on this base; synth-3tap dilutes the proven real-row mix. |
| bit-peel | START 2026-06-14 11:14:08 | sft warmstart 2000 ex x 3 ep, lr 0.0002, est $2.26 |
| bit-peel | ckpt step 100 | tinker://a8f1e51e-0d76-5bbe-8b64-d2dc1975ab9a:train:0/weights/bit-peel-step100 |
| bit-peel | ckpt step 186 | tinker://a8f1e51e-0d76-5bbe-8b64-d2dc1975ab9a:train:0/weights/bit-peel-step186 |
| bit-peel | DONE 2026-06-14 11:27:34 | state=tinker://a8f1e51e-0d76-5bbe-8b64-d2dc1975ab9a:train:0/weights/bit-peel-step186 sampler=tinker://a8f1e51e-0d76-5bbe-8b64-d2dc1975ab9a:train:0/sampler_weights/bit-peel-sampler | 13 min |
| eval bit-peel | 2026-06-14 11:28:06 | ckpt=tinker://a8f1e51e-0d76-5bbe-8b64-d2dc1975ab9a:train:0/sampler_weights/bit-peel-sampler | weighted 0.0152 | $0.13 | tinker/evals/bit-peel.csv |
| bit-2tap | START 2026-06-14 12:33:29 | sft warmstart 1750 ex x 1 ep, lr 0.0002, est $2.84 |
| crypt-honest-r1 | START 2026-06-15 08:29:28 | sft warmstart 1011 ex x 4 ep, lr 0.0002, est $1.38 |
| crypt-honest-r1 | ckpt step 100 | tinker://fda9f84c-7d37-537d-8048-ac364bc84e24:train:0/weights/crypt-honest-r1-step100 |
| crypt-honest-r1 | ckpt step 124 | tinker://fda9f84c-7d37-537d-8048-ac364bc84e24:train:0/weights/crypt-honest-r1-step124 |
| crypt-honest-r1 | DONE 2026-06-15 08:37:18 | state=tinker://fda9f84c-7d37-537d-8048-ac364bc84e24:train:0/weights/crypt-honest-r1-step124 sampler=tinker://fda9f84c-7d37-537d-8048-ac364bc84e24:train:0/sampler_weights/crypt-honest-r1-sampler | 8 min |
| eval crypt-honest-r1 | 2026-06-15 08:39:02 | ckpt=tinker://fda9f84c-7d37-537d-8048-ac364bc84e24:train:0/sampler_weights/crypt-honest-r1-sampler | weighted 0.0036 | $0.08 | tinker/evals/crypt-honest-r1.csv |
| crypt-honest-r2 | START 2026-06-15 10:02:04 | sft warmstart 719 ex x 4 ep, lr 0.0002, est $1.33 |
| crypt-honest-r2 | ckpt step 88 | tinker://f01259ef-8b4d-5b7f-9f5c-49129c840c03:train:0/weights/crypt-honest-r2-step88 |
| crypt-honest-r2 | DONE 2026-06-15 10:08:30 | state=tinker://f01259ef-8b4d-5b7f-9f5c-49129c840c03:train:0/weights/crypt-honest-r2-step88 sampler=tinker://f01259ef-8b4d-5b7f-9f5c-49129c840c03:train:0/sampler_weights/crypt-honest-r2-sampler | 6 min |
| eval crypt-honest-r2 | 2026-06-15 10:10:35 | ckpt=tinker://f01259ef-8b4d-5b7f-9f5c-49129c840c03:train:0/sampler_weights/crypt-honest-r2-sampler | weighted 0.0023 | $0.25 | tinker/evals/crypt-honest-r2.csv |
| crypt-honest-r3 | START 2026-06-15 10:42:43 | sft warmstart 775 ex x 8 ep, lr 0.0002, est $2.05 |
| crypt-honest-r3 | ckpt step 100 | tinker://bd72c04f-381a-584a-8d7b-6eeaa305791f:train:0/weights/crypt-honest-r3-step100 |
| crypt-honest-r3 | ckpt step 192 | tinker://bd72c04f-381a-584a-8d7b-6eeaa305791f:train:0/weights/crypt-honest-r3-step192 |
| crypt-honest-r3 | DONE 2026-06-15 10:56:24 | state=tinker://bd72c04f-381a-584a-8d7b-6eeaa305791f:train:0/weights/crypt-honest-r3-step192 sampler=tinker://bd72c04f-381a-584a-8d7b-6eeaa305791f:train:0/sampler_weights/crypt-honest-r3-sampler | 14 min |
| eval crypt-honest-r3 | 2026-06-15 10:57:54 | ckpt=tinker://bd72c04f-381a-584a-8d7b-6eeaa305791f:train:0/sampler_weights/crypt-honest-r3-sampler | weighted 0.0021 | $0.07 | tinker/evals/crypt-honest-r3.csv |
| v19-tinker | START 2026-06-15 11:27:21 | sft warmstart 10662 ex x 1 ep, lr 0.0002, est $8.91 |
| v19-tinker | ckpt step 100 | tinker://d82bc1ea-dc34-5fd4-8235-6cad9f8fca3c:train:0/weights/v19-tinker-step100 |
| v19-tinker | ckpt step 200 | tinker://d82bc1ea-dc34-5fd4-8235-6cad9f8fca3c:train:0/weights/v19-tinker-step200 |
| v19-tinker-r2 | START 2026-06-15 11:52:00 | sft warmstart 10662 ex x 2 ep, lr 5e-06, est $17.83 |
| v19-tinker-r2 | START 2026-06-15 11:53:43 | sft warmstart 10662 ex x 1 ep, lr 5e-06, est $8.91 |
| v19-tinker-final | START 2026-06-15 11:55:54 | sft warmstart 10662 ex x 1 ep, lr 0.0002, est $8.91 |
| v19-tinker-final | ckpt step 100 | tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/weights/v19-tinker-final-step100 |
| v19-tinker-final | ckpt step 200 | tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/weights/v19-tinker-final-step200 |
| v19-tinker-final | ckpt step 300 | tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/weights/v19-tinker-final-step300 |
| v19-tinker-final | ckpt step 333 | tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/weights/v19-tinker-final-step333 |
| v19-tinker-final | DONE 2026-06-15 13:10:07 | state=tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/weights/v19-tinker-final-step333 sampler=tinker://c2fad538-fe4a-57dc-a499-1bbb07df7377:train:0/sampler_weights/v19-tinker-final-sampler | 74 min |
| crypt-grpo-proof | START 2026-06-15 14:39:59 | curriculum=crypt_curriculum.jsonl adaptive=True | iters=20 batch=32 k=8 maxtok=7680 temp=0.9 | from=tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step230 | GRPO lr=1e-05 kl=0.0 |
| crypt-grpo-proof2 | START 2026-06-15 14:46:43 | curriculum=crypt_curriculum.jsonl adaptive=True | iters=6 batch=32 k=8 maxtok=7680 temp=0.9 | from=tinker://5379b39a-f6a5-56cb-8231-cbb1a2702e02:train:0/weights/crypt-boot-r8-step230 | GRPO lr=1e-05 kl=0.0 |
| eval crypt-grpo-proof2-it2 | 2026-06-15 14:52:01 | ckpt=tinker://d361c7eb-3500-5d2d-97df-fb4a0598fc13:train:0/sampler_weights/crypt-grpo-proof2-eval-it0002 | weighted 0.0035 | $0.08 | tinker/evals/crypt-grpo-proof2-it2.csv |
| eval crypt-grpo-proof2-it4 | 2026-06-15 14:55:38 | ckpt=tinker://d361c7eb-3500-5d2d-97df-fb4a0598fc13:train:0/sampler_weights/crypt-grpo-proof2-eval-it0004 | weighted 0.0035 | $0.08 | tinker/evals/crypt-grpo-proof2-it4.csv |
| crypt-grpo-proof2 | DOLLAR-CAP $1 at iter 5 |
| eval crypt-grpo-proof2-final | 2026-06-15 14:57:53 | ckpt=tinker://d361c7eb-3500-5d2d-97df-fb4a0598fc13:train:0/sampler_weights/crypt-grpo-proof2-eval-it0004 | weighted 0.0028 | $0.08 | tinker/evals/crypt-grpo-proof2-final.csv |
| crypt-grpo-proof2 | DONE 2026-06-15 14:57:54 | $1.02 |
| crypt-dense-ctrl | START 2026-06-16 14:30:02 | sft warmstart 775 ex x 4 ep, lr 0.0002, est $1.00 |
| crypt-dense-ctrl | ckpt step 96 | tinker://af034074-1b3a-5c4d-9513-ff92f0b2ac44:train:0/weights/crypt-dense-ctrl-step96 |
| crypt-dense-ctrl | DONE 2026-06-16 14:35:30 | state=tinker://af034074-1b3a-5c4d-9513-ff92f0b2ac44:train:0/weights/crypt-dense-ctrl-step96 sampler=tinker://af034074-1b3a-5c4d-9513-ff92f0b2ac44:train:0/sampler_weights/crypt-dense-ctrl-sampler | 5 min |
| eval crypt-llama3b | 2026-06-16 14:41:01 | ckpt=tinker://af034074-1b3a-5c4d-9513-ff92f0b2ac44:train:0/sampler_weights/crypt-dense-ctrl-sampler | weighted 0.0007 | $0.18 | tinker/evals/crypt-llama3b.csv |
| crypt-qwen30a3b | START 2026-06-16 14:42:08 | sft warmstart 775 ex x 4 ep, lr 0.0002, est $0.99 |
| crypt-qwen4b | START 2026-06-16 14:42:33 | sft warmstart 775 ex x 4 ep, lr 0.0002, est $1.02 |
| crypt-qwen4b | ckpt step 96 | tinker://474121f8-1e64-50d7-811e-e82fbb48098a:train:0/weights/crypt-qwen4b-step96 |
| crypt-qwen4b | DONE 2026-06-16 14:46:32 | state=tinker://474121f8-1e64-50d7-811e-e82fbb48098a:train:0/weights/crypt-qwen4b-step96 sampler=tinker://474121f8-1e64-50d7-811e-e82fbb48098a:train:0/sampler_weights/crypt-qwen4b-sampler | 4 min |
| eval crypt-qwen4b | 2026-06-16 14:47:47 | ckpt=tinker://474121f8-1e64-50d7-811e-e82fbb48098a:train:0/sampler_weights/crypt-qwen4b-sampler | weighted 0.0028 | $0.05 | tinker/evals/crypt-qwen4b.csv |
| crypt-gptoss20b | START 2026-06-16 15:08:56 | sft warmstart 775 ex x 4 ep, lr 0.0002, est $1.04 |
| crypt-gptoss20b | ckpt step 96 | tinker://c8303e3d-6eac-56be-afc8-724b1cc1fcfa:train:0/weights/crypt-gptoss20b-step96 |
| crypt-gptoss20b | DONE 2026-06-16 15:12:31 | state=tinker://c8303e3d-6eac-56be-afc8-724b1cc1fcfa:train:0/weights/crypt-gptoss20b-step96 sampler=tinker://c8303e3d-6eac-56be-afc8-724b1cc1fcfa:train:0/sampler_weights/crypt-gptoss20b-sampler | 4 min |
| eval crypt-gptoss20b | 2026-06-16 15:14:04 | ckpt=tinker://c8303e3d-6eac-56be-afc8-724b1cc1fcfa:train:0/sampler_weights/crypt-gptoss20b-sampler | weighted 0.0010 | $0.05 | tinker/evals/crypt-gptoss20b.csv |
| crypt-qwen30a3b | START 2026-06-16 19:46:13 | sft warmstart 775 ex x 4 ep, lr 0.0002, est $0.99 |
| eval crypt-frontier-deepseek-ai_DeepSeek-V3_1 | 2026-06-16 20:45:16 | ckpt=BASE | weighted 0.0035 | $0.02 | tinker/evals/crypt-frontier-deepseek-ai_DeepSeek-V3_1.csv |
| eval crypt-frontier-nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 2026-06-16 20:48:04 | ckpt=BASE | weighted 0.0000 | $0.05 | tinker/evals/crypt-frontier-nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-BF16.csv |
| frontier-full | START 2026-06-16 21:04:39 | sft warmstart 303 ex x 3 ep, lr 0.0002, est $0.10 |
| frontier-full | ckpt step 27 | tinker://d130795d-9965-5266-bc75-968534f02381:train:0/weights/frontier-full-step27 |
| frontier-full | DONE 2026-06-16 21:06:22 | state=tinker://d130795d-9965-5266-bc75-968534f02381:train:0/weights/frontier-full-step27 sampler=tinker://d130795d-9965-5266-bc75-968534f02381:train:0/sampler_weights/frontier-full-sampler | 2 min |
| eval frontier-full | 2026-06-16 21:07:41 | ckpt=tinker://d130795d-9965-5266-bc75-968534f02381:train:0/sampler_weights/frontier-full-sampler | weighted 0.0396 | $0.00 | tinker/evals/frontier-full.csv |
| frontier-half | START 2026-06-16 21:07:48 | sft warmstart 303 ex x 3 ep, lr 0.0002, est $0.10 |
| frontier-half | ckpt step 27 | tinker://75929d2b-ebf0-5fb6-8aed-508766d453cc:train:0/weights/frontier-half-step27 |
| frontier-half | DONE 2026-06-16 21:09:25 | state=tinker://75929d2b-ebf0-5fb6-8aed-508766d453cc:train:0/weights/frontier-half-step27 sampler=tinker://75929d2b-ebf0-5fb6-8aed-508766d453cc:train:0/sampler_weights/frontier-half-sampler | 2 min |
| eval frontier-half | 2026-06-16 21:09:42 | ckpt=tinker://75929d2b-ebf0-5fb6-8aed-508766d453cc:train:0/sampler_weights/frontier-half-sampler | weighted 0.0033 | $0.00 | tinker/evals/frontier-half.csv |
