# Verified Tinker facts (2026-06-11) + gotchas

- Model lineup includes Nemotron-3-Nano-30B-A3B-BF16 (+ Super 120B, Ultra 550B).
- Pricing per Mtok, 64K ctx, "limited-time 50% discount" flag: prefill $0.13 /
  sample $0.33 / train $0.40. Storage $0.10/GB-month. Three meters: rollouts
  pay prefill+sample; updates pay train. (thinkingmachines.ai/tinker)
- ~$4.50 per GRPO iter @ 128 prompts x K=16 x ~2.5k tok. Oracle eval ~$0.70.
- Loading: create_training_client_from_state(_with_optimizer) take tinker://
  paths ONLY -> no external-weight import; warm start = re-SFT in-platform.
- Export: weights.download() + weights.build_lora_adapter() remap to HF/PEFT.
  For THIS model the raw export is NOT submittable: fused qkv / Mamba gate+x
  must be SVD-compressed to rank 32 + key renames + in_proj merge + config
  alignment -> use the user's notebooks in analysis/notebooks/
  (tinker-adapter-to-ready-to-submit-adapter.ipynb FORCED_FUSED_RANK=32, then
  tinker-submission-notebook.ipynb verify+zip). SELECT ON CONVERTED ADAPTER.
- Proof of platform fit: the 0.86 reference adapter (asalhi-tinker) is a
  Tinker product: rank 32, lm_head LoRA included, passed the LB at 0.85-0.86.
- Cookbook: github.com/thinking-machines-lab/tinker-cookbook (nightly branch
  needed for weights/_adapter used by the conversion notebook). RL + SFT
  recipes live there; API names iterate quickly — verify against installed ver.
- Docs: tinker-docs.thinkingmachines.ai (save-load, lora-primer, export-hf).
