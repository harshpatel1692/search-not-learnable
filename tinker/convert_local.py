"""Local (CPU) Tinker->PEFT conversion — the analysis/notebooks/tinker-adapter-
to-ready-to-submit-adapter.ipynb flow, runnable WITHOUT the 60GB base model.

Two patches on tinker_cookbook.weights._adapter:
  1. _merge_fused_projections: SVD-compress fused modules to FORCED_RANK=32
     (VERBATIM from the notebook — huikang lineage).
  2. resolve_model_dir / get_model_state_shapes: the converter only ever needs
     base-model CONFIG + tensor SHAPES, so fetch config.json via hf_hub_download
     and read safetensors headers remotely (HTTP range reads) instead of
     snapshot_download'ing the full checkpoint.

Usage:
  ~/.venvs/tinker/bin/python tinker/convert_local.py \
      --adapter-dir /tmp/tinker_dummy --out-dir /tmp/dummy_converted \
      [--ref-dir /tmp/ref_adapter/submission] [--zip /tmp/submission.zip]

Key-diff vs the reference adapter must be CLEAN (0 missing / 0 extra) before
any submission.
"""
import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

FORCED_FUSED_RANK = 32
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def install_patches():
    import torch
    import tinker_cookbook.weights._adapter as A
    from tinker_cookbook.hyperparam_utils import (
        _list_param_shapes_from_safetensors_remote,
    )

    # ---- notebook cell 1, VERBATIM logic ----
    def _compress_lora_pair_to_rank(B, A_mat, rank):
        delta = B.float() @ A_mat.float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
        sroot = torch.sqrt(S)
        B_new = U * sroot.unsqueeze(0)
        A_new = sroot.unsqueeze(1) * Vh
        return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous()

    def patched_merge_fused_projections(fused_model_key, adapter_layer_prefix,
                                        components, model_state_shapes,
                                        peft_weights, target_modules, profile):
        fused_out_dim = model_state_shapes[fused_model_key][0]
        fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]
        component_order = None
        for target, comps in profile.fused_projection_map:
            if target == fused_target_name:
                component_order = comps
                break
        assert component_order is not None
        comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

        lora_A_parts, comp_slices = [], []
        merged_rank = row_offset = 0
        for comp_name in component_order:
            if comp_name not in comp_by_name:
                raise RuntimeError(
                    f"Missing component {comp_name!r} for fused target {fused_model_key!r}")
            lora_A, lora_B = comp_by_name[comp_name]
            r, out_dim = lora_A.shape[0], lora_B.shape[0]
            lora_A_parts.append(lora_A)
            comp_slices.append((row_offset, row_offset + out_dim, r))
            row_offset += out_dim
            merged_rank += r

        merged_lora_A = torch.cat(lora_A_parts, dim=0)
        merged_lora_B = torch.zeros(fused_out_dim, merged_rank,
                                    dtype=merged_lora_A.dtype,
                                    device=merged_lora_A.device)
        rank_offset = 0
        for i, (row_start, row_end, r) in enumerate(comp_slices):
            _, lora_B = comp_by_name[component_order[i]]
            merged_lora_B[row_start:row_end, rank_offset:rank_offset + r] = lora_B
            rank_offset += r

        final_rank = merged_rank
        if merged_rank > FORCED_FUSED_RANK:
            merged_lora_B, merged_lora_A = _compress_lora_pair_to_rank(
                merged_lora_B, merged_lora_A, FORCED_FUSED_RANK)
            final_rank = FORCED_FUSED_RANK
        peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"
        A._add_peft_weight(peft_target_key, merged_lora_A, merged_lora_B,
                           peft_weights, target_modules)
        return final_rank

    A._merge_fused_projections = patched_merge_fused_projections

    # ---- shapes-without-download ----
    from huggingface_hub import hf_hub_download

    def patched_resolve_model_dir(base_model):
        if os.path.isdir(base_model):
            return Path(base_model)
        cfg = hf_hub_download(repo_id=base_model, filename="config.json")
        return Path(cfg).parent  # dir contains config.json; shapes come remotely

    _shape_cache = {}

    def patched_get_model_state_shapes(model_dir):
        if "shapes" not in _shape_cache:
            print("[shapes] reading safetensors headers remotely (range reads)...",
                  flush=True)
            _shape_cache["shapes"] = _list_param_shapes_from_safetensors_remote(
                BASE_MODEL)
            print(f"[shapes] {len(_shape_cache['shapes'])} tensors", flush=True)
        return _shape_cache["shapes"]

    # the shape helpers live in _artifacts and are re-exported; patch BOTH
    # modules so intra-module calls (get_model_state_keys -> ..._shapes) hit us
    import tinker_cookbook.weights._artifacts as ART
    for mod in (A, ART):
        if hasattr(mod, "resolve_model_dir"):
            mod.resolve_model_dir = patched_resolve_model_dir
        if hasattr(mod, "get_model_state_shapes"):
            mod.get_model_state_shapes = patched_get_model_state_shapes
    print("[patch] _merge_fused_projections + remote-shapes installed")


def keyset(path):
    from safetensors import safe_open
    out = set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            out.add((key, tuple(sl.get_shape()), sl.get_dtype()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref-dir", default=None,
                    help="dir with reference adapter_model.safetensors for key-diff")
    ap.add_argument("--zip", default=None, help="write submission zip here")
    args = ap.parse_args()

    install_patches()
    from tinker_cookbook import weights
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    weights.build_lora_adapter(base_model=BASE_MODEL,
                               adapter_path=args.adapter_dir,
                               output_path=args.out_dir)
    print(f"[convert] wrote {args.out_dir}: {os.listdir(args.out_dir)}")
    cfg = json.load(open(os.path.join(args.out_dir, "adapter_config.json")))
    print(f"[convert] r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"target_modules={cfg.get('target_modules')}")

    if args.ref_dir:
        ours = keyset(os.path.join(args.out_dir, "adapter_model.safetensors"))
        ref = keyset(os.path.join(args.ref_dir, "adapter_model.safetensors"))
        missing, extra = ref - ours, ours - ref
        print(f"[key-diff] ref-only {len(missing)} | ours-only {len(extra)}")
        for k in sorted(missing)[:10]:
            print("  MISSING:", k)
        for k in sorted(extra)[:10]:
            print("  EXTRA  :", k)
        if not missing and not extra:
            print("[key-diff] CLEAN — identical keys/shapes/dtypes")

    if args.zip:
        os.makedirs(os.path.dirname(os.path.abspath(args.zip)), exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(args.out_dir):
                if os.path.isfile(os.path.join(args.out_dir, f)):
                    zf.write(os.path.join(args.out_dir, f), arcname=f)
        print(f"[zip] {args.zip} ({os.path.getsize(args.zip)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
