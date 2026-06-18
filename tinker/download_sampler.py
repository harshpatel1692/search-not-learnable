"""Download a Tinker checkpoint archive (the raw LoRA adapter) so it can be
uploaded to Kaggle and fed to analysis/notebooks/tinker-adapter-to-ready-to-
submit-adapter.ipynb (build_lora_adapter(adapter_path=...)).

The archive endpoint ONLY serves sampler_weights/ checkpoints (400 on weights/).
If you pass a state (weights/) path, this first loads it into a fresh client and
saves a sampler from it, then downloads that.

Usage:
  ~/.venvs/tinker/bin/python tinker/download_sampler.py <tinker://...path> \
      [--tar /tmp/v19_sampler.tar] [--out /tmp/v19_adapter]
"""
import argparse
import os
import tarfile
import urllib.request

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="tinker:// sampler_weights (preferred) or weights/ state path")
    ap.add_argument("--tar", default="/tmp/v19_sampler.tar")
    ap.add_argument("--out", default="/tmp/v19_adapter")
    args = ap.parse_args()
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set"
    import tinker

    sc = tinker.ServiceClient()
    sampler_path = args.path

    # If a state checkpoint was given, mint a sampler from it.
    if "/weights/" in sampler_path and "/sampler_weights/" not in sampler_path:
        print(f"[recover] {sampler_path} is a STATE ckpt; minting a sampler...")
        tc = sc.create_lora_training_client(
            base_model=BASE_MODEL, rank=32, seed=42,
            train_mlp=True, train_attn=True, train_unembed=True)
        tc.load_state(sampler_path).result()
        sp = tc.save_weights_for_sampler(name="v19-recover-sampler").result()
        sampler_path = sp.path
        print(f"[recover] new sampler: {sampler_path}")

    rest = sc.create_rest_client()
    url = rest.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    archive_url = getattr(url, "url", url)
    print(f"[dl] {sampler_path}\n  -> {args.tar}")
    urllib.request.urlretrieve(str(archive_url), args.tar)
    print(f"[dl] {args.tar} ({os.path.getsize(args.tar) / 1e6:.1f} MB)")

    os.makedirs(args.out, exist_ok=True)
    with tarfile.open(args.tar) as t:
        t.extractall(args.out)
    # If everything landed under a single top dir, surface that for clarity.
    entries = os.listdir(args.out)
    print(f"[extract] {args.out}: {entries}")
    for root, dirs, files in os.walk(args.out):
        for f in files:
            fp = os.path.join(root, f)
            print(f"    {os.path.relpath(fp, args.out)}  ({os.path.getsize(fp)/1e6:.2f} MB)")
    print(f"\n[ok] adapter dir = {args.out}")


if __name__ == "__main__":
    main()
