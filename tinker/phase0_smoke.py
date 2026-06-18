"""Phase-0 smoke test — run ONCE when TINKER_API_KEY lands (~$1, ~15 min).

Covers every key-required CHECKLIST Phase-0 box in one command:
  1. server capabilities -> confirm exact nemotron-3-nano model id
  2. create rank-32 LoRA training client (train_unembed=True = lm_head) + info
  3. tokenizer parity: Tinker's client tokenizer vs competition tokenizer.json
  4. sample 3 greedy completions from the BASE on real bit prompts
     (sanity: think-mode output, token counts)
  5. dummy save_state (0 train steps) + save_weights_for_sampler +
     checkpoint-archive download URL -> /tmp/tinker_dummy_ckpt.tar
     => feed that into the conversion chain (analysis/notebooks/) to prove
     the export path BEFORE any real spend.

Run:  ~/.venvs/tinker/bin/python tinker/phase0_smoke.py
(Local key-free checks — template parity, SDK surface — already PASSED
2026-06-11; see LOG.md.)
"""
import csv
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "tinker", "RUNS.md")
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
csv.field_size_limit(10 ** 9)
EVAL_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
               "For example: `\\boxed{your answer}`")


def main():
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set (~/.bashrc)"
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()

    # 1. capabilities / model list
    caps = sc.get_server_capabilities()
    models = [getattr(m, "model_name", str(m)) for m in caps.supported_models]
    nemotron = [m for m in models if "emotron" in str(m)]
    print(f"[1] {len(models)} supported models; nemotron entries: {nemotron}")
    assert any(BASE_MODEL in str(m) for m in models), \
        f"{BASE_MODEL} not in supported models!"

    # 2. rank-32 LoRA training client incl. unembed (lm_head)
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL, rank=32, seed=42,
        train_mlp=True, train_attn=True, train_unembed=True)
    info = tc.get_info()
    print(f"[2] training client OK; info: {info}")

    # 3. tokenizer parity vs competition tokenizer.json
    from tokenizers import Tokenizer
    comp = Tokenizer.from_file(os.path.join(ROOT, "competition_dataset/tokenizer.json"))
    tok = tc.get_tokenizer()
    probe = "ROTL3 XOR -> ok != 0b1011 \\boxed{10110001}"
    a = comp.encode(probe, add_special_tokens=False).ids
    b = tok.encode(probe, add_special_tokens=False)
    print(f"[3] tokenizer parity (client vs competition): {list(a) == list(b)}")
    assert list(a) == list(b)

    # 4. 3 greedy base completions on real bit prompts
    hub_tok = get_tokenizer(BASE_MODEL)
    base_sampler = sc.create_sampling_client(base_model=BASE_MODEL)
    params = tinker.SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0)
    rows = [r for r in csv.DictReader(open(os.path.join(
        ROOT, "competition_dataset/train_categorized.csv")))
        if r["category"] == "bit_manipulation"][:3]
    futs = []
    for r in rows:
        text = hub_tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"] + EVAL_SUFFIX}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = hub_tok.encode(text, add_special_tokens=False)
        futs.append(base_sampler.sample(tinker.ModelInput.from_ints(ids),
                                        num_samples=1, sampling_params=params))
    for i, fut in enumerate(futs):
        s = fut.result().sequences[0]
        txt = hub_tok.decode(s.tokens, skip_special_tokens=True)
        print(f"[4.{i}] {len(s.tokens)} tok, stop={s.stop_reason}: "
              f"{txt[:160]!r}...")

    # 5. dummy checkpoint round-trip (export-path proof)
    st = tc.save_state(name="phase0-dummy").result()
    sp = tc.save_weights_for_sampler(name="phase0-dummy-sampler").result()
    print(f"[5] state   : {st.path}")
    print(f"[5] sampler : {sp.path}")
    rest = sc.create_rest_client()
    # archive endpoint ONLY serves sampler_weights/ checkpoints (400 on weights/)
    url = rest.get_checkpoint_archive_url_from_tinker_path(sp.path).result()
    archive_url = getattr(url, "url", url)
    dest = "/tmp/tinker_dummy_ckpt.tar"
    urllib.request.urlretrieve(str(archive_url), dest)
    print(f"[5] downloaded {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    print("[5] NEXT: run analysis/notebooks/tinker-adapter-to-ready-to-submit-"
          "adapter.ipynb + tinker-submission-notebook.ipynb on this archive; "
          "key-diff vs reference must be clean.")

    with open(RUNS, "a") as f:
        f.write(f"| phase0-smoke | {time.strftime('%F %T')} | model={BASE_MODEL} "
                f"| dummy state={st.path} sampler={sp.path} |\n")
    print("\nPHASE 0 (key-required) SMOKE: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
