<div align="center">

# A Verifiable Search Is Not a Learnable Chain-of-Thought

[![arXiv](https://img.shields.io/badge/arXiv-2606.21884-b31b1b.svg)](https://arxiv.org/abs/2606.21884)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/main.pdf)
[![Site](https://img.shields.io/badge/site-nemotron.harshpatel.live-1f6feb.svg)](https://nemotron.harshpatel.live)
[![Benchmark](https://img.shields.io/badge/benchmark-Kaggle-20beff.svg)](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY--4.0-green.svg)](LICENSE)

</div>

> You can write a program that solves a task perfectly and still be unable to distil its **search**
> into a chain-of-thought: the model collapses to verdict-as-token across every architecture and
> scale tried here. The task is not unlearnable, though — it becomes learnable only by **removing
> the search from the trace**, precomputing its structure into a finite catalog the model
> *memorizes* and verifies against. What transfers is memorization and verification, not search.

A research study built on the **NVIDIA Nemotron Model Reasoning Challenge**. Nine reasoning
tasks come from deterministic generators, so a held-out slice of the public training data is a
faithful proxy for the hidden-test score. That makes it a clean laboratory for one question:
*does a verifiable solver's coverage predict whether its search can be distilled into a small
model's chain-of-thought?* It does not — and the competition's
[1st-place solution](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution)
(private LB 0.92) is the existence proof for the escape: it cracked cryptarithm and
bit-manipulation by *memorizing* the search's finite structure, not by teaching the model to search.

---

## The result in three numbers

| Finding | Evidence |
|---|---|
| **Search doesn't distill.** A 0.71 search-solver for cryptarithm yields **0.05** when its search is distilled into CoT, across 11 CoT designs + RLVR + STaR. | `paper/` §Results |
| **Not the architecture or scale.** The same corpus caps at **≤0.05** across 3B–30B (fine-tuned) and 120B–671B (in-context), under the competition's 7680-token budget. | `paper/evidence/crypt-*.csv` |
| **Forward-derivability is causal.** Revealing the cipher key on the *same* task — turning the derivation forward — lifts it **0.03 → 0.571**. | `paper/evidence/frontier-*.csv` |
| **The escape is memorization, not search.** The 1st-place solution reached **0.92** by memorizing the search's finite structure (a 4,205-entry signature catalog) and verifying in-trace. | [1st-place write-up](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution) |

The failure is not a capability gap: the model does the arithmetic correctly on 97–100% of lines
and ranks the right answer into its top-8 on 71% of instances. What it cannot do is carry the
search forward as a left-to-right derivation, so fine-tuning learns the *shape* of a verifiable
step while its verdicts become unconditional templates (the **verdict-as-token** failure —
browse real cases in the site's explorer).

## Repository layout

```
paper/                     The paper: main.tex, main.pdf, references.bib, main.bbl,
                           evidence/ (per-row eval CSVs behind every table)
docs/                      The interactive site (nemotron.harshpatel.live): walkthrough + trace explorer
pipeline/
  solvers/                 reverse-engineered per-category solvers (the data-generation oracle)
  bit_global.py            global single-rule bit_manipulation solver (98.88%)
  eq_fit.py                equation_numeric policy solver (95.1%)
  synth/                   synthetic-CoT renderers (incl. forward_frontier_gen.py, the intervention)
  data/                    bitmanip_solved.jsonl (basis), val.jsonl + *_eval.jsonl (held-out evals)
tinker/                    experiment runners + eval harness:
  run_crypt_arch_ctrl.sh   architecture control (any base; evals with that model's own tokenizer)
  run_crypt_frontier.sh    frontier in-context baseline (DeepSeek-V3.1, Nemotron-Super-120B)
  run_frontier.sh          forward-derivability intervention (full / half key)
  sft_warmstart.py, eval_oracle.py, RUNS.md, evals/
analysis/crypt_fidelity/   per-line fidelity audit (7,566 records) — drives the trace explorer
```
The benchmark data is **not** redistributed here — download it from the
[Kaggle competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## Reproduce

Experiments run on [Tinker](https://thinkingmachines.ai) (LoRA fine-tuning + sampling). Set
`TINKER_API_KEY`, then:

```bash
# Architecture control: same corpus, any base, evaluated with that model's own tokenizer
bash tinker/run_crypt_arch_ctrl.sh meta-llama/Llama-3.2-3B llama3b 4

# Frontier in-context baseline (no fine-tuning), under the competition's 7680-token budget
bash tinker/run_crypt_frontier.sh 20

# Forward-derivability intervention (reveal the cipher key on the same instances)
python3 pipeline/synth/forward_frontier_gen.py && bash tinker/run_frontier.sh
```

Per-row outputs land in `tinker/evals/` and are archived in `paper/evidence/`.

## Build the paper / site

```bash
# paper (Tectonic or pdflatex+bibtex); main.bbl is committed for arXiv
cd paper && tectonic -X compile main.tex
# site: serve docs/ locally
python3 -m http.server -d docs 8000   # then open http://localhost:8000
```

## Citation

```bibtex
@misc{patel2026verifiable,
  title         = {A Verifiable Search Is Not a Learnable Chain-of-Thought},
  author        = {Patel, Harsh},
  year          = {2026},
  eprint        = {2606.21884},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2606.21884},
  note          = {NVIDIA Nemotron Model Reasoning Challenge}
}
```

## License

Code is released under the **MIT License** ([`LICENSE`](LICENSE)). The paper text and figures
are released under **CC-BY-4.0**. The benchmark data belongs to the original competition and is
not redistributed here — download it from the
[Kaggle competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
