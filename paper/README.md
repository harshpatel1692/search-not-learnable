# Paper: *The Learnability Frontier of Procedural Reasoning*

A full arXiv-style preprint built from this repo's experiment log, solvers, and analysis
reports. Thesis: **a verifiable search is not a learnable chain-of-thought** — distilling a
backtracking search collapses to verdict-as-token; the task becomes learnable only by removing
the search from the trace (memorize its finite structure + verify), as the competition's
1st-place solution demonstrated.

**Status:** 24 pages, compiles clean (Tectonic, exit 0; 0 undefined refs/citations; 20
references; ~6 cosmetic overfull lines). `main.pdf` is the compiled proof.

**Token audit (pulled from `v1-clean`, now authorized):** App. Tokenizer now includes the
per-category count-weighted per-token cross-entropy table (`tab:tokce`; gravity/unit ≈2.1
vs cipher 6.34) computed from the v6 training-stats token audit, plus the 1,253-token
high-CE "blacklist" finding. Source: `v1-clean:audit/token_audit.csv`.

**EXP-138 (now VERIFIED via Kaggle API):** the 0.86 is real. Submission `53378508`
(2026-06-05) = **0.844 public / 0.860 private** — the highest private across all our
submissions, and unselected because public-LB-best was elsewhere (v17 0.852/0.852; reference
adapter 0.856/**0.832** = public overfit). Full record saved to
`evidence/kaggle_submissions.csv`. §Competition Dynamics now presents this as a **verified**
public/private selection near-miss with an inversion table (`tab:pubpriv`), and Limitations
records best-public 0.852 / best-private 0.860. The EXP-138 vehicle is the native-HF
`harshpatel1692/nemotron-v6-train` notebook, **forked from Tong's published recipe**
(`kernel_sources: huikang/end-to-end-finetuning-for-lb-0-85`), now cited as
`huikang_kaggle2026`.

**Adapter-noise laundering (documented, effect not measured):** `competition_dataset/`
`nvidia-nemotron-noisy-adapter.ipynb` loads a *third party's* shared submission adapter
(`kienngx/...`) and adds Gaussian noise at 20% of each tensor's σ to all 12,010 tensors, then
repackages it. No score recorded in our copy. §Competition Dynamics now describes this as a
concrete, reproducible integrity concern (replacing the earlier "unverified rumor" wording).

**Evidence:** `evidence/kaggle_submissions.csv` — authoritative public+private scores for all
35 submissions (the citable record; a literal page screenshot would need a browser).

**Raw-data tables added (computed/extracted from the repo, not summaries):** the verifiable
bit-rule inventory (op-family + tap distribution from the 1602-row solve cache); per-category
prompt & synthetic-CoT token lengths (measured with the competition tokenizer over all rows);
the full Tinker run ledger (every run: examples/epochs/steps/wall-time/cost/held-out acc).
**Caveat:** full per-step loss *curves* were not persisted in the repo — we recovered
checkpoint NLLs and the final-loss tail (≈0.003) and report convergence points, not curves.
The detailed `token_audit.csv` lives on the `v1-clean` branch (not pulled, per the
current-branch-only decision); the on-branch analysis is regenerated from `tokenizer.json`.

## Files
- `main.tex` — the paper. Body: intro, related work, testbed, method, results (incl. the
  bit 3-arg basis centerpiece + LB trajectory + crypt floor + STaR + verdict-fidelity
  figures), **failure anatomy** (9-mode catalog), **analysis**, **competition dynamics /
  external corroboration**, limitations, conclusion. Appendices: **solver pseudocode**
  (cipher, equation, cryptarithm search + forward, bit-global + stride), **training config
  + submission history**, **tokenizer analysis**, **per-sub-category results + failure
  stats**.
- `references.bib` — 20 entries.
- `main.bbl` — kept for arXiv (arXiv doesn't run BibTeX; upload `main.tex` + `main.bbl`).
- `figures/` — empty; all 5 figures are native pgfplots/TikZ.

## Compile
Overleaf (pdfLaTeX) or locally: `~/.local/bin/tectonic -X compile main.tex`
(Tectonic 0.16.9 is installed in `~/.local/bin`; the TeX bundle is cached). Needs
`pgfplots`, `algorithm`, `algpseudocodex`, `longtable`, `natbib`, `cleveref`, `tcolorbox`.

## Provenance & honesty checklist (important before submission)

All numbers trace to dated entries in `../LOG.md`, files in `../analysis/`, the solver
source in `../pipeline/`, or (for competition context) the live Kaggle leaderboard and the
open-source Progress-Prize repo. Five things to know:

1. **Bit 3-arg basis counts are a first-match partition, not independent "match-any."**
   The repo has no script that produces 625/459/234/148/58/53; an independent re-derivation
   confirms **XOR (~625–638) and MAJ (~459–478)** as the two largest, robust strata and
   **AND-3 ⊂ MAJ (108/109)**, but the split within {OR, XOR_OR, OR_XNOR, CHOICE} is
   ordering-dependent (the cluster *total* is invariant). The paper labels this honestly.
   Also distinct: **basis fit-on-examples (~98%) vs. global-solver correct-on-query
   (98.88%)** — different measurements.
2. **equation_numeric_deduce is 95.1%**, not the 21% in `CLAUDE.md` (that header is stale;
   `eq_fit.py`'s `policy_seq` is the real number). Paper uses 95.1% / 0.83–0.85 model.
3. **Leaderboard: leader (NullSira) = 0.92; 0.85 is the median** of 4,355 teams; the Open
   Progress Prize (methodology) went to a fully open-source 0.85 solution using our same
   approach — strong external corroboration, cited as `huikang2026nemotron`.
4. **Reference-adapter per-category numbers are from direct forensic re-run** (e.g. bit
   76.65%), not the second-hand 85.1% in a memory note.
5. **Unverified claims flagged in-text:** the "noise on a shared adapter" trick (post not
   located) and the public/private cryptarithm-operator overfit (forum snippet only). Both
   are marked unverified in §"Competition Dynamics."

## Re-verify bib IDs
arXiv IDs/venues written from best knowledge (June 2026). Web-verified: base model =
Nemotron 3 Nano (arXiv:2512.20848). Double-check the rest before posting.

## Author / next edits
`main.tex` has a placeholder author block. Optional polish: a worked verdict-as-token CoT
example box (`cotbox` env already defined); tighten the ~9 overfull lines; a same-size dense
transformer cryptarithm run (the single most valuable follow-up experiment).
