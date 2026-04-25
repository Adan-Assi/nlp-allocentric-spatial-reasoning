# LLM Evaluation Scripts Rewrite — 2026-04-25

The previous `scripts/evaluate_llm.py` and `scripts/evaluate_llm_masked.py` ran an LLM, but they asked a different question than the project's research question and read input files unrelated to the labeling pipeline. As a result, their outputs could not be compared to the symbolic oracle's labels at all — running them at scale would have spent SLURM GPU hours on data that doesn't answer the research question.

This pass rewrites both scripts so they (a) ask the 3-class classification question the oracle produces, (b) read the actual outputs of `batch_labeling.py` / `underspecify.py`, and (c) save results that can be diffed against `oracle_label` directly.

---

## TL;DR

| Concern | Before | After |
|---|---|---|
| Question asked of the LLM | *"What is the specific landmark or street name of the destination?"* — free-text | *"Classify as Answerable / Ambiguous / Contradictory"* — the oracle's exact label space |
| Input file (baseline) | `data/RVS_MASTER_GOLD_HYDRATED.parquet` (a different artifact, not produced by `batch_labeling.py`) | `data/<city>/<city>_silver_standard.parquet` (one or all 3 cities) |
| Input file (masked) | `reports/llm_audits/LLM_DEGRADATION_INPUT.parquet` (no producer in the repo) | `data/<city>/underspecified_variants.json` (the actual output of `underspecify.py`) |
| Output | `instruction, llm_output_raw` | adds `llm_prediction` (parsed label), `llm_parsed` flag, and (masked-only) `oracle_label_original`, `variant_type` |
| Comparison to oracle | Impossible (different schemas) | One-line merge / one-line confusion matrix |
| CLI | None | `--city`, `--model`, `--batch-size`, `--limit`, `--out` |
| Per-variant degradation summary | None | Printed at end of masked run, grouped by `variant_type` |

---

## What's new in `scripts/evaluate_llm.py`

1. **`build_prompt(instruction, city) -> str`** — short, T5-friendly classification prompt:
   ```
   Classify this <city> navigation instruction as exactly one of:
   Answerable (one valid destination),
   Ambiguous (multiple valid destinations),
   Contradictory (no valid destination).

   Instruction: <text>

   Classification:
   ```
   Kept short on purpose: flan-t5-base follows tight instructions better than verbose ones, and the budget is better spent on the actual instruction text. `max_new_tokens=8` since the answer is one word.

2. **`parse_label(raw) -> str | None`** — substring search in lowercased model output for any of the three canonical labels. Tolerates trailing punctuation, leading enumerators (`"1) Answerable"`), and verbose answers (`"the answer is contradictory"`). Returns `None` when none of the three labels appear; the caller flags those rows via `llm_parsed=False` instead of silently dropping them.

3. **`load_silver_standards(city) -> pd.DataFrame`** — concatenates `<city>_silver_standard.parquet` for every city present (or just the requested one). Sets `df["city"]` from the file context so the prompt builder always has it.

4. **`run_evaluation(...)`** — batched, no-sample, beam-1 decoding. Reports parse rate and overall oracle-vs-LLM agreement at the end. Output schema (added on top of the input):
   - `llm_output_raw` — model's raw decoded string
   - `llm_prediction` — parsed canonical label or `None`
   - `llm_parsed` — bool flag for clean separation in analysis

5. **CLI**: `--city`, `--model`, `--batch-size` (default 16), `--limit` (smoke test), `--out` (override default path).

Default output: `reports/llm_audits/llm_predictions_baseline.parquet`.

---

## What's new in `scripts/evaluate_llm_masked.py`

1. **`_flatten_variants(city) -> pd.DataFrame`** — reads `data/<city>/underspecified_variants.json` (what `underspecify.py` writes) and emits one row per variant with columns `sample_id, city, variant_type, removed_element, original_text, instruction (the masked text)`. The previous script's input file (`LLM_DEGRADATION_INPUT.parquet`) had no producer in the repo, so this script was effectively unrunnable.

2. **`_attach_original_oracle_label(variants) -> pd.DataFrame`** — joins each variant back to its source row's `oracle_label` (renamed to `oracle_label_original`). Since `underspecify.py` only masks rows the oracle already labeled Answerable, this'll mostly read "Answerable". The point isn't the label distribution — it's that downstream code can answer *"did the LLM still say what the oracle said when we hid information?"* by simple comparison.

3. **Re-uses** `build_prompt`, `parse_label`, `run_evaluation` from `evaluate_llm.py` — same model, same prompt, same parser. The only difference is the `instruction` column carries masked text (`"Meet at the [MASK] [DIR_MASK] of the park"`) instead of the original.

4. **Per-variant summary** at the end of the run — the *headline* degradation table:
   ```
   📊 Per-variant agreement with original oracle label:
     mask_landmark      n= 1234  agreement=42.31%
     mask_directions    n=  987  agreement=68.79%
     mask_both          n=  456  agreement=22.15%
   ```
   This is the most direct answer to the research question: as we strip information, does LLM agreement with the oracle's original label collapse?

5. **CLI**: `--city`, `--model`, `--batch-size`, `--limit`, `--out`. Default output: `reports/llm_audits/llm_predictions_masked.parquet`.

---

## SLURM script bug fix

`scripts/job_evaluate_llm_masked.sh` line 20 had:

```bash
cd $PROJECT_ROOT [cite: 64]
```

The `[cite: 64]` marker was a research-paper citation that got pasted into a real bash command (not a comment). Bash would have treated it as a literal directory argument to `cd` and failed. Fixed:

```bash
cd "$PROJECT_ROOT"
```

The `[cite: 60]` and `[cite: 56]` markers on adjacent lines are inside `#`-comments, so they were already harmless.

---

## Behavioural verification

Run `python -c "from scripts.evaluate_llm import build_prompt, parse_label; ..."` to confirm:

- `build_prompt("Meet me at the cafe on East 49th Street.", "manhattan")` produces a clean prompt under 200 chars.
- `parse_label("Answerable")`, `parse_label("Ambiguous")`, `parse_label("Contradictory.")`, `parse_label("  ambiguous  ")`, `parse_label("1) Answerable")`, `parse_label("the answer is contradictory")`, `parse_label("Answerable, with caveats")` all return their canonical labels.
- `parse_label("I am not sure")` and `parse_label("")` return `None` — the script flags these as `llm_parsed=False` so they don't pollute the confusion matrix.

`tests/sanity_check_logic.py` still passes (no regression in the NLP layer this rewrite touches).

---

## Environment caveat

On the local dev machine the import chain `scripts.evaluate_llm` → `transformers` → `torch` warns:

```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2
```

This is purely a local-env mismatch — not a code issue. On the SLURM cluster the conda env (`nlp_env` per `job_evaluate_llm_masked.sh`) needs torch ≥ 2.4 *or* a transformers version pinned to one that supports torch 2.2.x. Fix this before submitting the job, otherwise the model never loads and you get an empty results parquet.

---

## How to use

End-to-end, after `batch_labeling.py` produces silver standards for all three cities:

```bash
# 1. Generate masked variants from the silver standards
python scripts/underspecify.py

# 2. Smoke test on a small slice (CPU okay on flan-t5-base, slow but fine)
python scripts/evaluate_llm.py --city manhattan --limit 50
python scripts/evaluate_llm_masked.py --city manhattan --limit 50

# 3. Full SLURM run (or local with GPU)
sbatch scripts/job_evaluate_llm.sh
sbatch scripts/job_evaluate_llm_masked.sh
```

Outputs land in `reports/llm_audits/`. Both files have `oracle_label` and `llm_prediction` columns — confusion matrix is one `pd.crosstab(...)` call away.

## What changes downstream

- The analysis notebook (`notebooks/llm_degradation_analysis.ipynb`) needs to read these two parquets and use the new column names. The previous version expected `llm_output_raw` only; now it can compute `agreement = (df['llm_prediction'] == df['oracle_label']).mean()` directly.
- `audit_failures.py` is unaffected — it reads silver standards, not LLM outputs.
- `verify_label_quality.py` and `qc_ambiguous.py` still need their existing bugs fixed (separate task).
