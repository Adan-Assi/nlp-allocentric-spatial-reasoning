# Next Steps After `batch_labeling`

This document covers (1) what to do once `batch_labeling.py` finishes for Manhattan, and (2) an audit of the downstream scripts to see which are ready and which still need work.

---

## Pipeline overview (post-batch_labeling)

```
batch_labeling.py                 ← you are here
  └─ produces: data/<city>/<city>_silver_standard.parquet

verify_label_quality.py           ← QA: distribution check
underspecify.py                   ← generates masked variants
label_variants_with_oracle.py     ← re-labels variants with oracle (deprecated path)
evaluate_llm.py                   ← LLM inference on un-masked text
evaluate_llm_masked.py            ← LLM inference on masked variants
audit_failures.py                 ← samples Contradictory rows for inspection
qc_ambiguous.py                   ← inspects Ambiguous rows
notebooks/llm_degradation_analysis.ipynb  ← final plots
```

---

## Suggested order of operations

### 1. Verify the Manhattan output

```bash
python scripts/verify_label_quality.py
```

Quickly inspect the label distribution:

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/manhattan/manhattan_silver_standard.parquet'); print(df['oracle_label'].value_counts())"
```

Healthy ranges: ~50–70 % Ambiguous, ~20–40 % Answerable, ~5–15 % Contradictory.

### 2. Run the other two cities

```bash
python scripts/batch_labeling.py --city pittsburgh
python scripts/batch_labeling.py --city philadelphia
```

Pittsburgh and Philadelphia have sparser street grids — consider raising `SNAP_MAX_M` in `_prepare_poi_data` from `100.0` to `150.0` or `200.0` for them.

### 3. Generate underspecified variants

```bash
python scripts/underspecify.py
```

Produces `data/<city>/underspecified_variants.json` for each city. This is the masked-instruction set used in Phase 5.

### 4. Run LLM evaluation (Phase 5)

Two runs:

```bash
sbatch scripts/job_evaluate_llm.sh           # baseline, full text
sbatch scripts/job_evaluate_llm_masked.sh    # stress test, masked variants
```

Or interactively (needs a GPU for any reasonable speed):

```bash
python scripts/evaluate_llm.py
python scripts/evaluate_llm_masked.py
```

### 5. Failure analysis

```bash
python scripts/audit_failures.py     # inspects Contradictory rows
python scripts/qc_ambiguous.py       # inspects Ambiguous rows
```

### 6. Plots and write-up

`notebooks/llm_degradation_analysis.ipynb` — confusion matrices, degradation curves, per-category accuracy, cross-city comparison.

---

## Audit: are the downstream scripts ready?

I read each script and checked: do its inputs match what `batch_labeling.py` produces, do its column names line up, do its imports resolve, does it fit the new 3-class research question? Findings below.

### ✅ `scripts/underspecify.py` — ready

- Reads `data/<city>/<city>_silver_standard.parquet` ✓
- Uses `oracle_label`, `extracted_noun`, `instruction` (all present in batch output) ✓
- Filters to `oracle_label == 'Answerable'` for variant generation, which is the right design — masking already-Ambiguous text doesn't add experimental signal.
- Outputs `data/<city>/underspecified_variants.json` ✓

**Caveat (not a bug, just numbers shifting):** with the corrected oracle, Answerable will drop from the previously-inflated number to ~20–40 %. The "22 k variants" figure in `scripts/README.md` was based on the old salience-collapse behavior. Expect proportionally fewer variants now.

### ⚠️ `scripts/verify_label_quality.py` — heuristic is inverted

The audit prints the distribution correctly, but the warning logic is wrong:

```python
if stats.get('Answerable', 0) < 70:
    print("⚠️ WARNING: Low Answerability! ...")
```

This was written under the old semantics, where the buggy salience-collapse pushed everything into Answerable. Under the corrected 3-class oracle, Answerable should be ~20–40 % and Ambiguous should dominate. So this warning will fire on every healthy run.

**Fix needed**: invert the check — warn when Answerable is *too high* (>~80 %) or when Ambiguous is *too low* (<~30 %), since either signals a regression in the multi-candidate logic.

### ❌ `label_variants_with_oracle.py` — broken imports + schema mismatch

Two blocking issues:

1. **Missing module**:
   ```python
   from src.constraints.underspec_constraints import extract_constraints
   ```
   `src/constraints/` does not exist in this repo (only `extraction_utils.py`, `oracle_engine.py`, `symbolic_solver.py`, `utils.py`). The import fails immediately.
2. **Schema mismatch with `underspecify.py` output**:
   - This script expects a parquet with columns `region`, `variant_text`, `start_lat`, `start_lon`.
   - `underspecify.py` writes JSON with `sample_id`, `original_text`, `gold_goal_node`, `variants: [...]` — different format.

**Action**: this script was written for an older constraint-based pipeline. Either resurrect/reimplement `underspec_constraints.extract_constraints`, or — simpler — re-use `solver.solve(variant_text, start_node)` directly inside `underspecify.py` and write the per-variant oracle label into the JSON. The current `solve()` already handles "missing landmark" / "missing direction" gracefully via its 3-stage search; we don't need a separate constraint extractor.

### ⚠️ `scripts/evaluate_llm.py` — runs, but disconnected from Phase 5 and from the 3-class task

- Reads `data/RVS_MASTER_GOLD_HYDRATED.parquet` (which exists), **not** the per-city silver-standard parquets that `batch_labeling.py` produces.
- The prompt asks: *"What is the specific landmark or street name of the destination?"* — that's a free-form **destination identification** task, not the **3-class classification** task that the project's research question is built on.
- No comparison against `oracle_label` is computed — only a string output is saved.

**To make this script actually evaluate the project's research question**, two changes are needed:
1. Point `GOLD_PATH` at the city-level silver-standard parquets (or a concatenated version), so the LLM sees the same rows the oracle labeled.
2. Change the prompt to elicit a 3-class label, e.g.:
   > *"Given the navigation instruction below, decide whether it has (A) exactly one valid destination, (B) multiple valid destinations, or (C) no valid destination. Answer with one of: Answerable, Ambiguous, Contradictory."*
3. Parse the model's output to one of the three labels and write it next to `oracle_label` so the confusion matrix can be built.

### ⚠️ `scripts/evaluate_llm_masked.py` — same issues + input file probably doesn't exist

Mirror of `evaluate_llm.py`:

- Reads `reports/llm_audits/LLM_DEGRADATION_INPUT.parquet`, which is **not** what `underspecify.py` writes (that emits per-city JSON).
- Same destination-identification prompt — not the 3-class task.

**To make this work end-to-end after `underspecify.py`:**

- Add a small adapter that reads `data/<city>/underspecified_variants.json`, flattens the `variants` list to one row per variant (with `city`, `sample_id`, `variant_type`, `masked_instruction`), and writes a single parquet at the path this script expects.
- Apply the same prompt + label-parsing changes as `evaluate_llm.py`.

### ✅ `scripts/audit_failures.py` — works, minor cleanup recommended

- Reads `data/<city>/<city>_silver_standard.parquet` ✓
- Uses correct columns (`oracle_label`, `extracted_noun`, `start_node`, `instruction`, `sample_id`) ✓
- Picks Contradictory rows, excludes those where the noun is a bare cardinal direction, and re-runs `solver.solve()` on a sample of 10 to see if any get rescued ✓
- Has a defensive `_normalize_start_node` for cross-format start-node lookups, which is a reasonable belt-and-suspenders guard.

**Small thing**: only iterates over `['philadelphia', 'pittsburgh']` (Manhattan is hard-coded out). If you want to audit Manhattan failures too, add `'manhattan'` to that list.

### ❌ `scripts/qc_ambiguous.py` — two blockers

```python
parquet_path = os.path.join(os.path.dirname(config.RVS_DATA_JSON), "manhattan_silver_standard.parquet")
...
ambiguous_df = df[df['silver_label'] == 'ambiguous']
```

1. **`config.RVS_DATA_JSON`** was removed in the multi-city config refactor. The reference is commented out in `config.py:17`. This line will raise `AttributeError`.
2. **`df['silver_label']`** — `batch_labeling.py` writes the column under the name **`oracle_label`**, not `silver_label`. KeyError.
3. **City** is also hardcoded to Manhattan.

**Fix needed (one-line each):**

```python
parquet_path = os.path.join(config.BASE_DIR, "data", "manhattan", "manhattan_silver_standard.parquet")
...
ambiguous_df = df[df['oracle_label'] == 'Ambiguous']    # capital A, matches batch output
```

And consider adding a `--city` CLI argument like the other scripts have.

---

## Summary table

| Script | Status | What works | What needs fixing |
|---|---|---|---|
| `verify_label_quality.py` | ⚠️ Inverted heuristic | Distribution print | Invert the warning threshold |
| `underspecify.py` | ✅ Ready | Reads Silver Standard, generates 3 mask types | — |
| `label_variants_with_oracle.py` | ❌ Broken | — | Missing module import; schema mismatch. Easier to retire and re-use `solver.solve()` directly. |
| `evaluate_llm.py` | ⚠️ Disconnected | Loads + runs T5/Pythia inference | Wrong input file, wrong prompt task (destination free-text vs. 3-class). |
| `evaluate_llm_masked.py` | ⚠️ Disconnected | Mirror of above | Wrong input file, wrong prompt task; no adapter from `underspecify.py`'s JSON output. |
| `audit_failures.py` | ✅ Works | Re-runs oracle on Contradictory samples | Manhattan hard-coded out (cosmetic). |
| `qc_ambiguous.py` | ❌ Broken | — | Stale `config.RVS_DATA_JSON` reference; wrong column name `silver_label` vs `oracle_label`. |

## Concrete recommended order before the LLM phase

1. **Today**: fix `verify_label_quality.py` threshold and `qc_ambiguous.py` (both are 1–3 line edits) so QA reflects the new semantics.
2. **Today**: run Manhattan, Pittsburgh, Philadelphia through `batch_labeling.py`. Gives 3 silver-standard parquets.
3. **Today/Tomorrow**: run `underspecify.py` — this works as-is.
4. **Before Phase 5 LLM eval**: rewrite the LLM prompt and the input-loading paths in `evaluate_llm.py` and `evaluate_llm_masked.py` to match the 3-class task and consume `underspecified_variants.json`. Without this, the LLM eval doesn't measure what the research question asks.
5. **Optional**: retire `label_variants_with_oracle.py` (constraint-extraction approach is superseded by the new `solver.solve()` 3-stage search), or rebuild its missing dependency if there's a reason to keep it.

## Bottom line

The labeling phase is in good shape. The **LLM evaluation phase is not** — the two `evaluate_llm*.py` scripts ask the wrong question (free-text destination instead of 3-class) and read the wrong inputs. They will run, but their output won't compare cleanly to `oracle_label`. That's the most important fix before the SLURM jobs are submitted, otherwise GPU time gets spent on outputs that don't answer the research question.
