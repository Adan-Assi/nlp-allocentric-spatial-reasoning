# Project Flow — From Labeling to Submitted Paper

End-to-end roadmap for the NLP Allocentric Spatial Reasoning project. Each phase lists its **inputs**, **scripts**, **outputs**, and **wall-clock estimate**. Most of the time after Phase 1 is spent on analysis and writing — the compute is short.

> Companion docs:
> - [`CHANGES.md`](CHANGES.md) — earlier 3-class logic / 8-way direction fixes
> - [`CHANGES_2026_04_25.md`](CHANGES_2026_04_25.md) — POI mapping + iteration speed fixes
> - [`CHANGES_LLM_EVAL_2026_04_25.md`](CHANGES_LLM_EVAL_2026_04_25.md) — LLM evaluation rewrite
> - [`NEXT_STEPS.md`](NEXT_STEPS.md) — audit of every downstream script

---

## Critical-path timeline

```
T+0         Manhattan local + Pittsburgh/Philly cluster labeling
T+24h       Phase 2 audit
T+24.5h     Phase 3 underspecify
T+24.6h     Phase 4 LLM jobs submitted
T+30h       LLM jobs done; Phase 5 analysis starts
T+~3 days   Phase 6 error taxonomy
T+~10 days  Phase 7 draft paper
T+~14 days  Revise + submit
```

Total: ~2–3 weeks after Manhattan finishes. Bottleneck is *analysis quality*, not compute.

---

## Phase 1 — Complete Labeling (~24 h wall-clock)

**Goal:** produce silver-standard 3-class labels for all three cities under the corrected pipeline.

### Inputs
- `data/<city>/<city>_graph.gpickle`
- `data/<city>/<city>_poi.pkl`
- `data/<city>/<city>.json` (raw RVS instructions)

### Scripts
```bash
# Manhattan — already running locally, ~16h
python scripts/batch_labeling.py --city manhattan

# Pittsburgh + Philadelphia — submit as cluster array job
sbatch --array=1-2 scripts/submit_labeling.sh
```

### Outputs
- `data/manhattan/manhattan_silver_standard.parquet`
- `data/pittsburgh/pittsburgh_silver_standard.parquet`
- `data/philadelphia/philadelphia_silver_standard.parquet`

Schema: `sample_id, city, instruction, oracle_label, candidate_count, reachable_candidate_count, extracted_category, extracted_noun, extracted_direction, target_node, start_node, gold_goal_node, resolution_stage`.

### Per-city tunable
For Pittsburgh / Philadelphia (sparser grids than Manhattan), consider raising `SNAP_MAX_M` in `_prepare_poi_data` from `100.0` to `150.0` if the init log shows POI mapping rate < 50%.

---

## Phase 2 — Quality Audit (~30 min)

**Goal:** confirm the labeler produced a healthy distribution before sinking GPU hours into LLM evaluation.

### Scripts
```bash
python scripts/verify_label_quality.py
python scripts/qc_ambiguous.py --city manhattan --n 30
python scripts/audit_failures.py
```

### Healthy targets per city

| Bucket | Target range | Red flag |
|---|---|---|
| Ambiguous | 50–65% | < 30% (multi-candidate collapse bug) |
| Answerable | 25–40% | > 80% (same bug, different sign) |
| Contradictory | 5–15% | > 20% (NLP extraction failures) |

If any city is far off, fix the underlying issue and re-label that city before moving on.

### Manual spot-check
Eyeball ~30 Ambiguous and ~10 Contradictory rows from each city. The Contradictory ones especially should look "noisy text", not "the labeler dropped the ball".

---

## Phase 3 — Generate Underspecified Variants (~5 min)

**Goal:** produce masked instruction variants per city for the degradation experiment.

### Inputs
- `data/<city>/<city>_silver_standard.parquet` (Phase 1 output)

### Script
```bash
python scripts/underspecify.py
```

### Outputs
- `data/<city>/underspecified_variants.json` (per city)

For each Answerable instruction, up to 3 variants are generated:
| Variant | What's masked |
|---|---|
| `mask_landmark` | The landmark name → `[MASK]` |
| `mask_directions` | Cardinal direction words → `[DIR_MASK]` |
| `mask_both` | Both above |

Expected total: ~3 × Answerable_count ≈ **15–25k variant rows** across all 3 cities.

---

## Phase 4 — LLM Inference (~3–6 h on GPU)

**Goal:** get the LLM's 3-class prediction on (a) the un-masked instructions and (b) the masked variants.

### Scripts
```bash
sbatch scripts/job_evaluate_llm.sh           # baseline
sbatch scripts/job_evaluate_llm_masked.sh    # degradation
```

Both run in parallel, both ask the 3-class question (`Answerable / Ambiguous / Contradictory`), both pair `oracle_label` ↔ `llm_prediction`.

### Outputs
- `reports/llm_audits/llm_predictions_baseline.parquet`
- `reports/llm_audits/llm_predictions_masked.parquet`

Schema for both: `sample_id, city, instruction, oracle_label, llm_prediction, llm_output_raw, llm_parsed`. Masked file additionally has `variant_type, oracle_label_original`.

### Recommended: multi-model run
For a stronger paper, run with 2–3 model sizes:
- `google/flan-t5-base` (default)
- `google/flan-t5-large`
- `EleutherAI/pythia-1b` or similar

Easiest path: extend the SLURM script to a model array, write per-model output paths. ~2–3× the wall-clock if jobs parallelize across nodes; minimal extra code.

### Pre-flight checks (on the cluster, before submitting)
1. Today's code changes synced (POI mapping + extraction fixes + new LLM scripts).
2. Conda env (`nlp_env`) has **torch ≥ 2.4** — without this, `transformers` disables the torch backend silently and the run produces empty output.
3. Smoke test: `python scripts/evaluate_llm.py --city manhattan --limit 50` should produce a valid parquet in seconds.

---

## Phase 5 — Quantitative Analysis (~2–3 days)

**Goal:** turn the predictions into the figures and tables for the paper.

Build `notebooks/llm_degradation_analysis.ipynb`. Five core analyses:

### 5.1 — 3-class confusion matrix (baseline, per city)

For each city, build a 3×3 matrix of `oracle_label` × `llm_prediction`:

```
                     LLM →  Answerable  Ambiguous  Contradictory
Oracle ↓
Answerable                    [diag]      [over-Amb] [over-Cont]
Ambiguous                     [over-Ans]  [diag]     [over-Cont]
Contradictory                 [hallucin]  [over-Amb] [diag]
```

**Headline numbers:**
- Diagonal sum = overall agreement.
- **`oracle=Ambiguous, LLM=Answerable` cell** = the **overconfidence rate**. This is the project's central claim.

### 5.2 — Degradation curve (the hero plot)

For each variant_type ∈ {full text, mask_landmark, mask_directions, mask_both}, plot:
- LLM agreement with `oracle_label_original` (the un-masked label)
- % of predictions that shift to `Ambiguous`
- % of predictions that shift to `Contradictory`

Hypothesis: as text gets more masked, agreement drops AND the LLM should ideally shift toward `Ambiguous` (recognizing the rising under-specification). If it instead stays on `Answerable`, that's overconfidence.

### 5.3 — Per-category accuracy

Group by `extracted_category` (CHURCH, CAFE, BICYCLE, RESTAURANT, …). Compute LLM accuracy per category. Some categories are systematically harder.

### 5.4 — Per-city comparison

Manhattan (dense, many candidates) vs. Pittsburgh / Philadelphia (sparser). Test: is the LLM more overconfident in dense cities (where the right answer space is bigger but the LLM still picks one)?

### 5.5 — Direction handling

Filter to rows with non-null `extracted_direction`. Check whether the LLM correctly handles 8-way directions or collapses NE → N. Use the symbolic oracle's 8-way classifier as ground truth.

---

## Phase 6 — Error Taxonomy (~1–2 days)

**Goal:** turn the disagreements into a qualitative story.

Sample 50–100 disagreement rows (especially the `oracle=Ambiguous, LLM=Answerable` quadrant). For each row, ask:
- What did the oracle say?
- What did the LLM say?
- Why did they disagree?

Cluster the failures into types. Likely classes:

| Failure mode | What happens |
|---|---|
| **Hallucinated unique target** | LLM picks one POI when the text admits many — the headline failure |
| **Surface-level brand anchoring** | LLM grabs the first proper noun and ignores spatial constraints |
| **Direction-blind** | LLM ignores explicit `north / northeast / on my left` |
| **Mask-confused** | LLM treats `[MASK]` / `[DIR_MASK]` as a literal landmark |
| **Wrong landmark in multi-landmark text** | LLM picks the reference object instead of the goal |
| **Spatial relation inversion** | LLM swaps "north of" with "south of" |

Pull 2–3 verbatim examples per class for the paper.

---

## Phase 7 — Paper Writing (~1 week)

### Suggested structure

| Section | Length | Source material |
|---|---|---|
| Abstract | ~200 words | headline degradation %, overconfidence rate |
| 1. Introduction | 1 page | the research question in `README.md`; why allocentric reasoning matters; why under-specification is a clean probe |
| 2. Related Work | 1 page | RVS (Paz-Argaman et al., 2024), allocentric vs. egocentric reasoning, LLM evaluation under uncertainty, calibration / abstention work |
| 3. Methodology | 1.5 pages | symbolic oracle (3-class), masking protocol, evaluation protocol — leans on `CHANGES.md` and `CHANGES_2026_04_25.md` |
| 4. Experiments | 2–3 pages | Phase 5 analyses |
| 5. Error Analysis | 1 page | Phase 6 taxonomy with examples |
| 6. Discussion | 0.5 page | what does this say about LLM spatial reasoning? |
| 7. Limitations | 0.25 page | single graph, OSM tag noise, models tested, prompt sensitivity |
| 8. Conclusion | 0.25 page | one paragraph |

### Tables (minimum)

1. **Dataset stats per city** — sample count, label distribution, POI mapping rate, mean candidate_count.
2. **Baseline confusion matrix** — 3×3 per city.
3. **Degradation table** — agreement % by `variant_type` × city × model.

### Figures (minimum)

1. **Pipeline diagram** — Instructions → NLP → Oracle (3-class) → LLM → Comparison. The README's high-level pipeline is already a list; turn it into a flowchart.
2. **Degradation curve** — the hero plot from §5.2.
3. **Per-category bars** — §5.3.
4. **Example error cases** — text + brief description of why it fails. 1–2 mapped onto a Manhattan snippet would land hard if you have the time.

---

## Two things to do in parallel *now*

1. **Sketch the paper outline** as a 1-page bullet list. Decides which plots to produce, saves rework in Phase 5.
2. **Decide the model list**. Adding flan-t5-large + a Pythia variant to the SLURM array now (one config change) triples the comparison signal for almost no extra wall-clock if jobs parallelize.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Manhattan labeling crashes mid-run | The new per-row debug log makes recovery cheap — restart from where it died (rows already processed are independent) |
| `Contradictory` rate stays high after fixes | Most of those are NLP-extraction failures; iterate on `extraction_utils.py` patterns and re-label only the affected city |
| LLM SLURM run produces empty output | Almost always the torch/transformers version mismatch noted in `CHANGES_LLM_EVAL_2026_04_25.md` — check the conda env first |
| Pittsburgh / Philadelphia mapping rate < 50% | Raise `SNAP_MAX_M` to 150–200 m and re-run Phase 1 for that city only |
| Disagreement story too noisy to write up | Filter to `llm_parsed=True` and `oracle_label != Contradictory` first — those rows are where the experimental signal lives |
