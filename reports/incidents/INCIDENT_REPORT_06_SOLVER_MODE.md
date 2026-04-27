# 📄 Incident Report: 06-Solver Mode & Ambiguity Architecture

> **Salient** (Definition)
>
> In this context, *salient* refers to the most prominent or noticeable candidate among several valid options. The one that stands out as the most likely intended answer (for example, a well-known landmark or a location with stronger recognition signals such as a Wikipedia page).

## Background: The Previous Design

The previous `solve()` had no `mode` parameter. When multiple candidates were found,
it always resolved to a single target via salience-based selection (per the RVS
LANDMARK baseline), and returned `STATE_ANSWERABLE`. `STATE_AMBIGUOUS` was never
assigned anywhere in the pipeline.

This was intentional for the silver standard labeler, original RVS instructions are
human-validated to be unambiguous, so multiple solver candidates indicate solver
imprecision, not genuine instruction ambiguity. Salience-based resolution was the
correct behavior for Step 2 (From [steps_sequence](docs/steps_sequence.md))

**The gap:** `underspecify_instructions.py` generates masked variants (e.g. "meet me at the [MASK]
north of you") but never calls `solver.solve()` on them. The variants are saved with
no oracle label. There is no mechanism to measure whether a masked variant is still
`Answerable`, has become `Ambiguous`, or has become `Contradictory`.

---

## Why This Matters for the Research

The project proposal defines the research question as:

> "How robust are LLMs at allocentric spatial reasoning when natural-language
> descriptions of space are systematically underspecified?"

And explicitly defines the three labels:

> "labels each variant as **Answerable (unique solution), Ambiguous (multiple
> solutions), or Contradictory (no solution)**."

Without an oracle labeling the masked variants, there is no ground truth to evaluate
the LLM against. The LLM predicts a coordinate, but we cannot determine whether:
- The masked instruction still had a unique answer (Answerable → LLM should get it right)
- The masking created genuine ambiguity (Ambiguous → no single correct answer exists)
- The masking destroyed all valid candidates (Contradictory → instruction unsolvable)

This means Step 6 (analysis) is currently measuring LLM performance against an
incomplete gold standard.

---

## The Two Oracle Architecture

The fix requires recognizing that the pipeline needs **two separate "oracles"** with
different behaviors:

### Oracle 1 — Silver Standard Labeler (`batch_labeling.py`)
- **Input:** Original RVS instructions (human-validated unique)
- **Behavior:** When multiple candidates found → pick most salient → `Answerable`
- **Never returns:** `STATE_AMBIGUOUS`
- **Rationale:** Original instructions passed human validation (100m rule). Multiple
  solver candidates = solver imprecision, not genuine ambiguity. Mirrors RVS
  LANDMARK baseline salience hierarchy.
- **Output:** `Answerable` / `Contradictory` only

### Oracle 2 — Variant Labeler (`underspecify_instructions.py` or new `label_variants.py`)
- **Input:** Masked variants (landmark and/or direction removed)
- **Behavior:** When multiple candidates found → `Ambiguous` (preserve the signal)
- **Returns:** `Answerable` / `Ambiguous` / `Contradictory`
- **Rationale:** Masking genuinely destroys uniqueness. "Meet me at the [MASK] north
  of you" may match 5 cafes. That IS ambiguity, not solver imprecision.
- **Output:** All three labels

---

## How the `mode` Parameter Implements the Oracle Architecture

`mode` parameter maps directly onto the two oracle roles:

```python
solver.solve(instruction, start_node, mode="resolve")  # Oracle 1 behavior
solver.solve(instruction, start_node, mode="label")    # Oracle 2 behavior
```

| Aspect | `mode="resolve"` (Oracle 1) | `mode="label"` (Oracle 2) |
|--------|----------------------------|--------------------------|
| Multiple candidates | Pick most salient → Answerable | Preserve → Ambiguous |
| Reachability | Filtered first, then salience selection | Filtered first, then counted |
| Returns Ambiguous | Never | Yes, when reachable_count > 1 |
| Used in | `batch_labeling.py` (Step 2) | Variant labeling (Step 4/4.5) |

> **Implementation Note: Previous Reachability Bug**
>
> A previous issue existed in the Oracle 1 (`mode="resolve"`) logic due to the order of salience selection and reachability checking.
>
> Example: suppose there are 5 valid café candidates. The most salient one (for example, the one with a Wikipedia page) is located on a disconnected graph island and is therefore unreachable, while another candidate is fully reachable.
>
> In the earlier implementation, the system first selected the most salient candidate and only then checked reachability. As a result, it chose the unreachable café, failed the reachability check, and incorrectly returned **Contradictory**, even though a valid reachable answer still existed.
>
> The corrected logic ensures that reachability is considered before concluding contradiction, preventing false negative labels.

---

## How This Implementation Differs From `Final-Pipeline` Branch

The Final-Pipeline branch adds `mode` correctly but proposes using `mode="label"` as the
**default**, which would make `batch_labeling.py` label original instructions as
Ambiguous whenever the solver finds multiple candidates.

This is **incorrect for Step 2** because:
1. Original RVS instructions are human-validated unique; multiple solver candidates
   indicate solver imprecision, not genuine ambiguity
2. It would inflate Ambiguous % on original instructions, corrupting the baseline
3. The RVS paper confirms: "if a solver finds five matching cafes, it typically means
   it is failing to account for fine-grained configurational details that allowed the
   human validator to successfully isolate the unique target"

**The Updated implementation:**
- `mode="resolve"` is the default (correct for Step 2)
- `mode="label"` is explicitly passed in variant labeling (correct for Step 4/4.5)
- `batch_labeling.py` passes `mode="resolve"` explicitly for clarity

```python
# batch_labeling.py — Oracle 1
label_info = solver.solve(instruction, start_node, mode="resolve")

# label_variants.py or underspecify_instructions.py — Oracle 2
label_info = solver.solve(masked_instruction, start_node, mode="label")
```

---

## What Needs to Be Added to the Pipeline

A new labeling step is needed between Step 4 and Step 5 that runs Oracle 2 on every
masked variant:

```python
# New step: label_variants.py (or add to underspecify_instructions.py)
for experiment in all_experiments:
    start_node = experiment['start_node']
    for variant in experiment['variants']:
        label_info = solver.solve(variant['text'], start_node, mode="label")
        variant['oracle_label'] = label_info['state']
        variant['reachable_candidate_count'] = label_info.get('reachable_candidate_count', 0)
        variant['candidate_nodes'] = label_info.get('candidate_nodes', [])
```

This produces the ground truth that Step 6 uses to evaluate LLM predictions:
- `Answerable` variants → LLM should predict within 250m of the unique target
- `Ambiguous` variants → LLM prediction cannot be "wrong" (no unique answer exists)
- `Contradictory` variants → instruction is unsolvable, LLM should ideally abstain

---

## Files To Be Modified

| File | Change |
|------|--------|
| `src/symbolic_solver.py` | Add `mode` parameter to `solve()`, implement both behaviors |
| `scripts/batch_labeling.py` | Pass `mode="resolve"` explicitly |
| `scripts/label_variants.py` | Add Oracle 2 labeling loop after variant generation |
| Pipeline doc | Add Step 4.5 "Variant Oracle Labeling" between Steps 4 and 5 |

---

## Key Insight

The `mode` parameter is not primarily about the solver's internal logic — it is about
which oracle role the solver is playing at each pipeline step. The same solver,
same graph, same POI data, but two different contracts:

- **Oracle 1:** "Find the unique intended target, resolve ties by salience"
- **Oracle 2:** "Count how many valid targets remain, preserve ambiguity as signal"

Without both modes, the pipeline can generate masked variants but cannot measure
whether those variants are actually harder — which is the entire point of the research.