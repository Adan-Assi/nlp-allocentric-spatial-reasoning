## Research Advisor Review

### 1. Pipeline Review — What Needs Updating

The pipeline is architecturally sound after recent changes. One gap: **Step 4 currently generates masked variants but never labels them with Oracle 2 (`mode="label"`)**. Per INCIDENT_06, masked variants need oracle labels (Answerable/Ambiguous/Contradictory) before Step 5 can evaluate the LLM meaningfully. Add this to `underspecify_instructions.py` or as a new Step 4.5.

Everything else in the pipeline flow is correct and consistent with recent changes.

---

### 2. `underspecify_instructions.py` Review

**`extracted_noun` usage — correct.** Using the solver's extracted noun for masking is the right design — it masks exactly what the solver used for grounding, not a regex guess.

**`extracted_category` — missing opportunity.** You should include it in the output for later analysis:

```python
all_experiments.append({
    "sample_id": sample_dict.get('sample_id', 'N/A'),
    "city": city,
    "original_text": sample_dict.get('instruction', ''),
    "gold_goal_node": sample_dict.get('gold_goal_node'),
    "extracted_category": sample_dict.get('extracted_category'),  # ADD
    "extracted_direction": sample_dict.get('extracted_direction'), # ADD
    "start_node": sample_dict.get('start_node'),                  # ADD — needed for Oracle 2
    "variants": sample_variants
})
```

`extracted_category` enables Step 6 analysis like "does the LLM degrade more on RESTAURANT vs PHARMACY?" — this is exactly the "characteristic error patterns" your proposal mentions. `start_node` is needed for Oracle 2 labeling of variants.

**Filtering `oracle_label == "Answerable"` — correct.** Only Answerable originals have a well-defined unique goal node. Generating variants from Contradictory originals would have no valid evaluation target.

**`gold_goal_node` — insufficient for Step 6.** You save the graph node ID but Step 6 needs GPS coordinates (lat/lon) to compute 250m distance. Add coordinates:

```python
# In batch_labeling.py output, also save goal coordinates
# OR resolve them here from the graph
"gold_goal_node": sample_dict.get('gold_goal_node'),
"gold_goal_lat": sample_dict.get('gold_goal_lat'),   # add if available
"gold_goal_lon": sample_dict.get('gold_goal_lon'),   # add if available
```

Check if `gold_goal_node` is in the silver standard and whether you can resolve its coordinates from the graph at underspecify time.

**Missing: Oracle 2 labeling of variants.** Add this loop after generating variants:

```python
# After generating variants, label each with Oracle 2
# This requires loading the solver — add at top of __main__
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
import pickle

# Load solver once per city
G = load_graph(city)  
oracle = OracleEngine(G, config.get_poi_path(), ...)
solver = SymbolicSolver(oracle, ...)

# Then per variant:
for variant in sample_variants:
    start_node = sample_dict.get('start_node')
    if start_node and start_node in G.nodes:
        label_info = solver.solve(variant['text'], start_node, mode="label")
        variant['oracle_label'] = label_info['state']
        variant['reachable_candidate_count'] = label_info.get('reachable_candidate_count', 0)
```

Without this, Step 5 has no ground truth to evaluate against.

---

### 3. `evaluate_llm_masked.py` — Conceptual Issues

**Your suspicion is correct — this is the most important thing to fix.**

The current design asks the LLM: *"What is the landmark name?"* and saves raw text. But your research question is about **spatial reasoning and goal localization**, not named entity extraction. The evaluation needs coordinates, not text.

The correct pipeline is exactly what you proposed:

```
masked instruction → LLM → predicted landmark/entity → symbolic resolution → coordinates → 250m check
```

Here's why each step is necessary:

**Step A — LLM predicts a landmark name or description:**
```python
# Current prompt is actually fine for this
"What is the specific landmark or street name of the destination?"
# LLM outputs: "CVS pharmacy" or "the park on 5th" or "I don't know"
```

**Step B — Symbolic resolution maps prediction to graph node:**
```python
# Use oracle.resolve_landmark() to ground LLM text output to a node
predicted_node = oracle.resolve_landmark(
    llm_output_raw, 
    context_node=start_node, 
    radius_m=1500
)
```

**Step C — Distance check against gold:**
```python
# Compare predicted_node coordinates to gold_goal_node coordinates
predicted_coords = (G.nodes[predicted_node]['y'], G.nodes[predicted_node]['x'])
gold_coords = (G.nodes[gold_goal_node]['y'], G.nodes[gold_goal_node]['x'])
distance_m = haversine(predicted_coords, gold_coords)
success_250m = distance_m <= 250
```

**What to fix in `evaluate_llm_masked.py`:**

```python
# 1. Input needs start_node and gold_goal_node columns
# 2. After inference, add symbolic resolution step
# 3. Save structured results not just raw text

results.append({
    'sample_id': row['sample_id'],
    'city': row['city'],
    'variant_type': row['variant_type'],      # mask_landmark / mask_directions / mask_both
    'oracle_label': row['oracle_label'],       # from Oracle 2
    'masked_instruction': row['masked_instruction'],
    'llm_output_raw': decoded_text,
    'predicted_node': predicted_node,          # after symbolic resolution
    'distance_m': distance_m,                  # haversine to gold
    'success_250m': distance_m <= 250,
    'success_100m': distance_m <= 100,
})
```

**Also: the prompt needs improvement.** FLAN-T5-base at ~250M parameters struggles with long prompts. Simplify:

```python
# Current (too long, mixes task types):
f"Task: Follow the navigation instructions in {row['city']}.\n"
f"Instructions: {row['masked_instruction']}\n"
f"Question: What is the specific landmark or street name of the destination?\n"
f"Answer:"

# Better (concise, task-focused):
f"Navigation instruction: {row['masked_instruction']}\n"
f"Destination landmark:"
```

---

## What To Do Now While Manhattan Runs

**Priority order:**

1. **Fix `underspecify_instructions.py`** — add `start_node`, `extracted_category`, `extracted_direction` to output. This can be done now and doesn't need Manhattan results.

2. **Fix `evaluate_llm_masked.py`** — add symbolic resolution step (Step B above) and structured result saving. This is the most conceptually important fix.

3. **Phase 4 Layer 1** — create the 100-instruction gold-labeled test set for category extraction accuracy. You can do this manually for Pittsburgh now using the silver standard:

```python
# Sample 100 instructions for manual labeling
df = pd.read_parquet('data/pittsburgh/pittsburgh_silver_standard.parquet')
sample = df.sample(100, random_state=42)[['instruction', 'extracted_category', 'extracted_noun']]
sample.to_csv('data/eval/category_gold_100.csv', index=False)
```

4. **Wait for Manhattan** before running Phase 4 Layer 2 (MRR, Recall@k) — needs all city silver standards.