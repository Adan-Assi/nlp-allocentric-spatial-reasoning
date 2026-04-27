## 🚀 RVS Research Pipeline: End-to-End Workflow

**Step 1: `data_download.py` & `normalize_raw.py`**
* **Action:** Fetches raw RVS instructions and cleans GPS coordinates.
* **Output:** Standardized city JSON files with `[lat, lon]` goal points.

**Step 2: `batch_labeling.py` (The Main Engine)**
* **Action:** Executes the `SymbolicSolver` + `OracleEngine` to ground instructions into the street graph.
* **Output:** City-specific **`silver_standard.parquet`** files (contains the critical `extracted_noun` and `oracle_label` columns).

**Step 3: Geodetic Hydration (The "Transition" Step)**
* **Action:** Maps symbolic "Goal Nodes" from the graph back to verified $WGS84$ (Latitude/Longitude) coordinates.
* **Output:** `RVS_MASTER_GOLD_HYDRATED.parquet` (The "Clean Room" dataset).
* **Impact:** Ensures every sample contains verified real-world coordinates, effectively pruning unreachable paths and geometric contradictions from the evaluation set.

**Step 4: `underspecify_instructions.py` (The Masking Engine)**
* **Action:** Reads the Silver Standard Parquet and uses the `extracted_noun` to replace landmarks and directions with `[MASK]` tokens.
* **Output:** City-specific **`underspecified_variants.json`** files (now fully populated for sparse cities like Philly).

**Step 5: `evaluate_llm_masked.py` (The Stress-Test Engine)**
* **Action:** Sends all 22k+ masked variants to the LLM and records the raw coordinate predictions.
* **Output:** A consolidated **`LLM_DEGRADATION_RESULTS.parquet`** file.

**Step 6: Analysis & Visualization (Notebook)**
* **Action:** Merges the LLM results with the Oracle "Gold" nodes to calculate success rates (e.g., within 250m).
* **Output:** Stage 4 plots (The Grouped Bar Chart) showing the "Sparsity Bias" across Manhattan, Pittsburgh, and Philadelphia.

---

### 💡 Pro-Tip for Skimming:
* **The "Key" to the whole thing:** Step 2 MUST run before Step 3, because the Masking Engine now relies on the Solver's "wisdom" to know what to mask.

---

### 🔍 Clarification on Step 3's Necessity

*Why isn't silver enough?*

While the **OracleEngine** and **SymbolicSolver** handle logical and graph-based validity, Step 3 is required to resolve "Geodetic Mismatches" that logic alone cannot catch:

1. **The Graph/Reality Gap:** The Solver confirms a path exists in the NetworkX graph. However, if that graph component is a "disconnected island" (e.g., a pedestrian path on a bridge with no physical connection to the street below), the instruction is logically valid but physically impossible to navigate.

2. **KDTree Snapping Errors:** During batch labeling, GPS points are snapped to the nearest node. In dense urban environments, a point can snap to an "Overpass Node" (a highway) that is vertically close but functionally separated from the "Street-Level" destination mentioned in the text.

3. **Geometric Integrity:** The Hydration step prunes cases where the distance between the grounded landmark and the final GPS goal exceeds a realistic threshold (e.g., >200m), ensuring the dataset remains a "Gold Standard" free of misleading or contradictory spatial data.

---

### Why `underspecify_instructions.py` reads Silver instead of Gold

It seems counter-intuitive to use "Silver" (the intermediate) when "Gold" (the final) exists, but there is a specific reason for it in this research design: **The Scope of Masking.**

* **The Content:** The `silver_standard.parquet` contains the **`extracted_noun`** and **`extracted_category`** metadata produced by the Solver.
* **The Intent:** We need to mask the *specific* thing the solver found. If the Silver Standard says it grounded the instruction using the noun "pharmacy," we want to mask "pharmacy."

* **The Filter:** The Gold Hydrated file is a "Clean Room"; it often has rows pruned out because of the geodetic checks we discussed. If we only masked based on the Gold file, we might miss the opportunity to generate variants for samples that were "Answerable" but had slight coordinate offsets.

> **Conclusion:** It ensures the Masking Engine has access to the rawest form of the Solver's "thoughts" (the metadata) before the data is pruned for the final evaluation. You want your "Stress Test" (Step 4) to be as broad as possible, while your "Final Metric" (Step 6) is as strict as possible.
