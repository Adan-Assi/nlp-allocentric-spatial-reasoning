# 🏺 Project Notebooks (Knowledge Base)
This index serves as the central map for our symbolic navigation pipeline. Use these links to revisit our core design decisions, data audits, and topological optimizations.

## 🚀 Navigation Guide

| Notebook File | Scientific Goal / Purpose | When to consult it (Quick Reminder) |
| :--- | :--- | :--- |
| **🔍 Data Audits & Compatibility** | | |
| [`inspect_poi.ipynb`](./inspect_poi.ipynb) | **Legacy Data Recovery:** Implements a "Monkey Patch" for Pandas 2.0 to load legacy `.pkl` files. Audits 1,033 columns to find dense tags like `amenity` and `shop`. | Consult this to **verify OSM tag density** or if you encounter an `Int64Index` error when loading POIs. |
| [`integration_check.ipynb`](./integration_check.ipynb) | **ID Alignment & Connectivity:** Performs a "1# Sanity Check" to ensure the POI `osmid` matches the Graph's projected node IDs. | Consult this to **confirm the ID prefix** (e.g., `1#12345`) and verify that landmarks are actually connected to the road network. |
| [`targeted_discovery.ipynb`](./targeted_discovery.ipynb) | **Precision Engine Tuning:** Performs an information-density audit across all OSM tags. Identifies high-value columns (`amenity`, `shop`, `cuisine`) to replace slow `.apply()` loops with vectorized "Sniper" searches. | Consult this to **optimize search speeds** or to identify which columns contain the "Intel" for a new city's dataset. |
| [`asset_sanity_check.ipynb`](./asset_sanity_check.ipynb) | **Multi-City Asset Validation:** Performs a cross-city structural audit on `pittsburgh_graph.gpickle` and `philadelphia_graph.gpickle`. | Consult this to **verify graph compatibility** before running the multi-city expansion. |
| **📊 Linguistic & Spatial Analysis** | | |
| [`landmark_frequency_analysis.ipynb`](./landmark_frequency_analysis.ipynb) | **NLP Entity Extraction:** Uses NLTK POS-tagging to identify the most common landmark nouns in the Manhattan dataset. | Consult this to see the **top 30 landmark types** used by humans to ensure our `LANDMARK_GROUPS` config is exhaustive. |
| [`Landmark_Recall_Sensitivity.ipynb`](./Landmark_Recall_Sensitivity.ipynb) | **Hyperparameter Optimization:** Empirically identifies 1.5km as the optimal search radius to balance Recall vs. Computational Complexity. | Consult this for the **graphical proof of the 1.5km threshold** used to justify our pruning logic. |
| **🗺️ Geospatial Layering** | | |
| [`geo_paths_layers.ipynb`](./geo_paths_layers.ipynb) | **GeoPackage Inspection:** Uses `fiona` to audit the `manhattan_geo_paths.gpkg` file. Identifies layers for start/end points and pivot landmarks (Main, Near, Beyond). | Consult this to understand the **multi-layer structure** of our spatial paths and how pivot landmarks are categorized. |
| **🧭 Semantic Grounding & Testing** | | |
| [`Manhattan_Semantic_Navigator.ipynb`](./Manhattan_Semantic_Navigator.ipynb) | **Intelligence Validation:** Tests the `deep_search` logic by matching instruction text to physical coordinates using semantic scoring. | Consult this to see **how the Solver "thinks"** when resolving a vague goal name to a specific map coordinate. |
| [`debug_contradictory_samples.ipynb`](./debug_contradictory_samples.ipynb) | **Oracle V4 Development & Deep Search:** Forensic audit of 412 "Contradictory" samples. Implements the "Aggressive V4" loop (1.5km bounding box + multi-column deep search). | Consult this to understand the **Net Gain Calculation** (V1 vs. V4) and how searching across `amenity`, `shop`, and `tourism` tags simultaneously resolved the "Data Gap" issue. |
| [`validate_oracle_v4_update.ipynb`](./validate_oracle_v4_update.ipynb) | **Production Benchmarking (V4):** The final validation suite for the Proximity-Aware Oracle. Confirms the 92.53% accuracy and the "Rescue" of 310 samples. | Consult this for the **final performance metrics** of the Manhattan Silver Standard V4. |
| [`philly_rescue_931_ambiguous.ipynb`](./philly_rescue_931_ambiguous.ipynb) | **NLP Debugging & Silver Merge:** Diagnoses the Philly "None Noun" failure and executes the initial 3-city merge into the Silver Standard. | Consult this for the **V3 Extraction Logic**, the "Pretzel Factory" range-logic proof, and the logic used to create the foundation for the eventual Gold Standard. |
| **🏅 Final Gold Verification** | | |
| [`truth_convergence_audit.ipynb`](./truth_convergence_audit.ipynb) | **Geodetic Hydration & Baseline Validation:** Hydrates 9,301 Symbolic IDs into $WGS84$ coordinates. Benchmarks the dataset against the official RVS "STOP" Baseline (1,124m). | Consult this for the **Proof of Gold Status** and to verify that the final dataset perfectly replicates official research task difficulty. |
| **📉 LLM Benchmarking & Robustness** | | |
| [`read_evaluated_llm.ipynb`](./read_evaluated_llm.ipynb) | **Inference Post-Processing:** Parses raw LLM coordinate outputs from the cluster. Aligns model predictions with the Symbolic Gold Standard for error calculation. | Consult this to **verify the parsing logic** for coordinate strings and to see how raw model logs are converted into plottable DataFrames. |
| [`llm_degradation_analysis.ipynb`](./llm_degradation_analysis.ipynb) | **Information Decay Analysis:** Generates the "Worrisome Gap" and "Spatial Drift" (KDE) visualizations. Benchmarks masked performance against RVS official papers. | Consult this for the **core scientific plots** of the project and the statistical proof of the model's "Near-Sighted" intelligence. |
---

## 🗺️ Visualization Artifacts (.html)
These files are the rendered outputs of our spatial inference tasks. Open these in a browser to audit the agent's decision-making on an interactive Manhattan map.

* **`my_manhattan_trip.html`**: The full rendered trajectory of a sample navigation task.
* **`semantic_inference_task_457.html`**: A deep-dive into Task #457, showing how semantic keywords were grounded to specific POI nodes.
* **`spatial_inference_to_manhattan_trip.html`**: Visualizes the "Directional Wedges" and spatial logic used to narrow down the target path.

---

### 💡 Key Technical Insights

* **The V4 Deep Search Mask:** `debug_contradictory_samples` proved that categorical searches were too narrow. By implementing a **Multi-Column Mask** (searching `name`, `amenity`, `shop`, `tourism`, `leisure`, `historic`, and `man_made` simultaneously), the Oracle V4 reclaimed **310 instructions** previously lost to "Data Gaps" in OSM tagging.

* **Bounding Box Optimization:** To maintain efficiency during aggressive searches, we implemented a **Geospatial Bounding Box** ($\Delta \approx 0.0135^\circ$ or $1500m$). This pre-filter allows the "Deep Search" to remain computationally lightweight while maintaining a **98% landmark recall**.

* **The "None" Noun Resolution:** `philly_rescue_931_ambiguous` identified a critical NLP failure where the solver defaulted to ambiguous categorical searches. Implementing the **V3 Anchor-Based Parser** (clipping text at the *last* 'at' or 'to' token) rescued the Philadelphia yield from a **2.4%** success rate to over **80%**.

* **Range-Based Contradiction:** Audit of the "Pretzel Factory" case proved that "Contradictory" labels are often geographically accurate but spatially impossible. By rejecting landmarks >1.5km from the agent, the Oracle enforces a **Local Reasoning Horizon** consistent with human observable limits (Paz-Argaman et al., 2020).

* **Vectorization vs. Iteration:** `targeted_discovery` proved that searching high-density columns via vectorization is **40x faster** than whole-row `.apply()` iteration. This optimization was the prerequisite for scaling the pipeline to the 14,000-node Philadelphia graph without timeouts.

* **Schema Standardization:** During the final merge, we resolved a critical `ArrowTypeError` by forcing `sample_id` to **String** format. This ensured 100% schema compatibility when merging city-specific Parquet files into the unified `RVS_MASTER_GOLD_HYDRATED`.

* **Geodetic Convergence:** `truth_convergence_audit` validated that the hydrated "Gold" coordinates replicate the official RVS "STOP" baseline with **<1% variance** (1,133m vs 1,124m). This confirms the dataset provides a high-fidelity replica of the task difficulty found in state-of-the-art navigation research.

* **Data Provenance:** `geo_paths_layers` and `integration_check` confirmed the structural integrity of the pipeline, specifically the use of a **6-layer GeoPackage** and the mandatory **1# node prefixing** required for consistent graph lookups.

* **The "Worrisome Gap" Discovery:** `llm_degradation_analysis` identified a massive decoupling between topological and semantic intelligence. While the model maintains **~90% street-level grounding**, success drops to **<2%** for specific entities when landmarks are masked. This suggests LLMs navigate using a "Skeletal Map" rather than high-resolution entity memory.

* **Anatomy of a Near-Miss:** Forensic analysis in `llm_degradation_analysis` revealed a "Spatial Drift" hump between **20m and 150m**. This proves the model isn't hallucinating randomly; it is successfully navigating to the correct vicinity but lacks the "Terminal Precision" to identify the specific door or building without explicit landmark tokens.

* **Inference Alignment:** `read_evaluated_llm` resolved the challenge of mapping unstructured LLM text back to geodetic points. By implementing a **Coordinate Extraction Regex**, we successfully benchmarked 22,000+ variants against the symbolic ground truth with 100% parity.