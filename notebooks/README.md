# 🏺 Project Notebooks (Knowledge Base)
This index serves as the central map for our symbolic navigation pipeline. Use these links to revisit our core design decisions, data audits, and topological optimizations.

## 🚀 Navigation Guide

| Notebook File | Scientific Goal / Purpose | When to consult it (Quick Reminder) |
| :--- | :--- | :--- |
| **🔍 Data Audits & Compatibility** | | |
| [`inspect_poi.ipynb`](./inspect_poi.ipynb) | **Legacy Data Recovery:** Implements a "Monkey Patch" for Pandas 2.0 to load legacy `.pkl` files. Audits 1,033 columns to find dense tags like `amenity` and `shop`. | Consult this to **verify OSM tag density** or if you encounter an `Int64Index` error when loading POIs. |
| [`integration_check.ipynb`](./integration_check.ipynb) | **ID Alignment & Connectivity:** Performs a "1# Sanity Check" to ensure the POI `osmid` matches the Graph's projected node IDs. | Consult this to **confirm the ID prefix** (e.g., `1#12345`) and verify that landmarks are actually connected to the road network. |
| **📊 Linguistic & Spatial Analysis** | | |
| [`landmark_frequency_analysis.ipynb`](./landmark_frequency_analysis.ipynb) | **NLP Entity Extraction:** Uses NLTK POS-tagging to identify the most common landmark nouns in the Manhattan dataset. | Consult this to see the **top 30 landmark types** used by humans to ensure our `LANDMARK_GROUPS` config is exhaustive. |
| [`Landmark_Recall_Sensitivity.ipynb`](./Landmark_Recall_Sensitivity.ipynb) | **Hyperparameter Optimization:** Empirically identifies 1.5km as the optimal search radius to balance Recall vs. Computational Complexity. | Consult this for the **graphical proof of the 1.5km threshold** used to justify our pruning logic. |
| **🗺️ Geospatial Layering** | | |
| [`geo_paths_layers.ipynb`](./geo_paths_layers.ipynb) | **GeoPackage Inspection:** Uses `fiona` to audit the `manhattan_geo_paths.gpkg` file. Identifies layers for start/end points and pivot landmarks (Main, Near, Beyond). | Consult this to understand the **multi-layer structure** of our spatial paths and how pivot landmarks are categorized. |
| **🧭 Semantic Grounding & Testing** | | |
| [`Manhattan_Semantic_Navigator.ipynb`](./Manhattan_Semantic_Navigator.ipynb) | **Intelligence Validation:** Tests the `deep_search` logic by matching instruction text to physical coordinates using semantic scoring. | Consult this to see **how the Solver "thinks"** when resolving a vague goal name to a specific map coordinate. |

---

## 🗺️ Visualization Artifacts (.html)
These files are the rendered outputs of our spatial inference tasks. Open these in a browser to audit the agent's decision-making on an interactive Manhattan map.

* **`my_manhattan_trip.html`**: The full rendered trajectory of a sample navigation task.
* **`semantic_inference_task_457.html`**: A deep-dive into Task #457, showing how semantic keywords were grounded to specific POI nodes.
* **`spatial_inference_to_manhattan_trip.html`**: Visualizes the "Directional Wedges" and spatial logic used to narrow down the target path.

---

### 💡 Key Technical Insights
* **Layering:** `geo_paths_layers` confirmed the use of a 6-layer GeoPackage, separating path features from pivot points.
* **Prefix Management:** `integration_check` confirmed the mandatory `1#` prefix for node lookups within the `.gpickle` graph.
* **The 1.5km Decision:** Sensitivity Analysis proved that 1500m is the "Efficiency Peak," capturing **84%** of landmarks before candidate noise grows exponentially.
* **Data Density:** `inspect_poi` identified `amenity` (20k+ entries) and `shop` (3.5k+ entries) as the primary high-confidence columns for the Oracle.
