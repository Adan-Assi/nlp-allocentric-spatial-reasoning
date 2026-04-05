# 📂 Scripts Directory: RVS Spatial Reasoning Pipeline

This folder contains the core utility scripts for the **NLP Spatial Reasoning Project**. These scripts handle the end-to-end workflow: downloading instructions, loading RVS-provided graphs, and generating **Silver Standard** labels for multi-city navigation.

## 🛠️ Script Registry

| Script Path | Status | Goal / Purpose |
|:---|:---|:---|
| `data_download.py` | Core | Fetches the raw `tzufi/RVS` instruction dataset from Hugging Face. |
| `normalize_raw.py` | Core | Cleans raw dataset columns and extracts `[lat, lon]` pairs for start/goal. |
| `batch_labeling.py` | **Primary** | **The Main Engine:** Uses `SymbolicSolver` + `OracleEngine` on the RVS Google Drive assets. |
| `underspecify.py` | Research | Generates "Ambiguous" variants by masking landmarks to test model degradation (Phase 5). |
| `verify_label_quality.py` | QA | Audits the distribution of `Answerable` vs `Ambiguous` labels in Parquet outputs. |
| `stress_test_oracle.py` | QA | Diagnostic suite for validating Oracle logic across city boundaries. |
| `qc_ambiguous.py` | Tool | Quality Control utility for manual inspection of ambiguous samples. |

### 📦 Archive / Development Only
*Scripts that are NOT part of the standard RVS production pipeline:*
- `build_region_graphs.py`: **Do not use.** OSMnx builds live graphs; we must use the RVS Google Drive `.gpickle` files to avoid data misalignment.
- `attach_target_node_all_regions.py`: Legacy script; node snapping is now handled dynamically in `batch_labeling.py` via KDTree.
- `rvs_parser.py`: Superceded by the unified loading logic in `batch_labeling.py`.

---

## 🚀 Execution Flow (The "Silver Standard" Path)

To reproduce the dataset labels from scratch using the RVS-provided maps:

1.  **Map Acquisition:** Ensure the `.gpickle` and `_poi.pkl` files from the RVS Google Drive are placed in `data/[city_name]/`.
2.  **Data Preparation:** 
    * `python data_download.py`
    * `python normalize_raw.py`
3.  **Labeling (The Big Run):**
    * `python batch_labeling.py --city manhattan`
    * `python batch_labeling.py --city philadelphia`
    * `python batch_labeling.py --city pittsburgh`
4.  **Verification:**
    * `python verify_label_quality.py`

## 📋 Requirements
- **NetworkX / OSMnx**: For graph topology.
- **Geopy**: Geodesic distance (meters) for Success Radius checks.
- **Scipy**: `KDTree` for high-speed spatial node snapping.
- **Pandas / PyArrow**: For efficient Silver Standard Parquet storage.