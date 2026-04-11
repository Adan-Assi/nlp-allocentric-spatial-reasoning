# 📂 Scripts Directory: RVS Spatial Reasoning Pipeline

This folder contains the core utility scripts and HPC configurations for the **NLP Spatial Reasoning Project**. These scripts handle the end-to-end workflow: data ingestion, symbolic labeling, information degradation, and large-scale LLM benchmarking.

## 🛠️ Script Registry

| Script Path | Status | Goal / Purpose |
|:---|:---|:---|
| [`data_download.py`](./data_download.py) | Core | Fetches the raw `tzufi/RVS` instruction dataset from Hugging Face. |
| [`normalize_raw.py`](./normalize_raw.py) | Core | Cleans raw dataset columns and extracts `[lat, lon]` pairs. |
| [`batch_labeling.py`](./batch_labeling.py) | **Primary** | **The Main Engine:** Executes the `SymbolicSolver` + `OracleEngine` pipeline. |
| [`underspecify.py`](./underspecify.py) | Research | **Masking Engine:** Generates 22k+ landmark/direction masked variants. |
| [`evaluate_llm.py`](./evaluate_llm.py) | Research | **Inference Engine:** Standard evaluation of the Silver Standard (7k rows). |
| [`evaluate_llm_masked.py`](./evaluate_llm_masked.py) | Research | **Stress-Test Engine:** Benchmarks the 22,173 underspecified variants. |
| [`verify_label_quality.py`](./verify_label_quality.py) | QA | Audits the distribution of `Answerable` vs `Ambiguous` labels. |

## 🚀 Cluster Inference (HPC / SLURM)

To reproduce the Phase 5 analysis on a GPU cluster, utilize the following batch scripts:

* [`job_evaluate_llm.sh`](./job_evaluate_llm.sh): **Baseline Run.** Configured for a 1-hour window on a standard GPU partition to validate the core 7k samples.
* [`job_evaluate_llm_masked.sh`](./job_evaluate_llm_masked.sh): **Production Degradation Run.** Optimized for the `studentkillable` partition with a 4-hour window and 32GB RAM to process all 22,173 experimental variants.

**Example Usage:**
```bash
sbatch scripts/job_evaluate_llm_masked.sh
```

## 🚀 Execution Flow (The "Full Loop")

1. **Map Acquisition:** Place `.gpickle` and `_poi.pkl` assets in `data/[city_name]/`.
2. **Data Preparation:** * `python scripts/data_download.py`
    * `python scripts/normalize_raw.py`
3. **Labeling:**
    * `python scripts/batch_labeling.py --city [manhattan/philadelphia/pittsburgh]`
4. **Degradation & Inference (Phase 5):**
    * `python scripts/underspecify.py`
    * `sbatch scripts/job_evaluate_llm_masked.sh`
5. **Analysis:**
    * Open `notebooks/llm_degradation_analysis.ipynb` to generate final plots.

## 📋 Requirements
* **NetworkX / OSMnx**: Graph topology and navigation.
* **Scipy / KDTree**: High-speed spatial node snapping.
* **PyTorch / Transformers**: LLM inference (T5-base).
* **Pandas / PyArrow**: High-efficiency data storage (Parquet).

## 📦 Archive / Development Only
*Scripts that are NOT part of the standard RVS production pipeline:*

| Script Path | Reason for Archive |
|:---|:---|
| `build_region_graphs.py` | **Do not use.** OSMnx builds live graphs; we must use the RVS Google Drive `.gpickle` files to avoid data misalignment with the original research. |
| `attach_target_node_all_regions.py` | **Legacy.** Node snapping is now handled dynamically within `batch_labeling.py` via the high-speed `KDTree` implementation. |
| `rvs_parser.py` | **Superceded.** The logic has been unified and optimized inside the main `batch_labeling.py` and `underspecify.py` scripts for better consistency. |