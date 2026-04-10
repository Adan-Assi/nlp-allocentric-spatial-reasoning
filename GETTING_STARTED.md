# 🚀 Getting Started: RVS Spatial Reasoning Pipeline

This guide explains how to set up the environment, acquire the necessary map assets, and run the **Silver Standard Labeling** pipeline to replicate our multi-city results.

---

## 1. 🏗️ Environment Setup

First, clone the repository and set up the Conda environment. This ensures all dependencies (OSMnx, Geopy, NetworkX) are version-matched.

```bash
# Clone the repository
git clone https://github.com/Adan-Assi/nlp-allocentric-spatial-reasoning.git

# Enter the project directory
cd nlp-allocentric-spatial-reasoning

# Create and activate the environment
conda create --name rvs python=3.10 -y
conda activate rvs

# Install dependencies
pip install -r requirements.txt
```

---

## 2. 🗺️ Data Acquisition (Manual Step)
The RVS map files are too large for GitHub. You must download the official assets to ensure our Oracle matches the authors' original coordinate system.

1. Download Maps: Go to the [Official RVS Google Drive](https://drive.google.com/drive/folders/1bvxNeIlN1SKeup6aJgIUzWrQ8v-cL9Yq).

2. Download Instructions: Instructions are pulled automatically by `data_download.py`, but can be found here if needed.

3. Local Directory Structure: Place the files exactly as shown below:

```text
data/
├── manhattan/
│   ├── manhattan.json
│   ├── manhattan_graph.gpickle
│   └── manhattan_poi.pkl
├── philadelphia/
│   ├── philadelphia.json
│   ├── philadelphia_graph.gpickle
│   └── philadelphia_poi.pkl
└── pittsburgh/
    ├── pittsburgh.json
    ├── pittsburgh_graph.gpickle
    └── pittsburgh_poi.pkl
```

> ⚠️ CRITICAL: Do NOT run `build_region_graphs.py`. This script downloads "Live" OSM data which has drifted since the RVS dataset was created. Use only the .gpickle files from the Google Drive.

---

## 3. ⚙️ Configuration
Check `config.py` to ensure the `CITY_SETTINGS` reflect the correct density-aware parameters:

- Manhattan: salience_ratio: 0.7, success_radius: 80m

- Pittsburgh: salience_ratio: 0.5, success_radius: 100m

- Philadelphia: salience_ratio: 0.5, success_radius: 250m

---

## 4. 🏃 Execution Flow
Run the pipeline in this specific order to generate the Silver Standard labels and the experimental variants.

### Step A: Data Normalization
Prepares the raw HuggingFace data for our spatial engine.

```bash
python scripts/data_download.py
python scripts/normalize_raw.py
```

### Step B: Silver Standard Labeling (The "Judge")
This uses the `SymbolicSolver` to verify which instructions are actually solvable. Note that landmarks beyond 1500m are rejected as "Range Contradictions" per **Paz-Argaman et al. (2020).**

```bash
# Run for all cities (Recommended: Use Slurm for parallel execution)
python scripts/batch_labeling.py --city manhattan
python scripts/batch_labeling.py --city philadelphia
python scripts/batch_labeling.py --city pittsburgh
```

### Step C: Master Consolidation
Merges the individual city outputs into a unified training file.

```bash
python scripts/merge_silver_standards.py
```

### 📊 Expected Outputs
After running the pipeline, check the /data folder for:

- `RVS_MASTER_SILVER_STANDARD.parquet`: The unified labeled ground truth (7,263 Answerable rows).

- `reports/silver_standard.md`: Detailed audit report and evaluation of city-specific yield.


### 🆘 Troubleshooting
- ArrowTypeError: Occurs if `sample_id` columns are mixed (int vs string). The merge script handles this by forcing `astype(str)`.

- Memory Issues: The Manhattan graph requires ~4GB RAM. Close memory-heavy apps or use the Slurm cluster for batch runs.

- Coordinate Drift: If the Oracle finds 0 candidates for a known goal, verify you are using the Google Drive assets. Live OSM data will result in a ~15% higher "Contradictory" rate.


### 💡 Note for the Team
The Salience Ratio is the "magic knob" for this project. If you find too much ambiguity in a dense area, increase the ratio to **0.7**. For sparse residential areas, drop it to **0.5** to allow for more flexible landmark matching.

---

## 5. 🏗️ Cluster-Specific Setup (University Cluster)
If you are running on the cluster (e.g., node `c-008`), follow these steps to bypass home-directory storage limits and utilize GPUs.

### A. Environment Location
The environment is stored on the high-capacity volume to avoid `Disk Quota Exceeded` errors.

```bash
# Activation Handshake
cd /vol/joberant_nobck/data/NLP_368307701_2526a/<username>/
conda activate <environment_name>
```

### B. HuggingFace "Portal" (The Shield)
We have linked the `~/.cache` folder to the `/vol/` drive. This ensures that when you download large LLMs (Pythia, T5, etc.), they do not occupy your home directory space.

* **Verification:** `ls -ld ~/.cache` should show a symlink to `/vol/.../<username>/.cache`.

### C. Running LLM Inference
To evaluate the Silver Standard using neural models, use the cluster GPUs via Slurm or interactive sessions:

```bash
# Example: Run Pythia-70m Benchmarking
python scripts/evaluate_llm.py --model_name "EleutherAI/pythia-70m" --data_path "./RVS_MASTER_SILVER_STANDARD.parquet"
```
### 📂 Cluster Directory Map

| Path | Description |
| :--- | :--- |
| `/vol/.../<username>/anaconda3/` | The Conda installation and environments. |
| `/vol/.../<username>/.cache/` | Storage for model weights (Transformers). |
| `/vol/.../<username>/data/` | Your Map assets (.gpickle, .pkl, .json). |
| `/vol/.../<username>/*.parquet` | Your generated Silver Standard results. |

### 🛠️ Maintenance & Cleanup
If you need to free up space on the volume:

* **Clear the HF cache:** `rm -rf /vol/joberant_nobck/data/NLP_368307701_2526a/<username>/.cache/huggingface/hub/*`
* **Remove old logs:** `rm slurm-*.out`