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

1. Download Maps: Go to the Official RVS Google Drive.

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
Check `config.py` to ensure the CITY_SETTINGS reflect the correct filenames and success radii:

- Manhattan: 80m

- Philadelphia/Pittsburgh: 100m

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
This uses the SymbolicSolver to verify which instructions are actually solvable on the map.

```bash
# Run for all cities (Recommended: Use Slurm for parallel execution)
python scripts/batch_labeling.py --city manhattan
python scripts/batch_labeling.py --city philadelphia
python scripts/batch_labeling.py --city pittsburgh
```

### Step C: Underspecification (The "Experiment")
Once you have the Answerable labels, run this to create the masked variants for Phase 5.

```bash
python scripts/underspecify.py
```

### 📊 Expected Outputs
After running the pipeline, check the data/[city]/ folders for:

- `[city]_silver_standard.parquet`: The labeled ground truth.

- `ambiguity_report.csv`: Statistics on how many instructions were Ambiguous vs. Answerable.

### 🆘 Troubleshooting
- Memory Issues: The Manhattan graph requires ~4GB RAM. If running locally, close memory-heavy apps or use a Slurm cluster.

- Coordinate Drift: If the Oracle finds 0 candidates for a known goal, verify you are using the Google Drive `.gpickle` and not a locally generated one.


### 💡 Note for the Team
I have explicitly flagged the **"No `build_region_graphs.py`"** rule. This is the most common mistake in RVS replication; if we build our own graphs, the `node_ids` won't match the `landmarks` dictionary in the JSON, and the project will fail to validate.