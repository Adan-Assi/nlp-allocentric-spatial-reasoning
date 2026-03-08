# Robust Allocentric Spatial Reasoning under Underspecified Instructions

Course project (NLP): evaluating how robust large language models (LLMs) are at **allocentric spatial reasoning** when navigation instructions are **systematically underspecified**, while the underlying world (urban map graph) remains fixed and well-defined.

---

## 🔹 Guiding Principle

**De-risk early.**  
Prove feasibility quickly, freeze infrastructure, then focus on **analysis**, not engineering.

---

## 🧠 Project Overview

LLMs often perform well on clean, fully specified spatial instructions.  
However, real-world spatial language is frequently incomplete and relies on implicit shared knowledge.

This project studies whether LLMs **truly reason over spatial structure** or rely on surface-level cues by:

- grounding navigation instructions in **real urban map graphs** (RVS dataset),
- systematically **removing or weakening spatial information** in the instructions,
- comparing LLM outputs against **graph-derived ground truth**.

---

## 🗺️ Dataset

We use the **Rendezvous (RVS)** dataset:

- Human-written allocentric navigation instructions
- Grounded in real-world urban map graphs
- Clear, graph-backed ground truth (reachability, paths, directions)

The dataset allows us to modify instructions while keeping the environment fixed and answerable.

---

## 📖 Documentation Hub

Access the full project specifications and research strategy below:

### 🧠 Strategy & Vision
* [**Strategy & Calibration**](docs/STRATEGY_CALIBRATION.md) — *The "Big Picture."* Explains our "Brain Building" analogy, the 4 stages of the project, and how we use Manhattan as a logic laboratory.

### ⚙️ Technical Specifications
* [**Oracle Specification**](docs/ORACLE_SPEC.md) — *The "Math."* Deep dive into the Symbolic Solver's logic, including the Clamped Radius formula and Vector/Bearing filters.
* [**Pipeline Data Flow**](docs/PIPELINE_FLOW.md) — *The "Architecture."* A step-by-step trace of how raw instruction text turns into a final answerability label.

### 🗃️ Data & Implementation
* [**Data & Graph Guide**](docs/DATA_GUIDE.md) — *The "Reference."* Detailed breakdown of the 74k-node Manhattan graph, POI pickle structure, and node ID prefixing (`1#`).
* [**Task Tracker (TASKS.md)**](docs/TASKS.md) — *The "Roadmap."* Current sprint progress, pending logic implementations, and team assignments.

---

## 📁 Repository Structure

The repository is organized to clearly separate data, code, and environment setup:

```text
nlp-allocentric-spatial-reasoning/
├── data/
│   └── manhattan/
│       └── manhattan_graph.gpickle   # Fixed Manhattan street graph (OSM-derived)
├── scripts/
│   └── sanity_check.py               # Graph sanity check (Task 1)
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```
The virtual environment (`.venv/`) is created locally and is not tracked by Git.

---
## ⚙️ Setup & Graph Sanity Check

### 1. Environment setup

Create and activate a virtual environment:

    python -m venv .venv
    .venv\Scripts\activate             # Windows
    # source .venv/bin/activate        # macOS / Linux

### 2. Install dependencies

Ensure your environment is active, then install the required engines (including `pyarrow` for data and `scikit-learn` for spatial indexing):

    pip install -r requirements.txt

### 3. Run the graph sanity check

Verify the maps are loading correctly:

    #python tests/sanity_check.py
    python tests/sanity_check_all_graphs.py

## Load and preprocess the RVS dataset

Follow these steps to prepare the data. These steps load the RVS dataset from Hugging Face, extract coordinates, map them to the nearest graph node, and store the results for training.

**Normalize raw data:** Convert Hugging Face data into consistent internal formats.

```bash
python scripts/normalize_raw.py
```

**Ground instructions to graph nodes:** Map lat/lon coordinates to the nearest street graph nodes.

```bash
python scripts/attach_target_node_all_regions.py
```

> ⚠️ **Important Note:** The first time you run this for a region (e.g., Manhattan), the script builds a spatial index (BallTree). The progress bar may stay at **0% for 1–3 minutes** while indexing. **Do not interrupt it.** Once indexed, it will process thousands of rows per minute.

**Verify results:** Generate a report on the grounding success rate.

```bash
python scripts/grounding_report_all_regions.py
```

**What these steps achieve:**

1. **Load** the RVS dataset from Hugging Face.

2. **Extract** instruction text and goal coordinates.

3. **Map each** target latitude/longitude to the nearest graph node (grounding).

4. **Store** `target_node_id` in Parquet format for efficient training and evaluation.

After this stage, instructions are graph-grounded and ready for model training or robustness evaluation.

---

## 🔧 High-Level Pipeline

1. **Instruction** (original or underspecified)
2. **LLM inference** (small, open-weight models)
3. **LLM answer**
4. **Graph-based solver**
   - reachability
   - shortest path
   - coarse direction checks
5. **Comparison & error analysis**

The symbolic solver is intentionally minimal and does *not* perform language understanding.

---

## 🧩 Symbolic Solver Scope

The solver **does**:
- compute reachability
- compute shortest paths
- check coarse relative direction (N / S / E / W)

The solver **does not**:
- parse natural language
- perform logical inference
- resolve ambiguity
- go beyond graph queries

This scope is fixed early to avoid engineering creep.

---

## 🧠 Instruction Underspecification

We define a small number of systematic underspecification strategies, such as:

- removing spatial relations,
- replacing exact directions with vague ones,
- dropping intermediate landmarks.

These are applied to a controlled subset of instructions while preserving answerability via the graph.

---

## 📊 Evaluation Strategy

- LLM answers are compared against **graph-derived ground truth**.
- Outputs are labeled as:
  - correct
  - incorrect
  - inconsistent / hallucinated

We focus on **qualitative error patterns** and robustness trends rather than leaderboard performance.

---

## 🗓️ Project Plan (High-Level)

**Week 1 — Feasibility & Presentation Readiness**
- Load RVS graph and run basic queries
- Freeze solver scope
- Define underspecification strategies
- Prepare preliminary presentation

**Week 2 — Minimal End-to-End Pipeline**
- Run LLM inference (small models)
- Compare LLM outputs to graph answers
- Document failure modes

**Week 3 — Analysis & Scaling Plan**
- Define error taxonomy
- Plan larger-scale experiments and comparisons

---

## 👥 Team

- Adan Assi  
- Shaimaa Hoji
- Noor Mhajne

(Tel Aviv University – NLP course project)

---

## 📌 Notes

- The project prioritizes **clarity and feasibility** over model scale.
- We start with sub-1B parameter models and scale only if justified.
- All experiments are designed to be reproducible and graph-grounded.

---
