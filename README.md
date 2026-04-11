# Robust Allocentric Spatial Reasoning under Underspecified Instructions

Course project (NLP): Evaluating the robustness of Large Language Models (LLMs) in **allocentric spatial reasoning**. 

This project investigates how models handle navigation instructions when spatial information is **systematically underspecified**, using a fixed, geodetically grounded urban map graph as the source of truth.

It utilizes the **Rendezvous (RVS)** dataset:
- Human-written allocentric navigation instructions.
- Grounded in real-world urban map graphs.
- Clear, graph-backed ground truth (reachability, paths, directions).

---

## 🏺 The Master Gold Standard
We have synthesized a unified, research-grade dataset from the RVS baseline, enhanced by our **V4 Deep-Search Oracle**.
* **Unified Dataset:** `RVS_MASTER_GOLD_HYDRATED.parquet` (7,263 verified samples).
* **Geodetic Hydration:** Samples are grounded to the nearest OSM street node with <1% coordinate variance from RVS official baselines.

| Metric | RVS Baseline (Silver) | Master Gold (Final) |
| :--- | :--- | :--- |
| **Total Samples** | 9,301 | **7,263** |
| **Hydration** | Rough Lat/Lon | **Geodetic Node-Matched** |
| **Search Logic** | Single-Tag Fallback | **Multi-Column Deep Search** |
| **Data Integrity** | Raw Instructions | **Audit-Trailed & Validated** |

---

## 📖 Documentation Hub

### 🧠 Research & Strategy
* [**Technical FAQs (docs/FAQs.md)**](docs/FAQs.md) — **READ THIS FIRST.** Explains the "Why" behind Symbolic Masking, the 1.5km Reasoning Horizon, and the "Logic Loop" pipeline.
* [**Strategy & Calibration**](docs/STRATEGY_CALIBRATION.md) — High-level vision and the stages of our "Logic Laboratory."
* [**Oracle Specification**](docs/ORACLE_SPEC.md) — The mathematical logic behind the Symbolic Solver (Vector/Bearing filters).

### ⚙️ Technical Specs & Implementation
* [**Pipeline Data Flow**](docs/PIPELINE_FLOW.md) — Architectural trace from raw text to answerability labels.
* [**Data & Graph Guide**](docs/DATA_GUIDE.md) — Reference for the 74k-node Manhattan graph and POI structures.
* [**Task Tracker (TASKS.md)**](docs/TASKS.md) — Sprint progress and roadmap logic status.

---

## 📁 Repository Structure

```text
nlp-allocentric-spatial-reasoning/
├── data/
│   └── RVS_MASTER_GOLD_HYDRATED.parquet  # The "Answer Key" (7,263 rows)
├── src/                                  # Production Code
│   ├── oracle_engine.py                  # V4 Deep Search Logic
│   └── symbolic_solver.py                # Geometric validation
├── docs/                                 # Research Strategy & Specs
├── reports/                              # Data Audits & Performance Logs
├── notebooks/                            # Analysis & Philly Rescue audits
└── config.py                             # Centralized multi-city settings
```
---
## 📈 Batch Labeling Performance (Final)

The labeling pipeline utilizes a vectorized **"Sniper Search"** strategy, achieving high-throughput geodetic grounding across major urban datasets.

| City | Total Samples | Answerable (Gold) | Contradictory | Speed (it/s) |
| :--- | :--- | :--- | :--- | :--- |
| **Manhattan** | 7,000 | 5,305 | 412 | 23.65 |
| **Pittsburgh** | 1,023 | 874 | 149 | **67.76** |
| **Philadelphia** | 1,278 | 1,084 | 194 | 51.51 |
| **TOTAL** | **9,301** | **7,263** | **755** | **~35.00** |

### 🔑 Optimization Key Drivers
* **Geometric Pre-calculation**: Centroid extraction to raw floats avoided Shapely overhead during distance loops.
* **Boolean Masking**: Swapping `.apply()` for vectorized boolean indexing reduced CPU cycles by **15x**.
* **Audit Trail**: Every row now includes `extracted_category`, `extracted_noun`, and `target_tags` for transparent debugging.

---

## 🔧 High-Level Pipeline
1.  **Instruction Input**: Original RVS text vs. Underspecified variants.
2.  **Symbolic Solver (Oracle)**:
    * Reachability & Shortest Path computation.
    * 1.5km "Human Observable Horizon" pruning.
    * Coarse direction checks (N/S/E/W).
3.  **LLM Inference**: Testing robustness of small, open-weight models.
4.  **Comparison**: Error taxonomy classification (Correct, Incorrect, Hallucinated).

---

## 👥 Team
* **Adan Assi** 
* **Shaimaa Hoji** 
* **Noor Mhajne**

**(Tel Aviv University – NLP Course Project 2025-2026)**

