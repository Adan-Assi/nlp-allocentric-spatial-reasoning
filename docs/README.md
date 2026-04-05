# 📖 Documentation Index: Manhattan Spatial Reasoning

This directory contains the theoretical grounding, empirical justifications, and technical specifications for our study on LLM robustness under spatial underspecification.

---

## 📍 Quick Start & High-Level Overview
* **[`NORTH_STAR_GUIDE.md`](./NORTH_STAR_GUIDE.md)** 
    * **READ THIS FIRST.** The "North Star" for the project.  
    * *Purpose:* Explains the **Research Question**, the **Identity Resolution vs. Geometric Reasoning** distinction, and why we evaluate LLMs against the **Oracle Truth** rather than the original human goal.

---

## 🔬 Core Research & Theory
* **[`Literature_Review_and_Theoretical_Framework.md`](./Literature_Review_and_Theoretical_Framework.md)**
    * *Purpose:* Synthesizes **RVS**, **StepGame**, and **Li et al.** findings. 
    * *Key Logic:* Establishes the **1500m Geodesic Gatekeeper** as a cognitive filter and defines our 3-state Answerability logic (*Answerable, Ambiguous, Contradictory*).
* **[`Bibliography Files (PDFs)`](./bibliography)** *(Refer to uploaded files: 2402.16364v2.pdf, etc.)*
    * *Primary Sources:* The original papers for RVS (EACL 2024), StepGame (AAAI 2022), and the StepGame Rectification (AAAI 2024).

---

## 📈 Empirical Justification: The 1500m Threshold
* **The "Elbow" Finding:** Our sensitivity tests across the Manhattan road graph identified that **1500m** is the "Efficiency Peak."
* **Recall:** 84% (captures the vast majority of human-referenced landmarks).
* **Noise:** Filters out ~75% of geographically impossible candidates compared to a 2km radius.
* **Literature Support:** Aligns with Paz-Argaman et al.'s finding that model accuracy drops sharply when targets exceed the 2km range.

---

## 🗄️ Data & Environment Specifications
* **[`DATA_GUIDE.md`](./DATA_GUIDE.md)**: Single source of truth for the `.gpickle` graph weights and the `1#` node ID prefix.
* **[`DATA_LAYERS.md`](./DATA_LAYERS.md)**: Details the 40+ POI columns (amenity, cuisine, shop) used for semantic grounding.
* **[`DATASET_SPEC.md`](./DATASET_SPEC.md)**: Details on the RVS train/dev/test splits for Manhattan, Pittsburgh, and Philadelphia.

---

## ⚙️ Pipeline & Logic Specs
* **[`PIPELINE_FLOW.md`](./PIPELINE_FLOW.md)**: Visualizes the move from raw text → identity resolution → geometric reasoning → final labeling.
* **[`ORACLE_SPEC.md`](./ORACLE_SPEC.md)**: The mathematical protocol for labeling instructions as Ambiguous or Contradictory.
* **[`RVS_Data_and_Masking_Protocol.md`](./RVS_Data_and_Masking_Protocol.md)**: Explains our systematic perturbation strategy (masking landmarks vs. directions).

---

## 🛠️ Project Management
* **[`STRATEGY_CALIBRATION.md`](./STRATEGY_CALIBRATION.md)**: The 4-stage roadmap (Calibration → Labeling → LLM Testing → Generalization).
* **[`TASKS.md`](./TASKS.md)**: Live checklist of implemented features and remaining technical debt (e.g., SCC Optimization).

---
*Developed for the NLP Final Project: Robust Allocentric Spatial Reasoning.*