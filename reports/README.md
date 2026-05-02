# 📊 Research Reports & Technical Audits

This directory contains the analytical foundation of the RVS Master Gold Dataset. Each report documents a critical milestone in our transition from raw OSM data to a validated, research-grade navigation corpus.

## 🚀 Navigation Guide

| Report File | Research Domain | Key Insight / Purpose |
| :--- | :--- | :--- |
| **🧪 Evaluation & Degradation** | | |
| [`methodology_phase_5.md`](./methodology_phase_5.md) | **Experimental Design** | Documents the "Underspecification Pipeline," including the logic for the 22,173 variants and the HPC Slurm inference workflow. |
| [`llm_resolution_limit_report.md`](./llm_resolution_limit_report.md) | **Phase 5 Results** | Analysis of the "Worrisome Gap". Proves that LLMs possess "Near-Sighted Intelligence": topologically accurate but semantically fragile. |
| **🧐 Dataset Validation** | | |
| [`dataset_generation.md`](./dataset_generation.md) | **Provenance & Hydration** | The "Biography" of the Gold Dataset. Documents the transition from Silver to Gold, including geodetic hydration logic and the final 1,117m baseline proof. |
| [`bibli_reliance.md`](./bibli_reliance.md) | **Literature Review** | Maps our technical decisions (1.5km horizons, symbolic extraction) back to the original *Paz-Argaman et al. (2020)* methodology. |
| **🔍 Data & Taxonomy Audits** | | |
| [`landmark_taxonomy_analysis.md`](./landmark_taxonomy_analysis.md) | **Semantic Modeling** | Categorizes landmarks into functional groups. Validates the `LANDMARK_GROUPS` configuration used by the Symbolic Solver. |
| [`data_audits/all_discovered_landmarks.csv`](data_audits/all_discovered_landmarks.csv) | **Statistical Distribution** | A raw frequency audit of 170+ unique categories and entities across Manhattan. Essential for understanding "Search Noise" and target density. |

---

## 💡 Why These Reports Matter

In spatial reasoning research, the "How" is just as important as the "What." These reports provide the transparency needed to trust our metrics:

1. **Methodological Rigor:** `methodology_phase_5` explains how we isolated reasoning from memorization using controlled information decay.
2. **Scientific Validation:** `llm_resolution_limit_report` benchmarks our results against official RVS metrics, ensuring our 1.5% "Strict Match" is a verified replication, not a bug.
3. **Reproducibility:** All reports provide the formulas (Haversine) and logical constraints (1500m horizons) required to replicate the Master Gold standard.