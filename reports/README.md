# 📊 Research Reports & Technical Audits

This directory contains the analytical foundation of the RVS Master Gold Dataset. Each report documents a critical milestone in our transition from raw OSM data to a validated, research-grade navigation corpus.

## 🚀 Navigation Guide

| Report File | Research Domain | Key Insight / Purpose |
| :--- | :--- | :--- |
| **🧪 Dataset Validation** | | |
| [`dataset_generation.md`](./dataset_generation.md) | **Provenance & Hydration** | The "Biography" of the Gold Dataset. Documents the transition from Silver to Gold, including geodetic hydration logic and the final 1,117m baseline proof. |
| [`bibli_reliance.md`](./bibli_reliance.md) | **Literature Review** | Maps our technical decisions (1.5km horizons, symbolic extraction) back to the original *Paz-Argaman et al. (2020)* methodology. |
| **🔍 Data & Taxonomy Audits** | | |
| [`landmark_taxonomy_analysis.md`](./landmark_taxonomy_analysis.md) | **Semantic Modeling** | Categorizes landmarks into functional groups. Validates the `LANDMARK_GROUPS` configuration used by the Symbolic Solver. |
| [`data_audits/all_discovered_landmarks.csv`](data_audits/all_discovered_landmarks.csv) | **Statistical Distribution** | A raw frequency audit of 170+ unique categories and entities across Manhattan. Essential for understanding "Search Noise" and target density. |

---

## 💡 Why These Reports Matter

In spatial reasoning research, the "How" is just as important as the "What." These reports provide the transparency needed to trust the **92.53% Accuracy** metric:

1.  **Transparency:** The `landmark_frequency_report` proves we aren't "cherry-picking" easy landmarks.
2.  **Reproducibility:** The `dataset_generation` report provides the exact Haversine formulas and hydration steps needed to replicate our Master Gold file.
3.  **Linguistic Rigor:** The taxonomy analysis ensures our NLP parser (V3) is grounded in actual OSM tag distributions.
