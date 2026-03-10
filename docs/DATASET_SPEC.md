# 📊 RVS Dataset Specification

This document details the Remote Visual Speaker (RVS) instruction dataset used to calibrate and evaluate the Symbolic Oracle.

## 📁 File Structure & Splits
The dataset consists of four JSON files representing different experimental phases. We primarily use the **Manhattan Train** set for landmark frequency analysis and logic calibration.

| File | Context | Purpose |
| :--- | :--- | :--- |
| `train.json` | Manhattan (Seen) | **Calibration Lab**: Used for Task 2.5 (Landmark Frequency) and tuning spatial constants. |
| `dev.json` | Manhattan | **Verification**: Ensures logic generalizes to new instructions in the same city. |
| `unseen-city-dev.json` | Pittsburgh | **Generalization Test**: Proves the spatial reasoning is city-agnostic. |
| `test.json` | Philadelphia | **Final Evaluation**: The "hidden" set used for final performance reporting. |

## 🧩 Data Schema
Each entry in the JSON files contains the following key-value pairs:

* **`content`**: The raw natural language instruction (e.g., *"Walk past the deli and turn left"*).
* **`rvs_start_point`**: The [Latitude, Longitude] coordinates of the starting location.
* **`rvs_goal_point`**: The [Latitude, Longitude] coordinates of the intended destination.

## 🛠️ Usage in Phase 2
1.  **Landmark Discovery (Task 2.5)**: We parse the `content` field using `spaCy` to identify high-frequency nouns. These are then cross-referenced with `config.py` to ensure our "Brain" has the necessary vocabulary.
2.  **Oracle Validation**: The Oracle's success is defined by its ability to take the `rvs_start_point`, process the `content`, and arrive within 80 meters of the `rvs_goal_point`.

---