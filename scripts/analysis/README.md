# Data Analysis & Reporting

This directory contains scripts used to inspect the RVS dataset and report on the quality of the grounding process. These scripts are "read-only"; they do not modify the data, but provide the statistics needed for the Methodology section of our research paper.

## Files

* **`inspect_data.py`**: A discovery script to view the raw HuggingFace dataset structure, column names, and sample entries. Run this first to understand the source data.
* **`grounding_report_all_regions.py`**: Generates a statistical summary of the distance between raw coordinates and the nearest graph nodes across Manhattan, Philadelphia, and Pittsburgh.

## Usage

Run these scripts from the **project root** to ensure paths resolve correctly:

```bash
python scripts/analysis/inspect_data.py
python scripts/analysis/grounding_report_all_regions.py
```

## Key Metrics to Watch

* **Mean Distance (m)**: Represents the average distance between raw dataset coordinates and the snapped graph nodes. We aim for **< 75m** for a "successful" grounding to ensure spatial accuracy.
* **Region Distribution**: A breakdown of the sample count per city. This ensures we maintain a balanced dataset across **Manhattan, Philadelphia, and Pittsburgh** for unbiased evaluation.