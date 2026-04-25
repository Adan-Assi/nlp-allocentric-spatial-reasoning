"""
qc_ambiguous.py
Quality-control inspection of Ambiguous rows in the silver standard.

Fixes vs. the previous version:
  * `config.RVS_DATA_JSON` was removed in the multi-city config refactor;
    this script crashed on import. Replaced with `config.BASE_DIR` + the
    standard `data/<city>/<city>_silver_standard.parquet` layout.
  * The column was being read as `silver_label`; `batch_labeling.py`
    actually writes the column under `oracle_label`. Wrong-column read
    silently returned an empty DataFrame.
  * The label string was lowercase `'ambiguous'`; the oracle writes the
    canonical capitalized form `'Ambiguous'`. Fixed.
  * Hard-coded Manhattan; now accepts `--city` and `--n` CLI args.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def run_qc(city: str, n: int):
    parquet_path = os.path.join(
        config.BASE_DIR, "data", city, f"{city}_silver_standard.parquet"
    )

    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run batch_labeling.py --city {city} first.")
        return

    df = pd.read_parquet(parquet_path)
    ambiguous_df = df[df['oracle_label'] == 'Ambiguous']

    print(f"🔍 {city}: {len(ambiguous_df):,} Ambiguous samples (showing up to {n})\n")

    for _, row in ambiguous_df.head(n).iterrows():
        print(f"ID:    {row['sample_id']}")
        print(f"NOUN:  {row.get('extracted_noun')!r}  "
              f"DIR: {row.get('extracted_direction')}  "
              f"CANDS: {row.get('candidate_count')}")
        print(f"TEXT:  {row['instruction']}")
        print("-" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="manhattan",
                    help="manhattan / pittsburgh / philadelphia")
    ap.add_argument("--n", type=int, default=20,
                    help="How many ambiguous samples to print")
    args = ap.parse_args()
    run_qc(args.city, args.n)