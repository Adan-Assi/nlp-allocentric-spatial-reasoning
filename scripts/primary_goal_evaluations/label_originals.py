"""
Label original (unmasked) instructions under mode='label' (three-way oracle).
Unlike Oracle 1 (mode='resolve'), which returns only Answerable/Contradictory,
mode='label' returns all three labels: Answerable, Ambiguous, Contradictory.
This produces a comparable unmasked baseline for LLM evaluation.

Input:  data/{city}/{city}_silver_standard.parquet (all three cities)
Output: reports/llm_audits/ORIGINAL_ORACLE_LABELS.parquet
"""

import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree

import config
from src.symbolic_solver import SymbolicSolver
from src.oracle_engine import OracleEngine


def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    output_path = os.path.join(
        config.BASE_DIR, "reports", "llm_audits", "ORIGINAL_ORACLE_LABELS.parquet"
    )

    city_paths = {
        'manhattan':    os.path.join(config.BASE_DIR, "data", "manhattan",    "manhattan_silver_standard.parquet"),
        'pittsburgh':   os.path.join(config.BASE_DIR, "data", "pittsburgh",   "pittsburgh_silver_standard.parquet"),
        'philadelphia': os.path.join(config.BASE_DIR, "data", "philadelphia", "philadelphia_silver_standard.parquet"),
    }

    dfs = []
    for city_name, path in city_paths.items():
        city_df = pd.read_parquet(path)
        dfs.append(city_df)
    originals = pd.concat(dfs, ignore_index=True)

    print(f"Total original instructions: {len(originals)}")
    print(f"By city:\n{originals['city'].value_counts().to_string()}")
    print(f"Oracle 1 label distribution (resolve):\n{originals['oracle_label'].value_counts().to_string()}")

    results = []

    for city_name, city_df in originals.groupby('city'):
        print(f"\nProcessing {city_name} ({len(city_df)} samples)...")

        config.CURRENT_CITY = city_name
        graph_path = config.get_graph_path()
        poi_path = config.get_poi_path()

        G = load_graph(graph_path)
        oracle = OracleEngine(G, poi_path, config.get_node_prefix(), city_name)
        solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())

        for _, row in tqdm(city_df.iterrows(), total=len(city_df)):
            try:
                label_info = solver.solve(
                    row['instruction'],
                    row['start_node'],
                    mode="label"
                )
                results.append({
                    'sample_id':            row['sample_id'],
                    'city':                 city_name,
                    'original_text':        row['instruction'],
                    'start_node':           row['start_node'],
                    'gold_goal_node':       row['gold_goal_node'],
                    'gold_goal_lat':        row['gold_goal_lat'],
                    'gold_goal_lon':        row['gold_goal_lon'],
                    'extracted_category':   row['extracted_category'],
                    'extracted_direction':  row['extracted_direction'],
                    'extracted_noun':       row['extracted_noun'],
                    'oracle_label_resolve': row['oracle_label'],        # Oracle 1 label
                    'oracle_label':         label_info['state'],        # Oracle 2 three-way label
                    'candidate_count':      label_info['candidate_count'],
                    'variant_type':         'original',
                })
            except Exception as e:
                print(f"Failed on sample_id={row['sample_id']}: {e}")
                continue

        # Checkpoint after each city in case of preemption
        pd.DataFrame(results).to_parquet(output_path + ".checkpoint")
        print(f"Checkpoint saved after {city_name}. Total so far: {len(results)}")

    out_df = pd.DataFrame(results)
    out_df.to_parquet(output_path)
    print(f"\nSaved {len(out_df)} rows to {output_path}")
    print(f"\nOracle 2 label distribution (label mode):")
    print(out_df['oracle_label'].value_counts().to_string())
    print(f"\nOracle 1 vs Oracle 2 cross-tab:")
    print(pd.crosstab(out_df['oracle_label_resolve'], out_df['oracle_label']).to_string())


if __name__ == "__main__":
    main()