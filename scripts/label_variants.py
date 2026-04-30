"""
label_variants.py

Step 4.5 of the updated pipeline; Oracle 2 labeling of masked variants.

Reads underspecified_variants.json and labels each variant using
solver.solve(..., mode="label"):
  Answerable    = masking left a unique solution
  Ambiguous     = masking created multiple valid candidates  
  Contradictory = masking removed all valid candidates

Must run AFTER underspecify_instructions.py and BEFORE build_eval_input.py.

Oracle 2 is distinct from Oracle 1 (batch_labeling.py):
  Oracle 1: mode="resolve" — salience picks single candidate, never Ambiguous
  Oracle 2: mode="label"   — counts reachable candidates, preserves Ambiguous signal
"""

import sys
import json
import os
import pickle
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver


def label_city(city: str):
    print(f"\n🏙️  Oracle 2 Labeling: {city.upper()}")
    config.CURRENT_CITY = city

    city_dir = os.path.join(config.BASE_DIR, "data", city)
    input_path  = os.path.join(city_dir, "underspecified_variants.json")
    output_path = os.path.join(city_dir, "underspecified_variants_labeled.json")

    if not os.path.exists(input_path):
        print(f"❌ {input_path} not found. Run underspecify_instructions.py first!")
        return

    with open(input_path, 'r') as f:
        experiments = json.load(f)
    print(f"📂 Loaded {len(experiments)} experiments")

    # --- Initialize Oracle 2 ---
    print(f"🔧 Initializing Oracle 2 for {city}...")
    with open(config.get_graph_path(), 'rb') as f:
        G = pickle.load(f)

    oracle = OracleEngine(G, config.get_poi_path(),
                          config.get_node_prefix(), city)
    solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())
    print(f"✅ Oracle 2 ready.")

    # --- Label each variant ---
    labeled   = 0
    skipped   = 0
    label_counts = {"Answerable": 0, "Ambiguous": 0,
                    "Contradictory": 0, "error": 0}

    for experiment in tqdm(experiments, desc=f"Oracle 2 — {city}"):
        start_node = experiment.get('start_node')

        # Resolve start node
        if not start_node or str(start_node) == 'nan':
            skipped += 1
            for v in experiment['variants']:
                v['oracle_label'] = 'error'
                v['reachable_candidate_count'] = 0
                v['candidate_nodes'] = []
            continue

        # Ensure node is in graph
        if start_node not in G.nodes:
            resolved = oracle.get_graph_node(start_node)
            if not resolved or resolved not in G.nodes:
                skipped += 1
                for v in experiment['variants']:
                    v['oracle_label'] = 'error'
                    v['reachable_candidate_count'] = 0
                    v['candidate_nodes'] = []
                continue
            start_node = resolved

        for variant in experiment['variants']:
            try:
                label_info = solver.solve(
                    variant['text'],
                    start_node,
                    mode="label" # <-- This is the key difference from batch_labeling.py
                )
                variant['oracle_label']            = label_info['state']
                variant['reachable_candidate_count'] = label_info.get(
                    'reachable_candidate_count', 0)
                variant['candidate_nodes']         = label_info.get(
                    'candidate_nodes', [])
                label_counts[label_info['state']] += 1
                labeled += 1
            except Exception as e:
                variant['oracle_label']            = 'error'
                variant['reachable_candidate_count'] = 0
                variant['candidate_nodes']         = []
                label_counts['error']             += 1

    # --- Save ---
    with open(output_path, 'w') as f:
        json.dump(experiments, f, indent=4)

    print(f"\n✅ Labeled {labeled} variants | Skipped {skipped} experiments")
    print(f"📊 Label distribution: {label_counts}")
    total = sum(label_counts.values())
    for label, count in label_counts.items():
        if count > 0:
            print(f"   {label}: {count} ({count/total:.1%})")
    print(f"💾 Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle 2 — label masked variants with Answerable/Ambiguous/Contradictory")
    parser.add_argument(
        "--city", type=str,
        help="City to process: manhattan, pittsburgh, philadelphia")
    args = parser.parse_args()

    if args.city:
        if args.city in config.CITY_SETTINGS:
            label_city(args.city)
        else:
            print(f"❌ '{args.city}' not in config.CITY_SETTINGS")
    else:
        print("⚠️  No city specified. Running all cities sequentially...")
        for city in config.CITY_SETTINGS.keys():
            label_city(city)


if __name__ == "__main__":
    main()