import os
import sys
import time
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.symbolic_solver import SymbolicSolver
from src.oracle_engine import OracleEngine


def load_graph_safely(path):
    """Handles standard pickle vs legacy gpickle."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return nx.read_gpickle(path)


def process_city(city_name):
    print(f"\n🏙️  --- Processing City: {city_name.upper()} ---")

    config.CURRENT_CITY = city_name

    graph_path = config.get_graph_path()
    poi_path = config.get_poi_path()
    json_path = os.path.join(
        config.BASE_DIR,
        "data",
        city_name,
        config.CITY_SETTINGS[city_name]["raw_json"],
    )

    print(f"DEBUG: Real Path: {os.path.abspath(graph_path)}")
    print(f"DEBUG: File Size on Disk: {os.path.getsize(graph_path)} bytes")

    G = load_graph_safely(graph_path)
    print(f"DEBUG: Graph Loaded. Node count: {len(G.nodes())}")

    oracle = OracleEngine(G, poi_path, config.get_node_prefix(), city_name)
    solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())

    # Spatial index for fast coordinate → graph node snapping
    node_ids = list(G.nodes())
    coords = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids])
    spatial_index = KDTree(coords)

    def fast_snap(lat, lon):
        _, idx = spatial_index.query([lat, lon])
        return node_ids[idx]

    # Load JSON / JSONL
    try:
        df = pd.read_json(json_path)
    except ValueError as e:
        if "Trailing data" in str(e):
            print(f"ℹ️ Detected JSONL format for {city_name}. Switching to lines=True...")
            df = pd.read_json(json_path, lines=True)
        else:
            print(f"❌ Failed to load JSON for {city_name}. Error: {e}")
            raise e

    final_data = []

    label_counter = {"Answerable": 0, "Ambiguous": 0, "Contradictory": 0}

    for row_idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"Labeling {city_name}")
    ):
        try:
            instruction = row["content"]

            # ✅ FIX:
            # Current HF RVS data has rvs_start_point but rvs_start_node is null.
            # So we ground the start location by snapping GPS coordinates.
            s0_lat, s0_lon = row["rvs_start_point"]
            start_node = fast_snap(s0_lat, s0_lon)

            if start_node not in G.nodes:
                tqdm.write(f"[{row_idx}/{len(df)}] ⚠️  start_node {start_node} not in graph — skip")
                continue

            t0 = time.time()
            label_info = solver.solve(instruction, start_node)
            elapsed = time.time() - t0

            current_id = row.get("rvs_sample_number", row.get("key", "N/A"))

            goal_lat, goal_lon = row["rvs_goal_point"]
            gold_goal_node = fast_snap(goal_lat, goal_lon)

            final_data.append({
                "sample_id": current_id,
                "city": city_name,
                "instruction": instruction,
                "oracle_label": label_info["state"],
                "candidate_count": label_info.get("candidate_count"),
                "reachable_candidate_count": label_info.get("reachable_candidate_count"),
                "start_node": start_node,
                "gold_goal_node": gold_goal_node,

                "extracted_category": label_info.get("extracted_category"),
                "extracted_noun": label_info.get("extracted_noun"),
                "extracted_direction": label_info.get("extracted_direction"),
                "target_node": label_info.get("target_node"),
                "resolution_stage": label_info.get("resolution_stage"),
            })

            state = label_info["state"]
            label_counter[state] = label_counter.get(state, 0) + 1

            tqdm.write(
                f"[{row_idx:>5}/{len(df)}] id={current_id} "
                f"| {state:<13} "
                f"| cands={label_info.get('candidate_count'):>4} "
                f"reach={label_info.get('reachable_candidate_count'):>4} "
                f"| cat={label_info.get('extracted_category'):<10} "
                f"noun={str(label_info.get('extracted_noun'))[:25]:<25} "
                f"dir={str(label_info.get('extracted_direction')):<4} "
                f"| stage={label_info.get('resolution_stage'):<35} "
                f"| {elapsed:>5.2f}s "
                f"| A={label_counter['Answerable']} "
                f"Am={label_counter['Ambiguous']} "
                f"C={label_counter['Contradictory']}"
            )

        except Exception:
            import traceback
            tqdm.write(f"⚠️  Warning: Failed on sample {row.get('rvs_sample_number', 'N/A')}.")
            traceback.print_exc()
            continue

    if final_data:
        out_df = pd.DataFrame(final_data)
        out_dir = os.path.join(config.BASE_DIR, "data", city_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{city_name}_silver_standard.parquet")
        out_df.to_parquet(out_path)

        print(f"✅ Success! Generated {len(out_df)} labels for {city_name}.")
        print(f"💾 Saved to: {out_path}")
        print(f"📊 Label Distribution:\n{out_df['oracle_label'].value_counts()}\n")
    else:
        print(f"⚠️ No labels generated for {city_name}.")


def main():
    parser = argparse.ArgumentParser(description="Run RVS Silver Standard Labeling")
    parser.add_argument(
        "--city",
        type=str,
        help="Specific city to process: manhattan, pittsburgh, philadelphia",
    )
    args = parser.parse_args()

    if args.city:
        if args.city in config.CITY_SETTINGS:
            process_city(args.city)
        else:
            print(f"❌ Error: City '{args.city}' not found in config.CITY_SETTINGS")
    else:
        print("⚠️ No city specified. Running all cities sequentially...")
        for city in config.CITY_SETTINGS.keys():
            process_city(city)


if __name__ == "__main__":
    main()