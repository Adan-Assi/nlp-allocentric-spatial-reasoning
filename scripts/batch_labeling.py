import os
import sys
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
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # Fallback for older NetworkX versions (if needed)
        return nx.read_gpickle(path)

def process_city(city_name):
    print(f"\n🏙️  --- Processing City: {city_name.upper()} ---")
    
    # 1. Update Global Config Toggle
    config.CURRENT_CITY = city_name
    
    # 2. Dynamic Path Fetching
    graph_path = config.get_graph_path()
    poi_path = config.get_poi_path()    
    json_path = os.path.join(config.BASE_DIR, "data", city_name, config.CITY_SETTINGS[city_name]["raw_json"])

    if not os.path.exists(json_path):
        print(f"❌ Skipping {city_name}: JSON data not found at {json_path}")
        return

    # 3. Initialize Engines
    G = load_graph_safely(graph_path)
    oracle = OracleEngine(G, poi_path)
    # The solver now gets the city-specific radius (80 or 100) automatically
    solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())

    # 4. Spatial Index for Fast Node Snapping
    node_ids = list(G.nodes())
    coords = np.array([[G.nodes[n]['y'], G.nodes[n]['x']] for n in node_ids])
    spatial_index = KDTree(coords)

    def fast_snap(lat, lon):
        _, idx = spatial_index.query([lat, lon])
        return node_ids[idx]

    # 5. Load Dataset
    try:
        # Try standard JSON first
        df = pd.read_json(json_path)
    except ValueError as e:
        if "Trailing data" in str(e):
            # If it fails with trailing data, it's likely a JSONL file
            print(f"ℹ️ Detected JSONL format for {city_name}. Switching to lines=True...")
            df = pd.read_json(json_path, lines=True)
        else:
            print(f"❌ Failed to load JSON for {city_name}. Error: {e}")
            raise e
    final_data = []

    # 6. Core Labeling Loop
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Labeling {city_name}"):
        try:
            instruction = row['content']
            # Coordinates from the RVS JSON
            s0_lat, s0_lon = row['rvs_start_point']
            
            # Snap GPS to Graph Nodes
            start_node = fast_snap(s0_lat, s0_lon)
            
            # --- THE SYMBOLIC SOLVER CALL ---
            # This handles the extraction, resolution, and reachability in one go
            # label_info contains: {'state', 'candidates', 'count'}
            label_info = solver.solve(instruction, start_node)
            
            final_data.append({
                "sample_id": row.get('rvs_sample_number', 'N/A'),
                "city": city_name,
                "instruction": instruction,
                "oracle_label": label_info['state'],
                "candidate_count": label_info['candidate_count'],
                "start_node": start_node,
                "gold_goal_node": fast_snap(row['rvs_goal_point'][0], row['rvs_goal_point'][1]),
                
                # --- metadata columns ---
                "extracted_category": label_info.get('category'),
                "extracted_noun": label_info.get('raw_noun'),
                "target_tags": label_info.get('target_tags')
            })
        except Exception as e:
            # Silent fail for individual malformed rows to keep batch running
            print(f"⚠️  Warning: Failed to process sample {row.get('rvs_sample_number', 'N/A')} in {city_name}. Error: {e}")
            continue

    # 7. Export to Parquet (Superior for large multi-city datasets)
    if final_data:
        out_df = pd.DataFrame(final_data)
        out_dir = os.path.join(config.BASE_DIR, "data", city_name)
        out_path = os.path.join(out_dir, f"{city_name}_silver_standard.parquet")
        
        out_df.to_parquet(out_path)
        print(f"✅ Success! Generated {len(out_df)} labels for {city_name}.")
        print(f"📊 Label Distribution:\n{out_df['oracle_label'].value_counts()}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RVS Silver Standard Labeling")
    parser.add_argument("--city", type=str, help="Specific city to process (manhattan, pittsburgh, philadelphia)")
    args = parser.parse_args()

    if args.city:
        if args.city in config.CITY_SETTINGS:
            process_city(args.city)
        else:
            print(f"❌ Error: City '{args.city}' not found in config.CITY_SETTINGS")
    else:
        # Default behavior: Process all cities sequentially if no flag is provided
        print("⚠️ No city specified. Running all cities sequentially...")
        for city in config.CITY_SETTINGS.keys():
            process_city(city)

if __name__ == "__main__":
    main()