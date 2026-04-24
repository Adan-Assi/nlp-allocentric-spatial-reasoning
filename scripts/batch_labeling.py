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
        
    # 1. Force the config update immediately
    config.CURRENT_CITY = city_name 
    
    # 2. Get the paths explicitly
    graph_path = config.get_graph_path()
    poi_path = config.get_poi_path()

    # Define json_path here so it points to the right city folder
    json_path = os.path.join(config.BASE_DIR, "data", city_name, config.CITY_SETTINGS[city_name]["raw_json"])
    
    print(f"DEBUG: Real Path: {os.path.abspath(graph_path)}")
    print(f"DEBUG: File Size on Disk: {os.path.getsize(graph_path)} bytes")
    
    # 3. Initialize
    G = load_graph_safely(graph_path)
    node_count = len(G.nodes())

    print(f"DEBUG: Graph Loaded. Node count: {len(G.nodes())}") # Verify this isn't 31036!

    oracle = OracleEngine(G, poi_path, config.get_node_prefix(), city_name)
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
            s0_lat, s0_lon = row['rvs_start_point']
            rvs_id = row.get('rvs_start_node')

            # 1. Start Node Resolution: ID-First, Snap-Second
            # This uses your fuzzy prefix fix (1#, #, etc.)
            start_node = oracle.get_graph_node(rvs_id) 

            # CRITICAL CHECK: Even if get_graph_node returns something, 
            # we must ensure it is actually in the current loaded Graph G.
            if start_node is None or start_node not in G.nodes:
                # Fallback to physical snapping if ID mapping fails
                start_node = fast_snap(s0_lat, s0_lon)
            
            # FINAL SAFETY: If for some reason G is empty or snapping failed
            if start_node not in G.nodes:
                continue

            # 2. THE SYMBOLIC SOLVER CALL
            label_info = solver.solve(instruction, start_node)
            
            # Capture the correct ID for consistency
            current_id = row.get('rvs_sample_number', row.get('key', 'N/A'))

            # 3. Build Record with Matched Metadata Keys
            final_data.append({
                "sample_id": current_id,
                "city": city_name,
                "instruction": instruction,
                "oracle_label": label_info['state'],
                "candidate_count": label_info['candidate_count'],
                "start_node": start_node,
                "gold_goal_node": fast_snap(row['rvs_goal_point'][0], row['rvs_goal_point'][1]),
                
                # --- metadata columns (Matches solver.solve() keys) ---
                "extracted_category": label_info.get('extracted_category'),
                "extracted_noun": label_info.get('extracted_noun'),
                "extracted_direction": label_info.get('extracted_direction'),
                "target_node": label_info.get('target_node')
            })

            # 4. Debug: Periodic timing check (Every 50 iterations)
            if len(final_data) % 50 == 0:
                # USE current_id HERE so Philly shows 9126 instead of None
                print(f"DEBUG: Sample {current_id} -> Label: {label_info['state']} | Cands: {label_info['candidate_count']}", flush=True)

        except Exception as e:
            import traceback
            print(f"⚠️  Warning: Failed on sample {row.get('rvs_sample_number', 'N/A')}.")
            traceback.print_exc()
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