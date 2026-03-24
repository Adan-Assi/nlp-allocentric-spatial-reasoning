import os
import sys
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree

# Add the project root to path so we can import from /src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.extraction_utils import extract_rvs_target
from src.symbolic_solver import SymbolicSolver
from src.oracle_engine import OracleEngine

def load_graph_safely(path):
    """Handles standard pickle vs legacy gpickle."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return nx.read_gpickle(path)

def process_city(city_name, cfg):
    print(f"\n🌍 --- Launching Oracle for {city_name.upper()} ---")
    
    data_dir = os.path.join(config.BASE_DIR, "data", city_name)
    json_path = os.path.join(data_dir, cfg["raw_json"])
    graph_path = os.path.join(data_dir, cfg["graph_file"])
    poi_path = os.path.join(data_dir, cfg["poi_file"])

    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found. Check filename in config.")
        return

    # 1. Initialize Engines
    G = load_graph_safely(graph_path)
    oracle = OracleEngine(G, poi_path)
    # Pass city-specific success_radius to solver
    solver = SymbolicSolver(oracle, search_radius=cfg["success_radius"])

    # 2. Speed Optimization: KD-Tree
    node_ids = list(oracle.G.nodes())
    coords = np.array([[oracle.G.nodes[n]['y'], oracle.G.nodes[n]['x']] for n in node_ids])
    spatial_index = KDTree(coords)

    def fast_snap(lat, lon):
        _, idx = spatial_index.query([lat, lon])
        return node_ids[idx]

    # 3. Load Data
    df = pd.read_json(json_path) 
    final_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Labeling {city_name}"):
        try:
            target_name = extract_rvs_target(row['content'])
            
            start_node = fast_snap(row['rvs_start_point'][0], row['rvs_start_point'][1])
            goal_node = fast_snap(row['rvs_goal_point'][0], row['rvs_goal_point'][1])
            
            is_reachable = solver.check_reachability(start_node, goal_node)
            
            label = config.STATE_CONTRADICTORY
            if is_reachable:
                label = config.STATE_ANSWERABLE if target_name != "unknown" else config.STATE_AMBIGUOUS

            final_data.append({
                "key": row.get('key', row['rvs_sample_number']),
                "instruction": row['content'],
                "extracted_landmark": target_name,
                "silver_label": label,
                "city": city_name
            })
        except Exception:
            continue

    # 4. Export
    if final_data:
        out_df = pd.DataFrame(final_data)
        out_path = os.path.join(data_dir, f"{city_name}_silver_standard.parquet")
        out_df.to_parquet(out_path)
        print(f"✅ Saved to {out_path}\n{out_df['silver_label'].value_counts()}")

def main():
    for city, cfg in config.CITY_SETTINGS.items():
        process_city(city, cfg)

if __name__ == "__main__":
    main()