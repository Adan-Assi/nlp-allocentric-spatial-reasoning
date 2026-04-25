import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# 1. DYNAMIC PATH INJECTION
# This ensures 'src' and 'config' are discoverable regardless of entry point
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. IMPORT FROM SRC NAMESPACE
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
import config  # Centralized source of truth

def run_stress_test():
    """
    ACL Methodology: Validates 500 underspecified variants using 
    Geometric Reasoning and Identity Grounding.
    """
    # 3. Initialize engines
    oe = OracleEngine(config.GRAPH_PATH, config.POI_PATH)
    ss = SymbolicSolver(oe)

    # 4. Load data using config paths
    with open(config.VARIANTS_JSON, 'r') as f:
        variants = json.load(f)

    raw_lookup = {}
    print(f"📖 Reading RVS data line-by-line from {config.RVS_DATA_JSON}...")
    with open(config.RVS_DATA_JSON, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            
            # The notebooks show 'sample_id' is the standard for your variants
            # but the RVS data might use 'sample_number' or 'id'
            key = str(item.get('sample_number', item.get('sample_id', item.get('id', i))))
            raw_lookup[key] = item

    results = []

    for entry in tqdm(variants, desc="⚖️ Stress Testing Oracle"):
        sample_id = str(entry['sample_id'])
        original_meta = raw_lookup[sample_id]
        
        # Step A: Identify the "Start Node" for geometry
        # Based on our Manhattan_Semantic_Navigator notebook results, the key is 'rvs_start_point'
        start_coords = original_meta.get('rvs_start_point', original_meta.get('start_point'))
        
        if start_coords is None:
            print(f"⚠️ Warning: No start coordinates found for sample {sample_id}. Skipping.")
            continue

        start_lat, start_lon = start_coords
        
        # Ground the agent in the graph
        # Our Navigator notebook shows nearest node lookup is essential for Dijkstra logic
        start_node, _ = oe.find_nearest_node(start_lat, start_lon)


        for variant in entry['variants']:
            # Step B: Identity Resolution (Oracle)
            # Find which landmark group we are masking (e.g., 'CHURCH')
            removed_name = variant.get('removed_element', '').upper()
            tags = config.LANDMARK_GROUPS.get(removed_name, {})
            
            # Using the 'resolve_all_candidates' logic
            candidates = oe.resolve_all_candidates(tags)

            # Step C: Geometric Filtering (Solver)
            valid_candidates = []
            for c in candidates:
                # A: QUICK HEURISTIC (Straight-line distance)
                # Calculate distance in meters using your existing utils or math
                dist = oe.calculate_distance(start_lat, start_lon, c['coords'][0], c['coords'][1])
                
                # Only run expensive graph math if the POI is within 1.5km
                # (1500m is roughly the max range for allocentric instructions)
                if dist < 1500:
                    # B: EXPENSIVE GRAPH CHECK
                    if ss.check_reachability(start_node, c['node_id']):
                        valid_candidates.append(c)

            # Step D: Final Labeling
            count = len(valid_candidates)
            label = config.STATE_ANSWERABLE if count == 1 else config.STATE_AMBIGUOUS
            if count == 0: label = "Invalid"

            results.append({
                "sample_id": sample_id,
                "variant_type": variant['type'],
                "final_label": label,
                "candidate_count": count
            })

    # 5. SAVE USING CONFIG PATH
    # This resolves the "output_path is not defined" error
    df_results = pd.DataFrame(results)
    df_results.to_csv(config.AMBIGUITY_REPORT_CSV, index=False)
    
    print(f"✅ Scientific Stress Test Complete.")
    print(f"📊 Report saved to: {config.AMBIGUITY_REPORT_CSV}")
    print("\nSummary of Results:")
    print(df_results['final_label'].value_counts(normalize=True))


if __name__ == "__main__":
    # No arguments needed anymore because the function looks at config.py
    run_stress_test()