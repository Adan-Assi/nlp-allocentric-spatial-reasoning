import os
import sys
import pandas as pd

# --- BOOTSTRAP PATHS ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import config
from oracle_engine import OracleEngine
from symbolic_solver import SymbolicSolver


# Columns we depend on — checked at load time so failures are informative
REQUIRED_COLUMNS = {'oracle_label', 'extracted_noun', 'start_node', 'instruction', 'sample_id'}


def _normalize_start_node(val, prefix, graph_nodes):
    # 1. Get raw digits (e.g., "1#123" -> "123")
    raw_str = str(val).split('.')[0]
    digits = "".join(filter(str.isdigit, raw_str))
    
    if not digits:
        return str(val)

    # 2. List all possible "identities" this node might have in the graph
    candidates = [
        digits,              # String "7977067481"
        f"{prefix}{digits}", # String "1#7977067481"
        int(digits),         # Integer 7977067481
    ]
    
    # 3. Use the first one that actually exists in the graph
    for cand in candidates:
        if cand in graph_nodes:
            return cand
            
    # Fallback
    return f"{prefix}{digits}"


def audit_city(city_name: str) -> None:
    # --- 1. Force config to the target city ---
    config.CURRENT_CITY = city_name

    city_dir    = os.path.join(project_root, "data", city_name)
    graph_path  = config.get_graph_path()
    poi_path    = config.get_poi_path()
    prefix      = config.get_node_prefix()           # e.g. "1#" — city-aware
    parquet_path = os.path.join(city_dir, f"{city_name}_silver_standard.parquet")

    # --- 2. Initialise oracle + solver ---
    oracle = OracleEngine(graph_path, poi_path, prefix, city_name)
    solver = SymbolicSolver(oracle)

    # --- 3. Resolve solve method dynamically (don't hardcode) ---
    solve_method = next(
        (getattr(solver, m) for m in ('solve', 'resolve_all_candidates', 'find_candidates')
         if hasattr(solver, m)),
        None,
    )
    if solve_method is None:
        raise AttributeError(
            f"SymbolicSolver has none of the expected solve methods. "
            f"Available: {[m for m in dir(solver) if not m.startswith('_')]}"
        )

    # --- 4. Load parquet exactly once ---
    df = pd.read_parquet(parquet_path)

    # --- 5. Validate required columns up front ---
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{city_name}] Parquet is missing columns: {missing}")

    # --- 6. Build failure sample ---
    fails = df[df['oracle_label'] == 'Contradictory']

    # Exclude rows whose extracted noun is just a cardinal direction
    direction_mask = (
        fails['extracted_noun']
        .fillna('')           # treat NaN as empty string, not a crash
        .str.lower()
        .str.fullmatch(r'north|south|east|west')   # fullmatch = whole-cell, not substring
    )
    valid_fails = fails[~direction_mask]

    # Safe sample — never request more rows than exist
    population = valid_fails if len(valid_fails) > 0 else fails
    n_samples  = min(10, len(population))

    if n_samples == 0:
        print(f"\n[{city_name.upper()}] No contradictory samples found — nothing to audit.")
        return

    samples = population.sample(n_samples, random_state=None)

    # --- 7. Run audit ---
    print(f"\n{'='*60}\n🔍 AUDIT: {city_name.upper()}\n{'='*60}")

    for _, row in samples.iterrows():
        noun = row['extracted_noun']
        instr = row['instruction']  # Define here so 'except' can see it
        sid = row['sample_id']      # Use sample_id to match our columns
        
        print(f"\nID: {sid} | Noun: '{noun}'")

        try:
            # 1. Flexible ID lookup
            start_node_id = _normalize_start_node(row['start_node'], prefix, oracle.G.nodes)
            
            # 2. Check if it actually exists in the graph
            if start_node_id not in oracle.G.nodes:
                raise KeyError(f"{start_node_id}")

            # 3. Solve
            result = solve_method(instr, start_node_id)

            print(f"🕵️  Audit New State:       {result['state']}")
            print(f"🕵️  Audit Candidates Found: {result.get('candidate_count', 'N/A')}")

            if result['state'] == 'Answerable':
                print("✨ SUCCESS: This sample is now RESCUED!")

        except KeyError as e:
            # This handles "off-map" nodes or mismatched IDs
            print(f"❌ KEY ERROR: Node {e} not found in graph.")
            print(f"📝 INSTRUCTION: \"{instr}\"")
            print(f"🕵️  Audit New State: Contradictory (Invalid Starting Location)")
            
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    for city in ['philadelphia', 'pittsburgh']:
        try:
            audit_city(city)
        except Exception as e:
            print(f"CRITICAL FAILURE FOR {city.upper()}: {e}")