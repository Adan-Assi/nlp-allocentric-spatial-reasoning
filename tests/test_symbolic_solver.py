import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
from config import LANDMARK_GROUPS  # Import our new 2.5 Mapping

def run_integration_tests():
    print("=" * 70)
    print("🚀 SYMBOLIC SOLVER: CATEGORY-AWARE INTEGRATION TESTS")
    print("=" * 70)

    # 1. Setup
    graph_path = 'data/manhattan/manhattan_graph.gpickle'
    poi_path = 'data/manhattan/manhattan_poi.pkl'
    
    if not os.path.exists(graph_path):
        print(f"❌ Error: {graph_path} not found.")
        return

    oracle = OracleEngine(graph_path, poi_path)
    solver = SymbolicSolver(oracle)

    node_ids = list(solver.G.nodes)
    start_node = node_ids[0]

    # --- NEW TEST 4: KEYWORD RESOLUTION (Task 2.5 Verification) ---
    print("\n" + "="*70 + "\nTEST 4: KEYWORD RESOLUTION (2.5 MAPPING)\n" + "="*70)
    
    # We will test three levels of mapping we defined in config.py
    test_queries = [
        "CHURCH",            # Level 1: Pure Root
        "the small garden",  # Level 2: Messy string containing Root
        "POST OFFICE",       # Level 3: Multi-word specific match
        "7-ELEVEN"           # Level 4: Brand name fallback
    ]

    for query in test_queries:
        print(f"\n🔍 Testing Query: '{query}'")
        
        # This simulates what the Parser (3.1) sends to the Solver
        # The Solver should look at query, find the Root in LANDMARK_GROUPS, 
        # and then query the Oracle for those OSM tags.
        path_to_poi = solver.get_path_to_landmark(start_node, query)
        
        if path_to_poi:
            dist = solver.get_path_length(path_to_poi)
            print(f"✅ SUCCESS: '{query}' resolved to a coordinate. Path: {dist:.2f}m")
        else:
            print(f"❌ FAILURE: '{query}' could not be mapped to an OSM entity.")

    # --- NEW TEST 5: DIRECTIONAL REASONING ---
    print("\n" + "="*70 + "\nTEST 5: SPATIAL REASONING (RELATIVE BEARING)\n" + "="*70)
    # Testing if the solver can tell us where a landmark is relative to our path
    target_landmark = "BANK"
    bearing = solver.get_landmark_bearing(start_node, target_landmark)
    print(f"✅ The {target_landmark} is currently to your: {bearing}")

if __name__ == '__main__':
    run_integration_tests()