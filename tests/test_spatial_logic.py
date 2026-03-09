import sys
import os
from pathlib import Path

# 1. Force the ROOT_DIR to be the actual project root
# We look for 'config.py' to identify the root
current_path = Path(__file__).resolve()
ROOT_DIR = next(p for p in current_path.parents if (p / 'config.py').exists())

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
import src.utils as utils

def test_clamped_radius_and_search_limit():
    print("🚀 Starting Spatial Logic Sanity Check...")

    # 2. Use ROOT_DIR to build the data path
    graph_path = ROOT_DIR / "data" / "manhattan" / "manhattan_graph.gpickle"
    poi_path = ROOT_DIR / "data" / "manhattan" / "manhattan_poi.pkl"

    print(f"DEBUG: Root identified as: {ROOT_DIR}")
    print(f"DEBUG: Looking for graph at: {graph_path}")

    if not graph_path.exists():
        print(f"❌ ERROR: Graph file not found!")
        return

    # 3. Initialize
    oracle = OracleEngine(str(graph_path), str(poi_path))
    solver = SymbolicSolver(oracle)
    
    # 4. Test Step A: Search Limit (Human Error Buffer)
    dist = 500
    limit = solver.get_search_limit(dist)
    expected_limit = max(500 * 1.1, 500 + 80) # 580
    print(f"Checking Step A: Dist {dist}m -> Limit {limit}m (Expected: {expected_limit})")
    assert limit == expected_limit, "❌ Step A Failed: Search limit calculation is wrong."

    # 5. Test Step C: Clamped Radius for different sizes
    # Small landmark (approx area of a storefront)
    small_area = 50 
    r_small = utils.get_clamped_radius(small_area)
    
    # Large landmark (approx area of a city park)
    large_area = 50000 
    r_large = utils.get_clamped_radius(large_area)

    print(f"Checking Step C: Small Area ({small_area}m2) -> Radius: {r_small}m")
    print(f"Checking Step C: Large Area ({large_area}m2) -> Radius: {r_large}m")

    assert r_small == config.RADIUS_MIN, "❌ Step C Failed: Small area should be clamped to MIN."
    assert r_large == config.RADIUS_MAX, "❌ Step C Failed: Large area should be clamped to MAX."

    # 6. Test Step B: Candidate Search
    # Picking a random start node from the graph
    start_node = list(solver.G.nodes())[0]
    print(f"Checking Step B: Finding candidates within {limit}m of node {start_node}...")
    
    candidates = oracle.get_candidates_within_radius(start_node, limit)
    print(f"✅ Found {len(candidates)} candidate nodes.")
    
    assert len(candidates) > 0, "❌ Step B Failed: No candidates found (check graph loading)."

    print("\n✨ ALL SPATIAL LOGIC TESTS PASSED! ✨")
    
    max_d = max([utils.get_geodesic_dist_raw(*utils.get_node_coords(oracle.G, start_node), 
                                            *utils.get_node_coords(oracle.G, c)) for c in candidates])
    print(f"Verified: Furthest candidate is {max_d:.2f}m away.")


def test_vector_logic():
    # Start at (0,0), Landmark at (0,10)
    s0 = (0, 0)
    land = (0, 10)
    
    # This point is halfway to the landmark (Should be False)
    short = (0, 5)
    # This point is way past the landmark (Should be True)
    past = (0, 15)
    # This point is far away but in the WRONG direction (Should be False)
    wrong_way = (0, -5)
    
    assert utils.is_past_landmark(s0, land, short) == False
    assert utils.is_past_landmark(s0, land, past) == True
    assert utils.is_past_landmark(s0, land, wrong_way) == False
    print("✅ Task 2.3: Vector 'Past' Logic Verified!")

if __name__ == "__main__":
    try:
        test_clamped_radius_and_search_limit()
        test_vector_logic()
    except Exception as e:
        print(f"FATAL ERROR during test: {e}")