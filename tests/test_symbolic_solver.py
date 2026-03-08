import os
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver

def run_integration_tests():
    print("=" * 70)
    print("SYMBOLIC SOLVER INTEGRATION TESTS")
    print("=" * 70)

    # 1. Setup the environment
    graph_path = 'data/manhattan/manhattan_graph.gpickle'
    poi_path = 'data/manhattan/manhattan_poi.pkl'
    
    if not os.path.exists(graph_path):
        print(f"❌ Error: {graph_path} not found. Skipping tests.")
        return

    oracle = OracleEngine(graph_path, poi_path)
    solver = SymbolicSolver(oracle)

    # TEST 0: BASIC LOAD CHECK
    print(f"\n[TEST 0] Graph Stats:")
    print(f"Nodes: {len(solver.G.nodes)}")
    print(f"Edges: {len(solver.G.edges)}")

    # TEST 1: REACHABILITY
    print("\n" + "="*70 + "\nTEST 1: REACHABILITY\n" + "="*70)
    node_ids = list(solver.G.nodes)
    n1, n2 = node_ids[0], node_ids[100]
    
    result = solver.check_reachability(n1, n2)
    print(f"✅ Reachability {n1} → {n2}: {result}")

    # TEST 2: SHORTEST PATH & DISTANCE
    print("\n" + "="*70 + "\nTEST 2: SHORTEST PATH\n" + "="*70)
    path = solver.compute_shortest_path(n1, n2)
    if path:
        dist = solver.get_path_length(path)
        print(f"✅ Path found! Nodes: {len(path)}, Total Distance: {dist:.2f}m")
    else:
        print("⚠️ No path found between chosen sample nodes.")

    # TEST 3: DIRECTION
    print("\n" + "="*70 + "\nTEST 3: DIRECTION\n" + "="*70)
    # Using real nodes to test the new Node ID based direction logic
    direction = solver.get_coarse_direction(n1, n2)
    print(f"✅ Direction from {n1} to {n2}: {direction}")

    # TEST 4: ORACLE BRIDGE
    print("\n" + "="*70 + "\nTEST 4: LANDMARK NAVIGATION\n" + "="*70)
    sample_landmark = "Hell's Kitchen" # Ensure this exists in your POI csv
    path_to_poi = solver.get_path_to_landmark(n1, sample_landmark)
    if path_to_poi:
        print(f"✅ Successfully found path to {sample_landmark}!")
    else:
        print(f"❌ Could not resolve or find path to {sample_landmark}")

if __name__ == '__main__':
    run_integration_tests()