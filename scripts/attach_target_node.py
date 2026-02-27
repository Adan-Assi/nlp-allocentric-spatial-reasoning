import sys
from pathlib import Path
import pandas as pd
# add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.symbolic_solver import SymbolicSolver

GRAPH_PATH = "data/manhattan/manhattan_graph.gpickle"  # adjust if needed

IN_PATH = Path("data/processed/train_normalized.parquet")
OUT_PATH = Path("data/processed/train_manhattan_with_target_node.parquet")

MAX_DIST_M = 75.0  # quality filter threshold (tune later)

def main():
    df = pd.read_parquet(IN_PATH)
    df = df[df["region"] == "manhattan"].copy()
    print(f"Loaded train split, Manhattan-only rows: {len(df)}")

    solver = SymbolicSolver(GRAPH_PATH)

    target_nodes = []
    target_dists = []

    for lat, lon in zip(df["target_lat"].tolist(), df["target_lon"].tolist()):
        node_id, dist_m = solver.find_nearest_node(lat, lon)
        target_nodes.append(node_id)
        target_dists.append(dist_m)

    df["target_node_id"] = target_nodes
    df["target_node_distance_m"] = target_dists

    before = len(df)
    df = df[df["target_node_distance_m"] <= MAX_DIST_M].copy()
    after = len(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"✅ Saved: {OUT_PATH}")
    print(f"Filtered by distance <= {MAX_DIST_M}m: {before} -> {after}")

if __name__ == "__main__":
    main()
