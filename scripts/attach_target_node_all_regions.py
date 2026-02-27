import json
import pandas as pd
from pathlib import Path
import sys

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.symbolic_solver import SymbolicSolver  # your solver module

MANIFEST_PATH = Path("data/graphs/manifest.json")
IN_DIR = Path("data/processed")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "test", "validation_seen", "validation_unseen"]
MAX_DIST_M = 75.0

def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ground_region(df_region: pd.DataFrame, solver: SymbolicSolver) -> pd.DataFrame:
    nodes = []
    dists = []

    for lat, lon in zip(df_region["target_lat"].tolist(), df_region["target_lon"].tolist()):
        node_id, dist_m = solver.find_nearest_node(lat, lon)
        nodes.append(node_id)
        dists.append(dist_m)

    df_region = df_region.copy()
    df_region["target_node_id"] = nodes
    df_region["target_node_distance_m"] = dists
    return df_region

def main():
    graph_paths = load_manifest()

    # Cache solvers so we load each graph only once per run
    solvers = {}

    for split in SPLITS:
        in_path = IN_DIR / f"{split}_normalized.parquet"
        if not in_path.exists():
            print(f"⚠️ Missing normalized split: {in_path}")
            continue

        df = pd.read_parquet(in_path)
        df["region"] = df["region"].astype(str).str.lower()

        grounded_parts = []
        print(f"\n=== Split: {split} | rows={len(df)} ===")

        for region, df_region in df.groupby("region"):
            if region not in graph_paths:
                print(f"⚠️ No graph for region='{region}' — skipping {len(df_region)} rows")
                continue

            if region not in solvers:
                solvers[region] = SymbolicSolver(graph_paths[region])

            solver = solvers[region]
            print(f"Grounding region='{region}' rows={len(df_region)}")

            g = ground_region(df_region, solver)

            # filter by grounding quality
            before = len(g)
            g = g[g["target_node_distance_m"] <= MAX_DIST_M].copy()
            after = len(g)
            print(f"  Filter <= {MAX_DIST_M}m: {before} -> {after}")

            grounded_parts.append(g)

        if not grounded_parts:
            print(f"❌ No regions grounded for split={split}")
            continue

        out = pd.concat(grounded_parts, ignore_index=True)
        out_path = OUT_DIR / f"{split}_all_regions_grounded.parquet"
        out.to_parquet(out_path, index=False)

        print(f"✅ Saved grounded split: {out_path} | rows={len(out)}")

if __name__ == "__main__":
    main()
