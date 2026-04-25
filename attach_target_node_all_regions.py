"""
attach_target_node_all_regions.py
For each split parquet, snap (target_lat, target_lon) to the nearest node in
the appropriate city graph and attach two columns:
  target_node_id          : the snapped graph node ID
  target_node_distance_m  : distance from (lat, lon) to that node, in meters

Fix vs the original repo version:
  * Original called solver.find_nearest_node(lat, lon). That method lives on
    OracleEngine, not SymbolicSolver. Changed to use the Oracle directly.
"""

import json
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src.oracle_engine import OracleEngine

MANIFEST_PATH = Path("data/graphs/manifest.json")
IN_DIR = Path("data/processed")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "test", "validation_seen", "validation_unseen"]
MAX_DIST_M = 75.0  # rows farther than this from any node will be flagged downstream


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_oracle_for_region(region: str, graph_path: str) -> OracleEngine:
    """Build an OracleEngine for `region` using the standard config paths."""
    if region not in config.CITY_SETTINGS:
        raise ValueError(
            f"Region '{region}' is not in config.CITY_SETTINGS. "
            f"Known: {list(config.CITY_SETTINGS.keys())}"
        )
    prev_city = config.CURRENT_CITY
    config.CURRENT_CITY = region
    try:
        poi_path = config.get_poi_path()
        node_prefix = config.get_node_prefix()
    finally:
        config.CURRENT_CITY = prev_city
    return OracleEngine(graph_path, poi_path, node_prefix, region)


def ground_region(df_region: pd.DataFrame, oracle: OracleEngine) -> pd.DataFrame:
    """Snap each (target_lat, target_lon) row to the nearest node in `oracle.G`."""
    nodes = []
    dists = []
    print(f"Starting grounding for {len(df_region)} rows...")

    for lat, lon in tqdm(
        zip(df_region["target_lat"].tolist(), df_region["target_lon"].tolist()),
        total=len(df_region),
    ):
        node_id, dist_m = oracle.find_nearest_node(lat, lon)
        nodes.append(node_id)
        dists.append(dist_m)

    df_region = df_region.copy()
    df_region["target_node_id"] = nodes
    df_region["target_node_distance_m"] = dists
    return df_region


def main():
    graph_paths = load_manifest()

    # Cache one Oracle per region so each graph is loaded only once.
    oracles: dict[str, OracleEngine] = {}

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
                print(f"⚠️ No graph in manifest for region={region}; skipping {len(df_region)} rows.")
                continue

            if region not in oracles:
                oracles[region] = _build_oracle_for_region(region, graph_paths[region])

            grounded_parts.append(ground_region(df_region, oracles[region]))

        if not grounded_parts:
            print(f"⚠️ Nothing grounded for split={split}. Skipping write.")
            continue

        out = pd.concat(grounded_parts, axis=0).reset_index(drop=True)
        out_path = OUT_DIR / f"{split}_grounded.parquet"
        out.to_parquet(out_path, index=False)
        print(f"✅ Wrote {out_path} ({len(out)} rows)")

        # Quick QA: how many rows are farther than MAX_DIST_M from their snapped node?
        far = out[out["target_node_distance_m"] > MAX_DIST_M]
        if len(far):
            print(
                f"   ⚠️ {len(far)} rows snapped >{MAX_DIST_M}m away — "
                f"these may be off-graph and worth filtering."
            )


if __name__ == "__main__":
    main()
