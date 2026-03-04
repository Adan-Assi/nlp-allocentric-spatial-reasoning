from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import pandas as pd

from src.symbolic_solver import SymbolicSolver
from src.constraints.underspec_constraints import extract_constraints


def load_graph_gpickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def oracle_label_from_count(n: int) -> str:
    if n == 0:
        return "contradictory"
    if n == 1:
        return "answerable"
    return "ambiguous"


def main(
    in_path: str,
    out_path: str,
    graphs_dir: str = "data/graphs",
    graph_ext: str = ".gpickle",
    region_col: str = "region",
    text_col: str = "variant_text",
    start_lat_col: str = "start_lat",
    start_lon_col: str = "start_lon",
    enabled_types: list[str] | None = None,
    limit: int | None = None,
):
    enabled_types = enabled_types or ["direction", "radius", "proximity"]

    df = pd.read_parquet(in_path)
    needed = [region_col, text_col, start_lat_col, start_lon_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    if limit is not None:
        df = df.head(limit).copy()

    graphs_dir = Path(graphs_dir)
    solver_cache: dict[str, SymbolicSolver] = {}

    num_candidates = []
    used_radius_m = []
    used_direction = []
    oracle_labels = []

    # Normalize region key to match your graph filenames
    def region_key(r):
        return str(r).strip().lower()

    for _, row in df.iterrows():
        region = region_key(row[region_col])
        text = str(row[text_col])
        start_lat = float(row[start_lat_col])
        start_lon = float(row[start_lon_col])

        # Load solver/graph for region (cached)
        if region not in solver_cache:
            gpath = graphs_dir / f"{region}{graph_ext}"
            if not gpath.exists():
                raise FileNotFoundError(f"Graph not found for region='{region}': {gpath}")
            #G = load_graph_gpickle(str(gpath))
            #solver_cache[region] = SymbolicSolver(G)
            solver_cache[region] = SymbolicSolver(str(gpath))

        solver = solver_cache[region]

        # Extract constraints that remain in this variant text
        cons = extract_constraints(text, enabled=enabled_types)

        # Pull out direction + radius/proximity meters
        dirs = [c.meta["dir"] for c in cons if c.type == "direction" and "dir" in c.meta]
        radii = [float(c.meta["meters"]) for c in cons if c.type == "radius" and "meters" in c.meta]
        prox  = [float(c.meta["meters"]) for c in cons if c.type == "proximity" and "meters" in c.meta]

        direction = dirs[0] if dirs else None

        # Determine effective radius (v1): smallest constraint wins (most specific)
        radius_m = None
        if radii or prox:
            radius_m = min(radii + prox)

        # Candidate set
        if radius_m is not None:
            candidates = solver.nodes_within_radius(start_lat, start_lon, radius_m)
        else:
            # No radius-like constraint -> all nodes in the region graph
            candidates = list(solver.nodes.keys())

        if direction is not None:
            candidates = solver.filter_nodes_by_direction(start_lat, start_lon, candidates, direction)

        n = len(candidates)

        num_candidates.append(n)
        used_radius_m.append(radius_m)
        used_direction.append(direction)
        oracle_labels.append(oracle_label_from_count(n))

    df_out = df.copy()
    df_out["num_candidates"] = num_candidates
    df_out["used_radius_m"] = used_radius_m
    df_out["used_direction"] = used_direction
    df_out["oracle_label"] = oracle_labels

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    print(f"✅ Wrote labeled variants: {out_path}")
    print(df_out["oracle_label"].value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--graphs_dir", default="data/graphs")
    ap.add_argument("--graph_ext", default=".gpickle")
    ap.add_argument("--region_col", default="region")
    ap.add_argument("--text_col", default="variant_text")
    ap.add_argument("--start_lat_col", default="start_lat")
    ap.add_argument("--start_lon_col", default="start_lon")
    ap.add_argument("--enabled_types", default="direction,radius,proximity")
    ap.add_argument("--limit", type=int, default=None)  # use this for quick debug
    args = ap.parse_args()

    enabled = [s.strip() for s in args.enabled_types.split(",") if s.strip()]
    main(
        in_path=args.in_path,
        out_path=args.out_path,
        graphs_dir=args.graphs_dir,
        graph_ext=args.graph_ext,
        region_col=args.region_col,
        text_col=args.text_col,
        start_lat_col=args.start_lat_col,
        start_lon_col=args.start_lon_col,
        enabled_types=enabled,
        limit=args.limit,
    )