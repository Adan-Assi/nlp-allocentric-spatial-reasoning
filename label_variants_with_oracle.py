"""
label_variants_with_oracle.py
Apply the symbolic oracle to a parquet of underspecified variants.

For each variant row:
  1. Parse the remaining spatial constraints from the variant text
     (direction / radius / proximity) using underspec_constraints.
  2. Snap the row's start (lat, lon) to a graph node.
  3. Find all graph nodes within the effective radius.
  4. If a direction constraint remains, filter to nodes in that direction
     (using the 8-way matcher from utils.direction_matches).
  5. Label by candidate count:
        0 candidates  -> "contradictory"
        1 candidate   -> "answerable"
        >1 candidates -> "ambiguous"

Fixes vs the original repo version:
  * SymbolicSolver constructor was being called with a graph path; it actually
    requires a pre-built OracleEngine. Build the Oracle here and pass it.
  * The script previously called solver.nodes_within_radius() and
    solver.filter_nodes_by_direction(), which do not exist. Replaced with
    the Oracle-level equivalents (find_nearest_node, get_candidates_within_radius,
    filter_candidates_by_direction).
  * Direction constraints from extract_constraints come as words ("north",
    "southwest"); the Oracle filter expects abbreviations ("N", "SW").
    A small mapping is now applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
from src.constraints.underspec_constraints import extract_constraints


# Map the direction words returned by underspec_constraints to the 8-way
# abbreviations used by Oracle.filter_candidates_by_direction.
_DIRECTION_WORD_TO_ABBR = {
    "north":     "N",
    "northeast": "NE",
    "east":      "E",
    "southeast": "SE",
    "south":     "S",
    "southwest": "SW",
    "west":      "W",
    "northwest": "NW",
}


def oracle_label_from_count(n: int) -> str:
    if n == 0:
        return "contradictory"
    if n == 1:
        return "answerable"
    return "ambiguous"


def _build_oracle_for_region(region: str, graphs_dir: Path, graph_ext: str) -> OracleEngine:
    """
    Build a fully-loaded OracleEngine for a region, using the standard
    config.CITY_SETTINGS layout for graph + POI paths.
    """
    if region not in config.CITY_SETTINGS:
        raise ValueError(
            f"Region '{region}' is not in config.CITY_SETTINGS. "
            f"Known: {list(config.CITY_SETTINGS.keys())}"
        )

    # Prefer the explicit user-provided graphs_dir; fall back to the standard
    # data/<city>/<city>_graph.gpickle layout if a region-named file isn't there.
    explicit_path = graphs_dir / f"{region}{graph_ext}"
    if explicit_path.exists():
        graph_path = str(explicit_path)
    else:
        # Switch to the city and use config's path resolver
        prev_city = config.CURRENT_CITY
        config.CURRENT_CITY = region
        try:
            graph_path = config.get_graph_path()
        finally:
            config.CURRENT_CITY = prev_city
        if not Path(graph_path).exists():
            raise FileNotFoundError(
                f"Graph not found for region='{region}' at '{explicit_path}' or '{graph_path}'."
            )

    # Switch the city context for POI loading too.
    prev_city = config.CURRENT_CITY
    config.CURRENT_CITY = region
    try:
        poi_path = config.get_poi_path()
        node_prefix = config.get_node_prefix()
    finally:
        config.CURRENT_CITY = prev_city

    return OracleEngine(graph_path, poi_path, node_prefix, region)


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

    graphs_dir_path = Path(graphs_dir)

    # Per-region Oracle/Solver cache. Loading a graph + POIs is expensive; keep
    # them across all variants of the same region.
    oracle_cache: dict[str, OracleEngine] = {}
    solver_cache: dict[str, SymbolicSolver] = {}

    def region_key(r):
        return str(r).strip().lower()

    def get_solver(region: str) -> SymbolicSolver:
        if region not in solver_cache:
            oracle_cache[region] = _build_oracle_for_region(region, graphs_dir_path, graph_ext)
            solver_cache[region] = SymbolicSolver(oracle_cache[region])
        return solver_cache[region]

    num_candidates: list[int] = []
    used_radius_m: list[float | None] = []
    used_direction: list[str | None] = []
    oracle_labels: list[str] = []

    for _, row in df.iterrows():
        region = region_key(row[region_col])
        text = str(row[text_col])
        start_lat = float(row[start_lat_col])
        start_lon = float(row[start_lon_col])

        solver = get_solver(region)
        oracle = oracle_cache[region]

        # 1. Extract spatial constraints remaining in the variant.
        cons = extract_constraints(text, enabled=enabled_types)

        directions_raw = [
            c.meta["dir"] for c in cons
            if c.type == "direction" and "dir" in c.meta
        ]
        radii = [
            float(c.meta["meters"]) for c in cons
            if c.type == "radius" and "meters" in c.meta
        ]
        prox = [
            float(c.meta["meters"]) for c in cons
            if c.type == "proximity" and "meters" in c.meta
        ]

        # First direction wins (mirrors the old behavior). Map word -> abbrev.
        direction_word = directions_raw[0] if directions_raw else None
        direction_abbr = (
            _DIRECTION_WORD_TO_ABBR.get(direction_word.lower())
            if direction_word else None
        )

        # 2. Effective radius — smallest constraint wins (most specific).
        radius_m: float | None = None
        if radii or prox:
            radius_m = min(radii + prox)

        # 3. Snap start (lat, lon) to a graph node.
        start_node, _snap_dist = oracle.find_nearest_node(start_lat, start_lon)

        # 4. Candidate set.
        if radius_m is not None:
            candidate_ids = oracle.get_candidates_within_radius(start_node, radius_m)
        else:
            # No radius-like constraint -> every node in the region graph.
            candidate_ids = list(solver.G.nodes())

        # 5. Direction filter (if a direction is still in the text).
        if direction_abbr is not None and candidate_ids:
            candidate_ids = oracle.filter_candidates_by_direction(
                start_node, candidate_ids, direction_abbr
            )

        n = len(candidate_ids)

        num_candidates.append(n)
        used_radius_m.append(radius_m)
        used_direction.append(direction_abbr)
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
