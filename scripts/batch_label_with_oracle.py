"""
Task 3.3 — Batch Labeling Script (OracleEngine Integration)

Iterates through underspecified variants and labels each as:
  - Answerable:    target reachable AND candidate set is small (≤ threshold)
  - Ambiguous:     target reachable BUT candidate set is large
  - Contradictory: target NOT reachable from the filtered candidate set

Uses Shaimaa's OracleEngine for:
  - Landmark resolution via POI database
  - Candidate search within clamped radius
  - Directional filtering (N/S/E/W dominant axis)

Uses constraint extraction to parse direction + radius + proximity from text.

Usage:
    python -m scripts.batch_label_with_oracle \
        --in_dir data/processed \
        --out_dir data/processed \
        --graph_path data/manhattan/manhattan_graph.gpickle \
        --poi_path data/manhattan/manhattan_poi.pkl

    # Quick test on 100 rows:
    python -m scripts.batch_label_with_oracle \
        --in_dir data/processed \
        --out_dir data/processed \
        --graph_path data/manhattan/manhattan_graph.gpickle \
        --poi_path data/manhattan/manhattan_poi.pkl \
        --limit 100
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver
from src.constraints.underspec_constraints import extract_constraints


# ─── Thresholds ─────────────────────────────────────────────────────────

# If |candidates| ≤ this AND target is inside → Answerable
# Tuned for Manhattan density (~36k nodes)
ANSWERABLE_THRESHOLD = 50

# If a landmark was resolved, we allow a larger set to still count as answerable
# because landmark grounding significantly narrows the real-world ambiguity
ANSWERABLE_WITH_LANDMARK = ANSWERABLE_THRESHOLD * 5


# ─── Helpers ────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Quick haversine for start-to-target distance in meters."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Core Classification Logic ──────────────────────────────────────────

def classify_variant(
    oracle, solver, variant_text,
    start_lat, start_lon, target_lat, target_lon,
    target_node_id=None,
):
    """
    Full oracle pipeline for one variant:
      1. Extract remaining constraints from variant text
      2. Try to resolve landmarks via OracleEngine POI database
      3. Compute search radius (clamped or fallback)
      4. Find start node → get candidates within radius
      5. Filter by direction
      6. Check if target is in candidate set
      7. Classify based on |S| and target membership

    Returns:
        dict with keys: oracle_label, candidate_count, target_in_candidates,
                        used_radius_m, used_direction, landmark_resolved
    """

    # ── Step 1: Parse constraints from the (possibly masked) text ──────
    cons = extract_constraints(variant_text, enabled=("direction", "radius", "proximity"))
    dirs = [c.meta["dir"] for c in cons if c.type == "direction" and "dir" in c.meta]
    radii = [float(c.meta["meters"]) for c in cons if c.type == "radius" and "meters" in c.meta]
    prox = [float(c.meta["meters"]) for c in cons if c.type == "proximity" and "meters" in c.meta]

    direction = dirs[0] if dirs else None
    all_m = radii + prox
    extracted_distance_m = min(all_m) if all_m else None

    # ── Step 2: Try to resolve landmarks via OracleEngine ──────────────
    landmark_resolved = False
    text_lower = variant_text.lower()
    for keyword in config.LANDMARK_GROUPS:
        if keyword in text_lower:
            node = oracle.resolve_landmark(keyword)
            if node is not None:
                landmark_resolved = True
                break

    # ── Step 3: Determine search radius ────────────────────────────────
    if extracted_distance_m is not None:
        # Clamped radius from ORACLE_SPEC: max(D * 1.1, D + 80)
        search_radius = solver.get_search_limit(extracted_distance_m)
    else:
        # Fallback: actual start→target distance × 1.5
        actual_d = haversine(start_lat, start_lon, target_lat, target_lon)
        search_radius = min(actual_d * 1.5, 3000.0)

    # ── Step 4: Find start node and get candidates ─────────────────────
    start_node, _ = oracle.find_nearest_node(start_lat, start_lon)
    if start_node is None:
        return {
            "oracle_label": "contradictory",
            "candidate_count": 0,
            "target_in_candidates": False,
            "used_radius_m": round(search_radius, 1),
            "used_direction": direction,
            "landmark_resolved": landmark_resolved,
        }

    candidates = oracle.get_candidates_within_radius(start_node, search_radius)

    # ── Step 5: Filter by direction ────────────────────────────────────
    if direction and len(candidates) > 0:
        candidates = oracle.filter_candidates_by_direction(
            start_node, candidates, direction
        )

    # ── Step 6: Check target membership ────────────────────────────────
    target_in = False
    if target_node_id is not None:
        target_in = target_node_id in candidates
    else:
        t_node, _ = oracle.find_nearest_node(target_lat, target_lon)
        if t_node is not None:
            target_in = t_node in candidates

    # ── Step 7: Classify ───────────────────────────────────────────────
    n = len(candidates)

    if n == 0 or not target_in:
        label = "contradictory"
    elif n <= ANSWERABLE_THRESHOLD:
        label = "answerable"
    elif landmark_resolved and n <= ANSWERABLE_WITH_LANDMARK:
        # Landmark grounding narrows real-world ambiguity
        label = "answerable"
    else:
        label = "ambiguous"

    return {
        "oracle_label": label,
        "candidate_count": n,
        "target_in_candidates": target_in,
        "used_radius_m": round(search_radius, 1),
        "used_direction": direction,
        "landmark_resolved": landmark_resolved,
    }


# ─── Batch Processing ──────────────────────────────────────────────────

def label_split(df, oracle, solver, text_col="variant_text", limit=None):
    """Label all rows in a DataFrame, return with new columns."""
    if limit is not None:
        df = df.head(limit).copy()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        result = classify_variant(
            oracle=oracle,
            solver=solver,
            variant_text=str(row[text_col]),
            start_lat=float(row["start_lat"]),
            start_lon=float(row["start_lon"]),
            target_lat=float(row["target_lat"]),
            target_lon=float(row["target_lon"]),
            target_node_id=row.get("target_node_id"),
        )
        results.append(result)

    res_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    return out


# ─── Main ───────────────────────────────────────────────────────────────

def main(in_dir, out_dir, graph_path, poi_path, text_col="variant_text", limit=None):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Oracle + Solver once
    print("Initializing OracleEngine...")
    oracle = OracleEngine(graph_path, poi_path)
    solver = SymbolicSolver(oracle)
    print(f"Graph: {oracle.G.number_of_nodes()} nodes, {oracle.G.number_of_edges()} edges")
    print(f"POIs: {len(oracle.poi_df)} entries")
    print()

    # Process each split
    splits = ["train", "test", "validation_seen", "validation_unseen"]
    for split in splits:
        in_path = in_dir / f"{split}_variants.parquet"
        out_path = out_dir / f"{split}_variants_labeled.parquet"

        if not in_path.exists():
            print(f"⚠️  Skip {split}: {in_path} not found")
            continue

        df = pd.read_parquet(in_path)
        print(f"=== {split}: {len(df)} variants ===")

        labeled = label_split(df, oracle, solver, text_col=text_col, limit=limit)
        labeled.to_parquet(out_path, index=False)

        print(f"✅ Saved: {out_path}")
        print(labeled["oracle_label"].value_counts().to_string())

        # Degradation preview
        if "n_dropped" in labeled.columns:
            print("\nDegradation preview:")
            for nd in sorted(labeled["n_dropped"].unique()):
                sub = labeled[labeled["n_dropped"] == nd]
                dist = sub["oracle_label"].value_counts().to_dict()
                print(f"  n_dropped={nd} ({len(sub)} rows): {dist}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 3.3: Batch Oracle Labeling")
    ap.add_argument("--in_dir", default="data/processed",
                    help="Directory with *_variants.parquet files")
    ap.add_argument("--out_dir", default="data/processed",
                    help="Directory for *_variants_labeled.parquet output")
    ap.add_argument("--graph_path", required=True,
                    help="Path to manhattan_graph.gpickle")
    ap.add_argument("--poi_path", required=True,
                    help="Path to manhattan_poi.pkl")
    ap.add_argument("--text_col", default="variant_text")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N rows per split (for testing)")
    a = ap.parse_args()
    main(**vars(a))
