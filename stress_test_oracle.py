"""
stress_test_oracle.py
Validates a batch of underspecified variants against the symbolic oracle.

Two structural fixes vs the original repo version:
  1. The original referenced config.GRAPH_PATH / config.POI_PATH / config.VARIANTS_JSON /
     config.RVS_DATA_JSON. Those module-level constants were removed when config went
     multi-city. They are replaced by the per-city resolvers (get_graph_path / get_poi_path)
     and explicit per-city paths for the variant + raw JSON files.
  2. The original called OracleEngine(GRAPH_PATH, POI_PATH) with 2 args; the constructor
     now requires (graph_path, poi_path, node_prefix, city_name).

A behavioral improvement:
  * Instead of re-implementing candidate filtering inline, we now delegate to
    solver.solve(text, start_node, mode="label"), which returns the official
    Answerable / Ambiguous / Contradictory state per the project proposal. This
    keeps the stress test consistent with the rest of the pipeline.
"""

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

# 1. DYNAMIC PATH INJECTION
# Ensures 'src' and 'config' are discoverable regardless of entry point.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == "scripts" else current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver


def _city_paths(city_name: str) -> dict:
    """Resolve graph / POI / variants / raw-RVS JSON paths for a given city."""
    if city_name not in config.CITY_SETTINGS:
        raise ValueError(
            f"Unknown city '{city_name}'. Known: {list(config.CITY_SETTINGS.keys())}"
        )

    prev_city = config.CURRENT_CITY
    config.CURRENT_CITY = city_name
    try:
        graph_path = config.get_graph_path()
        poi_path = config.get_poi_path()
        node_prefix = config.get_node_prefix()
    finally:
        config.CURRENT_CITY = prev_city

    settings = config.CITY_SETTINGS[city_name]
    raw_json = os.path.join(config.BASE_DIR, "data", city_name, settings["raw_json"])
    variants_json = os.path.join(config.BASE_DIR, "data", city_name, "underspecified_variants.json")
    return {
        "graph_path": graph_path,
        "poi_path": poi_path,
        "node_prefix": node_prefix,
        "raw_json": raw_json,
        "variants_json": variants_json,
    }


def _load_raw_lookup(raw_json_path: str) -> dict:
    """
    Read RVS data line-by-line into a {sample_id -> row} dict. Tolerant to
    sample_number / sample_id / id keys.
    """
    raw_lookup: dict[str, dict] = {}
    print(f"📖 Reading RVS data from {raw_json_path}...")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = str(
                item.get("rvs_sample_number")
                or item.get("sample_number")
                or item.get("sample_id")
                or item.get("id")
                or i
            )
            raw_lookup[key] = item
    return raw_lookup


def run_stress_test(city_name: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Validate underspecified variants for `city_name` and emit a CSV of labels.
    Each row = one variant of one sample. The label is taken from the official
    solver in label mode (Answerable / Ambiguous / Contradictory).
    """
    paths = _city_paths(city_name)

    # 1. Engines
    oracle = OracleEngine(paths["graph_path"], paths["poi_path"], paths["node_prefix"], city_name)
    solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())

    # 2. Data
    if not os.path.exists(paths["variants_json"]):
        raise FileNotFoundError(
            f"Variants file not found for city='{city_name}': {paths['variants_json']}"
        )
    with open(paths["variants_json"], "r", encoding="utf-8") as f:
        variants_payload = json.load(f)

    raw_lookup = _load_raw_lookup(paths["raw_json"])

    results: list[dict] = []

    for entry in tqdm(variants_payload, desc=f"⚖️ Stress Testing {city_name}"):
        sample_id = str(entry.get("sample_id"))
        original_meta = raw_lookup.get(sample_id)
        if original_meta is None:
            # Variants without a matching raw row can't be grounded; skip them.
            continue

        start_coords = original_meta.get("rvs_start_point") or original_meta.get("start_point")
        if start_coords is None:
            print(f"⚠️ No start coordinates for sample {sample_id}; skipping.")
            continue
        start_lat, start_lon = start_coords

        # Snap the agent's start to a node so the solver has an anchor.
        start_node, _snap_dist = oracle.find_nearest_node(start_lat, start_lon)

        for variant in entry.get("variants", []):
            variant_text = variant.get("variant_text") or variant.get("text", "")
            if not variant_text:
                continue

            try:
                result = solver.solve(variant_text, start_node, mode="label")
            except Exception as exc:
                print(f"❌ solve() failed on sample={sample_id}: {type(exc).__name__}: {exc}")
                continue

            results.append({
                "sample_id": sample_id,
                "variant_type": variant.get("type"),
                "final_label": result["state"],
                "candidate_count": result.get("candidate_count"),
                "reachable_candidate_count": result.get("reachable_candidate_count"),
                "extracted_category": result.get("extracted_category"),
                "extracted_direction": result.get("extracted_direction"),
            })

    df_results = pd.DataFrame(results)

    out = output_path or config.AMBIGUITY_REPORT_CSV
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(out, index=False)

    print(f"\n✅ Stress test complete for {city_name}.")
    print(f"📊 Report saved to: {out}")
    print("\nLabel distribution:")
    if not df_results.empty:
        print(df_results["final_label"].value_counts(normalize=True).round(3))
    else:
        print("(no rows produced)")

    return df_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Symbolic oracle stress test on underspecified variants.")
    ap.add_argument(
        "--city",
        default=config.CURRENT_CITY,
        choices=list(config.CITY_SETTINGS.keys()),
        help="City to stress-test (default: config.CURRENT_CITY).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to config.AMBIGUITY_REPORT_CSV.",
    )
    args = ap.parse_args()
    run_stress_test(args.city, args.output)
