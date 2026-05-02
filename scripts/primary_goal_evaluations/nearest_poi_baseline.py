"""
Nearest-POI Baseline for Goal Localization.

For each sample, finds the nearest POI of the correct extracted_category
to the start point, and measures distance to gold goal.

This is the strongest baseline computable without LLM inference:
it uses oracle category knowledge (extracted_category) and the full POI index.

Input:  reports/llm_audits/LLM_DEGRADATION_INPUT.parquet
        data/{city}/{city}_silver_standard.parquet  (for start coordinates)
Output: reports/llm_audits/NEAREST_POI_BASELINE.parquet
"""

import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree

import config
from src.oracle_engine import OracleEngine
from src.utils import haversine_vectorized

# OSM keys that have corresponding clean_* columns in poi_df
CLEAN_COL_KEYS = ['amenity', 'shop', 'tourism', 'leisure', 'historic', 'man_made']

def get_category_mask(poi_df, category):
    """
    Derives POI filter mask from config.LANDMARK_GROUPS.
    Uses clean_* columns which are lowercased, whitespace-stripped
    versions of raw OSM tag values.
    """
    group = config.LANDMARK_GROUPS.get(category)
    if not group:
        return None  # Unknown category — caller falls back to full POI set

    mask = pd.Series(False, index=poi_df.index)

    for osm_key, osm_vals in group.items():
        clean_col = f"clean_{osm_key}"
        if clean_col not in poi_df.columns:
            continue

        # Normalize to list
        if isinstance(osm_vals, str):
            osm_vals = [osm_vals]

        if osm_key not in CLEAN_COL_KEYS:
            continue

        for val in osm_vals:
            # Clean value to match the cleaned column format
            clean_val = val.lower().replace(" ", "").replace("_", "")
            mask |= poi_df[clean_col].str.lower().str.replace(
                r'[\s_]', '', regex=True
            ).str.contains(clean_val, na=False)

    return mask


def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    input_path = os.path.join(
        config.BASE_DIR, "reports", "llm_audits", "LLM_DEGRADATION_INPUT.parquet"
    )
    output_path = os.path.join(
        config.BASE_DIR, "reports", "llm_audits", "NEAREST_POI_BASELINE.parquet"
    )

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    print(f"By city:\n{df['city'].value_counts().to_string()}")

    # Load start coordinates from silver standard parquets
    # (LLM_DEGRADATION_INPUT has start_node IDs, but not coordinates)
    silver = {}
    for city in ['manhattan', 'pittsburgh', 'philadelphia']:
        path = os.path.join(config.BASE_DIR, "data", city,
                            f"{city}_silver_standard.parquet")
        sdf = pd.read_parquet(path)[['sample_id', 'gold_goal_lat', 'gold_goal_lon']]
        silver[city] = sdf.set_index('sample_id')

    # We need start coordinates, so we get then from the original RVS JSON instruction files
    raw_start = {}
    for city in ['manhattan', 'pittsburgh', 'philadelphia']:
        json_path = os.path.join(config.BASE_DIR, "data", city,
                                 config.CITY_SETTINGS[city]["raw_json"])
        try:
            rdf = pd.read_json(json_path)
        except ValueError:
            rdf = pd.read_json(json_path, lines=True)
        # key = sample_id, rvs_start_point = [lat, lon]
        start_map = {
            row['key']: row['rvs_start_point']
            for _, row in rdf.iterrows()
            if 'key' in row and 'rvs_start_point' in row
        }
        raw_start[city] = start_map
        print(f"Loaded {len(start_map)} start coords for {city}")

    results = []

    for city_name, city_df in df.groupby('city'):
        print(f"\nProcessing {city_name} ({len(city_df)} samples)...")

        config.CURRENT_CITY = city_name
        G = load_graph(config.get_graph_path())
        oracle = OracleEngine(
            G, config.get_poi_path(),
            config.get_node_prefix(), city_name
        )

        poi_df = oracle.poi_df
        poi_coords = oracle.poi_coords  # shape (N, 2): [lat, lon]

        city_start = raw_start[city_name]

        for _, row in tqdm(city_df.iterrows(), total=len(city_df)):
            sample_id = row['sample_id']
            category = row.get('extracted_category')
            gold_lat = row['gold_goal_lat']
            gold_lon = row['gold_goal_lon']

            # Get start coordinates
            start_coords = city_start.get(sample_id)
            if start_coords is None:
                results.append(_null_result(row))
                continue

            start_lat, start_lon = start_coords[0], start_coords[1]

            # Filter POIs by category
            mask = get_category_mask(poi_df, category)
            if mask is None or mask.sum() == 0:
                # Fall back to full POI set
                filtered_coords = poi_coords
                filtered_idx = np.arange(len(poi_coords))
            else:
                filtered_idx = np.where(mask.values)[0]
                filtered_coords = poi_coords[filtered_idx]

            if len(filtered_coords) == 0:
                results.append(_null_result(row))
                continue

            # Find nearest POI to start point
            dists_to_start = haversine_vectorized(
                start_lat, start_lon,
                filtered_coords[:, 0],
                filtered_coords[:, 1]
            )
            nearest_local_idx = np.argmin(dists_to_start)
            nearest_lat = filtered_coords[nearest_local_idx, 0]
            nearest_lon = filtered_coords[nearest_local_idx, 1]

            # Measure distance from nearest POI to gold goal
            dist_to_gold = haversine_vectorized(
                nearest_lat, nearest_lon,
                np.array([gold_lat]),
                np.array([gold_lon])
            )[0]

            results.append({
                'sample_id':       sample_id,
                'city':            city_name,
                'variant_type':    row['variant_type'],
                'oracle_label':    row['oracle_label'],
                'extracted_category': category,
                'start_lat':       start_lat,
                'start_lon':       start_lon,
                'predicted_lat':   nearest_lat,
                'predicted_lon':   nearest_lon,
                'gold_goal_lat':   gold_lat,
                'gold_goal_lon':   gold_lon,
                'distance_m':      dist_to_gold,
                'success_250m':    dist_to_gold <= 250,
                'success_100m':    dist_to_gold <= 100,
            })

        # Checkpoint after each city
        pd.DataFrame(results).to_parquet(output_path + ".checkpoint")
        print(f"Checkpoint saved. Total so far: {len(results)}")

    out_df = pd.DataFrame(results)
    out_df.to_parquet(output_path)

    print(f"\nSaved {len(out_df)} rows to {output_path}")
    print(f"\nNearest-POI Baseline Results:")
    print(f"Success@250m: {out_df['success_250m'].mean():.1%}")
    print(f"Success@100m: {out_df['success_100m'].mean():.1%}")
    print(f"\nBy variant_type:")
    print(out_df.groupby('variant_type')['success_250m'].mean().round(3).to_string())
    print(f"\nBy oracle_label:")
    print(out_df.groupby('oracle_label')['success_250m'].mean().round(3).to_string())
    print(f"\nBy city:")
    print(out_df.groupby('city')['success_250m'].mean().round(3).to_string())


def _null_result(row):
    return {
        'sample_id':       row['sample_id'],
        'city':            row['city'],
        'variant_type':    row['variant_type'],
        'oracle_label':    row['oracle_label'],
        'extracted_category': row.get('extracted_category'),
        'start_lat':       None,
        'start_lon':       None,
        'predicted_lat':   None,
        'predicted_lon':   None,
        'gold_goal_lat':   row['gold_goal_lat'],
        'gold_goal_lon':   row['gold_goal_lon'],
        'distance_m':      None,
        'success_250m':    False,
        'success_100m':    False,
    }


if __name__ == "__main__":
    main()