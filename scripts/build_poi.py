"""
build_poi.py
Builds the per-city POI pickle the OracleEngine reads from
data/<city>/<city>_poi.pkl.

The OracleEngine expects a pandas/GeoPandas DataFrame with:
  - 'osmid'                      — OSM id (or string convertible)
  - 'centroid' OR 'geometry'     — shapely Point/Polygon (centroid preferred)
  - any subset of the OSM tag columns the LANDMARK_GROUPS / TEXT_TO_GROUP_MAP
    in config.py reference: amenity, shop, tourism, leisure, historic, name,
    brand, building, office, craft, healthcare, highway, boundary, natural,
    waterway, railway, building:material, roof:shape, roof:material, colour,
    building:colour, roof:colour, parking, wikipedia, wikidata, man_made.

Usage (from the repo root):
    python scripts/build_poi.py --city manhattan
    python scripts/build_poi.py --city pittsburgh
    python scripts/build_poi.py --city philadelphia

Notes:
  - This downloads tens of MB of OSM data on first run and caches it in
    ~/.osmnx/cache. Manhattan typically takes ~5–15 minutes the first time.
  - If your teammates already have a working manhattan_poi.pkl, prefer that
    file — the column set may have been hand-tuned during their original run.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import osmnx as ox
import pandas as pd

# Make repo root importable when running as `python scripts/build_poi.py`
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


# Geographic query string per city. These are the same names used by
# build_region_graphs.py so the POI footprint matches the graph footprint.
CITY_QUERIES = {
    "manhattan":    "Manhattan, New York City, New York, USA",
    "pittsburgh":   "Pittsburgh, Pennsylvania, USA",
    "philadelphia": "Philadelphia, Pennsylvania, USA",
}

# Tag dict passed to osmnx. Each key=True means "any value of this tag is
# fetched". This is intentionally broad — the Oracle filters down later via
# config.LANDMARK_GROUPS. Adding more tags doesn't slow the Oracle (it's a
# KDTree lookup over coordinates) but does increase the pickle size.
OSM_TAGS = {
    "amenity":       True,
    "shop":          True,
    "tourism":       True,
    "leisure":       True,
    "historic":      True,
    "office":        True,
    "craft":         True,
    "healthcare":    True,
    "building":      True,
    "man_made":      True,
    "natural":       True,
    "waterway":      True,
    "railway":       True,
    "highway":       True,
    "boundary":      True,
}

# Columns we always want the pickle to expose, even when the underlying OSM
# data didn't include them. The Oracle's _prepare_poi_data() tolerates missing
# columns but downstream search code is happier when they exist as empty.
ENSURED_COLUMNS = [
    "name",
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "historic",
    "man_made",
    "brand",
    "building",
    "office",
    "craft",
    "healthcare",
    "highway",
    "boundary",
    "natural",
    "waterway",
    "railway",
    "wikipedia",
    "wikidata",
    "parking",
    "building:material",
    "roof:shape",
    "roof:material",
    "colour",
    "building:colour",
    "roof:colour",
]


def _flatten_osmid(idx) -> str:
    """
    osmnx returns features as a GeoDataFrame whose index is a MultiIndex of
    (element_type, osmid). Flatten that to a single string id and prepend the
    element_type so we never collide a node id with a way id of the same
    number. The Oracle's normalize_node_id() strips these prefixes when it
    matches against the graph.
    """
    if isinstance(idx, tuple) and len(idx) == 2:
        element_type, osmid = idx
        return f"{element_type}/{osmid}"
    return str(idx)


def build_poi(city: str, force: bool = False) -> Path:
    if city not in CITY_QUERIES:
        raise ValueError(
            f"Unknown city '{city}'. Known: {list(CITY_QUERIES.keys())}"
        )

    out_dir = Path(config.BASE_DIR) / "data" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city}_poi.pkl"

    if out_path.exists() and not force:
        print(f"⚠️  {out_path} already exists. Use --force to rebuild.")
        return out_path

    query = CITY_QUERIES[city]
    print(f"📡 Querying OSM for POIs in: {query}")
    print(f"   (this can take 5–15 minutes the first time; tags are cached locally)")

    gdf = ox.features_from_place(query, tags=OSM_TAGS)
    print(f"✅ Downloaded {len(gdf):,} OSM features.")

    # ---- Schema normalization ----
    gdf = gdf.reset_index()  # turn the (element_type, osmid) MultiIndex into columns

    # Build a single 'osmid' column the Oracle can read.
    if "element_type" in gdf.columns and "osmid" in gdf.columns:
        gdf["osmid"] = gdf.apply(
            lambda r: f"{r['element_type']}/{r['osmid']}", axis=1
        )
    elif "osmid" not in gdf.columns:
        # Fall back to whatever the index was.
        gdf["osmid"] = [
            _flatten_osmid(i) for i in gdf.index
        ]

    # Build a 'centroid' column. The Oracle will use this in preference to
    # 'geometry' when both are present (see oracle_engine._prepare_poi_data).
    # Polygons get .centroid; points are returned as-is.
    gdf["centroid"] = gdf["geometry"].apply(
        lambda g: g if g.geom_type == "Point" else g.centroid
    )

    # Ensure expected columns exist (empty string fallback, NOT NaN — the
    # Oracle's downstream string-cleaning logic chokes on float NaN).
    for col in ENSURED_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = ""
        else:
            gdf[col] = gdf[col].astype(object).where(gdf[col].notna(), "")

    # Keep only POIs with at least one identifying tag — pure unnamed buildings
    # explode the pickle size and the Oracle never matches against them.
    has_signal = (
        gdf["name"].astype(str).str.len().gt(0)
        | gdf["amenity"].astype(str).str.len().gt(0)
        | gdf["shop"].astype(str).str.len().gt(0)
        | gdf["tourism"].astype(str).str.len().gt(0)
        | gdf["leisure"].astype(str).str.len().gt(0)
        | gdf["historic"].astype(str).str.len().gt(0)
        | gdf["office"].astype(str).str.len().gt(0)
        | gdf["healthcare"].astype(str).str.len().gt(0)
    )
    n_before = len(gdf)
    gdf = gdf[has_signal].copy()
    print(f"   Filtered to {len(gdf):,} POIs with at least one identifying tag "
          f"(dropped {n_before - len(gdf):,} unnamed/untagged features).")

    # Convert from GeoDataFrame to plain DataFrame before pickling. The Oracle
    # treats it as a regular pandas DataFrame anyway and not depending on
    # geopandas at load time avoids version-mismatch headaches.
    df = pd.DataFrame(gdf.drop(columns=[]))

    print(f"💾 Writing {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Done. {len(df):,} POIs · {out_path.stat().st_size/1e6:.1f} MB on disk.")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the per-city OSM POI pickle.")
    ap.add_argument(
        "--city",
        required=True,
        choices=list(CITY_QUERIES.keys()),
        help="Which city to build POIs for.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the output file already exists.",
    )
    args = ap.parse_args()
    build_poi(args.city, force=args.force)
