"""
prepare_city_jsons.py

Builds the per-city raw RVS instructions JSONL the batch_labeling script reads
from data/<city>/<city>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

HF_DATASET = "tzufi/RVS"

DEFAULT_COL_CANDIDATES = {
    "content": ["content", "instruction", "text"],
    "rvs_start_point": ["rvs_start_point", "start_point"],
    "rvs_start_node": ["rvs_start_node", "start_node", "start_osmid"],  # optional
    "rvs_sample_number": ["rvs_sample_number", "sample_number", "sample_id", "id", "key"],
    "rvs_goal_point": ["rvs_goal_point", "goal_point", "target_point"],
    "region": ["region", "city"],
}


def _pick_column(df, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_region(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    for known in ("manhattan", "pittsburgh", "philadelphia"):
        if known in s:
            return known
    return s


def write_city_jsons(
    splits: list[str] | None = None,
    only_city: str | None = None,
    overrides: dict[str, str] | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("❌ Please install: pip install datasets")

    overrides = overrides or {}
    print(f"📥 Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)

    splits = splits or list(ds.keys())
    print(f"   Using splits: {splits}")

    rows_by_city = {
        "manhattan": [],
        "pittsburgh": [],
        "philadelphia": [],
    }

    for split_name in splits:
        if split_name not in ds:
            continue

        df = ds[split_name].to_pandas()

        resolved = {}
        for canon, candidates in DEFAULT_COL_CANDIDATES.items():
            override = overrides.get(canon)
            if override and override in df.columns:
                resolved[canon] = override
            else:
                resolved[canon] = _pick_column(df, candidates)

        # ✅ FIX: only require essential columns
        required_cols = [
            "content",
            "rvs_start_point",
            "rvs_sample_number",
            "rvs_goal_point",
            "region",
        ]

        missing = [k for k in required_cols if resolved.get(k) is None]
        if missing:
            print(f"⚠️ Missing {missing} in split {split_name}, skipping")
            continue

        for _, row in df.iterrows():
            region = _normalize_region(row[resolved["region"]])
            if region not in rows_by_city:
                continue
            if only_city and region != only_city:
                continue

            start_point = row[resolved["rvs_start_point"]]
            goal_point = row[resolved["rvs_goal_point"]]

            rows_by_city[region].append({
                "content": row[resolved["content"]],
                "rvs_start_point": list(start_point) if hasattr(start_point, "__iter__") else start_point,
                "rvs_start_node": (
                    str(row[resolved["rvs_start_node"]])
                    if resolved.get("rvs_start_node")
                    else None
                ),
                "rvs_sample_number": (
                    int(row[resolved["rvs_sample_number"]])
                    if str(row[resolved["rvs_sample_number"]]).isdigit()
                    else str(row[resolved["rvs_sample_number"]])
                ),
                "rvs_goal_point": list(goal_point) if hasattr(goal_point, "__iter__") else goal_point,
                "split": split_name,
            })

    # Write files
    for city, rows in rows_by_city.items():
        if only_city and city != only_city:
            continue

        out_dir = Path(config.BASE_DIR) / "data" / city
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{city}.json"

        if not rows:
            print(f"⚠️ No data for {city}")
            continue

        print(f"💾 Writing {len(rows):,} rows → {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"✅ Done: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["manhattan", "pittsburgh", "philadelphia"])
    ap.add_argument("--splits", nargs="*", default=None)

    for canon in DEFAULT_COL_CANDIDATES.keys():
        ap.add_argument(f"--{canon.replace('_', '-')}", dest=canon, default=None)

    args = ap.parse_args()

    overrides = {
        canon: getattr(args, canon)
        for canon in DEFAULT_COL_CANDIDATES.keys()
        if getattr(args, canon)
    }

    write_city_jsons(
        splits=args.splits,
        only_city=args.city,
        overrides=overrides,
    )