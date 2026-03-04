from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.constraints.underspec_constraints import generate_variants_for_text

DEFAULT_ENABLED = ["direction", "radius", "proximity"]

def main(
    in_path: str,
    out_path: str,
    text_col: str = "instruction",
    enabled_types: list[str] | None = None,
    max_variants_per_example: int | None = None,
):
    enabled_types = enabled_types or DEFAULT_ENABLED

    in_path = str(in_path)
    out_path = str(out_path)

    df = pd.read_parquet(in_path)
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not found. Columns: {df.columns.tolist()}")

    rows = []

    for idx, r in df.iterrows():
        text = r[text_col]
        if not isinstance(text, str) or not text.strip():
            continue

        variants = generate_variants_for_text(
            text,
            enabled_types=enabled_types,
            drop_sets=None,  # default = all non-empty subsets
        )

        # Optional cap (useful if you later add many constraints)
        if max_variants_per_example is not None:
            variants = variants[:max_variants_per_example]

        # Create one output row per variant
        for j, v in enumerate(variants):
            out_row = {
                # identifiers
                "example_id": r.get("example_id", r.get("key", idx)),
                "variant_id": j,

                # text
                "original_text": text,
                "variant_text": v["variant_text"],

                # constraint metadata
                "enabled_types": list(enabled_types),
                "kept_types": v["kept_types"],
                "dropped_types": v["dropped_types"],
            }

            # Carry through useful fields if they exist
            for col in [
                "region",
                "instruction_id",
                "start_lat", "start_lon",
                "target_lat", "target_lon",
                "target_node_id", "target_node_distance_m",
            ]:
                if col in df.columns:
                    out_row[col] = r[col]

            rows.append(out_row)

    out_df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"✅ Wrote variants: {out_path}")
    print(f"Original examples: {len(df)}")
    print(f"Variant rows: {len(out_df)}")
    print("Dropped-types distribution:")
    print(out_df["dropped_types"].astype(str).value_counts().head(10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--text_col", default="instruction")
    ap.add_argument("--enabled_types", default="direction,radius,proximity")
    ap.add_argument("--max_variants_per_example", type=int, default=None)
    args = ap.parse_args()

    enabled = [s.strip() for s in args.enabled_types.split(",") if s.strip()]
    main(
        in_path=args.in_path,
        out_path=args.out_path,
        text_col=args.text_col,
        enabled_types=enabled,
        max_variants_per_example=args.max_variants_per_example,
    )