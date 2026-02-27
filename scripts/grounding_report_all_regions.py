import pandas as pd
from pathlib import Path

FILES = [
    "data/processed/train_all_regions_grounded.parquet",
    "data/processed/test_all_regions_grounded.parquet",
    "data/processed/validation_seen_all_regions_grounded.parquet",
    "data/processed/validation_unseen_all_regions_grounded.parquet",
]

def main():
    for f in FILES:
        p = Path(f)
        if not p.exists():
            print(f"⚠️ Missing: {p}")
            continue
        df = pd.read_parquet(p)

        print("\n===", p.name, "===")
        print("Total rows:", len(df))
        print("Regions:", df["region"].value_counts().to_dict())
        print("Max dist (m):", float(df["target_node_distance_m"].max()))
        print("Mean dist (m):", float(df["target_node_distance_m"].mean()))

if __name__ == "__main__":
    main()
