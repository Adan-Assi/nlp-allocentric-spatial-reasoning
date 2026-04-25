import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/splits")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "test", "validation_seen", "validation_unseen"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # required columns from tzufi/RVS
    required = ["content", "rvs_goal_point", "key", "region", "rvs_start_point"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    out = pd.DataFrame()
    out["instruction_id"] = df["key"].astype(str)
    out["region"] = df["region"].astype(str).str.lower()
    out["instruction"] = df["content"].astype(str)

    # goal point: [lat, lon]
    out["target_lat"] = df["rvs_goal_point"].apply(lambda x: float(x[0]))
    out["target_lon"] = df["rvs_goal_point"].apply(lambda x: float(x[1]))

    # start point (optional but useful later)
    out["start_lat"] = df["rvs_start_point"].apply(lambda x: float(x[0]))
    out["start_lon"] = df["rvs_start_point"].apply(lambda x: float(x[1]))

    # globally unique id
    out["example_id"] = out["region"] + "_" + out["instruction_id"]
    return out

def main():
    for split in SPLITS:
        p = RAW_DIR / f"{split}.parquet"
        if not p.exists():
            print(f"⚠️  Skip missing split file: {p}")
            continue
        df = pd.read_parquet(p)
        norm = normalize(df)
        out_path = OUT_DIR / f"{split}_normalized.parquet"
        norm.to_parquet(out_path, index=False)
        print(f"✅ Normalized {split}: {len(norm)} rows -> {out_path}")

if __name__ == "__main__":
    main()
