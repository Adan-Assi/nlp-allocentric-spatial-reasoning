from datasets import load_dataset
from pathlib import Path

HF_DATASET = "tzufi/RVS"
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset(HF_DATASET)

for split_name in ds.keys():
    df = ds[split_name].to_pandas()
    out_path = OUT_DIR / f"{split_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {split_name}: {len(df)} rows -> {out_path}")
