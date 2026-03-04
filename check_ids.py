import pandas as pd

df = pd.read_parquet("data/raw_hf/train.parquet")

print("Columns:")
print(df.columns.tolist())

print("\nFirst row:")
print(df.head(1).to_dict())