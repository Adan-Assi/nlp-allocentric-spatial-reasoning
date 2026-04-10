''' Quality Control Script for Ambiguous Samples in Silver Standard'''

import pandas as pd
import os
import sys

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def run_qc():
    parquet_path = os.path.join(os.path.dirname(config.RVS_DATA_JSON), "manhattan_silver_standard.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Error: {parquet_path} not found.")
        return

    df = pd.read_parquet(parquet_path)
    ambiguous_df = df[df['silver_label'] == 'ambiguous']
    
    print(f"🔍 Analyzing {len(ambiguous_df)} Ambiguous Samples...\n")
    
    for _, row in ambiguous_df.head(20).iterrows():
        print(f"ID: {row['sample_id']}")
        print(f"TEXT: {row['instruction']}") # Now this will exist!
        print("-" * 40)

if __name__ == "__main__":
    run_qc()