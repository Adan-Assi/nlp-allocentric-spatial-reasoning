import pandas as pd
import os

def run_quality_audit(file_path):
    df = pd.read_parquet(file_path)
    total = len(df)
    
    stats = df['oracle_label'].value_counts(normalize=True) * 100
    
    print(f"\n--- Quality Audit: {os.path.basename(file_path)} ---")
    print(f"Total Samples: {total}")
    for label, percent in stats.items():
        print(f" - {label}: {percent:.2f}%")
        
    # Heuristic Check
    if stats.get('Answerable', 0) < 70:
        print("⚠️ WARNING: Low Answerability! Check your search radius or OSM tag density.")
    else:
        print("✅ PASS: High Answerability maintained.")

# Run for all cities
for city in ['manhattan', 'pittsburgh', 'philadelphia']:
    path = f"data/{city}/{city}_silver_standard.parquet"
    if os.path.exists(path):
        run_quality_audit(path)