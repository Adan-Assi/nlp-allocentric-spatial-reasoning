import os
import pandas as pd
import config
from src.extraction_utils import extract_rvs_target

def debug_philly():
    # 1. Path to Philly Data
    json_path = os.path.join(config.BASE_DIR, "data/philadelphia/philadelphia.json")
    
    print(f"🔍 Loading Philly Dataset: {json_path}")
    if not os.path.exists(json_path):
        print("❌ Error: Path not found.")
        return

    df = pd.read_json(json_path, lines=True)
    
    # 2. Track stats
    unknown_count = 0
    total = len(df)
    
    print(f"\n--- EXTRACTION RESULTS (Top 50 Samples) ---\n")
    print(f"{'ID':<6} | {'CATEGORY':<12} | {'NOUN':<15} | {'DIR':<4} | {'INSTRUCTION'}")
    print("-" * 100)

    for i, row in df.iterrows():
        instruction = row['content']
        sample_id = row.get('key', 'N/A')
        
        # Run the actual utility used by the solver
        category, raw_noun, target_dir = extract_rvs_target(instruction)
        
        if category == "UNKNOWN":
            unknown_count += 1
            
        # Print first 50 to see the patterns
        if i < 50:
            print(f"{sample_id:<6} | {str(category):<12} | {str(raw_noun):<15} | {str(target_dir):<4} | {instruction[:60]}...")

    # 3. Final Stats
    print("-" * 100)
    print(f"📊 SUMMARY FOR PHILADELPHIA:")
    print(f"Total Samples: {total}")
    print(f"Unknown Categories: {unknown_count} ({(unknown_count/total)*100:.2f}%)")
    print(f"Mapped Categories: {total - unknown_count}")

if __name__ == "__main__":
    debug_philly()