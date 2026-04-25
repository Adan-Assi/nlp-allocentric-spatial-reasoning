import pandas as pd
import os

def analyze_categories(csv_path='all_discovered_landmarks.csv'):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Filter for Metadata (Ground Truth Categories)
    # The Metadata column in RVS contains the semantic labels we want to map to OSM.
    meta_df = df[df['Type'] == 'Metadata'].copy()

    # 3. Clean up the Landmark names
    # Normalize to uppercase and strip whitespace
    meta_df['Landmark'] = meta_df['Landmark'].str.upper().str.strip()

    # 4. Filter out specific "Named Entities" vs "Generic Categories"
    # Rule of thumb: Generic categories are usually short (1-2 words).
    # Specific buildings (e.g., "7 WORLD TRADE CENTER") are longer.
    meta_df['Word_Count'] = meta_df['Landmark'].apply(lambda x: len(str(x).split()))
    
    # We prioritize shorter names as they represent "Categories" (Pharmacy, Church, etc.)
    # but we still want to see the high-frequency specific targets.
    categories = meta_df.groupby('Landmark')['Count'].sum().reset_index()
    categories = categories.sort_values(by='Count', ascending=False)

    # 5. Calculate Cumulative Coverage
    total_samples = categories['Count'].sum()
    categories['Coverage_Pct'] = (categories['Count'] / total_samples) * 100
    categories['Cumulative_Percentage'] = categories['Coverage_Pct'].cumsum()

    # 6. Extract Top 40
    top_40 = categories.head(40)

    # Output the results
    print("\n" + "="*70)
    print(f"{'RANK':<5} | {'LANDMARK CATEGORY':<30} | {'COUNT':<8} | {'CUMULATIVE %'}")
    print("="*70)
    
    for i, row in enumerate(top_40.itertuples(), 1):
        print(f"{i:<5} | {row.Landmark:<30} | {row.Count:<8} | {row.Cumulative_Percentage:.2f}%")
    print("="*70)
    print(f"✅ Analysis complete. The Top 40 categories cover {top_40['Coverage_Pct'].sum():.2f}% of all landmarks.")
    print("👉 Use these strings to build your LANDMARK_GROUPS in config.py.")

if __name__ == "__main__":
    analyze_categories()