import pandas as pd
import os

def analyze_true_coverage(csv_path='all_discovered_landmarks.csv'):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    meta_df = df[df['Type'] == 'Metadata'].copy()
    
    # 1. ROOT CONSOLIDATION
    # We take the LAST word of the metadata string (e.g., "TWO BENCHES" -> "BENCHES")
    # and then we singularize it (simplistically for analysis)
    def get_root(text):
        words = str(text).upper().split()
        if not words: return "UNKNOWN"
        root = words[-1]
        # Basic plural removal
        if root.endswith('ES'): root = root[:-2]
        elif root.endswith('S') and not root.endswith('SS'): root = root[:-1]
        return root

    meta_df['Root_Category'] = meta_df['Landmark'].apply(get_root)

    # 2. Group by Root
    roots = meta_df.groupby('Root_Category')['Count'].sum().reset_index()
    roots = roots.sort_values(by='Count', ascending=False)

    # 3. Coverage Math
    total = roots['Count'].sum()
    roots['Coverage_Pct'] = (roots['Count'] / total) * 100
    roots['Cumulative_Pct'] = roots['Coverage_Pct'].cumsum()

    print("\n" + "="*70)
    print(f"{'RANK':<5} | {'ROOT CATEGORY':<30} | {'COUNT':<8} | {'CUMULATIVE %'}")
    print("="*70)
    for i, row in enumerate(roots.head(40).itertuples(), 1):
        print(f"{i:<5} | {row.Root_Category:<30} | {row.Count:<8} | {row.Cumulative_Pct:.2f}%")
    
    print("="*70)
    print(f"📊 By mapping these 40 ROOTS, you cover {roots.head(40)['Coverage_Pct'].sum():.2f}% of metadata instances.")
    print("💡 Note: Coverage of INSTRUCTIONS will be significantly higher than instance coverage.")

if __name__ == "__main__":
    analyze_true_coverage()