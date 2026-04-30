"""
Consolidates underspecified_variants_labeled.json files from all cities
into a single LLM_DEGRADATION_INPUT.parquet for evaluate_llm_masked.py's input.

Run AFTER label_variants.py, BEFORE evaluate_llm_masked.py.
"""
import sys, os, json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

rows = []
for city in config.CITY_SETTINGS.keys():
    path = os.path.join(config.BASE_DIR, "data", city,
                        "underspecified_variants_labeled.json")
    if not os.path.exists(path):
        print(f"⚠️  Missing: {path} — skipping {city}")
        continue

    with open(path) as f:
        experiments = json.load(f)

    city_rows = 0
    for exp in experiments:
        for variant in exp['variants']:
            rows.append({
                # Identity
                'sample_id':           exp['sample_id'],
                'city':                exp['city'],
                # Instruction
                'original_text':       exp['original_text'],
                'masked_instruction':  variant['text'],
                'variant_type':        variant['type'],
                'removed_element':     variant['removed_element'],
                # Solver metadata
                'extracted_category':  exp.get('extracted_category'),
                'extracted_direction': exp.get('extracted_direction'),
                'extracted_noun':      exp.get('extracted_noun'),
                # Graph anchors
                'start_node':          exp.get('start_node'),
                'gold_goal_node':      exp.get('gold_goal_node'),
                'gold_goal_lat':       exp.get('gold_goal_lat'),
                'gold_goal_lon':       exp.get('gold_goal_lon'),
                # Oracle 2 ground truth
                'oracle_label':        variant.get('oracle_label'),
                'reachable_candidate_count': variant.get('reachable_candidate_count', 0),
            })
        city_rows += len(exp['variants'])
    print(f"✅ {city}: {len(experiments)} experiments, {city_rows} variants")

df = pd.DataFrame(rows)
out = os.path.join(config.BASE_DIR, "reports", "llm_audits",
                   "LLM_DEGRADATION_INPUT.parquet")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_parquet(out)

print(f"\n✅ Total: {len(df)} rows → {out}")
print(f"Cities: {df['city'].value_counts().to_dict()}")
print(f"Variant types: {df['variant_type'].value_counts().to_dict()}")
print(f"Oracle labels: {df['oracle_label'].value_counts().to_dict()}")
print(f"gold_goal_lat nulls: {df['gold_goal_lat'].isna().sum()}")