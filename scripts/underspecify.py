import sys
import json
import re
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_variants(sample_row):
    """
    FIXED: Uses extracted_noun from the Silver Standard metadata 
    to ensure landmark masking works in sparse cities like Philadelphia.
    """
    original_text = sample_row['instruction']
    variants = []
    
    # 1. Use the 'extracted_noun' (the generic or proper name found by the Solver)
    # This was previously missing in the JSON-only logic
    target_landmark = sample_row.get('extracted_noun')
    applied_landmark_mask = False

    if target_landmark and str(target_landmark).lower() in original_text.lower():
        pattern = re.compile(re.escape(str(target_landmark)), re.IGNORECASE)
        landmark_masked_text = pattern.sub("[MASK]", original_text)
        
        variants.append({
            "type": "mask_landmark",
            "text": landmark_masked_text,
            "removed_element": target_landmark
        })
        applied_landmark_mask = True
    else:
        # Fallback to the original text if no noun was extracted
        landmark_masked_text = original_text

    # 2. Directional Masking (Kept the original regex because it was working fine)
    directions_regex = r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\b"
    dir_masked_text = re.sub(directions_regex, "[DIR_MASK]", original_text, flags=re.IGNORECASE)
    
    if dir_masked_text != original_text:
        variants.append({
            "type": "mask_directions",
            "text": dir_masked_text,
            "removed_element": "cardinal_directions"
        })

    # 3. Mask Both (Now works in Philly because applied_landmark_mask will be True)
    if applied_landmark_mask and dir_masked_text != original_text:
        hard_mode_text = re.sub(directions_regex, "[DIR_MASK]", landmark_masked_text, flags=re.IGNORECASE)
        variants.append({
            "type": "mask_both",
            "text": hard_mode_text,
            "removed_element": "landmarks_and_directions"
        })

    return variants


if __name__ == "__main__":
    cities_to_process = list(config.CITY_SETTINGS.keys())

    for city in cities_to_process:
        print(f"\n🏙️  Processing: {city.upper()}")
        config.CURRENT_CITY = city
        
        # FIXED: Points to the Silver Standard Parquet instead of Raw JSON
        city_dir = os.path.join(config.BASE_DIR, "data", city)
        input_parquet = os.path.join(city_dir, f"{city}_silver_standard.parquet")
        output_json = os.path.join(city_dir, "underspecified_variants.json")

        if not os.path.exists(input_parquet):
            print(f"❌ Error: {input_parquet} not found. Run batch_labeling.py first!")
            continue

        # Load the Silver Standard
        print(f"📂 Loading {city} Silver Standard...")
        df = pd.read_parquet(input_parquet)
        
        all_experiments = []

        # We only want to generate variants for 'Answerable' rows to keep the test clean
        answerable_df = df[df['oracle_label'] == 'Answerable']

        print(f"🎭 Generating variants for {len(answerable_df)} answerable samples...")
        for _, row in answerable_df.iterrows():
            # Convert row to dict for the generator
            sample_dict = row.to_dict()
            sample_variants = generate_variants(sample_dict)
            
            all_experiments.append({
                "sample_id": sample_dict.get('sample_id', 'N/A'),
                "city": city,
                "original_text": sample_dict.get('instruction', ''),
                "gold_goal_node": sample_dict.get('gold_goal_node'),
                "variants": sample_variants
            })

        # Save city-specific variants
        with open(output_json, 'w') as f:
            json.dump(all_experiments, f, indent=4)
        
        print(f"✅ Saved {city.upper()} variants to {output_json}")

    print("\n🚀 All cities successfully underspecified via Silver Standard logic!")