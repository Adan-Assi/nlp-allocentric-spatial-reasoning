import config
import json
import re
import os
from rvs_parser import load_rvs_accurate

def generate_variants(sample):
    """
    Creates underspecified versions of a single instruction.
    Includes a 'Hard Mode' (mask_both) for later research stages.
    """
    original_text = sample['instruction']
    landmarks = sample['landmarks']
    variants = []

    # 1. Landmark Masking Logic (Improved)
    landmark_masked_text = original_text
    applied_landmark_mask = False
    
    # Extract ALL unique landmark strings from the sample
    # This grabs "church", "garden", "3 World Trade Center", etc.
    all_landmark_names = set()
    for val in sample.get('landmarks', {}).values():
        if isinstance(val, list) and len(val) > 1:
            all_landmark_names.add(str(val[1]))
        elif isinstance(val, str):
            all_landmark_names.add(val)

    # Sort landmarks by length, longest first, to avoid partial masking
    sorted_landmarks = sorted(list(all_landmark_names), key=len, reverse=True)

    for p_name in sorted_landmarks:
        if p_name and p_name != "None" and p_name.lower() in original_text.lower():
            # Use regex for case-insensitive replacement to catch "Church" vs "church"
            pattern = re.compile(re.escape(p_name), re.IGNORECASE)
            single_mask = pattern.sub("[MASK]", original_text)
            
            variants.append({
                "type": "mask_landmark",
                "text": single_mask,
                "removed_element": p_name
            })
            
            landmark_masked_text = pattern.sub("[MASK]", landmark_masked_text)
            applied_landmark_mask = True

    # 2. Directional Masking Logic
    directions_regex = r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\b"
    dir_masked_text = re.sub(directions_regex, "[DIR_MASK]", original_text, flags=re.IGNORECASE)
    
    if dir_masked_text != original_text:
        variants.append({
            "type": "mask_directions",
            "text": dir_masked_text,
            "removed_element": "cardinal_directions"
        })

    # 3. HARD MODE: Mask Both (Landmarks + Directions)
    # This combines both masks into a single, highly ambiguous instruction
    if applied_landmark_mask and dir_masked_text != original_text:
        hard_mode_text = re.sub(directions_regex, "[DIR_MASK]", landmark_masked_text, flags=re.IGNORECASE)
        variants.append({
            "type": "mask_both",
            "text": hard_mode_text,
            "removed_element": "landmarks_and_directions",
            "research_stage": "advanced" # Flag to remind team to skip for now
        })

    return variants

if __name__ == "__main__":
    # We use the keys from CITY_SETTINGS: ['manhattan', 'pittsburgh', 'philadelphia']
    cities_to_process = list(config.CITY_SETTINGS.keys())

    for city in cities_to_process:
        print(f"\n🏙️  Processing: {city.upper()}")
        
        # Set global context so config getters work correctly
        config.CURRENT_CITY = city
        
        # Resolve Paths using config settings
        city_dir = os.path.join(config.BASE_DIR, "data", city)
        input_json = os.path.join(city_dir, config.CITY_SETTINGS[city]["raw_json"])
        graph_path = config.get_graph_path()
        output_json = os.path.join(city_dir, "underspecified_variants.json")

        if not os.path.exists(input_json):
            print(f"❌ Error: {input_json} not found. Skipping...")
            continue

        # Load data using our specialized RVS parser
        print(f"📂 Loading {city} data...")
        data = load_rvs_accurate(input_json, graph_path)
        
        all_experiments = []

        print(f"🎭 Generating variants for {len(data)} samples...")
        for sample in data:
            sample_variants = generate_variants(sample)
            
            # Store everything needed for the Oracle later
            all_experiments.append({
                "sample_id": sample.get('sample_number', 'N/A'),
                "city": city,
                "original_text": sample['instruction'],
                "rvs_start_point": sample['start_point'],
                "rvs_goal_point": sample['goal_point'],
                "variants": sample_variants
            })

        # Save city-specific variants
        with open(output_json, 'w') as f:
            json.dump(all_experiments, f, indent=4)
        
        print(f"✅ Saved {city.upper()} variants to {output_json}")

    print("\n🚀 All cities successfully underspecified!")