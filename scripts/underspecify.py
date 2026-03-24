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

    # 1. Landmark Masking Logic
    landmark_masked_text = original_text
    applied_landmark_mask = False
    
    for p_type, p_name in landmarks.items():
        if p_name and p_name != "None" and p_name in original_text:
            # We track a specific mask for individual variants
            single_mask = original_text.replace(p_name, "[MASK]")
            variants.append({
                "type": f"mask_{p_type}",
                "text": single_mask,
                "removed_element": p_name
            })
            # We also update the 'cumulative' mask for Hard Mode
            landmark_masked_text = landmark_masked_text.replace(p_name, "[MASK]")
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "..", "data", "manhattan", "manhattan.json")
    GRAPH_PATH = os.path.join(BASE_DIR, "..", "data", "manhattan", "manhattan_graph.gpickle")
    OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "manhattan", "underspecified_variants.json")

    data = load_rvs_accurate(JSON_PATH, GRAPH_PATH)
    all_experiments = []

    print("Generating variants (including Hard Mode)...")
    for sample in data[:500]:
        sample_variants = generate_variants(sample)
        all_experiments.append({
            "sample_id": sample['sample_number'],
            "original_text": sample['instruction'],
            "goal_node": sample['goal_point'],
            "variants": sample_variants
        })

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_experiments, f, indent=4)
    
    print(f"Success! Generated variants for {len(all_experiments)} samples.")