import json
import spacy
import os
from collections import Counter
from tqdm import tqdm

def run_extraction():
    print("🚀 Initializing Hybrid NLP & Metadata Engine...")
    # Load spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    
    # Path configuration - Updated to your local path
    DATA_DIR = os.path.join('data', 'manhattan') 
    TRAIN_PATH = os.path.join(DATA_DIR, 'manhattan.json') 

    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: Could not find {TRAIN_PATH}")
        return

    instruction_nlp_counts = Counter()
    metadata_exact_counts = Counter()
    instructions = []
    
    BLACKLIST = {
        'BLOCK', 'TURN', 'WAY', 'LEFT', 'RIGHT', 'STRAIGHT', 'AHEAD',
        'METERS', 'FEET', 'SIDE', 'CORNER', 'END', 'DIRECTION', 'FRONT',
        'NORTH', 'SOUTH', 'EAST', 'WEST', 'NORTHWEST', 'NORTHEAST', 'SOUTHWEST', 'SOUTHEAST'
    }

    print(f"📖 Reading Manhattan JSONL...")
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if 'content' in entry:
                    instructions.append(entry['content'])
                if 'landmarks' in entry:
                    for value in entry['landmarks'].values():
                        if isinstance(value, list) and len(value) > 1:
                            raw_name = str(value[1]).upper().strip()
                            if len(raw_name) > 2:
                                metadata_exact_counts.update([raw_name])
            except json.JSONDecodeError:
                continue

    print(f"🧠 NLP Analysis of {len(instructions)} instructions...")
    # Fix: Ensure all arguments after the list are keyword arguments
    for doc in tqdm(nlp.pipe(instructions, batch_size=100), total=len(instructions)):
        for ent in doc.ents:
            name = ent.text.upper().strip()
            if name not in BLACKLIST and len(name) > 3:
                instruction_nlp_counts.update([name])

    # --- REPORT GENERATION ---
    print("\n" + "="*60)
    print(f"{'TOP METADATA TARGETS (Ground Truth)':<45} | {'COUNT'}")
    print("="*60)
    for item, count in metadata_exact_counts.most_common(20):
        print(f"{item:<45} | {count}")

    print("\n" + "="*60)
    print(f"{'TOP NLP EXTRACTED (User Descriptions)':<45} | {'COUNT'}")
    print("="*60)
    for item, count in instruction_nlp_counts.most_common(20):
        print(f"{item:<45} | {count}")

    # ... (inside run_extraction after the loops) ...

    # 1. Save EVERY found landmark to a CSV for your teammate
    import pandas as pd
    
    all_data = []
    for name, count in metadata_exact_counts.items():
        all_data.append({'Type': 'Metadata', 'Landmark': name, 'Count': count})
    for name, count in instruction_nlp_counts.items():
        all_data.append({'Type': 'NLP_User', 'Landmark': name, 'Count': count})
        
    df = pd.DataFrame(all_data)
    df.to_csv('all_discovered_landmarks.csv', index=False)
    
    print(f"\n✅ Analysis complete! {len(metadata_exact_counts)} unique metadata targets found.")
    print(f"✅ {len(instruction_nlp_counts)} unique user-described entities found.")
    print("📁 All results saved to 'all_discovered_landmarks.csv'")

if __name__ == "__main__":
    run_extraction()