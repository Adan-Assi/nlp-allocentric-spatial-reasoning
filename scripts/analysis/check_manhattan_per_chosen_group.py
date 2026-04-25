import json
import os
from pathlib import Path

# 1. SETUP PATHS
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "manhattan" / "manhattan.json"

TOP_ROOTS = {
    "BUILDING", "CHURCH", "SHOP", "PARK", "RESTAURANT", "LIBRARY", "HOTEL", 
    "BENCH", "GARDEN", "THEATRE", "OFFICE", "MUSEUM", "STREET", "PARKING", 
    "TOWER", "SQUARE", "CENTER", "PHARMACY", "SCHOOL", "AVENUE", "RENTAL", 
    "BANK", "CAFE", "STATION", "PLAZA", "CINEMA", "DISTRICT", "ATTRACTION", 
    "HOUSE", "ATM", "CLINIC", "BRIDGE", "ART", "FOUNTAIN", "UNIVERSITY", 
    "CENTRE", "THEATER", "PLAYGROUND", "MEMORIAL", "BROADWAY"
}

def check_instruction_coverage(json_path=DEFAULT_DATA_PATH):
    if not os.path.exists(json_path):
        print(f"❌ Error: Could not find file at {json_path}")
        return

    total_instructions = 0
    covered_instructions = 0
    uncovered_examples = []

    print(f"Reading JSONL file: {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            
            try:
                entry = json.loads(line)
                total_instructions += 1
                
                # RVS JSONL usually stores landmarks inside 'clues' or 'tokens' 
                # or a specific 'metadata' field. 
                # Let's try to find them dynamically or check 'landmarks'
                landmarks = entry.get('landmarks', [])
                
                # SANITY CHECK: Print the first line's data structure
                if total_instructions == 1:
                    print(f"DEBUG: Keys found in JSON: {list(entry.keys())}")
                    print(f"DEBUG: Sample landmarks found: {landmarks}")

                is_covered = False
                
                if isinstance(landmarks, dict):
                    for key, value in landmarks.items():
                        # value[1] is the actual name (e.g., 'garden')
                        if isinstance(value, list) and len(value) > 1:
                            clean_landmark = str(value[1]).upper().strip()
                            if any(root in clean_landmark for root in TOP_ROOTS):
                                is_covered = True
                                break
                
                #if landmarks:
                #    for landmark in landmarks:
                #        clean_landmark = str(landmark).upper().strip()
                #        if any(root in clean_landmark for root in TOP_ROOTS):
                #            is_covered = True
                #            break
                
                if is_covered:
                    covered_instructions += 1
                else:
                    if len(uncovered_examples) < 5 and landmarks:
                        uncovered_examples.append(landmarks)

            except json.JSONDecodeError:
                continue

    if total_instructions == 0:
        print("❌ ERROR: No instructions were processed. Check if the file is empty or path is wrong.")
        return

    coverage_pct = (covered_instructions / total_instructions) * 100

    print("\n" + "="*50)
    print("🎯 ACTUAL PROJECT METRIC: INSTRUCTION COVERAGE")
    print("="*50)
    print(f"Total Instructions: {total_instructions}")
    print(f"Covered: {covered_instructions}")
    print(f"Final Coverage: {coverage_pct:.2f}%")
    print("="*50)

    if not uncovered_examples and coverage_pct < 100:
        print("💡 Note: Some instructions had NO landmarks listed in the JSON at all.")

if __name__ == "__main__":
    check_instruction_coverage()