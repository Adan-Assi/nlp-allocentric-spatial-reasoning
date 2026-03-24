import json
import pickle
import os

def load_rvs_accurate(json_path, gpickle_path):
    """
    Parses the RVS Manhattan JSON and pairs it with the graph.
    """
    # 1. Load the Oracle Graph
    if not os.path.exists(gpickle_path):
        print(f"Warning: {gpickle_path} not found. Graph will be None.")
        G = None
    else:
        with open(gpickle_path, 'rb') as f:
            G = pickle.load(f)

    # 2. Load the JSON (Handling line-by-line format)
    dataset = []
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return []

    with open(json_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            pivots = item.get('landmarks', {})
            
            # Extract the string names from the pivot lists [coords, name]
            # We use .get() and list slicing to avoid KeyErrors if a pivot is missing
            landmark_info = {
                "main": pivots.get('main_pivot', [None, "None"])[1],
                "near": pivots.get('near_pivot', [None, "None"])[1],
                "beyond": pivots.get('beyond_pivot', [None, "None"])[1]
            }

            dataset.append({
                "sample_number": item.get('rvs_sample_number'),
                "instruction": item.get('content'),
                "goal_point": item.get('rvs_goal_point'),
                "landmarks": landmark_info,
                "graph": G
            })
    
    return dataset

# --- THE TEST/RESULT BLOCK ---
if __name__ == "__main__":
    # Get the directory where rvs_parser.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Correct the paths to point into the data/manhattan/ directory correctly
    # '..' moves up from 'scripts' to 'project_repo', then into 'data/manhattan'
    JSON_PATH = os.path.join(BASE_DIR, "..", "data", "manhattan", "manhattan.json")
    GRAPH_PATH = os.path.join(BASE_DIR, "..", "data", "manhattan", "manhattan_graph.gpickle")

    print(f"--- Running RVS Parser Test ---")
    print(f"Checking JSON at: {os.path.abspath(JSON_PATH)}")
    print(f"Checking Graph at: {os.path.abspath(GRAPH_PATH)}")
    
    results = load_rvs_accurate(JSON_PATH, GRAPH_PATH)

    if results:
        print(f"Total samples parsed: {len(results)}\n")
        
        # Test result for the first 2 samples
        for i in range(2):
            sample = results[i]
            print(f"Result for Sample #{sample['sample_number']}:")
            print(f"  > Text: {sample['instruction'][:100]}...")
            print(f"  > Extracted Landmarks for Masking: {sample['landmarks']}")
            print("-" * 30)
    else:
        print("No results found. Check your file paths.")