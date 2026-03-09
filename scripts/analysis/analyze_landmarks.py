import pandas as pd
import spacy
from collections import Counter
import re

# Load spaCy English model
# Run 'python -m spacy download en_core_web_sm' if not installed
nlp = spacy.load("en_core_web_sm")

def extract_landmark_candidates(file_path):
    # Load your Manhattan RVS dataset
    df = pd.read_parquet(file_path) # or pd.read_csv
    
    all_instructions = df['instruction'].tolist()
    noun_counts = Counter()
    
    print(f"Analyzing {len(all_instructions)} instructions...")

    for doc in nlp.pipe(all_instructions, disable=["ner", "parser"]):
        # We want Nouns (NOUN) and Proper Nouns (PROPN)
        nouns = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        noun_counts.update(nouns)

    # Filter out common non-landmark words (stopwords/navigation verbs)
    blacklist = {'walk', 'turn', 'street', 'avenue', 'block', 'blocks', 'left', 'right', 'north', 'south'}
    
    filtered_counts = {k: v for k, v in noun_counts.items() if k not in blacklist and len(k) > 2}
    
    return Counter(filtered_counts).most_common(50)

if __name__ == "__main__":
    DATA_PATH = "data/rvs_manhattan.parquet"
    top_landmarks = extract_landmark_candidates(DATA_PATH)
    
    print("\n--- Top 50 Potential Landmarks Found ---")
    for word, count in top_landmarks:
        print(f"{word}: {count}")