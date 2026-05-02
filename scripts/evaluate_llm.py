import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import BASE_DIR

def run_evaluation(input_path, model_name="google/flan-t5-base", batch_size=8, limit=None):
    """
    Runs batched LLM inference.
    :param limit: Set to an integer (e.g. 100) for testing, or None for the full dataset.
    """
    # Load Data
    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit).copy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # --- Loading Block ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # Ensure tokenizer has a padding token (T5 usually uses eos_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []

    # --- Batched Inference Loop ---
    print(f"Evaluating {len(df)} samples (Batch Size: {batch_size})...")
    
    for i in tqdm(range(0, len(df), batch_size)):
        # Extract batch
        batch_df = df.iloc[i : i + batch_size]
        
        # Prepare prompts
        prompts = [
            f"Task: Follow the navigation instructions in {row['city']}.\n"
            f"Instructions: {row['instruction']}\n"
            f"Question: What is the specific landmark or street name of the destination?\n"
            f"Answer:" for _, row in batch_df.iterrows()
        ]

        # Tokenize batch with padding
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        # Generate for the whole batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=25, 
                do_sample=False
            )
        
        # Decode the batch
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend([text.strip() for text in decoded])

    # --- Save Logic ---
    df['llm_output_raw'] = results

    report_dir = os.path.join(BASE_DIR, "reports", "llm_audits")
    os.makedirs(report_dir, exist_ok=True)

    output_path = os.path.join(report_dir, "LLM_TEST_RESULTS.parquet")
    df.to_parquet(output_path)

    print(f"\nSuccess! Results saved to: {output_path}")
    print("\nSample Output (Last 5 rows):")
    print(df[['instruction', 'llm_output_raw']].tail())

if __name__ == "__main__":
    GOLD_PATH = os.path.join(BASE_DIR, "data", "RVS_MASTER_GOLD_HYDRATED.parquet")
    
    if os.path.exists(GOLD_PATH):
        # Change limit=100 to limit=None and batch_size=8 to batch_size=32 when we are ready for the full 7,000+ run
        run_evaluation(GOLD_PATH, batch_size=32, limit=None)
    else:
        print(f"Error: Could not find Gold Master at {GOLD_PATH}")