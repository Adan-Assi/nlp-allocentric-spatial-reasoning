"""
Pipeline per sample:
  masked instruction → FLAN-T5 → predicted landmark text
                     → oracle.resolve_landmark() → graph node
                     → GPS coordinates
                     → haversine distance to gold_goal
                     → success_250m / success_100m

Input:  reports/llm_audits/LLM_DEGRADATION_INPUT.parquet
Output: reports/llm_audits/LLM_DEGRADATION_RESULTS.parquet
"""

import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
from src.oracle_engine import OracleEngine
import src.utils as utils


def build_oracle_map(cities: list) -> dict:
    """Load one OracleEngine per city for symbolic resolution."""
    oracles = {}
    for city in cities:
        print(f"🔧 Loading oracle for {city}...", flush=True)
        config.CURRENT_CITY = city
        with open(config.get_graph_path(), 'rb') as f:
            G = pickle.load(f)
        oracles[city] = {
            'oracle': OracleEngine(G, config.get_poi_path(),
                                   config.get_node_prefix(), city),
            'G': G,
        }
    return oracles


def resolve_prediction(llm_text: str, city: str,
                       start_node: str, oracle_map: dict):
    """
    Resolve LLM text output → GPS coordinates via oracle.
    Returns (lat, lon) or (None, None) if resolution fails.
    Generic outputs like 'cafe' or 'supermarket' will return None
    when no specific POI can be uniquely identified — this is expected
    and tracked in the resolution rate metric.
    """
    if not llm_text or not llm_text.strip():
        return None, None
    city_data = oracle_map.get(city)
    if not city_data:
        return None, None
    oracle = city_data['oracle']
    G = city_data['G']
    try:
        predicted_node = oracle.resolve_landmark(
            llm_text.strip(),
            context_node=start_node,
            radius_m=config.GLOBAL_SEARCH_HORIZON_METERS,
        )
        if predicted_node and predicted_node in G.nodes:
            node_data = G.nodes[predicted_node]
            return node_data['y'], node_data['x']
    except Exception:
        pass
    return None, None


def run_evaluation(input_path: str,
                   model_name: str = "google/flan-t5-base",
                   batch_size: int = 32,
                   limit: int = None):
    """
    Full evaluation pipeline.
    Set limit=100 for testing, limit=None for full run.
    """
    # --- Load input ---
    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit).copy()
        print(f"Test mode: {limit} samples")
    print(f"📂 Loaded {len(df)} samples | cities: {df['city'].unique().tolist()}")

    # --- Validate required columns ---
    required = ['city', 'masked_instruction', 'variant_type',
                'oracle_label', 'start_node', 'gold_goal_lat', 'gold_goal_lon']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing} — some metrics will be None")

    # --- Load oracles for symbolic resolution ---
    cities = df['city'].unique().tolist()
    oracle_map = build_oracle_map(cities)

    # --- Load LLM ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Loading {model_name} on {device}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Inference + resolution loop ---
    all_results = []
    print(f"🚀 Running inference on {len(df)} samples (batch_size={batch_size})...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i: i + batch_size]

        # Concise prompt — FLAN-T5 works better with short prompts
        prompts = [
            f"Navigation instruction: {row['masked_instruction']}\n"
            f"Destination landmark:"
            for _, row in batch_df.iterrows()
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (_, row), llm_text in zip(batch_df.iterrows(), decoded):
            llm_text = llm_text.strip()

            # Symbolic resolution: LLM text → graph node → GPS
            pred_lat, pred_lon = resolve_prediction(
                llm_text, row['city'],
                row.get('start_node'), oracle_map)

            # Distance + success metrics
            gold_lat = row.get('gold_goal_lat')
            gold_lon = row.get('gold_goal_lon')
            distance_m = None
            success_250m = None
            success_100m = None

            if (pred_lat is not None and pred_lon is not None
                    and gold_lat is not None and gold_lon is not None):
                distance_m = utils.haversine(
                    gold_lat, gold_lon, pred_lat, pred_lon)
                success_250m = distance_m <= 250
                success_100m = distance_m <= 100

            all_results.append({
                # Identity
                'sample_id':            row.get('sample_id'),
                'city':                 row['city'],
                'variant_type':         row.get('variant_type'),
                # Oracle ground truth
                'oracle_label':         row.get('oracle_label'),
                'extracted_category':   row.get('extracted_category'),
                # Input / output
                'masked_instruction':   row['masked_instruction'],
                'llm_output_raw':       llm_text,
                # Symbolic resolution
                'resolution_succeeded': pred_lat is not None,
                'predicted_lat':        pred_lat,
                'predicted_lon':        pred_lon,
                # Evaluation metrics
                'gold_goal_lat':        gold_lat,
                'gold_goal_lon':        gold_lon,
                'distance_m':           distance_m,
                'success_250m':         success_250m,
                'success_100m':         success_100m,
            })

    # --- Save ---
    results_df = pd.DataFrame(all_results)
    report_dir = os.path.join(config.BASE_DIR, "reports", "llm_audits")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "LLM_DEGRADATION_RESULTS.parquet")
    results_df.to_parquet(output_path)

    # --- Summary ---
    print(f"\n✅ Saved {len(results_df)} rows → {output_path}")
    print(f"\n{'='*50}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*50}")

    # Resolution rate — key diagnostic metric
    resolved = results_df['resolution_succeeded'].sum()
    total = len(results_df)
    print(f"\n🔗 Symbolic resolution rate: {resolved}/{total} = {resolved/total:.1%}")
    print(f"   Unresolved (generic/empty LLM output): {total - resolved}")

    # Resolution rate by oracle label
    print(f"\n🔗 Resolution rate by oracle label:")
    print(results_df.groupby('oracle_label')['resolution_succeeded']
          .mean().round(3).to_string())

    # 250m accuracy — only on resolved samples
    resolved_df = results_df[results_df['success_250m'].notna()]
    if len(resolved_df) > 0:
        print(f"\n📈 250m accuracy (resolved only, n={len(resolved_df)}):")
        print(f"   Overall: {resolved_df['success_250m'].mean():.1%}")

        print(f"\n📈 250m accuracy by variant type:")
        print(results_df.groupby('variant_type')['success_250m']
            .apply(lambda x: x.dropna().mean()).round(3).to_string())

        print(f"\n📈 250m accuracy by oracle label:")
        print(results_df.groupby('oracle_label')['success_250m']
            .apply(lambda x: x.dropna().mean()).round(3).to_string())

        if 'extracted_category' in results_df.columns:
            print(f"\n📈 250m accuracy by category (min 10 samples):")
            cat_acc = (results_df.groupby('extracted_category')
                    .agg(mean=('success_250m', lambda x: x.dropna().mean()),
                            count=('success_250m', lambda x: x.dropna().count()))
                    .query('count >= 10')
                    .sort_values('mean', ascending=False))
            print(cat_acc.round(3).to_string())
    else:
        print("\n⚠️  No resolved predictions — check resolve_landmark()")
        print("Sample LLM outputs:")
        print(results_df[['masked_instruction', 'llm_output_raw']].head(10))

    # Sample outputs for inspection
    print(f"\n📋 Sample outputs (last 5):")
    print(results_df[['variant_type', 'oracle_label',
                       'llm_output_raw', 'resolution_succeeded',
                       'success_250m']].tail(5).to_string())


if __name__ == "__main__":
    DEGRADATION_PATH = os.path.join(
        config.BASE_DIR, "reports", "llm_audits",
        "LLM_DEGRADATION_INPUT.parquet")

    if os.path.exists(DEGRADATION_PATH):
        run_evaluation(
            DEGRADATION_PATH,
            model_name="google/flan-t5-base",
            batch_size=32,
            limit=None,   # Usaully None, set to 50 for testing
        )
    else:
        print(f"❌ Input not found: {DEGRADATION_PATH}")
        print("Run build_eval_input.py first.")