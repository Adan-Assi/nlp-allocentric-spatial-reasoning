"""
Pipeline per sample (FLAN-T5-Large):
  masked instruction → FLAN-T5-large → predicted landmark text
                     → oracle.resolve_landmark() → graph node
                     → GPS coordinates
                     → haversine distance to gold_goal
                     → success_250m / success_100m

Input:  reports/llm_audits/LLM_DEGRADATION_INPUT.parquet
Output: reports/llm_audits/LLM_DEGRADATION_RESULTS_LARGE.parquet

Comparison target: LLM_DEGRADATION_RESULTS.parquet (FLAN-T5-base)
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
                   model_name: str = "google/flan-t5-large",
                   batch_size: int = 16,
                   limit: int = None):
    """
    Full evaluation pipeline for FLAN-T5-large.
    batch_size reduced to 16 (vs 32 for base) due to larger model size.
    Set limit=50 for testing, limit=None for full run.
    """
    # --- Load input ---
    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit).copy()
        print(f"⚠️  Test mode: {limit} samples")
    print(f"📂 Loaded {len(df)} samples | cities: {df['city'].unique().tolist()}")

    required = ['city', 'masked_instruction', 'variant_type',
                'oracle_label', 'start_node', 'gold_goal_lat', 'gold_goal_lon']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")

    cities = df['city'].unique().tolist()
    oracle_map = build_oracle_map(cities)

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

    all_results = []
    print(f"🚀 Running inference on {len(df)} samples (batch_size={batch_size})...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i: i + batch_size]

        # Identical prompt to base evaluation — controlled comparison
        prompts = [
            f"Navigation instruction: {row['masked_instruction']}\n"
            f"Identify the destination landmark by name. "
            f"Do not repeat [MASK] or [DIR_MASK].\n"
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

            pred_lat, pred_lon = resolve_prediction(
                llm_text, row['city'],
                row.get('start_node'), oracle_map)

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
                'sample_id':            row.get('sample_id'),
                'city':                 row['city'],
                'variant_type':         row.get('variant_type'),
                'oracle_label':         row.get('oracle_label'),
                'extracted_category':   row.get('extracted_category'),
                'masked_instruction':   row['masked_instruction'],
                'llm_output_raw':       llm_text,
                'resolution_succeeded': pred_lat is not None,
                'predicted_lat':        pred_lat,
                'predicted_lon':        pred_lon,
                'gold_goal_lat':        gold_lat,
                'gold_goal_lon':        gold_lon,
                'distance_m':           distance_m,
                'success_250m':         success_250m,
                'success_100m':         success_100m,
            })

    results_df = pd.DataFrame(all_results)
    report_dir = os.path.join(config.BASE_DIR, "reports", "llm_audits")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "LLM_DEGRADATION_RESULTS_LARGE.parquet")
    results_df.to_parquet(output_path)

    print(f"\n✅ Saved {len(results_df)} rows → {output_path}")
    print(f"\n{'='*50}")
    print(f"📊 EVALUATION SUMMARY — FLAN-T5-large")
    print(f"{'='*50}")

    resolved = results_df['resolution_succeeded'].sum()
    total = len(results_df)
    print(f"\n🔗 Symbolic resolution rate: {resolved}/{total} = {resolved/total:.1%}")
    print(f"   Unresolved: {total - resolved}")

    print(f"\n🔗 Resolution rate by oracle label:")
    print(results_df.groupby('oracle_label')['resolution_succeeded']
          .mean().round(3).to_string())

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

    # --- Comparison with base results ---
    base_path = os.path.join(report_dir, "LLM_DEGRADATION_RESULTS.parquet")
    if os.path.exists(base_path):
        base_df = pd.read_parquet(base_path)
        base_resolved = base_df['resolution_succeeded'].sum()
        base_total = len(base_df)
        base_acc = base_df['success_250m'].dropna().mean()

        print(f"\n{'='*50}")
        print(f"📊 BASE vs LARGE COMPARISON")
        print(f"{'='*50}")
        print(f"{'Metric':<30} {'Base (250M)':>12} {'Large (780M)':>13}")
        print(f"{'-'*55}")
        print(f"{'Resolution rate':<30} "
              f"{base_resolved/base_total:>12.1%} "
              f"{resolved/total:>13.1%}")
        print(f"{'Acc@250m (resolved)':<30} "
              f"{base_acc:>12.1%} "
              f"{resolved_df['success_250m'].mean():>13.1%}")

        for label in ['Answerable', 'Ambiguous', 'Contradictory']:
            base_label_acc = base_df[base_df['oracle_label'] == label][
                'success_250m'].dropna().mean()
            large_label_acc = results_df[results_df['oracle_label'] == label][
                'success_250m'].dropna().mean()
            print(f"{'Acc@250m ' + label:<30} "
                  f"{base_label_acc:>12.1%} "
                  f"{large_label_acc:>13.1%}")
    else:
        print(f"\n⚠️  Base results not found for comparison at {base_path}")

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
            model_name="google/flan-t5-large",
            batch_size=16,
            limit=None,  # set to 50 for testing
        )
    else:
        print(f"❌ Input not found: {DEGRADATION_PATH}")
        print("Run build_eval_input.py first.")