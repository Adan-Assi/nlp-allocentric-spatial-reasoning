"""
evaluate_llm_masked.py
LLM stress-test on the masked variants produced by underspecify.py.

For each variant of each Answerable instruction, asks the LLM to predict
one of: Answerable, Ambiguous, Contradictory. The variant's text has the
landmark, the directions, or both replaced with [MASK]/[DIR_MASK]. The
research question:

    As we strip information from the instruction, does the LLM keep
    confidently picking a single destination (over-confident), or does it
    correctly recognize the rising ambiguity?

Reads:
    data/<city>/underspecified_variants.json   (output of underspecify.py)
    data/<city>/<city>_silver_standard.parquet (for the original oracle_label)

Writes:
    reports/llm_audits/llm_predictions_masked.parquet

Each output row carries: city, sample_id, variant_type, masked_instruction,
oracle_label_original (the label the oracle gave the un-masked text), and
the LLM's prediction on the masked text.

Usage:
    python scripts/evaluate_llm_masked.py
    python scripts/evaluate_llm_masked.py --city manhattan
    python scripts/evaluate_llm_masked.py --limit 100
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
# Re-use the prompt builder, parser and inference loop from the baseline
# script — same model, same parsing logic, only the input source differs.
from scripts.evaluate_llm import build_prompt, parse_label, run_evaluation  # noqa: E402


def _flatten_variants(city: str) -> pd.DataFrame:
    """Read underspecify.py's per-city JSON and emit one row per variant."""
    path = os.path.join(config.BASE_DIR, "data", city, "underspecified_variants.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/underspecify.py first."
        )
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        sid = rec.get("sample_id")
        original = rec.get("original_text")
        for v in rec.get("variants", []):
            rows.append({
                "sample_id": sid,
                "city": city,
                "variant_type": v.get("type"),
                "removed_element": v.get("removed_element"),
                "original_text": original,
                "instruction": v.get("text"),  # masked text — what the LLM sees
            })
    df = pd.DataFrame(rows)
    print(f"📂 Loaded {len(df):,} variant rows from {city}")
    return df


def _attach_original_oracle_label(variants: pd.DataFrame) -> pd.DataFrame:
    """Join each variant back to its source row's oracle_label.

    underspecify.py only generates variants for Answerable rows, so the
    original label is almost always 'Answerable'. We still attach it
    explicitly so the analysis notebook can verify and so the join is
    auditable.
    """
    frames = []
    for c in variants["city"].unique():
        path = os.path.join(config.BASE_DIR, "data", c, f"{c}_silver_standard.parquet")
        if not os.path.exists(path):
            print(f"⚠️  {path} missing — variants from {c} will lack oracle_label_original")
            continue
        ss = pd.read_parquet(path)[["sample_id", "oracle_label"]]
        ss = ss.rename(columns={"oracle_label": "oracle_label_original"})
        ss["city"] = c
        frames.append(ss)
    if not frames:
        return variants
    silver = pd.concat(frames, ignore_index=True)
    return variants.merge(silver, on=["sample_id", "city"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None,
                    help="manhattan/pittsburgh/philadelphia. Default: all 3.")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N variants (smoke test).")
    ap.add_argument("--out", default=os.path.join(
        config.BASE_DIR, "reports", "llm_audits", "llm_predictions_masked.parquet"))
    args = ap.parse_args()

    cities = [args.city] if args.city else list(config.CITY_SETTINGS.keys())
    frames = []
    for c in cities:
        try:
            frames.append(_flatten_variants(c))
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
    if not frames:
        raise SystemExit("No variant files found. Run scripts/underspecify.py first.")

    df = pd.concat(frames, ignore_index=True)
    df = _attach_original_oracle_label(df)

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"🔍 Smoke-test mode: limited to {len(df)} rows")

    # `run_evaluation` expects an 'instruction' column and a 'city' column —
    # both already populated by _flatten_variants.
    out = run_evaluation(
        df,
        model_name=args.model,
        batch_size=args.batch_size,
        out_path=args.out,
    )

    # Summary by variant type — the headline degradation story.
    if "variant_type" in out.columns and "oracle_label_original" in out.columns:
        print("\n📊 Per-variant agreement with original oracle label:")
        for vtype, sub in out.groupby("variant_type"):
            agree = (sub["llm_prediction"] == sub["oracle_label_original"]).mean()
            print(f"  {vtype:<18s}  n={len(sub):>5,}  agreement={100*agree:.2f}%")


if __name__ == "__main__":
    main()
