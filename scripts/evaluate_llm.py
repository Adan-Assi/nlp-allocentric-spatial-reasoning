"""
evaluate_llm.py
LLM evaluation on the un-masked Silver Standard.

For each labeled instruction, asks the LLM to predict one of:
    Answerable, Ambiguous, Contradictory
and saves both the oracle label and the LLM prediction so the confusion
matrix is a single merge away in the analysis notebook.

Reads:
    data/<city>/<city>_silver_standard.parquet  (output of batch_labeling.py)

Writes:
    reports/llm_audits/llm_predictions_baseline.parquet

Usage:
    python scripts/evaluate_llm.py                          # all 3 cities
    python scripts/evaluate_llm.py --city manhattan         # single city
    python scripts/evaluate_llm.py --limit 50               # quick smoke test
    python scripts/evaluate_llm.py --model google/flan-t5-base
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config

LABEL_SET = ("Answerable", "Ambiguous", "Contradictory")


def build_prompt(instruction: str, city: str) -> str:
    """3-class classification prompt — the project's research question.

    Kept short on purpose: flan-t5 follows tight instructions better than
    long ones, and we want every bit of the budget for the actual text.
    """
    return (
        f"Classify this {city} navigation instruction as exactly one of: "
        f"Answerable (one valid destination), "
        f"Ambiguous (multiple valid destinations), "
        f"Contradictory (no valid destination).\n\n"
        f"Instruction: {instruction}\n\n"
        f"Classification:"
    )


def parse_label(raw: str) -> str | None:
    """Normalize free-form model output to one of LABEL_SET (or None)."""
    if not raw:
        return None
    s = raw.strip().lower()
    # Order matters: 'answerable' must be checked before 'ambiguous' is not
    # a substring concern, but 'contradictory' should be checked before
    # 'contra' which it isn't a problem here. Just iterate.
    for canonical in LABEL_SET:
        if canonical.lower() in s:
            return canonical
    return None


def load_silver_standards(city: str | None) -> pd.DataFrame:
    """Concat the silver-standard parquets that exist on disk."""
    cities = [city] if city else list(config.CITY_SETTINGS.keys())
    frames = []
    for c in cities:
        path = os.path.join(config.BASE_DIR, "data", c, f"{c}_silver_standard.parquet")
        if not os.path.exists(path):
            print(f"⚠️  {path} missing — skipping {c}")
            continue
        df = pd.read_parquet(path)
        # Some Silver Standards may not carry the city column explicitly;
        # set it from the load context so downstream prompts are correct.
        df["city"] = c
        frames.append(df)
        print(f"📂 Loaded {len(df):,} rows from {c}")
    if not frames:
        raise FileNotFoundError(
            "No silver-standard parquets found. Run batch_labeling.py first."
        )
    return pd.concat(frames, ignore_index=True)


def run_evaluation(
    df: pd.DataFrame,
    model_name: str = "google/flan-t5-base",
    batch_size: int = 16,
    out_path: str | None = None,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Running on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_outputs: list[str] = []
    parsed_labels: list[str | None] = []

    print(f"🤖 Evaluating {len(df):,} samples (batch_size={batch_size}, model={model_name})")

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        prompts = [build_prompt(r["instruction"], r["city"]) for _, r in batch.iterrows()]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,        # one word — anything more is parser noise
                do_sample=False,
                num_beams=1,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for raw in decoded:
            raw = raw.strip()
            raw_outputs.append(raw)
            parsed_labels.append(parse_label(raw))

    df = df.copy()
    df["llm_output_raw"] = raw_outputs
    df["llm_prediction"] = parsed_labels
    df["llm_parsed"] = df["llm_prediction"].notna()

    n_parsed = int(df["llm_parsed"].sum())
    print(f"✅ Parsed {n_parsed:,}/{len(df):,} model outputs into a label "
          f"({100*n_parsed/len(df):.1f}%)")

    if "oracle_label" in df.columns:
        agreement = (df["llm_prediction"] == df["oracle_label"]).mean()
        print(f"📊 Oracle/LLM agreement (over all rows): {100*agreement:.2f}%")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
        print(f"💾 Saved: {out_path}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None,
                    help="manhattan/pittsburgh/philadelphia. Default: all 3.")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (smoke test).")
    ap.add_argument("--out", default=os.path.join(
        config.BASE_DIR, "reports", "llm_audits", "llm_predictions_baseline.parquet"))
    args = ap.parse_args()

    df = load_silver_standards(args.city)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"🔍 Smoke-test mode: limited to {len(df)} rows")

    run_evaluation(df, model_name=args.model, batch_size=args.batch_size, out_path=args.out)


if __name__ == "__main__":
    main()
