"""
Answerability Classification Evaluation — OLMo-2-0425-1B-Instruct (Shuffled 4-Shot ICL)

Pipeline per sample:
  masked instruction → OLMo-2-0425-1B-Instruct → predicted label
                     → parse to {Answerable, Ambiguous, Contradictory}
                     → compare against oracle_label

Evaluation uses 4-shot in-context learning with the same
four labeled demonstrations as the main setup, but in a
different order to test prompt-order sensitivity.

Input:  reports/llm_audits/LLM_DEGRADATION_INPUT.parquet
Output: reports/llm_audits/LLM_ANSWERABILITY_RESULTS_SHUF.parquet
"""

import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
)

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from sklearn.metrics import classification_report, confusion_matrix

# ── Label parsing ─────────────────────────────────────────────────────────────
VALID_LABELS = {'Answerable', 'Ambiguous', 'Contradictory'}

LABEL_ALIASES = {
    'answerable':    'Answerable',
    'answer':        'Answerable',
    'unique':        'Answerable',
    'solvable':      'Answerable',
    'yes':           'Answerable',
    'ambiguous':     'Ambiguous',
    'ambiguity':     'Ambiguous',
    'multiple':      'Ambiguous',
    'unclear':       'Ambiguous',
    'vague':         'Ambiguous',
    'contradictory': 'Contradictory',
    'contradiction': 'Contradictory',
    'impossible':    'Contradictory',
    'unsolvable':    'Contradictory',
    'no':            'Contradictory',
    'none':          'Contradictory',
    'invalid':       'Contradictory',
}


def parse_label(raw: str) -> str:
    if not raw or not raw.strip():
        return 'UNPARSEABLE'
    cleaned = raw.strip().lower()
    for label in VALID_LABELS:
        if cleaned == label.lower():
            return label
    for token in cleaned.split():
        token = token.strip('.,!?;:\n')
        if token in LABEL_ALIASES:
            return LABEL_ALIASES[token]
    for alias, label in LABEL_ALIASES.items():
        if alias in cleaned:
            return label
    return 'UNPARSEABLE'


def build_prompt(masked_instruction: str) -> str:
    return (
        f"A navigation instruction may contain [MASK] (a masked landmark) or "
        f"[DIR_MASK] (a masked direction). Based only on the remaining information, "
        f"classify whether the instruction identifies exactly one destination, "
        f"multiple possible destinations, or none.\n\n"
        f"- Answerable: exactly one valid destination can be identified\n"
        f"- Ambiguous: multiple valid destinations remain possible\n"
        f"- Contradictory: no valid destination exists\n\n"
        f"Example 1:\n"
        f"Instruction: Get on Liberty Avenue past basketball pitch located on your northeast. "
        f"I'm at the [MASK] just before storage rental shop.\n"
        f"Label: Contradictory\n\n"
        f"Example 2:\n"
        f"Instruction: Meet me at the [MASK] on Penn Avenue. "
        f"It is the building right next to the atm. "
        f"The atm is on the same side of the street as the confectionery shop.\n"
        f"Label: Ambiguous\n\n"
        f"Example 3:\n"
        f"Instruction: Meet me at the supermarket on Penn Avenue. "
        f"It is the building right next to the atm. "
        f"The atm is on the same side of the street as the confectionery shop.\n"
        f"Label: Answerable\n\n"
        f"Example 4:\n"
        f"Instruction: Head northeast to meet me at the [MASK] on East 49th Street. "
        f"United Nations is on my south and a hotel is located on my northeast.\n"
        f"Label: Answerable\n\n"
        f"Instruction: {masked_instruction}\n"
        f"Label:"
    )


def run_evaluation(input_path: str,
                   model_name: str = "allenai/OLMo-2-0425-1B-Instruct",
                   output_filename: str = "LLM_ANSWERABILITY_RESULTS_SHUF.parquet",
                   batch_size: int = 8,
                   limit: int = None):

    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit).copy()
        print(f"Test mode: {limit} samples")
    print(f"Loaded {len(df)} samples | cities: {df['city'].unique().tolist()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU available — aborting"
    print(f"Loading {model_name} on {device}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Left-padding required for causal LM batch inference
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    all_results = []
    print(f"Running classification on {len(df)} samples "
          f"(batch_size={batch_size})...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i: i + batch_size]

        prompts = [
            build_prompt(row['masked_instruction'])
            for _, row in batch_df.iterrows()
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        # Decode only generated tokens, not the prompt
        decoded = [
            tokenizer.decode(
                outputs[i][prompt_length:],
                skip_special_tokens=True
            ).strip()
            for i in range(len(outputs))
        ]

        for (_, row), raw_output in zip(batch_df.iterrows(), decoded):
            predicted_label = parse_label(raw_output)
            oracle_label    = row['oracle_label']
            is_correct      = predicted_label == oracle_label

            all_results.append({
                'sample_id':          row.get('sample_id'),
                'city':               row['city'],
                'variant_type':       row.get('variant_type'),
                'oracle_label':       oracle_label,
                'extracted_category': row.get('extracted_category'),
                'masked_instruction': row['masked_instruction'],
                'llm_output_raw':     raw_output,
                'predicted_label':    predicted_label,
                'is_correct':         is_correct,
            })

    results_df = pd.DataFrame(all_results)
    report_dir = os.path.join(config.BASE_DIR, "reports", "llm_audits")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, output_filename)
    results_df.to_parquet(output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSaved {len(results_df)} rows → {output_path}")
    print(f"\n{'='*55}")
    print(f"ANSWERABILITY CLASSIFICATION SUMMARY — {model_name}")
    print(f"{'='*55}")

    parseable   = results_df[results_df['predicted_label'] != 'UNPARSEABLE']
    unparseable = results_df[results_df['predicted_label'] == 'UNPARSEABLE']
    print(f"\nParse rate: {len(parseable)}/{len(results_df)} = "
          f"{len(parseable)/len(results_df):.1%}")
    print(f"Unparseable outputs: {len(unparseable)}")

    overall_acc = results_df['is_correct'].mean()
    parseable_acc = parseable['is_correct'].mean() if len(parseable) > 0 else 0
    print(f"\nOverall accuracy (all): {overall_acc:.1%}")
    print(f"Accuracy (parseable only): {parseable_acc:.1%}")

    majority = results_df['oracle_label'].value_counts().index[0]
    majority_acc = (results_df['oracle_label'] == majority).mean()
    print(f"\nMajority class baseline ({majority}): {majority_acc:.1%}")

    print(f"\nPer-class results:")
    for label in ['Answerable', 'Ambiguous', 'Contradictory']:
        subset = results_df[results_df['oracle_label'] == label]
        acc = subset['is_correct'].mean()
        pred_dist = subset['predicted_label'].value_counts().to_dict()
        print(f"  {label:<15}: Acc={acc:.1%} | N={len(subset)} | "
              f"Pred dist={pred_dist}")

    print(f"\nAccuracy by variant type:")
    print(results_df.groupby('variant_type')['is_correct']
          .mean().round(3).to_string())

    if len(unparseable) > 0:
        print(f"\nSample unparseable outputs:")
        print(unparseable['llm_output_raw'].value_counts().head(10).to_string())

    print(f"\nPredicted label distribution:")
    print(results_df['predicted_label'].value_counts().to_string())

    print("\nFull classification report:")
    parseable_df = results_df[results_df['predicted_label'] != 'UNPARSEABLE']
    print(classification_report(
        parseable_df['oracle_label'],
        parseable_df['predicted_label'],
        labels=['Answerable', 'Ambiguous', 'Contradictory'],
        zero_division=0
    ))

    print("\nConfusion matrix (rows=oracle, cols=predicted):")
    cm = confusion_matrix(
        parseable_df['oracle_label'],
        parseable_df['predicted_label'],
        labels=['Answerable', 'Ambiguous', 'Contradictory']
    )
    cm_df = pd.DataFrame(cm,
        index=['Oracle: Answerable', 'Oracle: Ambiguous', 'Oracle: Contradictory'],
        columns=['Pred: Answerable', 'Pred: Ambiguous', 'Pred: Contradictory'])
    print(cm_df.to_string())


if __name__ == "__main__":
    DEGRADATION_PATH = os.path.join(
        config.BASE_DIR, "reports", "llm_audits",
        "LLM_DEGRADATION_INPUT.parquet"
    )
    if os.path.exists(DEGRADATION_PATH):
        run_evaluation(
            DEGRADATION_PATH,
            model_name="allenai/OLMo-2-0425-1B-Instruct",
            output_filename="LLM_ANSWERABILITY_RESULTS_SHUF.parquet",
            batch_size=8,
            limit=None,
        )
    else:
        print(f"Input not found: {DEGRADATION_PATH}")