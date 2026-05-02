"""
Answerability Classification on Original (Unmasked) Instructions.

Evaluates FLAN-T5-large on original RVS instructions labeled under
mode='label' (three-way oracle), producing a comparable unmasked baseline
for the masked variant evaluation.

Input:  reports/llm_audits/ORIGINAL_ORACLE_LABELS.parquet
Output: reports/llm_audits/LLM_ANSWERABILITY_RESULTS_ORIGINAL.parquet
"""

import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
from sklearn.metrics import classification_report, confusion_matrix


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
        token = token.strip('.,!?;:')
        if token in LABEL_ALIASES:
            return LABEL_ALIASES[token]
    for alias, label in LABEL_ALIASES.items():
        if alias in cleaned:
            return label
    return 'UNPARSEABLE'


def build_prompt(instruction: str) -> str:
    return (
        f"A navigation instruction may contain [MASK] (a masked landmark) or "
        f"[DIR_MASK] (a masked direction). Based only on the remaining information, "
        f"classify whether the instruction identifies exactly one destination, "
        f"multiple possible destinations, or none.\n\n"
        f"- Answerable: exactly one valid destination can be identified\n"
        f"- Ambiguous: multiple valid destinations remain possible\n"
        f"- Contradictory: no valid destination exists\n\n"
        f"Example 1:\n"
        f"Instruction: Meet me at the supermarket on Penn Avenue. "
        f"It is the building right next to the atm. "
        f"The atm is on the same side of the street as the confectionery shop.\n"
        f"Label: Answerable\n\n"
        f"Example 2:\n"
        f"Instruction: Head northeast to meet me at the [MASK] on East 49th Street. "
        f"United Nations is on my south and a hotel is located on my northeast.\n"
        f"Label: Answerable\n\n"
        f"Example 3:\n"
        f"Instruction: Meet me at the [MASK] on Penn Avenue. "
        f"It is the building right next to the atm. "
        f"The atm is on the same side of the street as the confectionery shop.\n"
        f"Label: Ambiguous\n\n"
        f"Example 4:\n"
        f"Instruction: Get on Liberty Avenue past basketball pitch located on your northeast. "
        f"I'm at the [MASK] just before storage rental shop.\n"
        f"Label: Contradictory\n\n"
        f"Instruction: {instruction}\n"
        f"Label:"
    )


def run_evaluation(input_path, model_name="google/flan-t5-large",
                   output_filename="LLM_ANSWERABILITY_RESULTS_ORIGINAL.parquet",
                   batch_size=8):

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples | cities: {df['city'].unique().tolist()}")
    print(f"Oracle 2 label distribution:\n{df['oracle_label'].value_counts().to_string()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    print(f"Running classification on {len(df)} samples (batch_size={batch_size})...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i: i + batch_size]
        prompts = [
            build_prompt(row['original_text'])
            for _, row in batch_df.iterrows()
        ]
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
                max_new_tokens=5,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (_, row), raw_output in zip(batch_df.iterrows(), decoded):
            predicted_label = parse_label(raw_output.strip())
            oracle_label = row['oracle_label']
            all_results.append({
                'sample_id':          row['sample_id'],
                'city':               row['city'],
                'variant_type':       'original',
                'oracle_label':       oracle_label,
                'oracle_label_resolve': row.get('oracle_label_resolve'),
                'extracted_category': row.get('extracted_category'),
                'original_text':      row['original_text'],
                'llm_output_raw':     raw_output.strip(),
                'predicted_label':    predicted_label,
                'is_correct':         predicted_label == oracle_label,
            })

    results_df = pd.DataFrame(all_results)
    report_dir = os.path.join(config.BASE_DIR, "reports", "llm_audits")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, output_filename)
    results_df.to_parquet(output_path)

    print(f"\nSaved {len(results_df)} rows → {output_path}")
    print(f"\n{'='*55}")
    print(f"ANSWERABILITY CLASSIFICATION SUMMARY — {model_name} (ORIGINAL)")
    print(f"{'='*55}")

    parseable = results_df[results_df['predicted_label'] != 'UNPARSEABLE']
    print(f"\nParse rate: {len(parseable)}/{len(results_df)} = "
          f"{len(parseable)/len(results_df):.1%}")

    overall_acc = results_df['is_correct'].mean()
    print(f"\nOverall accuracy: {overall_acc:.1%}")

    majority = results_df['oracle_label'].value_counts().index[0]
    majority_acc = (results_df['oracle_label'] == majority).mean()
    print(f"Majority class baseline ({majority}): {majority_acc:.1%}")

    print(f"\nPer-class results:")
    for label in ['Answerable', 'Ambiguous', 'Contradictory']:
        subset = results_df[results_df['oracle_label'] == label]
        if len(subset) == 0:
            continue
        acc = subset['is_correct'].mean()
        pred_dist = subset['predicted_label'].value_counts().to_dict()
        print(f"  {label:<15}: Acc={acc:.1%} | N={len(subset)} | "
              f"Pred dist={pred_dist}")

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
    input_path = os.path.join(
        config.BASE_DIR, "reports", "llm_audits",
        "ORIGINAL_ORACLE_LABELS.parquet"
    )
    if os.path.exists(input_path):
        run_evaluation(
            input_path,
            model_name="google/flan-t5-large",
            output_filename="LLM_ANSWERABILITY_RESULTS_ORIGINAL_LARGE.parquet",
            batch_size=8,
        )
    else:
        print(f"Input not found: {input_path}")