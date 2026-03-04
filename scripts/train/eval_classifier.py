"""
Evaluate trained classifier on test split.

Loads:
    - final_model/
    - label_map.json

Evaluates:
    - accuracy
    - classification report

Saves:
    runs/.../metrics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main(
    test_path="data/splits/test.parquet",
    run_dir="runs/manhattan_nodeclf",
    text_col="instruction",
    label_col="target_node_id",
):
    run_dir = Path(run_dir)
    model_dir = run_dir / "final_model"

    test_df = pd.read_parquet(test_path)
    test_df = test_df.dropna(subset=[text_col, label_col]).copy()

    # Load label map
    with open(run_dir / "label_map.json", "r") as f:
        maps = json.load(f)

    label2id = maps["label2id"]

    # Filter out labels not seen during training
    test_df = test_df[test_df[label_col].isin(label2id.keys())].copy()

    y_true = test_df[label_col].map(label2id).astype(int).values

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    preds = []

    # Predict one by one (can batch later for speed)
    with torch.no_grad():
        for text in test_df[text_col].tolist():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=192,
            )
            outputs = model(**inputs)
            pred = int(torch.argmax(outputs.logits, dim=-1).item())
            preds.append(pred)

    acc = accuracy_score(y_true, preds)

    print("Test accuracy:", acc)
    print(classification_report(y_true, preds, digits=3))

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"test_accuracy": float(acc)}, f, indent=2)

    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
