"""
Train a text classification model to predict target_node_id
from navigation instruction text.

Task Formulation:
-----------------
Input:  Natural language navigation instruction
Output: Discrete graph node ID (target_node_id)

We treat this as a multi-class classification problem where:
- Each unique node ID = one class
- The model predicts which node the instruction refers to

Assumptions:
------------
- train.parquet and val.parquet exist in data/splits/
- Both contain:
    - instruction (text column)
    - target_node_id (label column)
- Splits are leakage-safe (grouped by instruction_id)

Output:
-------
runs/manhattan_nodeclf/
    label_map.json
    config.json
    checkpoints/
    final_model/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset


# ------------------------------
# Metric function
# ------------------------------
# This is used by HuggingFace Trainer to evaluate model performance.
# We compute simple accuracy as a baseline metric.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main(
    train_path="data/splits/train.parquet",
    val_path="data/splits/val.parquet",
    run_dir="runs/manhattan_nodeclf",
    model_name="distilbert-base-uncased",
    text_col="instruction",
    label_col="target_node_id",
    max_len=192,
):
    """
    Parameters
    ----------
    train_path : str
        Path to training parquet file.

    val_path : str
        Path to validation parquet file.

    run_dir : str
        Folder where checkpoints + logs + label map will be saved.

    model_name : str
        HuggingFace model checkpoint to use.

    text_col : str
        Column containing instruction text.

    label_col : str
        Column containing discrete node ID label.

    max_len : int
        Maximum token length for tokenizer truncation.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Load data
    # ------------------------------
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Drop rows with missing text or labels
    train_df = train_df.dropna(subset=[text_col, label_col]).copy()
    val_df = val_df.dropna(subset=[text_col, label_col]).copy()

    # ------------------------------
    # Create label mapping
    # ------------------------------
    # IMPORTANT: label map must be created ONLY from training data
    # to avoid information leakage.
    labels = sorted(train_df[label_col].unique().tolist())

    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Save mapping for reproducibility
    with open(run_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # Encode labels into integers
    train_df["label"] = train_df[label_col].map(label2id)
    val_df["label"] = val_df[label_col].map(label2id)

    # Remove validation rows whose labels were unseen in training
    val_df = val_df.dropna(subset=["label"]).copy()

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # ------------------------------
    # Tokenizer
    # ------------------------------
    # Converts text into token IDs that transformer models can process.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    # Convert pandas DataFrame → HF Dataset
    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]]).map(tokenize, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------
    # Model
    # ------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------
    # Training configuration
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ------------------------------
    # Train
    # ------------------------------
    trainer.train()

    # Save final model
    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save config for reproducibility
    config = {
        "model_name": model_name,
        "max_len": max_len,
        "num_labels": len(labels),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
