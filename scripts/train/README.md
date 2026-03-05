# Model Training & Evaluation

This directory contains the execution scripts for training and evaluating the spatial reasoning models. These scripts treat navigation as a **Global Node Classification** task, where the model learns to map natural language instructions to specific coordinates (Node IDs) in a city graph.

## Files

* **`train_classifier.py`**: The main training entry point. It loads the grounded Parquet files, maps Node IDs to label indices, and fine-tunes a Transformer model (e.g., BERT or T5) using the Hugging Face `Trainer` API.
* **`eval_classifier.py`**: Used to benchmark a trained model against the test set. It generates accuracy metrics and classification reports to measure how well the model generalizes to "Unseen" instructions or regions.

## Setup & Requirements

Before running these scripts, ensure you have:
1. Generated the grounded datasets using `scripts/data_prep/attach_target_node_all_regions.py`.
2. Verified that `data/processed/` contains the required `.parquet` splits.
3. Installed the dependencies listed in the root `requirements.txt` (specifically `torch`, `transformers`, and `scikit-learn`).

## Usage

Run the training script from the **project root**:

```bash
python scripts/train/train_classifier.py
```

## Output Artifacts

The scripts will create a `runs/` directory in the project root containing:

* **`final_model/`**: Optimized model weights and configuration.
* **`label_map.json`**: The mapping between graph Node IDs and the model's output neurons.
* **`checkpoints/`**: Intermediate training states.

## Note on Hardware

Training these models is computationally intensive. It is recommended to run these scripts on a machine with a **CUDA-enabled GPU**. The `train_classifier.py` script is configured to automatically detect and use a GPU if available.