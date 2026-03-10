"""
Task 4.2 — Automated Degradation Analysis

Analyzes how oracle labels shift as constraints are progressively removed.
Generates plots and statistics showing Answerable → Ambiguous → Contradictory flips.

Outputs (saved to --output_dir):
  - degradation_curve.png         : Accuracy/F1 by n_dropped (if predictions exist)
  - label_distribution_by_drop.png: Stacked bar chart of label proportions by n_dropped
  - label_shift_heatmap.png       : How labels change from n_dropped=0 to n_dropped=k
  - per_type_impact.png           : Which constraint type hurts most when dropped
  - degradation_report.json       : All numbers in machine-readable format

Usage:
    # Analyze oracle labels only (no model needed):
    python -m scripts.analysis.degradation_analysis \
        --labeled_path data/processed/train_variants_labeled.parquet

    # Analyze with model predictions (after training):
    python -m scripts.analysis.degradation_analysis \
        --labeled_path data/processed/test_variants_labeled.parquet \
        --pred_col pred_label
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


LABELS = ["answerable", "ambiguous", "contradictory"]
COLORS = {"answerable": "#2ecc71", "ambiguous": "#f39c12", "contradictory": "#e74c3c"}


# ─── Plot 1: Label Distribution by n_dropped (Stacked Bar) ─────────────

def plot_label_distribution(df, output_dir):
    """
    Stacked bar chart showing proportion of each label at each masking level.
    This is the main degradation figure for the paper.
    """
    if "n_dropped" not in df.columns:
        print("⚠️  No n_dropped column — skipping label distribution plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    drops = sorted(df["n_dropped"].unique())
    proportions = {label: [] for label in LABELS}

    for nd in drops:
        sub = df[df["n_dropped"] == nd]
        total = len(sub)
        for label in LABELS:
            count = len(sub[sub["oracle_label"] == label])
            proportions[label].append(count / total * 100 if total > 0 else 0)

    x = np.arange(len(drops))
    width = 0.6
    bottom = np.zeros(len(drops))

    for label in LABELS:
        vals = proportions[label]
        ax.bar(x, vals, width, bottom=bottom, label=label.capitalize(), color=COLORS[label])
        # Add percentage text on each segment
        for i, v in enumerate(vals):
            if v > 5:  # only show if segment is big enough
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottom += np.array(vals)

    ax.set_xlabel("Number of Constraint Types Dropped", fontsize=12)
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax.set_title("Label Distribution by Masking Level", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in drops])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    path = output_dir / "label_distribution_by_drop.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return proportions


# ─── Plot 2: Label Shift Heatmap ───────────────────────────────────────

def plot_label_shift_heatmap(df, output_dir):
    """
    For each example, compare its label at n_dropped=0 vs n_dropped=k.
    Shows how many instructions flip from answerable → ambiguous, etc.
    """
    if "n_dropped" not in df.columns or "example_id" not in df.columns:
        print("⚠️  Missing columns — skipping shift heatmap")
        return

    # Get original labels (n_dropped=0)
    originals = df[df["n_dropped"] == 0][["example_id", "oracle_label"]].copy()
    originals = originals.rename(columns={"oracle_label": "original_label"})

    # Get maximally-dropped labels
    max_drop = df["n_dropped"].max()
    max_masked = df[df["n_dropped"] == max_drop][["example_id", "oracle_label"]].copy()
    max_masked = max_masked.rename(columns={"oracle_label": "masked_label"})

    merged = originals.merge(max_masked, on="example_id", how="inner")
    if merged.empty:
        print("⚠️  No matching example_ids — skipping shift heatmap")
        return

    # Build transition matrix
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for _, row in merged.iterrows():
        i = LABELS.index(row["original_label"]) if row["original_label"] in LABELS else -1
        j = LABELS.index(row["masked_label"]) if row["masked_label"] in LABELS else -1
        if i >= 0 and j >= 0:
            matrix[i][j] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([l.capitalize() for l in LABELS], fontsize=10)
    ax.set_yticklabels([l.capitalize() for l in LABELS], fontsize=10)
    ax.set_xlabel(f"Label After Dropping {max_drop} Types", fontsize=11)
    ax.set_ylabel("Original Label (n_dropped=0)", fontsize=11)
    ax.set_title("Label Shifts: Original → Maximally Masked", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if val > matrix.max() * 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = output_dir / "label_shift_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return matrix


# ─── Plot 3: Per-Type Impact ───────────────────────────────────────────

def plot_per_type_impact(df, output_dir):
    """
    Bar chart: for each constraint type, how often does dropping it alone
    flip the label from answerable to ambiguous/contradictory?
    """
    if "dropped_types" not in df.columns:
        print("⚠️  No dropped_types column — skipping per-type impact")
        return

    types = ["direction", "radius", "proximity", "landmark"]
    flip_rates = {}

    for ctype in types:
        # Find rows where ONLY this one type was dropped (n_dropped=1)
        mask = df["dropped_types"].astype(str).apply(
            lambda x: x == f"['{ctype}']"
        )
        if mask.sum() == 0:
            continue

        sub = df[mask]
        # What fraction became non-answerable?
        non_answerable = len(sub[sub["oracle_label"] != "answerable"])
        flip_rates[ctype] = non_answerable / len(sub) * 100

    if not flip_rates:
        print("⚠️  No single-type drops found — skipping per-type impact")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    types_found = list(flip_rates.keys())
    rates = [flip_rates[t] for t in types_found]
    bars = ax.bar(types_found, rates, color=["#3498db", "#9b59b6", "#1abc9c", "#e67e22"][:len(types_found)])

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("% Non-Answerable After Dropping", fontsize=11)
    ax.set_xlabel("Constraint Type Dropped (alone)", fontsize=11)
    ax.set_title("Impact of Dropping Each Constraint Type", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    path = output_dir / "per_type_impact.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return flip_rates


# ─── Plot 4: Model Accuracy Degradation Curve ──────────────────────────

def plot_degradation_curve(df, pred_col, output_dir):
    """
    Line plot: model accuracy and F1 at each n_dropped level.
    Only works if pred_col exists (after model inference).
    """
    if pred_col not in df.columns:
        print(f"⚠️  No '{pred_col}' column — skipping accuracy degradation curve")
        return

    if "n_dropped" not in df.columns:
        print("⚠️  No n_dropped column — skipping degradation curve")
        return

    from sklearn.metrics import accuracy_score, f1_score

    drops = sorted(df["n_dropped"].unique())
    accs = []
    f1s = []

    for nd in drops:
        sub = df[df["n_dropped"] == nd]
        acc = accuracy_score(sub["oracle_label"], sub[pred_col])
        f1 = f1_score(sub["oracle_label"], sub[pred_col], average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(drops, accs, "o-", color="#2ecc71", linewidth=2, markersize=8, label="Accuracy")
    ax.plot(drops, f1s, "s--", color="#3498db", linewidth=2, markersize=8, label="Macro F1")

    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.annotate(f"{a:.2f}", (drops[i], a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Number of Constraint Types Dropped", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Degradation", fontsize=14, fontweight="bold")
    ax.set_xticks(drops)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "degradation_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return {"accuracy": dict(zip(drops, accs)), "f1": dict(zip(drops, f1s))}


# ─── Report Generation ─────────────────────────────────────────────────

def generate_report(df, pred_col, output_dir):
    """Generate full JSON report with all degradation statistics."""
    report = {}

    # Overall label distribution
    report["total_rows"] = len(df)
    report["label_distribution"] = df["oracle_label"].value_counts().to_dict()

    # Distribution by n_dropped
    if "n_dropped" in df.columns:
        by_drop = {}
        for nd in sorted(df["n_dropped"].unique()):
            sub = df[df["n_dropped"] == nd]
            by_drop[int(nd)] = {
                "n": len(sub),
                "labels": sub["oracle_label"].value_counts().to_dict(),
            }
        report["by_n_dropped"] = by_drop

    # Per-type statistics
    if "dropped_types" in df.columns:
        per_type = {}
        for ctype in ["direction", "radius", "proximity", "landmark"]:
            mask = df["dropped_types"].astype(str).str.contains(ctype)
            if mask.sum() > 0:
                sub = df[mask]
                per_type[ctype] = {
                    "n": int(mask.sum()),
                    "labels": sub["oracle_label"].value_counts().to_dict(),
                }
        report["per_type"] = per_type

    # Model metrics (if predictions exist)
    if pred_col in df.columns and "n_dropped" in df.columns:
        from sklearn.metrics import accuracy_score, f1_score
        model_metrics = {}
        for nd in sorted(df["n_dropped"].unique()):
            sub = df[df["n_dropped"] == nd]
            model_metrics[int(nd)] = {
                "accuracy": round(accuracy_score(sub["oracle_label"], sub[pred_col]), 4),
                "f1_macro": round(f1_score(sub["oracle_label"], sub[pred_col], average="macro", zero_division=0), 4),
                "n": len(sub),
            }
        report["model_degradation"] = model_metrics

    path = output_dir / "degradation_report.json"
    json.dump(report, open(path, "w"), indent=2)
    print(f"  ✅ Saved: {path}")
    return report


# ─── Main ───────────────────────────────────────────────────────────────

def main(labeled_path, output_dir="runs/degradation", pred_col="pred_label"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(labeled_path)
    print(f"Loaded: {len(df)} rows from {labeled_path}")
    print(f"Labels: {df['oracle_label'].value_counts().to_dict()}")
    print()

    print("Generating plots...")
    plot_label_distribution(df, output_dir)
    plot_label_shift_heatmap(df, output_dir)
    plot_per_type_impact(df, output_dir)
    plot_degradation_curve(df, pred_col, output_dir)

    print("\nGenerating report...")
    report = generate_report(df, pred_col, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("DEGRADATION SUMMARY")
    print("=" * 60)
    if "by_n_dropped" in report:
        for nd, info in report["by_n_dropped"].items():
            labels = info["labels"]
            total = info["n"]
            ans_pct = labels.get("answerable", 0) / total * 100
            amb_pct = labels.get("ambiguous", 0) / total * 100
            con_pct = labels.get("contradictory", 0) / total * 100
            print(f"  n_dropped={nd}: answerable={ans_pct:5.1f}%  ambiguous={amb_pct:5.1f}%  contradictory={con_pct:5.1f}%  (n={total})")

    print(f"\n✅ All outputs saved to {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 4.2: Degradation Analysis")
    ap.add_argument("--labeled_path", required=True,
                    help="Path to a *_variants_labeled.parquet file")
    ap.add_argument("--output_dir", default="runs/degradation",
                    help="Directory to save plots and report")
    ap.add_argument("--pred_col", default="pred_label",
                    help="Column with model predictions (if available)")
    a = ap.parse_args()
    main(**vars(a))
