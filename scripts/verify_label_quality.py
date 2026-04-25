"""
verify_label_quality.py

Sanity-checks the per-city silver-standard distribution. Under the project's
3-class oracle (Answerable / Ambiguous / Contradictory):

  - Ambiguous SHOULD dominate (50–70%). Most natural-language navigation
    instructions are underspecified.
  - Answerable should be moderate (20–40%). Cases where a unique brand or
    name singles out exactly one POI.
  - Contradictory should be small (5–15%). Higher than that usually means
    the NLP extractor failed (noun=None, junk fragments).

If Answerable is too high or Ambiguous is too low, the multi-candidate
collapse bug (or its equivalent) is back. If Contradictory is too high,
the NLP layer is mis-extracting nouns or the POI->graph mapping is broken.
"""

import os

import pandas as pd


def run_quality_audit(file_path):
    df = pd.read_parquet(file_path)
    total = len(df)

    stats = df['oracle_label'].value_counts(normalize=True) * 100

    print(f"\n--- Quality Audit: {os.path.basename(file_path)} ---")
    print(f"Total Samples: {total}")
    for label, percent in stats.items():
        print(f" - {label}: {percent:.2f}%")

    answerable    = stats.get('Answerable', 0)
    ambiguous     = stats.get('Ambiguous', 0)
    contradictory = stats.get('Contradictory', 0)

    warnings = []
    if answerable > 80:
        warnings.append(
            f"Answerable {answerable:.1f}% > 80% — multi-candidate collapse "
            f"bug may be back (oracle should preserve ambiguity, not pick one)."
        )
    if ambiguous < 30:
        warnings.append(
            f"Ambiguous {ambiguous:.1f}% < 30% — natural-language navigation "
            f"in dense cities should produce many under-specified rows."
        )
    if contradictory > 20:
        warnings.append(
            f"Contradictory {contradictory:.1f}% > 20% — NLP extractor likely "
            f"returning noun=None / junk fragments, OR POI->graph mapping "
            f"degraded. Check the init log for mapping rate."
        )

    if warnings:
        for w in warnings:
            print(f"⚠️  WARNING: {w}")
    else:
        print("✅ PASS: distribution looks healthy under 3-class semantics.")


if __name__ == "__main__":
    for city in ('manhattan', 'pittsburgh', 'philadelphia'):
        path = f"data/{city}/{city}_silver_standard.parquet"
        if os.path.exists(path):
            run_quality_audit(path)
        else:
            print(f"\n(skip) {path} not found — run batch_labeling.py for {city}.")