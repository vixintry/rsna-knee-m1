#!/usr/bin/env python3
"""Audit EXP-007 report-derived weak labels against fully labelled gold studies.

This is diagnostic only: gold studies are evaluated, never emitted as weak-label
training examples. Unknown/uncertain/conflicting parser outputs are uncovered,
not coerced to negative.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from derive_report_weak_labels import TARGETS, classify_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/exp007_gold_agreement",
    )
    parser.add_argument(
        "--expected-gold-studies",
        type=int,
        default=58,
        help="Fail closed if the number of complete-label gold studies differs.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"StudyInstanceUID", "Report", *TARGETS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if df["Report"].isna().any():
        raise ValueError("Report contains missing values; inspect source first")

    gold = df.loc[df[TARGETS].notna().all(axis=1)].copy()
    if len(gold) != args.expected_gold_studies:
        raise ValueError(
            f"Expected {args.expected_gold_studies} fully labelled gold studies, "
            f"found {len(gold)}. Check dataset/version before interpreting results."
        )

    records = []
    for row in gold[["StudyInstanceUID", "Report", *TARGETS]].itertuples(index=False, name=None):
        study_id, report, *gold_values = row
        for target, gold_value in zip(TARGETS, gold_values):
            weak_value, status, positive_cues, negative_cues = classify_report(report, target)
            records.append({
                "StudyInstanceUID": study_id,
                "target": target,
                "gold_label": int(gold_value),
                "weak_label": int(weak_value) if pd.notna(weak_value) else np.nan,
                "status": status,
                "positive_cues": positive_cues,
                "negative_cues": negative_cues,
            })

    detail = pd.DataFrame(records)
    summaries = []
    for target in TARGETS:
        part = detail.loc[detail["target"] == target]
        covered = part.loc[part["weak_label"].notna()]
        y = covered["gold_label"].astype(int)
        pred = covered["weak_label"].astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        accuracy = (tp + tn) / len(covered) if len(covered) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else (0.0 if precision == 0 and recall == 0 else None)
        )
        counts = part["status"].value_counts().to_dict()
        summaries.append({
            "target": target,
            "gold_studies": int(len(part)),
            "covered": int(len(covered)),
            "coverage": float(len(covered) / len(part)) if len(part) else None,
            "uncovered": int(len(part) - len(covered)),
            "positive_status": int(counts.get("positive", 0)),
            "negative_status": int(counts.get("negative", 0)),
            "unknown_status": int(counts.get("unknown", 0)),
            "uncertain_status": int(counts.get("uncertain", 0)),
            "conflict_status": int(counts.get("conflict", 0)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_on_covered": precision,
            "recall_on_covered": recall,
            "specificity_on_covered": specificity,
            "accuracy_on_covered": accuracy,
            "f1_on_covered": f1,
            "gold_positive_count": int((part["gold_label"] == 1).sum()),
            "gold_negative_count": int((part["gold_label"] == 0).sum()),
        })

    summary = pd.DataFrame(summaries)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail_path = outdir / "gold_agreement_detail.csv"
    summary_path = outdir / "gold_agreement_summary.csv"
    report_path = outdir / "gold_agreement_report.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "experiment": "EXP-007 gold agreement audit",
        "status": "COMPLETED_DIAGNOSTIC",
        "input_csv": str(Path(args.input_csv)),
        "n_input_studies": int(len(df)),
        "n_gold_studies": int(len(gold)),
        "expected_gold_studies": int(args.expected_gold_studies),
        "targets": TARGETS,
        "semantics": {
            "coverage": "fraction of gold study-target pairs with explicit parser output 0 or 1",
            "precision_recall": "computed only on covered pairs; uncovered outputs are not counted as negatives",
            "training_use": "none; gold studies are evaluation-only in this script",
        },
        "outputs": {
            "detail_csv": str(detail_path),
            "summary_csv": str(summary_path),
        },
        "summary": summaries,
    }
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Gold studies: {len(gold)}")
    print("\nPer-target agreement and coverage:")
    print(summary[[
        "target", "covered", "coverage", "tp", "fp", "tn", "fn",
        "precision_on_covered", "recall_on_covered",
        "unknown_status", "uncertain_status", "conflict_status",
    ]].to_string(index=False))
    print(f"\nWrote: {detail_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print("DIAGNOSTIC ONLY: do not use these gold studies as weak-label training data.")


if __name__ == "__main__":
    main()
