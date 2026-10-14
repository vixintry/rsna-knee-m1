#!/usr/bin/env python3
"""Compare baseline and experimental parser on fully labelled EXP-007 studies.

Gold rows are used for evaluation only and are never exported as weak-training
examples. Metrics are conditional on parser coverage; unknown/uncertain/conflict
outputs are not counted as negative predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.derive_report_weak_labels import TARGETS, classify_report
from scripts.derive_report_weak_labels_v2 import classify_report_v2


def metrics(labels, predictions):
    known = [(int(y), int(p)) for y, p in zip(labels, predictions) if pd.notna(y) and pd.notna(p)]
    tp = sum(y == 1 and p == 1 for y, p in known)
    fp = sum(y == 0 and p == 1 for y, p in known)
    tn = sum(y == 0 and p == 0 for y, p in known)
    fn = sum(y == 1 and p == 0 for y, p in known)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "covered": len(known),
        "coverage": len(known) / len(labels) if len(labels) else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--output-dir", default="/kaggle/working/exp007_parser_v2_comparison")
    parser.add_argument("--expected-gold-studies", type=int, default=58)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"StudyInstanceUID", "Report", *TARGETS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if df["Report"].isna().any():
        raise ValueError("Report contains missing values")

    gold = df.loc[df[TARGETS].notna().all(axis=1)].copy()
    if len(gold) != args.expected_gold_studies:
        raise ValueError(
            f"Expected {args.expected_gold_studies} complete-label studies, found {len(gold)}. "
            "Inspect the dataset version and pass --expected-gold-studies explicitly if justified."
        )

    details = []
    for row in gold[["StudyInstanceUID", "Report", *TARGETS]].itertuples(index=False, name=None):
        study_id, report, *gold_values = row
        for target, gold_value in zip(TARGETS, gold_values):
            base_label, base_status, base_pos, base_neg = classify_report(report, target)
            v2_label, v2_status, v2_pos, v2_neg = classify_report_v2(report, target)
            details.append({
                "StudyInstanceUID": study_id, "target": target, "gold_label": gold_value,
                "baseline_label": base_label, "baseline_status": base_status,
                "baseline_positive_cues": base_pos, "baseline_negative_cues": base_neg,
                "v2_label": v2_label, "v2_status": v2_status,
                "v2_positive_cues": v2_pos, "v2_negative_cues": v2_neg,
                "changed_label": (
                    (pd.isna(base_label) != pd.isna(v2_label))
                    or (pd.notna(base_label) and pd.notna(v2_label) and base_label != v2_label)
                ),
            })
    detail = pd.DataFrame(details)
    summaries = []
    for target in TARGETS:
        subset = detail.loc[detail["target"] == target]
        for version, col, status_col in (
            ("baseline", "baseline_label", "baseline_status"),
            ("v2", "v2_label", "v2_status"),
        ):
            m = metrics(subset["gold_label"].tolist(), subset[col].tolist())
            status_counts = subset[status_col].value_counts().to_dict()
            summaries.append({"target": target, "version": version, **m, **{
                f"status_{key}": int(value) for key, value in status_counts.items()
            }})

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outdir / "parser_v2_gold_detail.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "parser_v2_gold_summary.csv", index=False)
    changed = detail.loc[detail["changed_label"]]
    report = {
        "gold_studies": len(gold),
        "target_rows": len(detail),
        "changed_predictions": int(detail["changed_label"].sum()),
        "changed_by_target": changed.groupby("target").size().astype(int).to_dict(),
        "note": "Metrics are conditional on parser coverage; gold studies are evaluation-only.",
    }
    (outdir / "parser_v2_gold_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nChanged target predictions: {report['changed_predictions']}")
    print(f"Saved diagnostic outputs to {outdir}")
    print("No model training performed; do not use these gold studies as weak-label training data.")


if __content__ := __name__ == "__main__":
    main()
