#!/usr/bin/env python3
"""Create deterministic study-level OOF fold assignments from complete labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.validation import fold_support, make_multilabel_folds

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, help="Path to official train.csv")
    parser.add_argument("--output", required=True, help="Output CSV path for complete-label fold assignments")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.train_csv)
    required = ["StudyInstanceUID", *LABELS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    complete = data.dropna(subset=LABELS).copy()
    if complete["StudyInstanceUID"].duplicated().any():
        raise SystemExit("Duplicate StudyInstanceUID values found in complete-label rows")
    if complete.empty:
        raise SystemExit("No studies have complete labels")

    folds = make_multilabel_folds(
        complete["StudyInstanceUID"].astype(str).tolist(),
        complete[LABELS].to_numpy(),
        n_splits=args.folds,
        seed=args.seed,
    )
    output = complete[["StudyInstanceUID"]].copy()
    output["fold"] = folds
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    print(f"Wrote {len(output)} study assignments to {path}")
    print(f"Seed: {args.seed}; folds: {args.folds}; complete-label studies: {len(output)}")
    support = fold_support(complete[LABELS].to_numpy(), folds)
    for fold, counts in support.items():
        print(f"fold={fold} n={counts['n_studies']}")
        for label, positive, negative in zip(LABELS, counts["positive"], counts["negative"]):
            print(f"  {label}: positive={positive} negative={negative}")


if __name__ == "__main__":
    main()
