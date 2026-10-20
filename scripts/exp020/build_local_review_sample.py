#!/usr/bin/env python3
"""Create a local-only, stratified human-review sample for EXP-020.

This script does not modify parser rules, create labels, train a model, or upload data.
The output contains report text and must stay in the local Kaggle working directory.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

TARGETS = ["Medial OA", "Lateral OA", "PF OA", "Synovitis"]
STATUSES = ["positive", "negative", "unknown", "uncertain", "conflict"]
DEFAULT_INPUT = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"
DEFAULT_OUT = "/kaggle/working/exp020_review"
DEFAULT_SEED = 20261020


def load_parser_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp020_report_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_review_sample(train_csv: Path, parser_path: Path, output_dir: Path,
                        per_stratum: int, seed: int):
    parser = load_parser_module(parser_path)
    df = pd.read_csv(train_csv)
    required = {"StudyInstanceUID", "Report", *parser.TARGETS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if df["Report"].isna().any():
        raise ValueError("Report contains missing values; inspect source first")

    # Exclude all official-label studies before examining weak-label candidates.
    gold_mask = df[parser.TARGETS].notna().any(axis=1)
    source = df.loc[~gold_mask, ["StudyInstanceUID", "Report"]].copy()
    records = []
    for row in source.itertuples(index=False):
        study_id, report = row
        for target in TARGETS:
            label, status, positive_cues, negative_cues = parser.classify_report(report, target)
            records.append({
                "StudyInstanceUID": str(study_id),
                "target": target,
                "parser_status": status,
                "parser_label": label,
                "positive_cues": positive_cues,
                "negative_cues": negative_cues,
                # Raw report is intentionally present only in the local review file.
                "report_text_LOCAL_ONLY": str(report),
            })

    all_candidates = pd.DataFrame(records)
    samples = []
    for target in TARGETS:
        for status in STATUSES:
            group = all_candidates[
                (all_candidates["target"] == target) &
                (all_candidates["parser_status"] == status)
            ]
            if group.empty:
                continue
            take = min(per_stratum, len(group))
            samples.append(group.sample(n=take, random_state=seed).copy())

    if not samples:
        raise RuntimeError("No review rows were produced")
    review = pd.concat(samples, ignore_index=True)
    review = review.sort_values(["target", "parser_status", "StudyInstanceUID"]).reset_index(drop=True)
    review.insert(0, "review_row_id", [f"R{i:05d}" for i in range(1, len(review) + 1)])
    review["reference_label"] = ""  # reviewer fills: positive / negative / uncertain / not_stated
    review["reviewer_notes"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "exp020_review_sample_LOCAL_ONLY.csv"
    review.to_csv(out_csv, index=False)

    counts = (
        all_candidates.groupby(["target", "parser_status"]).size()
        .rename("available").reset_index()
    )
    sampled = (
        review.groupby(["target", "parser_status"]).size()
        .rename("sampled").reset_index()
    )
    summary = counts.merge(sampled, on=["target", "parser_status"], how="left")
    summary["sampled"] = summary["sampled"].fillna(0).astype(int)
    summary.to_csv(output_dir / "exp020_sampling_summary.csv", index=False)

    provenance = [
        "EXP-020 local review sample",
        f"input_csv_sha256={sha256_file(train_csv)}",
        f"parser_sha256={sha256_file(parser_path)}",
        f"seed={seed}",
        f"per_stratum={per_stratum}",
        f"total_studies={len(df)}",
        f"official_label_studies_excluded={int(gold_mask.sum())}",
        f"unlabelled_studies_considered={len(source)}",
        f"review_rows={len(review)}",
        "report_text_included=True; LOCAL ONLY; DO NOT COMMIT OR UPLOAD",
        "No parser rules changed. No training or model evaluation performed.",
    ]
    (output_dir / "exp020_provenance_LOCAL_ONLY.txt").write_text(
        "\n".join(provenance) + "\n", encoding="utf-8"
    )
    print("\n".join(provenance))
    print(f"\nSaved local review CSV: {out_csv}")
    print(f"Saved sampling counts: {output_dir / 'exp020_sampling_summary.csv'}")
    print("\nKeep the output directory local. Do not upload these files to GitHub or hosted services.")
    return review, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", default=DEFAULT_INPUT)
    ap.add_argument("--parser", default="scripts/derive_report_weak_labels.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--per-stratum", type=int, default=20,
                    help="maximum rows per target x parser-status stratum (default: 20)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    if args.per_stratum < 1:
        ap.error("--per-stratum must be >= 1")
    build_review_sample(Path(args.train_csv), Path(args.parser), Path(args.out),
                        args.per_stratum, args.seed)


if __name__ == "__main__":
    main()
