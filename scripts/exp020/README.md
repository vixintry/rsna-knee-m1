# EXP-020 local report review sample

This script creates a deterministic, stratified review sample for **Medial OA, Lateral OA, PF OA, and Synovitis**. It does not change the parser, assign reference labels, train a model, or evaluate model performance.

## Run in the Kaggle notebook environment

From the repository root (or adjust `--parser` to the checked-out parser path):

```bash
python scripts/exp020/build_local_review_sample.py \
  --train-csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --parser scripts/derive_report_weak_labels.py \
  --out /kaggle/working/exp020_review \
  --per-stratum 20 \
  --seed 20261020
```

The output includes up to 20 studies for each target × parser status (positive, negative, unknown, uncertain, conflict). The sample is deterministic for a fixed input, parser, seed, and pandas version. Empty strata are omitted. All studies with any official target label are excluded before sampling.

## Local outputs

- `exp020_review_sample_LOCAL_ONLY.csv`: includes the full report text so a reviewer can interpret negation, uncertainty, history, and compartment context. Fill `reference_label` with exactly one of `positive`, `negative`, `uncertain`, or `not_stated`; use `reviewer_notes` for ambiguity.
- `exp020_sampling_summary.csv`: counts available and sampled per target/status.
- `exp020_provenance_LOCAL_ONLY.txt`: input and parser SHA-256 hashes plus run settings.

**Privacy rule:** the CSV and provenance file are local-only outputs. Never commit them, upload them to GitHub, or send report text to hosted APIs. Only commit code and documentation. Review sample labels are not gold labels; this is an audit sample, not permission to train.

## Interpretation safeguards

- A parser status is not a human reference label.
- Unknown means the parser found no accepted cue; it does not mean negative.
- Do not broaden regexes from the review sample and then evaluate those changes on the same rows. Freeze any proposed rules before using a separate held-out review sample.
- This utility is a sampling aid only. It does not establish that report-derived weak labels are allowed by competition rules or accurate enough for training.
