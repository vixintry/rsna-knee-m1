# EXP-007 — Conservative report-derived weak labels

## Objective
Test whether reports can supply noisy supervision for studies with no official target labels. All studies with any official target value are excluded from weak-label generation; the 58 fully labelled studies remain the trusted gold-label pool.

## Label semantics
- 1 = explicit positive phrase found.
- 0 = explicit negative/normal phrase found.
- blank = no cue, uncertain evidence, or conflicting positive and negative cues.

Silence is never interpreted as a negative. The rule set is multilingual but incomplete and not a validated clinical NLP system. A match may still be wrong due to negation scope, historical findings, anatomy, language variation, or report structure.

## Run in Kaggle
GPU: OFF. This is CPU-only report processing.

Run from the repository root:
    python scripts/derive_report_weak_labels.py

Defaults:
- Input: /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv
- Output directory: /kaggle/working/exp007_weak_labels/

Outputs:
- weak_labels.csv: one row per unlabelled study; blank target values mean unknown.
- weak_label_audit.csv: one row per study-target with status and matched cue.
- weak_label_summary.csv: counts per target and status.

## Required audit before training
1. Inspect positive, negative, unknown, uncertain, and conflict examples for every target.
2. Check false positives and false negatives manually, particularly compartment-specific OA and cross-language negation.
3. Measure agreement with the 58 gold-labelled reports as a diagnostic only. Do not include those studies in weak-label training.
4. Keep unknown values masked during training; never fill them with zero.
5. Evaluate only against gold labels and keep all scans from a study in one partition.
6. Use MRI-only inference if test reports are unavailable.

## Status
Implementation committed for audit; not yet run against the Kaggle data. No weak-label counts or performance claims have been made.
