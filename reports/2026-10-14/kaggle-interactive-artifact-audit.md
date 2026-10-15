# Kaggle interactive inference artifact audit — 2026-10-14

## Observed environment

The notebook's competition input directory reported:

| File | Rows | Columns |
|---|---:|---:|
| `train.csv` | 4,407 | 14 |
| `test.csv` | 3 | 1 |
| `sample_submission.csv` | 3 | 13 |
| `train_series.csv` | 24,371 | 5 |
| `test_series.csv` | 15 | 5 |

These are user-observed counts from the Kaggle interactive environment. They establish that the accessible `test.csv` and sample submission each contained three studies at the time of the check.

## Real-mode inference report

The attached `run_report.json` records a non-simulated run with 3 studies, 3 model predictions, 0 prior-filled rows, CUDA, and 88.7 seconds total elapsed time. The test-series selection used 2 sagittal-fluid and 1 coronal-fluid series; all 3 used InstanceNumber ordering. No events were recorded.

The attached `real_submission_audit.json` records PASS for row count, ID/order match to the available `test.csv`, finite predictions, and prediction values within [0, 1]. The submission SHA-256 is `0897242c9f931a1f12f7f597288b6243ebe4cf3ede9d8430145f3ccba66c5ab7`.

## Interpretation and limits

- The three-row output matches the three-row test file available in that interactive environment; it is not evidence that the inference loop prematurely stopped.
- It is plausible that the interactive environment exposes a small placeholder test set while competition scoring uses a different evaluation setup, but these observations alone do **not** verify the hidden evaluation mechanism.
- A schema/artifact PASS is not a measure of predictive quality and does not establish competition performance.
- A separate 200-study report was marked `simulated_run: true`; it is a smoke test, not a real test-set evaluation.
- The original full-run runtime/event report was overwritten by a later simulation according to the audit provenance note; it has not been reconstructed.

## Next research step

Resume the planned EXP-007 weak-label audit against the 58 gold-labelled studies. Keep uncertain, unknown, conflicting, and silent reports distinct; evaluate explicit per-target classifications against gold labels before deciding whether weak supervision merits an experiment. Preserve fixed study-level folds and prevent held-out gold-fold leakage. Compare any justified experiment against the image-only pooled OOF macro AUC baseline of 0.589270, without claiming improvement unless measured.

## Provenance

These files document user-provided Kaggle outputs and row counts observed on 2026-10-14. No model code, labels, predictions, or competition data were changed by this documentation update.
