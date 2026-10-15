# 2026-10-14 — Run audit and next experiment gate

## Kaggle smoke test (user-provided output)
- `SIMULATE_N=100`; status OK.
- 100 studies processed, 100 with model predictions, 0 prior-filled.
- Total elapsed 123.5 seconds; reported extraction time 0.606 seconds/study; device CUDA.
- Test slice ordering: InstanceNumber for 100/100.
- Series used: sagittal_fluid 97, coronal_fluid 3.
- Event and note lists were empty.
- The notebook explicitly stated: no real submission was written.

This establishes successful inference for the sampled 100 studies only. It does not measure predictive accuracy or guarantee full-test reliability.

## Real submission status
The user reports that a prior `SIMULATE_N=0` run and `submission.csv` exist. The file has not yet been independently validated in this session. The simulation report used the same `run_report.json` path and may have overwritten the real run's provenance. Do not infer that the real submission is invalid, nor claim that it passes checks, until inspected.

## Immediate next action
Run the CPU-only validation cell against the existing `/kaggle/working/submission.csv` and competition `test.csv`. Check expected columns, row count, exact study ID/order match, finite predictions, range [0,1], and SHA-256. Save `real_submission_audit.json` separately. No GPU is required and no model rerun is intended.

## Next research steps
1. Audit EXP-007's report-derived weak labels on the 58 officially labelled studies. Measure per-target precision, recall and coverage on explicitly classified cases; inspect unknown, uncertain and conflicting labels separately; test whether unknown/report-silent cases differ in gold-positive rate. Weak labels remain audit-only.
2. If quality supports further work, test weak supervision against the frozen canonical study-level folds. Exclude held-out gold-fold studies from every supervised training path, including their report-derived labels.
3. Compare pooled OOF macro-AUC and uncertainty with the image-only EXP-011 reference of 0.589270. Preserve the original baseline and all per-target supports. Do not accept weak supervision without comparable evidence.

## Cell metadata convention for future instructions
Every new Kaggle cell instruction should specify: cell type (Markdown or Code), whether it is persistent project logic or a one-off temporary diagnostic, GPU on/off, expected output/artifact, and whether it changes model behaviour. Keep artifact audits CPU-only unless computation genuinely requires a GPU.
