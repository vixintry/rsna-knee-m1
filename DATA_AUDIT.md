# Dataset Audit — Label Scarcity and MRI Structure

Date: 2026-10-10

This note records findings from the initial Kaggle notebook audit. These are observed dataset facts, not model-performance results.

## Label coverage

- `train.csv` has 4,407 studies and 14 columns: `StudyInstanceUID`, `Report`, and 12 target columns.
- Only 58/4,407 studies (about 1.32%) have known values for the target labels.
- Those 58 studies have all 12 targets populated, giving 696 known target values.
- The other 4,349 studies have all 12 targets missing.
- **Missing targets must remain unknown, not be converted to negative labels.**
- Among the 58 labelled studies there are 240 positive and 456 negative target values. These are not 696 independent studies; all targets for a given study are correlated.

### Positive counts among the 58 labelled studies

| Target | Positive studies |
|---|---:|
| Effusion | 35 |
| Synovitis | 27 |
| Medial Meniscus | 26 |
| ACL | 24 |
| Lateral Meniscus | 23 |
| PF OA | 21 |
| Contusion | 19 |
| Fracture | 18 |
| Medial OA | 15 |
| Baker's | 12 |
| Lateral OA | 11 |
| MCL | 9 |

These counts make reliable model selection and per-target validation difficult, especially for MCL and Lateral OA. Any validation score from a tiny holdout will have high uncertainty.

## Series metadata

- `train_series.csv` has 24,371 series across the training studies.
- Plane counts: sagittal 9,864; coronal 8,609; axial 5,898.
- The 58 labelled studies contain 336 series: 136 sagittal, 119 coronal, and 81 axial.
- In the labelled subset, series per study: mean 5.79, median 5, minimum 4, maximum 9.
- Slice count for those 336 series: median 30, mean 31.33, minimum 11, maximum 320.
- The longest series are not automatically corrupt. One inspected 320-slice series is described as a sagittal 3D proton-density acquisition, with approximately 0.8 mm slice thickness and 0.4 mm spacing. Treat these as cases to investigate, not outliers to discard by default.
- In one inspected 30-slice series, sorting by physical slice position agreed with InstanceNumber. This has only been verified for that series. DICOM filenames are UID-like and should not be assumed to encode slice order; validate geometric sorting across more series before using it in a loader.

## Reports and weak supervision: next experiment

Reports could potentially provide useful weak labels for studies without target labels, but this has not been tested. Report-derived labels must not be treated as ground truth without validation.

1. Inspect `test.csv` columns first; establish which inputs are actually available for test-time prediction.
2. Measure report presence, length, and language distribution in labelled and unlabelled training studies.
3. Read a small, deliberately varied sample of reports from both groups.
4. Prototype explicit rule-based extraction for one target (ACL is a reasonable starting point), including negation, uncertainty, and non-English text.
5. Compare extracted labels with the 58 ground-truth-labelled studies and manually inspect disagreements. With only 58 labelled studies, precision estimates will be uncertain.
6. Only expand the method if the prototype shows credible precision and competition rules permit using reports. Keep unknown or ambiguous cases unlabeled rather than forcing a 0/1 decision.

## Modelling implications

- Do not train a conventional image CNN from scratch on only 58 labelled studies and trust the resulting score.
- Do not use random slice-level splits. All series/slices from one study must remain in the same split to avoid leakage.
- Keep a record of the exact split and seed; compare experiments on the same validation studies.
- Report-derived weak supervision and pretrained image models are hypotheses to test, not guaranteed improvements.
- The earlier audit of all series folders was interrupted before completion. The 336-series labelled-subset audit completed; no complete all-series slice-count summary should be inferred from it.

## Not yet established

- The official competition metric and current rules have not been independently re-verified in this note.
- It has not yet been established whether test reports are available or usable for prediction.
- No model has been trained, no validation score has been measured, and no submission has been produced.
- The attached third-party JPEG dataset is excluded from experiments until its licence and competition-rule compatibility are checked.
