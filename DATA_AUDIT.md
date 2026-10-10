# Dataset Audit — Label Scarcity and MRI Structure

Date: 2026-10-10

This note records findings from the initial Kaggle notebook audit. These are observed dataset facts, not model-performance results.

## Label coverage

- `train.csv` has 4,407 studies and 14 columns: `StudyInstanceUID`, `Report`, and 12 target columns.
- Only 58/4,407 studies (about 1.32%) have known values for the target labels.
- Those 58 studies have all 12 targets populated, giving 696 known target values.
- The other 4,349 studies have all 12 target labels missing.
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

## Reports and weak supervision

### Observed report availability

- All 4,407 training studies have a non-empty `Report` field in the audited CSV.
- Labelled studies: 58/58 non-empty reports; median length 1,205 characters; 25th/75th percentiles 817.5/1,777 characters.
- Unlabelled studies: 4,349/4,349 non-empty reports; median length 973 characters; 25th/75th percentiles 586/1,455 characters.
- Reports are multilingual. The inspected sample included English, Spanish, Turkish, and Greek. The sample is illustrative, not a complete language-distribution audit.
- The inspected reports contain clinically descriptive findings and impressions, so report-derived weak labels may be worth testing.

### Test-time availability warning

- In the audited notebook, `test.csv` had shape `(3, 1)` and only one column: `StudyInstanceUID`. The three displayed rows were a sample, not the full test set.
- Therefore reports are **not supplied as a column in `test.csv`**. Do not assume test reports are available for direct prediction. Check official competition rules and the submission/data interface before deciding how reports can be used.
- Training reports may still be useful for understanding the data or for permitted training-time weak supervision, but that does not by itself make a report-only predictor usable at test time.

### Important report/label disagreements

The initial sample revealed possible mismatches between report wording and supplied targets:
- One report describes a lateral meniscus tear while the supplied `Lateral Meniscus` target is 0.
- One report describes a partial ACL tear while the supplied `ACL` target is 0.

These examples do not establish whether the cause is a different target definition, report/label disagreement, annotation error, or a misunderstanding of the report. They do establish that naive keyword matching should not be trusted without checking target definitions and evaluating disagreements. Do not overwrite supplied labels with report-derived labels.

### Proposed next experiment

1. Verify the official metric, competition rules, and permitted data use.
2. Audit report languages and inspect a small, varied sample systematically.
3. Prototype rule-based extraction for one target (ACL is a reasonable starting point), handling negation, uncertainty, historical mentions, and non-English text.
4. Compare extracted labels with the 58 labelled studies and manually inspect disagreements. With only 58 labelled studies, precision estimates will be uncertain.
5. Keep ambiguous cases unknown instead of forcing 0/1 labels. Expand only if the prototype shows credible precision and the method is permitted.
6. Do not build a report-only test predictor unless the needed test-time information is legitimately available.

## Modelling implications

- Do not train a conventional image CNN from scratch on only 58 labelled studies and trust the resulting score.
- Do not use random slice-level splits. All series/slices from one study must remain in the same split to avoid leakage.
- Keep a record of the exact split and seed; compare experiments on the same validation studies.
- Report-derived weak supervision and pretrained image models are hypotheses to test, not guaranteed improvements.
- The earlier audit of all series folders was interrupted before completion. The 336-series labelled-subset audit completed; no complete all-series slice-count summary should be inferred from it.

## Not yet established

- The official competition metric and current rules have not been independently re-verified in this note.
- The report-language distribution across all studies has not been measured.
- No model has been trained, no validation score has been measured, and no submission has been produced.
- The attached third-party JPEG dataset is excluded from experiments until its licence and competition-rule compatibility are checked.
