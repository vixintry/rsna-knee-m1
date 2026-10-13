# EXP-011 — DINOv2 image-only baseline with five-fold study-level OOF validation

- **Status:** completed; preliminary baseline re-evaluation.
- **Purpose:** evaluate frozen DINOv2 features using the fixed, audited EXP-010 folds.
- **Validation protocol:** five folds; 58 fully labelled studies; split unit is StudyInstanceUID. The saved EXP-010 fold assignment was reused unchanged. Fold sizes: 13, 10, 12, 12, 11.
- **Inputs:** cached DINOv2 Base features, mean of nine slice embeddings per study; 768 features per study.
- **Classifier:** one binary logistic-regression model per target, using StandardScaler followed by LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42). The scaler is fitted only on each fold's training studies.
- **GPU:** off for this classifier evaluation; image feature cache was prepared separately.
- **Targets:** all 12 official targets; evaluation uses only the 58 studies with complete official labels. Studies with missing target labels were not treated as negatives.
- **Pooled OOF macro ROC-AUC:** 0.589270.
- **Mean per-fold macro ROC-AUC:** 0.594066.
- **Standard deviation of per-fold macro ROC-AUC:** 0.054762.
- **Important:** pooled OOF macro-AUC and mean per-fold macro-AUC are different aggregations and are reported separately.

## Per-target pooled OOF ROC-AUC

| Target | Studies | Positive | Negative | OOF ROC-AUC |
|---|---:|---:|---:|---:|
| ACL | 58 | 24 | 34 | 0.609069 |
| MCL | 58 | 9 | 49 | 0.421769 |
| Medial Meniscus | 58 | 26 | 32 | 0.474760 |
| Lateral Meniscus | 58 | 23 | 35 | 0.506832 |
| Medial OA | 58 | 15 | 43 | 0.755039 |
| Lateral OA | 58 | 11 | 47 | 0.740812 |
| PF OA | 58 | 21 | 37 | 0.543115 |
| Effusion | 58 | 35 | 23 | 0.642236 |
| Synovitis | 58 | 27 | 31 | 0.467145 |
| Baker's | 58 | 12 | 46 | 0.695652 |
| Contusion | 58 | 19 | 39 | 0.539811 |
| Fracture | 58 | 18 | 40 | 0.675000 |

## Per-fold macro ROC-AUC

| Fold | Validation studies | Macro ROC-AUC |
|---:|---:|---:|
| 0 | 13 | 0.680060 |
| 1 | 10 | 0.595198 |
| 2 | 12 | 0.580911 |
| 3 | 12 | 0.586152 |
| 4 | 11 | 0.528009 |

## Interpretation and limitations

- The strongest per-target scores in this run were Medial OA (0.755), Lateral OA (0.741), Baker's cyst (0.696), and Fracture (0.675); weakest were MCL (0.422), Synovitis (0.467), and Medial Meniscus (0.475).
- Only 58 fully labelled studies are available. Per-target support is small and uneven, so estimates have substantial uncertainty.
- These values are a baseline, not evidence of competition-level performance. Do not over-interpret small score differences or tune against individual folds.
- Keep the EXP-010 folds fixed for subsequent controlled comparisons. Any methodology changes must be documented as a new experiment.
- Runtime and confidence intervals were not recorded in the output; do not infer them.

## Artifacts

The Kaggle run saved six files under `/kaggle/working/exp011_cv_results`:
- `exp011_config.json`
- `exp011_fold_metrics.csv`
- `exp011_oof_predictions.csv`
- `exp011_per_fold_macro_auc.csv`
- `exp011_per_target_auc.csv`
- `exp011_summary.csv`

Five raw artifacts have now been committed from the actual uploaded files:
- [`exp011_config.json`](exp011_config.json)
- [`exp011_fold_metrics.csv`](exp011_fold_metrics.csv)
- [`exp011_per_fold_macro_auc.csv`](exp011_per_fold_macro_auc.csv)
- [`exp011_per_target_auc.csv`](exp011_per_target_auc.csv)
- [`exp011_summary.csv`](exp011_summary.csv)

The actual per-study image-only OOF file (`exp011_oof_predictions.csv`, 696 rows) is now committed in this directory. It was checked directly: 696 rows, 58 unique studies, 12 targets per study, no duplicate study–target pairs, and all `y_prob` values finite and within [0, 1]. Recomputed pooled OOF macro ROC-AUC matches the recorded 0.589270. Preserve this file as the raw prediction artifact; do not reconstruct it from aggregate metrics.


## EXP-011 (continued) — TF-IDF report-text diagnostic

This is a separate diagnostic baseline evaluated using the **same fixed EXP-010 folds** as the image-only baseline. It is not a test-time model because reports are unavailable for test studies.

- **Input:** report text from the 58 fully labelled training studies.
- **Text vectorizer:** TF-IDF, word unigrams and bigrams, lowercase, Unicode accent stripping, min_df=1, max_features=20,000, sublinear TF. A separate vocabulary was fitted on each fold's training reports only.
- **Classifier:** per-target LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42).
- **Pooled OOF macro ROC-AUC:** 0.555506.
- **Mean fold macro ROC-AUC:** 0.576350.
- **Population SD of fold macro ROC-AUC:** 0.024897.

### Per-target pooled OOF ROC-AUC

| Target | Studies | Positive | Negative | OOF ROC-AUC |
|---|---:|---:|---:|---:|
| ACL | 58 | 24 | 34 | 0.681373 |
| MCL | 58 | 9 | 49 | 0.371882 |
| Medial Meniscus | 58 | 26 | 32 | 0.671875 |
| Lateral Meniscus | 58 | 23 | 35 | 0.509317 |
| Medial OA | 58 | 15 | 43 | 0.719380 |
| Lateral OA | 58 | 11 | 47 | 0.599613 |
| PF OA | 58 | 21 | 37 | 0.522523 |
| Effusion | 58 | 35 | 23 | 0.563975 |
| Synovitis | 58 | 27 | 31 | 0.500597 |
| Baker's | 58 | 12 | 46 | 0.384058 |
| Contusion | 58 | 19 | 39 | 0.530364 |
| Fracture | 58 | 18 | 40 | 0.611111 |

### Per-fold macro ROC-AUC

| Fold | Validation studies | Macro ROC-AUC |
|---:|---:|---:|
| 0 | 13 | 0.582970 |
| 1 | 10 | 0.546806 |
| 2 | 12 | 0.547492 |
| 3 | 12 | 0.604384 |
| 4 | 11 | 0.600099 |

### Comparison and decision

- Image-only pooled OOF macro-AUC: **0.589270**.
- Text-only pooled OOF macro-AUC: **0.555506**.
- Difference (text minus image): **-0.033764**.
- The text score is lower in this small sample. This does not prove text has no value, and the two scores should be treated as preliminary because only 58 studies have complete official labels.
- **Decision:** retain as a diagnostic result only. Test reports are unavailable, so the final submission must predict from permitted test-time inputs such as MRI images. Report-derived training supervision can be evaluated separately with strict fold isolation and explicit handling of noisy labels.

### Text-run artifacts

The Kaggle run wrote these six files to `/kaggle/working/exp011_text_cv_results`:
- `exp011_text_config.json`
- `exp011_text_fold_metrics.csv`
- `exp011_text_oof_predictions.csv`
- `exp011_text_per_fold_macro_auc.csv`
- `exp011_text_per_target_auc.csv`
- `exp011_text_summary.csv`

**Raw artifact status:** the six text-run CSV/JSON artifacts are not yet committed. Add them only from the actual saved Kaggle outputs; do not reconstruct per-study predictions from aggregate metrics.
