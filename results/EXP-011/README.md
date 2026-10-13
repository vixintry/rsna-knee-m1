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

A verified ZIP backup was created at `/kaggle/working/exp011_cv_results_backup.zip`. The repository record above transcribes the metrics visible in the run output; the complete per-study OOF prediction file should only be committed when its actual contents are available, not reconstructed from summary metrics.
