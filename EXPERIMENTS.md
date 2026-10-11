# Experiment Register

This register contains results from actual Kaggle notebook runs reported in the project conversation. Scores are exploratory and must not be treated as robust estimates because only 58 studies have complete expert labels and the validation set contains 15 studies.

| ID | Date | Purpose | Status | Result |
|---|---|---|---|---|
| EXP-001 | 2026-10-10 | Three-slice slice-level CNN baseline | Completed | Validation macro ROC-AUC 0.5018; training macro ROC-AUC 0.6782 |
| EXP-002 | 2026-10-10 | Study-level CNN with 3 slices as channels | Completed | Validation macro ROC-AUC 0.4628 |
| EXP-003 | 2026-10-10 | Study-level CNN with 9 slices as channels | Completed | Validation macro ROC-AUC 0.5684 |
| EXP-004 | 2026-10-11 | Frozen DINOv2 Base embeddings + logistic regression | Completed | Validation macro ROC-AUC 0.5717 |
| EXP-005 | 2026-10-11 | TF-IDF report-only logistic regression | Completed | Validation macro ROC-AUC 0.5351 |
| EXP-006 | 2026-10-11 | DINOv2 image embeddings + TF-IDF report features | Completed | Validation macro ROC-AUC 0.5665 |
| EXP-007 | 2026-10-11 | Conservative report-derived weak-label generation | Implementation committed; not run | No result yet |

## Shared data and validation context

- Competition data root in Kaggle: `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- Training CSV contains 4,407 studies; 58 have all 12 target labels present.
- The 58 fully labelled studies were split by study ID into 43 training and 15 validation studies using seed 42 and stratification on ACL only. This is not multi-label stratification.
- The split was disjoint by study ID. All slices from each study remained within that study's partition.
- Validation is very small and per-target prevalence differs substantially from training. For example, Fracture positives were 8/43 in training and 10/15 in validation.
- All three models were trained from scratch on a Kaggle Tesla T4 using PyTorch. No pretrained weights were used.
- The chosen series was the first fluid-sensitive sagittal series per study if available, otherwise a fluid-sensitive coronal series. The current selector's series metadata showed 4,150 sagittal and 257 coronal chosen series across 4,407 studies.
- Images were intensity-scaled per slice using the 1st and 99th percentiles, then resized to 96×96 by nearest-neighbour indexing. Slice order was based on DICOM InstanceNumber. DICOM geometry-based ordering/orientation handling has not yet been validated.
- Only 58 fully labelled studies were used for supervised model training; unlabelled studies were not treated as negatives.
- No model checkpoint, runtime, peak GPU memory, or exact notebook version/commit was recorded in the supplied outputs. These omissions should be fixed for future runs.

## EXP-001 — Three-slice slice-level CNN baseline

- **Date:** 2026-10-10
- **Status:** completed; exploratory baseline
- **Hypothesis:** a small CNN trained on three sampled MRI slices can establish an end-to-end supervised baseline.
- **Change from baseline:** first model; three slices from one selected series per study. Each slice was a separate training example inheriting its study-level labels; validation probabilities were averaged across the three slices.
- **Data version:** official Kaggle competition input as mounted in the notebook; CSV/data hashes not recorded.
- **Validation split ID:** SPLIT-42-ACL-43-15
- **Model and configuration:** small 2D CNN (channels 3→16→32→64→96, adaptive average pooling, 64-unit head, dropout 0.3); AdamW learning rate 1e-3, weight decay 1e-3; batch size 12; 5 epochs; BCE-with-logits loss.
- **Random seed:** 42
- **Environment:** Kaggle, PyTorch 2.11.0+cu128, Python 3.13.15, Tesla T4 GPU.
- **Runtime / peak memory:** not recorded.
- **Primary metric:** validation macro ROC-AUC across 12 targets, mean of target AUCs.
- **Training loss:** 0.6738, 0.6395, 0.6127, 0.5928, 0.5943.
- **Training macro ROC-AUC diagnostic:** 0.6782.
- **Validation macro ROC-AUC:** 0.5018 (12/12 targets had both classes in this split).
- **Per-target validation AUC:** ACL 0.315; MCL 0.389; Medial Meniscus 0.464; Lateral Meniscus 0.704; Medial OA 0.346; Lateral OA 0.556; PF OA 0.583; Effusion 0.820; Synovitis 0.296; Baker's 0.722; Contusion 0.426; Fracture 0.400.
- **Prediction diagnostic:** mean per-target score SD 0.0268 on validation and 0.0319 on training.
- **Failure modes / caveats:** 43 training studies only; slice-level training labels may not be visible on every slice; strong train-validation gap; highly uncertain single-split estimate.
- **Conclusion:** pipeline trained and scored end-to-end, but overall validation ranking was near random.
- **Decision:** retain as baseline for comparison; do not claim competitive performance.
- **Code commit:** notebook code was executed in Kaggle but no corresponding repository commit or notebook export was recorded.

## EXP-002 — Study-level CNN, three slices as channels

- **Date:** 2026-10-10
- **Status:** completed; exploratory comparison
- **Hypothesis:** computing loss once per study may better align training with study-level labels.
- **Change from EXP-001:** three ordered grayscale slices were stacked as three input channels, with one training example and one loss vector per study.
- **Data version / split:** same official Kaggle input and SPLIT-42-ACL-43-15.
- **Model and configuration:** small CNN with 3 input channels, convolution widths 8/16/24, adaptive average pooling, dropout 0.4; AdamW learning rate 5e-4, weight decay 0.02; batch size 8; up to 30 epochs; patience 5 based on training loss; BCE-with-logits loss.
- **Random seed:** 42 for torch and NumPy.
- **Environment:** Kaggle Tesla T4; exact versions not reprinted for this run.
- **Runtime / peak memory:** not recorded.
- **Validation macro ROC-AUC:** 0.4628.
- **Per-target validation AUC:** ACL 0.333; MCL 0.361; Medial Meniscus 0.661; Lateral Meniscus 0.278; Medial OA 0.923; Lateral OA 0.500; PF OA 0.333; Effusion 0.320; Synovitis 0.389; Baker's 0.472; Contusion 0.463; Fracture 0.520.
- **Prediction diagnostic:** mean validation prediction SD 0.0168.
- **Failure modes / caveats:** scores lower than EXP-001; very compressed predictions; tiny validation sample, particularly only 2 Medial OA positives; early stopping on training loss is not validation-based model selection.
- **Conclusion:** the study-level loss setup alone did not improve this particular small CNN.
- **Decision:** retain as a comparison; do not adopt based on score.
- **Code commit:** notebook code was executed in Kaggle but no corresponding repository commit or notebook export was recorded.

## EXP-003 — Study-level CNN, nine slices as channels

- **Date:** 2026-10-10
- **Status:** completed; exploratory comparison
- **Hypothesis:** wider sampling of the selected series may expose more anatomy than three slices.
- **Change from EXP-002:** nine evenly spaced slices per study, stacked as nine channels; otherwise similar small study-level CNN.
- **Data version / split:** same official Kaggle input and SPLIT-42-ACL-43-15.
- **Data check:** 58/58 labelled studies decoded; 43 training and 15 validation studies; 522 cached slices (9 per study).
- **Model and configuration:** CNN with 9 input channels, convolution widths 8/16/24, adaptive average pooling, dropout 0.4; AdamW learning rate 5e-4, weight decay 0.02; batch size 8; up to 30 epochs; patience 5 based on training loss; BCE-with-logits loss.
- **Random seed:** 42 for torch and NumPy.
- **Environment:** Kaggle Tesla T4; exact versions not reprinted for this run.
- **Runtime / peak memory:** not recorded.
- **Training BCE losses:** epochs 1–20 as reported: 0.6865, 0.6848, 0.6832, 0.6756, 0.6694, 0.6602, 0.6423, 0.6446, 0.6295, 0.6276, 0.6269, 0.6296, 0.6253, 0.6270, 0.6028, 0.6267, 0.6216, 0.6192, 0.6139, 0.6078. (The notebook output lists epoch 19 as 0.6078 and epoch 20 as 0.6139; preserve the displayed values as reported rather than infer a corrected order.)
- **Validation macro ROC-AUC:** 0.5684.
- **Per-target validation AUC:** ACL 0.389; MCL 0.417; Medial Meniscus 0.661; Lateral Meniscus 0.852; Medial OA 0.885; Lateral OA 0.583; PF OA 0.778; Effusion 0.240; Synovitis 0.481; Baker's 0.556; Contusion 0.500; Fracture 0.480.
- **Prediction diagnostic:** mean validation prediction SD was not reported for this run.
- **Failure modes / caveats:** nine-channel input changes both sampling and first-layer architecture, so this is not a perfectly isolated slice-count experiment; one 15-study validation split is noisy and has a marked train/validation prevalence shift; the very high Medial OA AUC is based on only two validation positives.
- **Conclusion:** best of the three exploratory runs on this split, but the difference is not sufficient evidence of generalisation or competition readiness.
- **Decision:** keep as current experimental leader, not as a validated final model. Next prioritise robust validation, more principled MRI slice/series selection, and a permitted pretrained encoder rather than blindly tuning epochs.
- **Code commit:** notebook code was executed in Kaggle but no corresponding repository commit or notebook export was recorded.

## Cross-experiment summary

| Experiment | Validation macro ROC-AUC | Validation mean prediction SD | Interpretation |
|---|---:|---:|---|
| EXP-001: 3-slice slice-level CNN | 0.5018 | 0.0268 | Near-random overall; train macro AUC 0.6782 suggests a train/validation gap |
| EXP-002: 3-slice study-level CNN | 0.4628 | 0.0168 | Worse on this split; predictions more compressed |
| EXP-003: 9-slice study-level CNN | 0.5684 | Not reported | Highest observed score among scratch CNNs; still highly uncertain |
| EXP-004: DINOv2 Base + logistic regression | 0.5717 | Not reported | Highest observed single-split score; only +0.0033 vs EXP-003 |
| EXP-005: TF-IDF text-only | 0.5351 | Not reported | Below both image baselines on this split |
| EXP-006: DINOv2 + TF-IDF | 0.5665 | Not reported | Below image-only EXP-004 by 0.0052 |

## Experiment record template

Copy for each meaningful experiment:

### EXP-XXX — Short title
- **Date:**
- **Status:** planned / running / completed / failed
- **Hypothesis:**
- **Change from baseline:**
- **Data version:**
- **Validation split ID:**
- **Model and configuration:**
- **Random seed:**
- **Environment:**
- **Runtime / peak memory:**
- **Primary metric:**
- **Per-target metrics:**
- **Failure modes / caveats:**
- **Conclusion:**
- **Decision:** keep / reject / investigate
- **Code commit:**

### EXP-004 — DINOv2 Base embeddings + logistic regression
- **Date:** 2026-10-11
- **Status:** completed; exploratory baseline
- **Hypothesis:** frozen pretrained visual embeddings may transfer better than a CNN trained from scratch on 43 labelled studies.
- **Change from baseline:** DINOv2 Base (Kaggle Models, metaresearch/dinov2, PyTorch Base Version 1; Apache 2.0 as displayed on the model page) used as a frozen encoder. Nine evenly spaced slices per study; CLS token (768 dimensions) extracted per slice and averaged to one study vector; logistic regression per target.
- **Data:** 58 fully labelled studies; 43 train / 15 validation; split ID SPLIT-42-ACL-43-15. Weak/unlabelled studies were not used for this experiment.
- **Feature cache:** shape (58, 9, 768); all values finite; unique study IDs; 522 slice embeddings; extraction completed in 48.79 seconds on CUDA. Cache remained in Kaggle working storage and was not committed.
- **Validation macro ROC-AUC:** 0.5717 across 12 targets.
- **Per-target AUC:** ACL 0.2778; MCL 0.5833; Medial Meniscus 0.5179; Lateral Meniscus 0.6667; Medial OA 0.7308; Lateral OA 0.8056; PF OA 0.5833; Effusion 0.8200; Synovitis 0.5741; Baker's 0.5000; Contusion 0.4815; Fracture 0.3200.
- **Conclusion:** +0.0033 versus EXP-003, not meaningful evidence of improvement given the tiny validation set.
- **Decision:** keep as the best observed single-split result, but do not claim competition readiness.

### EXP-005 — TF-IDF report-only baseline
- **Date:** 2026-10-11
- **Status:** completed; exploratory baseline
- **Hypothesis:** diagnostic reports may encode target information.
- **Method:** TF-IDF word unigrams/bigrams, vocabulary fitted on the 43 training reports only, one balanced logistic regression per target. Official labels only; no weak labels used.
- **Split:** SPLIT-42-ACL-43-15.
- **Validation macro ROC-AUC:** 0.5351.
- **Per-target AUC:** ACL 0.4815; MCL 0.6389; Medial Meniscus 0.5179; Lateral Meniscus 0.5741; Medial OA 0.9231; Lateral OA 0.5278; PF OA 0.3889; Effusion 0.6200; Synovitis 0.3889; Baker's 0.2778; Contusion 0.4630; Fracture 0.6200.
- **Conclusion:** lower than image-only baselines on this split. This does not establish that report-derived supervision is useless because only 43 reports were used for supervised fitting.

### EXP-006 — Combined DINOv2 and TF-IDF features
- **Date:** 2026-10-11
- **Status:** completed; exploratory baseline
- **Hypothesis:** text features may add complementary information to image embeddings.
- **Method:** averaged DINOv2 study embeddings concatenated with TF-IDF report features; balanced logistic regression with C=0.1; transformations fitted on the training split only.
- **Split:** SPLIT-42-ACL-43-15.
- **Validation macro ROC-AUC:** 0.5665.
- **Per-target AUC:** ACL 0.2593; MCL 0.5833; Medial Meniscus 0.4821; Lateral Meniscus 0.6667; Medial OA 0.6923; Lateral OA 0.8333; PF OA 0.5833; Effusion 0.8400; Synovitis 0.5926; Baker's 0.4444; Contusion 0.5000; Fracture 0.3200.
- **Conclusion:** 0.0052 below EXP-004. Concatenation did not improve the score on this split.
- **Decision:** do not tune repeatedly on the same 15-study validation set; investigate weak supervision from the 4,349 unlabelled reports instead.

### EXP-007 — Conservative report-derived weak-label generation
- **Date:** 2026-10-11
- **Status:** implementation committed; not yet run
- **Hypothesis:** explicit report findings may provide noisy labels for some of the 4,349 studies that lack official labels.
- **Method:** conservative multilingual rules emit positive (1), explicit negative (0), or unknown (blank). Reports with any official target label are excluded from weak-label generation. Silence is never treated as negative; uncertain/conflicting cues remain unknown.
- **Outputs:** weak_labels.csv, weak_label_audit.csv, and weak_label_summary.csv in Kaggle working storage.
- **Validation:** none yet. No label coverage, label accuracy, or model performance has been claimed.
- **Required next step:** run the script with GPU OFF; manually inspect a sample of cues/statuses and measure agreement against the 58 gold-labelled reports as a diagnostic only. Do not use those 58 studies as weak-label training examples.
- **Code:** scripts/derive_report_weak_labels.py; experiment plan: reports/2026-10-11-exp007-weak-label-plan.md.
