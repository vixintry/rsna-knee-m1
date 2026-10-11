# Experiment Register

This register contains results from actual Kaggle notebook runs reported in the project conversation. Scores are exploratory and must not be treated as robust estimates because only 58 studies have complete expert labels and the validation set contains 15 studies.

| ID | Date | Purpose | Status | Result |
|---|---|---|---|---|
| EXP-001 | 2026-10-10 | Three-slice slice-level CNN baseline | Completed | Validation macro ROC-AUC 0.5018; training macro ROC-AUC 0.6782 |
| EXP-002 | 2026-10-10 | Study-level CNN with 3 slices as channels | Completed | Validation macro ROC-AUC 0.4628 |
| EXP-003 | 2026-10-10 | Study-level CNN with 9 slices as channels | Completed | Validation macro ROC-AUC 0.5684 |

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
| EXP-003: 9-slice study-level CNN | 0.5684 | Not reported | Highest observed score so far; still highly uncertain |

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

### EXP-004 — DINOv2 Base feature extraction (setup in progress)
- **Date:** 2026-10-10
- **Status:** setup complete; inference smoke test pending
- **Hypothesis:** a pretrained visual encoder will provide more useful image features than the scratch CNNs given only 58 fully labelled studies.
- **Change from baseline:** use the public Meta DINOv2 Base checkpoint as a frozen feature encoder before training a small 12-label classifier.
- **Model source / licence:** Kaggle Models, `metaresearch/dinov2`, PyTorch Base, Version 1; Apache 2.0 as shown on the model page.
- **Checkpoint location in Kaggle:** `/kaggle/input/models/metaresearch/dinov2/pytorch/base/1`
- **Files observed:** `config.json`, `preprocessor_config.json`, `README.md`, `pytorch_model.bin`, `.gitattributes`; weights reported as 346.39 MB on the model page.
- **Environment:** Kaggle notebook; Tesla T4 was available in previous experiments, but GPU availability for this run has not yet been confirmed.
- **Validation split:** intended to reuse SPLIT-42-ACL-43-15 (43 train / 15 validation studies).
- **Results:** none yet. Checkpoint attachment and file discovery are confirmed; successful model loading, feature extraction, training, and validation are not yet recorded.
- **Next steps:** load with `local_files_only=True`; test one decoded knee slice; verify preprocessing and feature shape; then cache fixed-slice embeddings and train/evaluate the small multi-label classifier.
- **Guardrail:** do not record an AUC or claim improvement until the run actually completes; compare on the same split and report all 12 target AUCs.


## EXP-008 — DINOv2 test inference and submission generation

- **Date added:** 2026-10-11
- **Status:** smoke-tested on the three example test studies; full hidden-test run not yet executed
- **Hypothesis:** the EXP-004 frozen DINOv2 embeddings plus per-target logistic regression can produce a structurally valid submission from permitted test-time MRI inputs.
- **Change:** extract nine evenly spaced slices from the selected fluid-sensitive sagittal series (coronal fluid-sensitive fallback), average DINOv2 CLS embeddings, fit one standardized logistic-regression classifier per target on all 58 fully labelled studies, and predict the test studies.
- **Data/split:** all 58 fully labelled studies used for final fitting after EXP-004's fixed split has served for exploratory validation; test studies are used only for inference.
- **Output:** `/kaggle/working/submission.csv`; intermediate test embeddings and metadata are saved under `/kaggle/working/exp008_test_features`.
- **Integrity checks in code:** target-column order, row count, study-ID order/uniqueness, finite values, and probabilities in [0, 1]; strict failure on missing eligible series or invalid labels/features.
- **Smoke-test run:** executed in Kaggle with GPU enabled (CUDA). EXP-004 feature extraction completed for 58 labelled studies in 37.45 s; test feature extraction completed for 3 example studies in 1.87 s. Test feature shape `(3, 9, 768)`.
- **Output validation:** `/kaggle/working/submission.csv`, shape `(3, 13)`, exact target columns/order, all predictions finite and within `[0, 1]`.
- **Scope limitation:** Kaggle's checked-in sample test files intentionally contain 3 example studies; scoring substitutes the hidden test set. The full scoring-environment run has not yet been executed.
- **Leaderboard result:** none. This is an inference smoke test, not evidence of predictive performance.
- **Next action:** evaluate models under the planned study-level out-of-fold protocol, then verify full hidden-test inference and runtime.


## Competitive research roadmap (planned, not completed)

The detailed rationale and acceptance criteria are in [docs/COMPETITIVE_RESEARCH_PLAN.md](docs/COMPETITIVE_RESEARCH_PLAN.md). Planned experiments:

| ID | Work | Status |
|---|---|---|
| EXP-009 | DICOM geometry, slice order, series/sequence coverage audit | Planned |
| EXP-010 | Deterministic study-level multi-label-aware OOF folds and metric tests | Planned |
| EXP-011 | Re-evaluate image and text baselines on identical OOF folds | Planned |
| EXP-012 | 2.5D adjacent-slice windows | Planned |
| EXP-013 | Multi-plane/sequence ablations | Planned |
| EXP-014 | LLM-assisted report-label quality audit | Planned |
| EXP-015 | Weak-supervised image training with strict gold-fold isolation | Planned |
| EXP-016 | Clinical-state auxiliary targets | Planned |
| EXP-017 | Verify and evaluate a medical-image foundation encoder | Planned |
| EXP-018 | Evaluate a complementary ensemble using OOF predictions | Planned |

None of EXP-009–EXP-018 has been run by virtue of being listed here. Do not assign them scores until measured.
