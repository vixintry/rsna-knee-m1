# Research Log

This log records observations supported by actual inspection or execution. Separate confirmed facts from hypotheses.

## 2026-10-10 — Project initialization and competition audit

### Competition facts supplied by the user
- The task predicts 12 binary abnormality labels per knee MRI study.
- `train.csv` contains study identifiers and radiology reports, plus the 12 labels as described by the user.
- `train_series.csv` describes series metadata, including fluid sensitivity, fat suppression, and anatomical plane.
- DICOM files are organized by study, series, and slice.
- The supplied dataset description says test reports are unavailable during scoring.
- The supplied description reports approximately 569.76 GB and about 820,000 files.
- The supplied description says only a small subset of training studies has per-condition labels, while reports are provided for additional studies.

These data details come from the competition description pasted by the user; they have not yet been independently audited against downloaded files.

### Official competition page checked on 2026-10-10
Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/

- **Metric:** macro-averaged ROC AUC across the 12 targets, with each target weighted equally.
- **Entry/rules acceptance deadline:** 15 October 2026, 23:59 UTC.
- **Team merger deadline:** 15 October 2026, 23:59 UTC.
- **Final submission deadline:** 22 October 2026, 23:59 UTC.
- **Submission format:** Kaggle Notebook submission, output file named `submission.csv`.
- **Notebook limits:** CPU or GPU notebook runtime must be no more than 9 hours; internet access must be disabled for submission execution.
- **External resources:** the competition overview says freely and publicly available external data and pretrained models are allowed. Read the full rules and check each resource's licence before using any.

The official page may change; recheck deadlines and requirements before the final submission. The user's screenshot showed an identity-verification prompt. The user must complete any required account/competition eligibility steps themselves; do not assume entry is complete until Kaggle confirms it.

### Local compute reported from screenshots
- CPU: Intel Core i5-12400F, 6 cores / 12 logical processors.
- RAM: 16 GB DDR4; screenshots showed roughly 13.4–13.6 GB in use at the time.
- GPU: AMD Radeon RX 6600, 8 GB dedicated GPU memory.
- Storage: 1 TB NVMe SSD, approximately 954 GB formatted capacity.

These specifications are read from user-provided screenshots. Actual free memory/storage and throughput must be checked before experiments. The dataset's stated 569.76 GB size makes a full local download risky given the available drive capacity and other system use; prefer first inspecting files through Kaggle and evaluating hosted compute.

### Current status
- Repository governance documents committed.
- Competition metric and main deadlines checked against the official overview.
- No dataset audit has been run.
- No model has been trained.
- No validation score or leaderboard result has been measured.
- No submission has been verified.

### Next actions
1. Accept the competition rules and resolve Kaggle's required identity verification if not already done.
2. Open the Data tab in Kaggle and confirm the actual CSV schema and label coverage.
3. Avoid downloading the entire DICOM dataset locally; start with small CSV audits.
4. Audit label missingness, prevalence, and study/series identifier consistency.
5. Choose a validation protocol based on the number of genuinely labelled studies.


## 2026-10-11 — Corrected current status and competitive direction

This entry supersedes the earlier initialization snapshot above where it conflicts with later executed work. The earlier snapshot is retained as historical context.

### Confirmed data audit

- train.csv contains 4,407 studies and 12 target columns plus Report.
- 58 studies have complete official labels. The other 4,349 studies must not be treated as having negative labels.
- The mounted test.csv and sample_submission.csv contain three example studies. The official competition scoring workflow substitutes the hidden test set; the three-row file is not a misconfigured full test set.
- train_series.csv contains 24,371 series rows; test_series.csv contains 15 rows in the example data.
- The series metadata includes fluid sensitivity, fat suppression and anatomical plane. Correctly using this metadata and the DICOM geometry is an important next step.

### Measured baseline results

The current experiment register contains EXP-001 through EXP-006 with fixed-split macro AUCs between 0.4628 and 0.5717. The fixed split contains 43 training and 15 validation studies. These results are highly uncertain and are not a reliable estimate of hidden-test performance.

### EXP-007 weak-report-label audit

The deterministic report parser and its seven regression cases were run in Kaggle; all 7/7 passed. The audit processed the 4,349 studies without complete official labels. Report-derived labels are noisy: some targets have very few usable negatives, and uncertain/conflicting cases remain. Decision: retain this as an audit, not as a source of indiscriminately hard training labels.

### EXP-008 inference smoke test

- The EXP-004 DINOv2 feature cache was regenerated for 58 fully labelled studies on CUDA in 37.45 seconds; cached feature shape was (58, 9, 768) and values were finite.
- EXP-008 extracted features for the three example test studies on CUDA in 1.87 seconds; test feature shape was (3, 9, 768).
- It wrote /kaggle/working/submission.csv, shape (3, 13), with the expected column order and finite probabilities in [0, 1].
- This confirms only that the pipeline ran on the example data. It does not establish model quality, full hidden-test execution, or leaderboard performance.

### Competitive research notes recovered

The user-provided Library document “🦴 RSNA Knee 2026.docx” contains a much more ambitious approach than the current DINOv2 baseline. It describes: report-derived soft/probabilistic labels from unlabelled studies; richer clinical-state targets; adjacent-slice 2.5D windows; multiple sagittal/coronal/axial planes and complementary sequences; DICOM geometry and physical-scale handling; label-specific attention across image windows; and medical-image foundation encoders. The document also reports high leaderboard/AUC figures, but those claims, external weights and exact reproducibility have **not** been independently verified in this project. They are hypotheses/reference material, not our results.

The intended strategy is to use an LLM as a structured, auditable report-labeling assistant to unlock useful supervision from reports, then train an image model to predict from permitted MRI inputs. At validation time, held-out gold studies must be excluded from every training path, including report-derived targets. Test reports remain unavailable and must never be used.

### Approved next phase

Follow docs/COMPETITIVE_RESEARCH_PLAN.md:
1. Audit DICOM geometry, series selection and slice order.
2. Build deterministic study-level out-of-fold validation with per-label support and explicit undefined-AUC handling.
3. Re-evaluate baselines on the same folds.
4. Test 2.5D/multi-plane inputs and report-derived soft/state supervision through controlled ablations.

No EXP-009 or later experiment is claimed complete in this entry.


## 2026-10-13 — EXP-010 audit and EXP-011 fixed-fold baseline

### EXP-010 fold integrity

- The saved fold assignment was verified: 58 rows, 58 unique StudyInstanceUIDs, and exact coverage of the 58 complete-label studies.
- Fold sizes were {0: 13, 1: 10, 2: 12, 3: 12, 4: 11}.
- Independently recomputed per-target positive/negative support matched the saved fold-support artifact.

### EXP-011 measured results

- Reused the saved EXP-010 folds unchanged; no fold regeneration for model evaluation.
- Used cached frozen DINOv2 Base mean-of-nine-slice features (768 dimensions per study) and a per-target StandardScaler + balanced logistic regression pipeline (C=1.0, max_iter=2000, random_state=42). Scaling was fitted on training studies within each fold.
- GPU was off for this classifier evaluation.
- 58 fully labelled studies and 12 targets were evaluated. Missing labels were not converted to negatives.
- Pooled OOF macro ROC-AUC: 0.589270.
- Mean per-fold macro ROC-AUC: 0.594066; standard deviation across folds: 0.054762.
- Per-fold macro AUCs: fold 0 = 0.680060, fold 1 = 0.595198, fold 2 = 0.580911, fold 3 = 0.586152, fold 4 = 0.528009.
- Per-target AUCs and class support are recorded in [results/EXP-011/README.md](results/EXP-011/README.md).
- Six raw result files were saved under Kaggle's `/kaggle/working/exp011_cv_results` and a ZIP was verified. This repository record transcribes the exact metrics available in the execution output; raw per-study OOF predictions are not recreated from aggregates.

### Interpretation / next action

The result is preliminary because only 58 fully labelled studies are available and target support is uneven. Do not claim robust generalisation. Preserve the EXP-010 folds and compare any next model on those same folds. Before a text comparison, inspect the existing report source and EXP-005 implementation, and verify that all preprocessing is fitted within training folds. Test-time reports are unavailable and must not be used as inference inputs.


### EXP-011 continued — TF-IDF report-text diagnostic (2026-10-13)

- Reused the existing EXP-010 five-fold study-level assignments unchanged: 58 studies, fold sizes {0: 13, 1: 10, 2: 12, 3: 12, 4: 11}.
- Used TF-IDF word unigrams and bigrams (lowercase, Unicode accent stripping, min_df=1, max_features=20,000, sublinear TF). Vocabulary fitting occurred independently within each training fold; validation reports were transformed using that fold's training vocabulary.
- Fit one balanced logistic-regression classifier per target per fold (C=1.0, max_iter=2000, random_state=42).
- Pooled OOF macro ROC-AUC: 0.555506. Mean fold macro ROC-AUC: 0.576350; population SD across fold scores: 0.024897.
- Fold macro-AUCs: 0.582970, 0.546806, 0.547492, 0.604384, 0.600099.
- Per-target pooled OOF AUCs: ACL 0.681373; MCL 0.371882; Medial Meniscus 0.671875; Lateral Meniscus 0.509317; Medial OA 0.719380; Lateral OA 0.599613; PF OA 0.522523; Effusion 0.563975; Synovitis 0.500597; Baker's 0.384058; Contusion 0.530364; Fracture 0.611111.
- Image-only EXP-011 pooled OOF macro-AUC was 0.589270, so the text-only result was 0.033764 lower. These are exploratory estimates based on only 58 fully labelled studies.
- Inference restriction confirmed from competition data description: test reports are unavailable at scoring. This baseline is diagnostic only and must not be used as a required test-time input. Report-derived weak supervision remains a separate possible approach, subject to noise audits and gold-fold isolation.
- The run saved six output files under Kaggle's `/kaggle/working/exp011_text_cv_results`. Raw outputs are not yet committed to Git; do not reconstruct per-study predictions from aggregate metrics.
