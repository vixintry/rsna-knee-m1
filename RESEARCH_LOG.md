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
