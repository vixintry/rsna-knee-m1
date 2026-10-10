# MRI Pipeline Findings — 2026-10-10

This report records outputs from the first three Kaggle MRI model experiments. Results were pasted from executed notebook cells; the notebook source/revision, runtime, peak memory, and checkpoints were not recorded. These are exploratory results, not evidence of competition readiness.

## Data and split
- Fully labelled studies: 58, all IDs unique; no duplicate study IDs.
- Split: 43 training / 15 validation studies, seed 42, stratified on ACL only. Study IDs are disjoint. This is not multi-label stratification.
- The series metadata has 4,407 studies. A selector found preferred fluid-sensitive sagittal/coronal series for all 4,407: 4,150 sagittal and 257 coronal. This is not the count of fully labelled studies.
- For the 58 labelled studies, the selected series had 1,949 DICOM slices: mean 33.60, median 29, min 15, max 320.
- Nine evenly spaced slices decoded successfully for all 58 labelled studies (522 slices cached).
- Kaggle GPU: Tesla T4. Python 3.13.15 and PyTorch 2.11.0+cu128 were reported for the first run.

## Experiment results

| ID | Method | Training result | Validation macro ROC-AUC | Mean validation prediction SD |
|---|---|---:|---:|---:|
| EXP-001 | Small CNN; 3 slices trained as separate examples, study probabilities averaged | Training macro AUC 0.6782; BCE losses 0.6738, 0.6395, 0.6127, 0.5928, 0.5943 | 0.5018 | 0.0268 |
| EXP-002 | Small CNN; 3 slices stacked as channels, one loss per study | Stopped on training loss; see experiment register | 0.4628 | 0.0168 |
| EXP-003 | Small CNN; 9 slices stacked as channels, one loss per study | Stopped on training loss; see experiment register | 0.5684 | Not reported |

### EXP-001 validation AUC by target
ACL 0.315; MCL 0.389; Medial Meniscus 0.464; Lateral Meniscus 0.704; Medial OA 0.346; Lateral OA 0.556; PF OA 0.583; Effusion 0.820; Synovitis 0.296; Baker's 0.722; Contusion 0.426; Fracture 0.400.

### EXP-002 validation AUC by target
ACL 0.333; MCL 0.361; Medial Meniscus 0.661; Lateral Meniscus 0.278; Medial OA 0.923; Lateral OA 0.500; PF OA 0.333; Effusion 0.320; Synovitis 0.389; Baker's 0.472; Contusion 0.463; Fracture 0.520.

### EXP-003 validation AUC by target
ACL 0.389; MCL 0.417; Medial Meniscus 0.661; Lateral Meniscus 0.852; Medial OA 0.885; Lateral OA 0.583; PF OA 0.778; Effusion 0.240; Synovitis 0.481; Baker's 0.556; Contusion 0.500; Fracture 0.480.

## Interpretation
- EXP-003 is the highest observed single-split score (0.5684), but 15 validation studies are too few for a reliable estimate.
- Validation prevalence differs substantially for some targets: Fracture positives are 8/43 in training and 10/15 in validation. Medial OA has only two validation positives, so its AUC is especially unstable.
- EXP-001's training macro AUC exceeds validation macro AUC, consistent with overfitting and/or sampling/distribution mismatch.
- EXP-003 changes both slice sampling and the input-channel architecture, so it does not isolate slice count as the sole cause of any score change.
- All three models were trained from scratch. Each uses a single selected fluid-sensitive series and a small number of slices. DICOM slice order was based on InstanceNumber and geometry-based ordering has not yet been validated.
- No competition submission has been generated or verified. None of these models should be described as competition-ready.

## Next actions
1. Save/export the Kaggle notebook and record a stable revision/configuration so runs are reproducible.
2. Audit series selection and DICOM spatial ordering/orientation.
3. Improve validation reliability using repeated/fold-based study-level evaluation and explicit per-target class-count checks.
4. Test a permitted pretrained encoder or more principled slice aggregation against a fixed protocol.
5. Keep all runs, including worse scores, in the experiment register.
