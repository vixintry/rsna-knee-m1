# MRI Baseline and Compute Plan

**Added:** 2026-10-10  
**Status:** Proposed plan only. No model has been trained and no score has been measured.

## Objective

Build a reproducible first MRI model for the RSNA Knee Abnormality Detection competition, while treating missing target labels correctly and respecting the official rules.

## Research starting points

- Stanford MRNet project/paper: https://stanfordmlgroup.github.io/projects/mrnet/
- Public MRNet implementation: https://github.com/MisaOgura/MRNet
- Multi-View Abnormality Detection in Clinical Knee MRI Studies Using Transformers: https://www.lfb.rwth-aachen.de/bibtexupload/pdf/KHA23c.pdf

MRNet provides a useful design pattern: apply a 2D CNN to slices, aggregate within each view, and combine sagittal/coronal/axial information. Its original task is not the same as our 12-target task, so published scores are not predictions of our performance. A transformer is a later option, only after a simple baseline works.

## Ordered implementation plan

### 1. Verify rules and evaluation first
- Confirm current competition deadline, entry/rules acceptance, permitted pretrained weights/external data/LLMs, inference restrictions, official metric, and submission schema.
- Do not assume ROC-AUC is the official metric until the current evaluation documentation confirms it.
- Re-run the audit against current official files and confirm target missingness.

### 2. Freeze a valid study-level split
- Keep every series and slice belonging to a study in the same partition.
- Report per-target positive/negative counts in each fold.
- Missing target values remain unknown; never convert them to zero.
- For expert-supervised loss, compute each target's loss only where that target is labelled.
- With only 58 fully labelled studies in the previous audit, validation estimates will be highly uncertain. Verify the count and do not overclaim from one split.

### 3. Build a small DICOM inspection notebook in Kaggle
- Inspect a small, reproducibly selected sample of labelled studies.
- Order slices using DICOM geometry/metadata, not filenames alone.
- Record series plane, sequence flags, slice counts, image dimensions, pixel-value handling, and failures.
- Profile runtime and memory before scaling up.

### 4. Train the smallest feasible image baseline
- Start with one orientation and a bounded number of slices per series.
- Use a small 2D CNN and permitted pretrained weights if appropriate.
- Define resizing, intensity normalisation, slice selection, and study-level aggregation explicitly.
- Compare on the frozen split and record per-target metrics, class counts, runtime, memory, seed, configuration, commit, and caveats in EXPERIMENTS.md.
- Do not assume 150–300 slices per series; profile actual series first.

### 5. Extend to multi-view
- Only after the single-view pipeline works, add sagittal, coronal, and axial views.
- First test simple probability averaging or feature/prediction fusion.
- Handle missing planes/series explicitly.
- Consider transformer-based fusion only if the simpler model is reliable and time permits.

### 6. Investigate report-derived weak labels separately
- First check rules and allowed tools.
- Compare a small rule-based extractor (including negation and uncertainty handling) with a permitted local multilingual model/LLM if feasible.
- Keep expert labels, weak labels, uncertain cases, and “not mentioned” distinct.
- Evaluate against expert labels and inspect errors by target/language; the 58-study sample is too small for confident generalisation.
- Never use reports as test-time inputs if they are unavailable for test studies, and never overwrite expert labels with generated labels.

## Local PC + Kaggle workflow

Use both, but do not attempt distributed training between them.

- **Local Windows PC (i5-12400F, 16 GB RAM, AMD RX 6600 8 GB):** code editing, Git, unit tests, small CSV/metadata inspection, report-parser experiments on small samples, and analysis of compact results. Assume CPU training unless GPU support for the chosen framework is verified.
- **Kaggle:** official dataset access, DICOM inspection at scale, GPU training if available, and end-to-end offline notebook testing.
- **Transfer:** version code through Git; share small configs and metrics. Do not commit the MRI dataset or large model checkpoints to Git. Recheck Kaggle runtime and internet restrictions before final runs.
- This split workflow is expected to be simpler than distributed training, but its speed advantage has not been measured.

## Immediate next actions

1. Verify live rules, deadline, metric, and submission schema.
2. Confirm current local/GitHub sync and run tests.
3. Create the small Kaggle DICOM inspection notebook.
4. Confirm missing-label semantics and freeze a study-level validation split.
5. Implement the official metric and test it on synthetic data.
6. Run a tiny single-view proof of concept only if the data/split audit supports it.
7. Record actual results in EXPERIMENTS.md before adding multi-view fusion or weak labels.

## Integrity notes

- No model has been trained; no score or submission is verified.
- Public ideas may be adapted, but verify licences, attribution, competition rules, and validation quality.
- Report-derived labels are noisy supervision, not ground truth.
- Keep hypotheses separate from measured results and record negative experiments.
