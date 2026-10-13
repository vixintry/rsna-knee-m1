# RSNA Knee Abnormality Detection

A reproducible machine-learning research project for the Kaggle RSNA Knee Abnormality Detection competition.

## Objective
Maximise legitimate competition performance while preserving evaluation integrity, reproducibility, and clear experiment records.

## Status
**EXP-011 exploratory five-fold evaluation recorded.** The image-only run reports pooled OOF macro ROC-AUC 0.589270; the text-only diagnostic reports 0.555506. These are results from only 58 fully labelled studies and are not reliable estimates of hidden-test performance. The saved EXP-010 fold file passed the 2026-10-13 integrity and deterministic-reproduction audit, but this does not prove it was the exact historical assignment used in the first EXP-011 run. The raw canonical fold CSV still needs to be committed; see [the audit report](results/EXP-010/FOLD_AUDIT.md) and [DECISIONS.md](DECISIONS.md). EXP-007 weak-label code and historical notebook-run counts are preserved, but their exact source-version relationship is not yet verified; see [the provenance note](reports/2026-10-13-exp007-provenance-reconciliation.md). Weak labels remain audit-only and have not been used for training. No leaderboard score is recorded.

## Results at a glance
All scores are macro ROC-AUC over 12 targets. Single-split and five-fold OOF results use different protocols and are not directly comparable.

| Protocol | Experiment | Score |
|---|---|---:|
| Single split (43 train / 15 validation) | Small CNNs from scratch (EXP-001, 002, 003) | 0.5018, 0.4628, 0.5684 |
| Single split | Frozen DINOv2 Base + logistic regression (EXP-004) | 0.5717 |
| Single split, diagnostic only | TF-IDF report baseline (EXP-005) | 0.5351 |
| Single split, diagnostic only | DINOv2 + TF-IDF (EXP-006) | 0.5665 |
| Five-fold OOF, reported | DINOv2 image-only (EXP-011) | 0.589270 pooled; 0.594066 mean fold (SD 0.054762) |
| Five-fold OOF, reported diagnostic | TF-IDF report text (EXP-011) | 0.555506 pooled; 0.576350 mean fold (SD 0.024897) |

EXP-005/006 and the text-only EXP-011 diagnostic cannot be used as test-time predictors because test reports are unavailable. EXP-007 is an audit only and has not been used for training. See [EXPERIMENTS.md](EXPERIMENTS.md) for methods and limitations.

## Repository layout
```
src/                 label definitions and validation/fold utilities
scripts/              fold creation and report-label prototype
tests/                pytest suite (python -m pytest tests)
notebooks/            data audit, MRI baseline, and DINOv2 experiments
notebooks/archive/    superseded notebook versions kept for reference
configs/folds/        EXP-010 fold provenance record
results/              curated experiment outputs
reports/              dated findings write-ups
docs/                 research plans and prototype notes
```

## Branch workflow
- `main` is the default branch.
- `notebook-cleanup` contains the current experiment work.
- Use a separate branch and pull request for cleanup or experiment changes. Review provenance conflicts before merging related experiment-register changes.

## Running on Kaggle
- Competition data is mounted at `/kaggle/input/competitions/rsna-knee-abnormality-detection`; DINOv2 Base is loaded from the attached Kaggle model `metaresearch/dinov2`.
- Follow the GPU ON/OFF note above each notebook cell. Do not use Run All blindly.
- `/kaggle/working` is session-local. Save a notebook version or download outputs before ending a session.
- The DINOv2 inference cells depend on feature caches under `/kaggle/working`; those caches must exist in the same session or be recreated from the documented extraction step.
- `requirements.txt` covers the local helper code and tests. Kaggle notebooks additionally use packages such as pydicom, torch, transformers, and scikit-learn; record the actual environment for each run.

## Research principles
- Never fabricate results or claim code ran unless it was executed.
- Keep training, validation, and test data roles separate.
- Split at study level; keep all series and slices from one study in the same partition.
- Never assume a missing label is negative.
- Record experiments, configurations, seeds, metrics, runtime, and conclusions.
- Compare models on a consistent validation protocol.
- Treat public leaderboard feedback as supplementary.
- Check current rules before using external data or pretrained weights.
- Preserve prior experiments and document methodology changes.
- Begin with low-cost audits and baselines before expensive training.

## Project documents
- [PROJECT_SPEC.md](PROJECT_SPEC.md) — project rules and workflow
- [DECISIONS.md](DECISIONS.md) — approved decisions and changes
- [RESEARCH_LOG.md](RESEARCH_LOG.md) — observations and conclusions
- [EXPERIMENTS.md](EXPERIMENTS.md) — experiment index and results
- [docs/COMPETITIVE_RESEARCH_PLAN.md](docs/COMPETITIVE_RESEARCH_PLAN.md) — out-of-fold validation, 2.5D/multi-plane MRI, LLM-assisted report supervision, and the next experiments

## Initial workflow
1. Confirm competition rules, eligibility, metric, and deadlines.
2. Inspect CSV schemas, label availability, study IDs, series descriptors, and data access.
3. Establish a reproducible study-level validation split.
4. Preserve existing baseline results, then establish deterministic study-level out-of-fold validation.
5. Audit DICOM geometry and series selection; compare multi-plane/2.5D inputs and report-derived supervision with controlled experiments.
6. Select the final model using out-of-fold evidence and verify full hidden-test inference and runtime.

## Data and artifacts
Do not commit competition data, trained weights, predictions, or other large artifacts. Keep them in approved storage and record versions/checksums where practical.

## Compute context
The local computer reported by the user has an Intel Core i5-12400F, 16 GB RAM, AMD Radeon RX 6600 with 8 GB VRAM, and a 1 TB NVMe SSD. This informs planning only; actual runtime and compatibility must be measured. Kaggle-hosted compute may be preferable for accessing the full dataset.

## Results policy
A result is valid only when supported by an actual run and identifiable configuration/data split. Report uncertainty and preserve the exact aggregation method used.
