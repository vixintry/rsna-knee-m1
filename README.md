# RSNA Knee Abnormality Detection

A reproducible machine-learning research project for the Kaggle RSNA Knee Abnormality Detection competition.

## Objective
Maximise legitimate competition performance while preserving evaluation integrity, reproducibility, and clear experiment records.

## Status
**Exploratory modelling in progress.** EXP-001 to EXP-006 have been run on the fixed 43/15 split of the 58 fully labelled studies. The best observed single-split validation macro ROC-AUC is 0.5717 (frozen DINOv2 features + logistic regression). This tiny validation set is not a reliable estimate of leaderboard performance. EXP-007 adds a conservative report-derived weak-label audit; it is committed but not yet run.

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

## Current experiment status

| Experiment | Validation macro ROC-AUC | Status |
|---|---:|---|
| EXP-001: 3-slice CNN | 0.5018 | Exploratory baseline |
| EXP-002: 3-slice study-level CNN | 0.4628 | Exploratory baseline |
| EXP-003: 9-slice study-level CNN | 0.5684 | Exploratory baseline |
| EXP-004: DINOv2 features + logistic regression | 0.5717 | Best observed on current split |
| EXP-005: TF-IDF report-only | 0.5351 | Completed |
| EXP-006: Image + TF-IDF | 0.5665 | Completed |
| EXP-007: Report-derived weak-label audit | — | Implemented; not run |

All six AUC values use the same 15-study validation partition and should be treated as highly uncertain. The 4,349 reports without official labels are not ground truth; EXP-007 only proposes conservative weak labels and must be audited before training.

## Initial workflow
1. Confirm competition rules, eligibility, metric, and deadlines.
2. Inspect CSV schemas, label availability, study IDs, series descriptors, and data access.
3. Establish a reproducible study-level validation split.
4. Implement and evaluate a baseline.
5. Run hypothesis-driven experiments and record outcomes.
6. Select the final model using validation evidence and verify submission formatting.

## Data and artifacts
Do not commit competition data, trained weights, predictions, or other large artifacts. Keep them in approved storage and record versions/checksums where practical.

## Compute context
The local computer reported by the user has an Intel Core i5-12400F, 16 GB RAM, AMD Radeon RX 6600 with 8 GB VRAM, and a 1 TB NVMe SSD. This informs planning only; actual runtime and compatibility must be measured. Kaggle-hosted compute may be preferable for accessing the full dataset.

## Results policy
A result is valid only when supported by an actual run and identifiable configuration/data split. At initialization, there are no verified model results.
