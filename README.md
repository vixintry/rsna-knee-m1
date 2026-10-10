# RSNA Knee Abnormality Detection

A reproducible machine-learning research project for the Kaggle RSNA Knee Abnormality Detection competition.

## Objective
Maximise legitimate competition performance while preserving evaluation integrity, reproducibility, and clear experiment records.

## Status
**Phase 0 — project setup and rules/data audit.** No model performance has been measured and no submission has been verified.

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
