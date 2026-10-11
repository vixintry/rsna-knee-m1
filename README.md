# RSNA Knee Abnormality Detection

A reproducible machine-learning research project for the Kaggle RSNA Knee Abnormality Detection competition.

## Objective
Maximise legitimate competition performance while preserving evaluation integrity, reproducibility, and clear experiment records.

## Status
**Baseline research underway.** EXP-001–EXP-006 have exploratory fixed-split results; EXP-007 audited weak report labels; EXP-008 passed a three-example inference smoke test. No full hidden-test run or verified competitive leaderboard result is recorded.

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
6. Select the final model using out-of-fold evidence and verify full hidden-test submission execution.

## Data and artifacts
Do not commit competition data, trained weights, predictions, or other large artifacts. Keep them in approved storage and record versions/checksums where practical.

## Compute context
The local computer reported by the user has an Intel Core i5-12400F, 16 GB RAM, AMD Radeon RX 6600 with 8 GB VRAM, and a 1 TB NVMe SSD. This informs planning only; actual runtime and compatibility must be measured. Kaggle-hosted compute may be preferable for accessing the full dataset.

## Results policy
A result is valid only when supported by an actual run and identifiable configuration/data split. At initialization, there are no verified model results.
