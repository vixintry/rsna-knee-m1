# Project Specification

Version: 1.0  
Created: 2026-10-10  
Project: RSNA Knee Abnormality Detection

## 1. Objective
Develop a reproducible ML pipeline to predict probabilities for the competition's 12 knee-abnormality targets and achieve the strongest legitimate ranking feasible within the deadline and available compute.

Primary objective: competition performance. Secondary objective: sound, explainable, reproducible research. A leaderboard score alone is not evidence of generalisation.

## 2. Governing rules
1. **No fabricated results.** Never invent metrics, dataset statistics, experiments, execution success, or submissions.
2. **No silent methodology changes.** Record changes to labels, splits, preprocessing, model selection, or evaluation, with reasons.
3. **No leakage.** Keep all series and slices belonging to one study in the same partition. Fit learned transformations using training data only.
4. **No leaderboard overfitting.** Use local validation as the main development signal; leaderboard feedback is supplementary.
5. **Traceability.** Record code/configuration versions, data split, seed, runtime, metrics, and conclusions for meaningful experiments.
6. **Fair comparison.** Compare candidate models using the same split and metric unless a documented protocol change is necessary.
7. **Preserve history.** Do not overwrite the only copy of successful code, weights, logs, or configurations.
8. **Explain decisions.** State hypothesis, rationale, expected benefit, risks, and success criteria before meaningful experiments.
9. **Respect compute constraints.** Audit storage, RAM, VRAM, runtime, and platform limits before expensive jobs.
10. **Competition compliance.** Follow current official rules, eligibility, external-data restrictions, inference limits, and submission requirements.

## 3. Data integrity
The competition describes 12 binary targets: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, and Fracture.

- Audit missingness and prevalence separately for every target.
- Never interpret an absent label as negative without evidence from the data specification.
- Verify study/series/DICOM identifier relationships.
- Treat radiology reports and report-derived labels as potentially noisy.
- The supplied competition description says test reports are unavailable at inference; do not depend on them for test predictions.
- Document use of pretrained weights or external data and confirm it is permitted.
- Do not download the entire dataset locally without a justified storage and compute plan.

## 4. Evaluation
Confirm the official scoring metric and exact submission schema from competition documentation before implementing evaluation. Do not assume a metric based on convention alone.

Use a fixed, reproducible study-level validation split. If labelled sample size supports it, consider grouped cross-validation to assess stability.

Record the primary metric, per-target metrics, label counts, runtime, memory, seed, and split identifier. If a target lacks both classes in a validation fold, its AUC is undefined; report this and follow the official scoring definition.

Training data fits model parameters. Validation data supports model comparison. Hidden test data must not be used for fitting or tuning.

## 5. Research sequence
### Phase 0 — Setup and rules audit
- Verify competition rules, eligibility, deadlines, metric, and data access.
- Record compute limits.

### Phase 1 — Data audit
- Inspect CSV schemas, label coverage, prevalence, missingness, series descriptors, and identifier consistency.
- Estimate image/storage workload without loading the full dataset into memory.

### Phase 2 — Evaluation design
- Establish a deterministic study-level split.
- Add tests preventing study overlap across partitions.
- Define metrics from official documentation.

### Phase 3 — Baseline
- Produce a schema-valid baseline submission.
- Establish a locally measured baseline where labelled data permits.
- Save configuration and outputs.

### Phase 4 — Research
Candidate directions, subject to the data audit:
- metadata-only diagnostic baseline;
- pretrained image models and feasible MRI sampling;
- combining series, sequences, and planes;
- carefully validated report-derived weak supervision;
- ensembling only after complementary standalone models are demonstrated.

### Phase 5 — Robustness and selection
- Inspect per-target errors and validation stability.
- Select using the predeclared validation procedure, not public leaderboard score alone.

### Phase 6 — Submission
- Verify inference inputs, output columns/order, study IDs, row count, finite probabilities, runtime, and competition constraints.
- Record the exact model/configuration used.

## 6. Experiment requirements
Each meaningful experiment records:
- unique ID and date;
- hypothesis and change from baseline;
- data version and split identifier;
- model/configuration and seed;
- compute environment and runtime;
- primary and per-target metrics;
- failures, interpretation, and decision.

Retain negative results. Do not silently discard failed runs or report only favourable outcomes.

## 7. Coding standards
- Prefer small testable modules over a monolithic notebook.
- Keep configuration separate from implementation.
- Test label handling, split integrity, metrics, and submission generation.
- Inspect existing files and Git history before modifying established code.
- Never state code executed successfully unless it did.
- Fix bugs without silently changing the research question or methodology.
- Keep large datasets and model artifacts out of Git.

## 8. Stage gates
- No expensive training before the data audit and compute estimate.
- No claimed improvement without comparable validation evidence.
- No final selection based solely on public leaderboard.
- No submission considered complete until schema and execution checks pass.

## 9. Repository source of truth
- PROJECT_SPEC.md: governing rules and workflow.
- DECISIONS.md: approved decisions and changes.
- RESEARCH_LOG.md: observations and conclusions.
- EXPERIMENTS.md: experiment index and comparative results.
- configs/: exact experiment configurations.
- tests/: automated integrity checks.

If conversation instructions conflict with a documented decision, flag the conflict before proceeding.
