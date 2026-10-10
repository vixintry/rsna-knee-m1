# Decision Log

Record approved project decisions and meaningful changes. Do not silently rewrite earlier decisions.

## DEC-001 — Separate repository
- **Date:** 2026-10-10
- **Decision:** Keep RSNA knee ML work in vixintry/rsna-knee-m1, separate from the quantitative trading/backtesting project.
- **Reason:** Different data, objectives, evaluation risks, dependencies, and research methods.
- **Status:** Accepted.

## DEC-002 — Repository as canonical project record
- **Date:** 2026-10-10
- **Decision:** Use this repository's specification, decision log, research log, and experiment register to track approved methodology and evidence.
- **Reason:** Preserve traceability and avoid contradictory informal decisions.
- **Status:** Accepted.

## DEC-003 — No training before initial audit
- **Date:** 2026-10-10
- **Decision:** Inspect competition rules, labelled sample size, schemas, split feasibility, and compute needs before expensive training.
- **Reason:** Prevent wasted compute and unreliable evaluation.
- **Status:** Accepted.

## Open decisions
- Official metric and exact aggregation: pending verification from current competition documentation.
- Validation split strategy: pending label-count and grouping audit.
- Modelling approach: pending data audit.
- Training environment: compare Kaggle-hosted compute with feasible local options.
