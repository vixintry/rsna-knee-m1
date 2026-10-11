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

## DEC-004 — Upgrade evaluation before further model selection
- **Date:** 2026-10-11
- **Decision:** Keep the historical SPLIT-42-ACL-43-15 results unchanged, but use a deterministic study-level out-of-fold protocol for new model comparisons. Prefer multi-label-aware stratification where feasible and audit per-label positive/negative counts in every fold.
- **Reason:** The 15-study validation set is too small and stratifies only on ACL, making model ranking highly uncertain.
- **Constraints:** Never split slices/series from one study across folds. Exclude held-out studies from every supervised training path, including report-derived labels. Explicitly report undefined per-label AUCs rather than inventing values. If the proposed fold count is infeasible for rare labels, document the evidence before changing it.
- **Status:** Accepted as the next evaluation design; implementation and feasibility checks are still pending.

## DEC-005 — Prioritise geometry-aware, multi-plane MRI and audited report supervision
- **Date:** 2026-10-11
- **Decision:** Investigate DICOM geometry/order, 2.5D adjacent-slice windows, multiple anatomical planes/sequences, and LLM-assisted probabilistic report labels/clinical-state targets before committing to expensive fine-tuning or an ensemble.
- **Reason:** The current model uses nine slices from one selected series and only 58 complete official label vectors exist; reports from additional studies may provide useful but noisy supervision.
- **Constraints:** Report-derived labels remain distinct from official labels; uncertain/conflicting reports are not forced into hard negatives/positives; test reports are never used. Treat claims in the user-provided research document and any external pretrained weights as unverified until provenance, permissions, and reproducibility are checked.
- **Status:** Accepted research direction; no performance gain is claimed.

## Open decisions
- Exact implementation of deterministic multi-label-aware folds, including feasibility for rare labels.
- Which available medical-image foundation weights are permitted, accessible offline, and computationally viable.
- Whether LLM-derived soft/state targets improve out-of-fold image performance after manual label-quality audit.
