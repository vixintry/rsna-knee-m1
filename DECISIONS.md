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
- **Status:** Accepted and implemented in EXP-010; the saved assignment passed integrity/support checks on 2026-10-13. Historical identity with the first reported EXP-011 run's fold assignment remains unproven; see DEC-006.

## DEC-005 — Prioritise geometry-aware, multi-plane MRI and audited report supervision
- **Date:** 2026-10-11
- **Decision:** Investigate DICOM geometry/order, 2.5D adjacent-slice windows, multiple anatomical planes/sequences, and LLM-assisted probabilistic report labels/clinical-state targets before committing to expensive fine-tuning or an ensemble.
- **Reason:** The current model uses nine slices from one selected series and only 58 complete official label vectors exist; reports from additional studies may provide useful but noisy supervision.
- **Constraints:** Report-derived labels remain distinct from official labels; uncertain/conflicting reports are not forced into hard negatives/positives; test reports are never used. Treat claims in the user-provided research document and any external pretrained weights as unverified until provenance, permissions, and reproducibility are checked.
- **Status:** Accepted research direction; no performance gain is claimed.

## DEC-006 — EXP-010 folds: regenerated, audited, and frozen
- **Date:** 2026-10-11; audit completed 2026-10-13
- **Decision:** Freeze the saved five-fold study-level assignment for the 58 complete-label studies. Use the saved CSV joined by `StudyInstanceUID`; do not regenerate folds mid-project.
- **Generator:** `src/validation.py`, Git blob SHA `baff60e34423f7de7f3bb66fcb69488a342bb5de`, seed 42.
- **Saved fold CSV SHA-256:** `5694a719f79a45c343ea6348f369b4e9154b3ff33bcc1f37e1f137fb96c161b6`.
- **Audit outcome (2026-10-13):** all printed checks passed against the saved CSV and `train.csv`: 58 complete-label IDs matched exactly; each study appeared once; fold IDs were exactly 0–4 with sizes 13/10/12/12/11; labels were binary; every label had both classes in every fold and at least five positives and negatives overall; saved support matched the previously reported table; and the saved CSV was reproduced exactly by the repository generator under both train-row and StudyInstanceUID-sorted orderings.
- **Important limitation:** this proves integrity and reproducibility of the saved assignment, not that it is the original historical assignment from the first EXP-010 run. The generator depends on row order. Fold 0 contains only one MCL-positive study (1 positive / 12 negative), so its per-fold MCL AUC is not meaningful; use pooled OOF AUC for primary comparisons.
- **Data hashes reported by Kaggle:** `train.csv` SHA-256 `8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`.
- **Artifact status:** the canonical `results/EXP-010/exp010_folds.csv`, `results/EXP-010/exp010_fold_support.csv`, `results/EXP-010/exp010_fold_audit.json`, and `results/EXP-010/FOLD_AUDIT.md` are committed. The fold CSV SHA-256 is `5694a719f79a45c343ea6348f369b4e9154b3ff33bcc1f37e1f137fb96c161b6`. The separate `fold_support.csv` is a transcription and must not be treated as the canonical fold assignment.
- **Status:** fold integrity audit passed; assignment frozen; canonical artifacts preserved; historical-membership caveat remains.

## Open decisions
- EXP-010 artifact preservation is complete: canonical fold CSV, support CSV, and audit JSON/report are committed. Do not regenerate the frozen assignment; the remaining issue is historical identity with the first EXP-011 run, not missing files.
- Which available medical-image foundation weights are permitted, accessible offline, and computationally viable.
- Whether LLM-derived soft/state targets improve out-of-fold image performance after manual label-quality audit.
