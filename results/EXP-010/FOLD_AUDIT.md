# EXP-010 fold integrity audit

- **Audit status:** PASS — all checks printed by the Kaggle audit cell passed.
- **Scope:** the saved five-fold assignment for the 58 studies with all 12 official labels; no model was trained by this audit cell.
- **Fold sizes:** fold 0 = 13, fold 1 = 10, fold 2 = 12, fold 3 = 12, fold 4 = 11.
- **Seed:** 42.
- **Saved fold CSV SHA-256 reported by Kaggle:** `5694a719f79a45c343ea6348f369b4e9154b3ff33bcc1f37e1f137fb96c161b6`.
- **train.csv SHA-256 reported by Kaggle:** `8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`.
- **Generator source:** `src/validation.py` downloaded from GitHub, blob SHA `baff60e34423f7de7f3bb66fcb69488a342bb5de`, reproduced the saved CSV exactly under both train.csv row order and StudyInstanceUID-sorted order. The notebook-global generator also reproduced it under both tested orderings; its source was marked unverified by the audit.
- **Integrity checks passed:** expected columns; all 12 label columns and study ID present in train.csv; unique train study IDs; exactly 58 complete-label studies; labels limited to 0/1; no blank fold IDs; every study assigned exactly once; fold IDs exactly 0–4; no empty folds; fold-file IDs exactly equal complete-label IDs; one-to-one ID join; each label has both classes in every fold; each label has at least five positives and five negatives overall; saved fold support matches the earlier reported support table; deterministic reproduction by at least one generator/order combination.
- **Caveat:** fold 0 has only one MCL-positive study (1 positive / 12 negative), so fold 0's per-fold MCL AUC is not meaningful. Use pooled OOF AUC for the primary comparison.
- **Provenance limitation:** this establishes integrity and reproducibility of the saved assignment, not proof that it is the original historical assignment from the first EXP-010 run. The generator is row-order dependent. Treat the saved CSV joined by `StudyInstanceUID` as canonical; do not regenerate folds mid-project.
- **Artifact limitation:** this report is recorded from the user-provided Kaggle cell output. The raw `exp010_folds.csv` and `exp010_fold_support.csv` files have not yet been committed to the repository, and this report is not a substitute for those files.
- **Exception after audit:** the notebook subsequently raised `SystemExit` because the output file already existed and the cell asked to rerun the audit cell. This occurred after the printed `VERDICT: ALL CHECKS PASSED`; it is not evidence of an audit-check failure.

## Per-fold label support

See [exp010_fold_support.csv](exp010_fold_support.csv) for the committed support artifact. The separate [fold_support.csv](fold_support.csv) is a transcription of the earlier printed table, not the canonical audit artifact.
