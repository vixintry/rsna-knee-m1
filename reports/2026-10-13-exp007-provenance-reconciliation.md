# EXP-007 implementation and result provenance

The repository contains two kinds of EXP-007 evidence that must not be conflated:

1. `scripts/derive_report_weak_labels.py` is the conservative multilingual report parser copied from the `exp007-weak-report-labels` branch. Its companion regression tests cover seven parser cases. The script's own plan states that the competition dataset run has not been performed by that script.
2. The experiment register records a Kaggle weak-label audit over 4,349 unlabelled studies, including selected candidate counts and a seven-case regression test result. Those outputs were reported from a Kaggle notebook run; the exact executed notebook source has not yet been matched byte-for-byte to the committed script.

Therefore:
- Preserve both the committed script and the historical notebook-run results.
- Do not attribute the recorded coverage counts to this script until its outputs are reproduced from the same input CSV and its code version is confirmed.
- These parser labels remain audit-only. They have not been used for model training and are not verified clinical ground truth.
- Unknown, uncertain, and conflicting evidence must not be converted to negative labels.
- Before using weak supervision, run this script on the official CSV, save its outputs and hashes, manually inspect cues, and confirm strict gold-fold isolation.
