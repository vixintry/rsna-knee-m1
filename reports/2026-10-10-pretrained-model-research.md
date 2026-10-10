# Pretrained Model Research and Next Experiment — 2026-10-10

## Competition constraints checked
The public Kaggle competition page states that submissions must be made through notebooks, internet access is disabled for the committed submission run, CPU/GPU runtime must be at most 9 hours, and freely/publicly available external data and pretrained models are allowed. The evaluation metric is macro-averaged ROC-AUC over 12 labels. Recheck the live rules before final submission.

Sources:
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation

## Why change approach
The scratch CNN experiments use only 58 fully labelled studies and achieved validation macro AUCs of 0.5018, 0.4628, and 0.5684 on one small 43/15 split. These results are not robust, and training a visual encoder from scratch is poorly matched to this labelled sample size.

## Candidate: DINOv2 pretrained visual encoder
A public DINOv2 model is available on Kaggle under the Apache 2.0 licence. DINOv2 is a self-supervised pretrained image encoder intended for feature extraction or fine-tuning; it does not include a task-specific head. A small variant is a reasonable first candidate because it is less expensive than base/large variants. Source: https://www.kaggle.com/models/metaresearch/dinov2

The competition also has public reference notebooks using DINOv2 and reporting stronger public leaderboard scores than the scratch baseline. Public leaderboard scores are not equivalent to reliable local validation, and notebook methods may use additional public datasets, report-derived labels, or checkpoints. We should inspect provenance and licences before reusing any assets.

## Next experiment (planned, not yet run)
1. Add a publicly available DINOv2-small pretrained model to a separate Kaggle notebook; do not overwrite the existing baseline.
2. First run a short inference/feature-extraction smoke test on a handful of the 58 labelled studies. Confirm checkpoint loading, image preprocessing, GPU compatibility, and feature tensor shapes.
3. Cache embeddings for a fixed set of slices per study, then train only a small multi-label classifier on the 43 training studies and evaluate on the existing 15-study validation split. Keep the split and metrics identical for comparability; explicitly note the severe small-validation uncertainty.
4. Compare against a simple constant/prevalence baseline and the existing EXP-003 result. Report all 12 per-target AUCs, not only macro AUC.
5. Only after this works, investigate carefully sourced report-derived weak labels for the remaining studies. The reports are multilingual and can disagree with image-reviewed labels, so do not treat extracted labels as ground truth.

## Guardrails
- Do not train on the validation labels and then present the same validation score as independent evidence.
- Do not use any external weights or dataset until its public availability, licence, provenance, and competition-rule compatibility are checked.
- Do not rely on leaderboard score alone.
- Final inference must not use report text because the competition says test reports will not be provided.
