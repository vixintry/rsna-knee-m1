# Competitive Research Plan — RSNA Knee Abnormality Detection

**Status:** research plan, not a report of completed experiments  
**Created:** 2026-10-11  
**Goal:** maximise legitimate macro ROC-AUC while preserving a defensible validation process and the competition's offline/runtime rules.

## 1. Where we actually stand

### Measured results so far

| Experiment | Method | Macro AUC | Evaluation |
|---|---|---:|---|
| EXP-001 | Small CNN, three slices treated as separate examples | 0.5018 | Existing fixed 15-study validation set |
| EXP-002 | Study-level CNN, three slices as channels | 0.4628 | Same fixed validation set |
| EXP-003 | Study-level CNN, nine slices as channels | 0.5684 | Same fixed validation set |
| EXP-004 | Frozen DINOv2 features + logistic regression | 0.5717 | Same fixed validation set |
| EXP-005 | TF-IDF report baseline | 0.5351 | Same fixed validation set |
| EXP-006 | DINOv2 + TF-IDF | 0.5665 | Same fixed validation set |

These numbers are exploratory, not reliable estimates of leaderboard performance. The validation set contains only 15 studies, and some class counts are very small. Do not compare future cross-validation scores directly with these single-split numbers as if they were the same evaluation protocol.

EXP-007 produced report-derived weak-label audits but did not establish that those labels are safe to train on. EXP-008 has now run on Kaggle's three example test studies: it generated a structurally valid 3-row, 13-column candidate CSV with finite probabilities in [0, 1]. Kaggle replaces those example rows with the hidden test set during scoring. This smoke test is not a performance result and is not full-test validation.

## 2. Key modelling insight: the report can supervise the image model

Only 58 studies have complete official labels; the other 4,349 studies have reports but no complete official target vector. Missing official labels are **unknown**, not negative.

The reports can potentially provide additional supervision, but we must not reduce every phrase to a careless hard yes/no label. A report-aware pipeline should distinguish:

- **Official targets:** the competition's 12 binary labels. These remain the evaluation ground truth.
- **Report-derived soft targets:** per-condition probabilities/confidence values, with explicit uncertainty and conflict states.
- **Clinical states:** richer intermediate categories where supported by the report, such as absent, equivocal, mild/partial, definite/high-grade, or chronic/degenerative versus acute/traumatic. Exact categories must be defined per finding and reviewed; they are not interchangeable with the official binary labels.
- **Evidence and provenance:** the report span/rule that supported a target, confidence, uncertainty, and parser/model version.

An LLM is most useful here as a **structured report-labeling assistant**, not as a magic image learner. Use it to convert the 4,349 training reports into auditable, probabilistic clinical targets. Compare it against the existing deterministic parser, manually audit a stratified sample, and preserve conflicts/uncertain cases rather than forcing a label. The vision model can then learn from the extra image/report-derived supervision. At inference, use only permitted image/series inputs; test reports are unavailable and must not be used.

### Anti-leakage rules for report supervision

1. Never derive a training target from a study's official labels.
2. When evaluating on a held-out gold fold, exclude those held-out studies from all supervised training paths, including any report-derived labels for those studies.
3. Keep the official labels, report-derived labels, and model predictions in separate columns/files.
4. A model-generated pseudo-label is not ground truth. Record its confidence and source.
5. Do not train on test reports or use test labels/predictions to fit preprocessing, select thresholds, or tune the model.
6. Do not claim that the LLM's report interpretation is correct without auditing it against reports and, where available, official labels.

## 3. MRI representation: geometry, multiple planes, and 2.5D

A knee study contains variable numbers of DICOM series and slices, often in different anatomical planes and pulse sequences. Treating it as nine arbitrary images from one series risks discarding information and may misorder slices.

The next image pipeline should audit and then use, where available:

- **Sagittal, coronal, and axial planes** rather than relying on a single plane.
- **Fluid-sensitive and non-fluid-sensitive sequences** as complementary sources. Fluid-sensitive sequences can help reveal edema, effusion and acute injury; other sequences can better depict morphology/cartilage. The value of each slot must be measured, not assumed.
- **DICOM geometry:** sort slices using ImageOrientationPatient and ImagePositionPatient (projection onto the slice normal) rather than trusting InstanceNumber alone. Check orientation, spacing, duplicate slices, missing files and MONOCHROME1 handling.
- **Physical-scale normalisation:** investigate consistent millimetre-based crops and slice-spacing normalisation so that scanner pixel spacing does not create avoidable scale variation.
- **2.5D adjacent-slice windows:** stack neighbouring slices as image channels. This provides local through-plane context at lower compute cost than a full 3D CNN. Compare against the existing 2D/fixed-slice baseline.
- **Study-level, label-specific aggregation:** different findings may be visible in different slices/planes. Test attention pooling per target over slice/window features instead of averaging every slice identically.
- **Missing-series masks:** if a study lacks a plane or sequence, represent that absence explicitly; do not silently substitute an unrelated series.

These are hypotheses to test. More slices, more planes, higher resolution, and larger backbones are not automatically better under the 9-hour inference budget.

## 4. Model families to investigate

Investigate in an order that isolates the source of any improvement:

1. **Reference baseline:** reproduce EXP-004 and existing CNN baselines under the new cross-validation protocol.
2. **Geometry-corrected DINOv2:** keep the encoder frozen initially; improve series/slice selection and study aggregation first.
3. **2.5D multi-plane model:** neighbouring slices per window, sequence/plane metadata, and label-specific window aggregation.
4. **Medical-image foundation encoder:** evaluate a permitted, available pretrained musculoskeletal/medical encoder (for example, an OrthoFoundation-type encoder if its exact weights, licence and provenance are verified). Compare on identical folds with DINOv2.
5. **Report-derived soft supervision:** compare deterministic parser, LLM-derived soft targets, and carefully combined targets. Begin with frozen image features/light heads before considering encoder fine-tuning.
6. **Clinical-state auxiliary heads:** test whether richer report states help the image representation learn distinctions that the binary targets hide. Map state predictions to official targets only through a predeclared, training-only mapping.
7. **Fine-tuning:** only after the feature and label pipeline is sound. Use small learning rates, early stopping on validation predictions, multiple seeds where feasible, and checkpoints. Never fine-tune on the validation fold.
8. **Ensembling:** only after at least two genuinely complementary methods show stable out-of-fold performance. Evaluate probability averaging versus rank blending using out-of-fold predictions only.

Do not copy a reported public leaderboard score or a model recipe into our results. A user-provided research document describes a more complex dual-model/2.5D/multi-plane approach and reports high scores, but those claims and its exact reproducibility have not been independently verified here. Treat it as a source of hypotheses, not as evidence of our performance or a guarantee that its external weights are available/permitted.

## 5. Validation protocol upgrade

### Why change it

The existing split (43 train / 15 validation, seed 42, stratified on ACL) is useful historical evidence but is too small and only stratifies one target. Model choices made against it can overfit to a few studies.

### Proposed protocol (new experiments only; do not overwrite old scores)

1. Keep all 58 officially labelled studies together at study level.
2. Build deterministic 5-fold out-of-fold (OOF) predictions, using multi-label-aware stratification if feasible. Inspect per-target positive/negative counts in every fold before training.
3. If any target has one class in a validation fold, mark its AUC undefined for that fold; do not invent a value. Follow the competition's official macro-AUC handling and report per-target support.
4. Each study must receive predictions only from a model that did not train on that study, including any report-derived training path.
5. Compare all candidate models on the same folds, seeds, preprocessing, and official macro-AUC implementation.
6. Report overall OOF macro-AUC, all 12 per-target AUCs, class counts, fold-level scores, variability, runtime, memory and failures.
7. Because there are only 58 studies, treat fold variability as substantial uncertainty. Do not over-tune architecture/hyperparameters to one OOF result.
8. After choosing a method, retrain using all eligible training data for the final inference model. The hidden test set is never used for model selection.

A 5-fold design is proposed, not yet implemented or approved by a successful test. If label counts make a fold design invalid, document that before changing the number of folds.

## 6. Experiment order and gates

| ID | Experiment | Hypothesis | Success evidence |
|---|---|---|---|
| EXP-009 | DICOM geometry and series-selection audit | Correct slice order and selection avoid lost anatomy and inconsistent inputs | Audit tests pass; report series coverage, spacing/orientation and failure counts |
| EXP-010 | Study-level multi-label-aware OOF evaluation harness | More reliable estimates reduce split luck | Deterministic, disjoint folds; tests for leakage, missing labels and undefined AUC |
| EXP-011 | Re-evaluate DINOv2 and CNN baselines on identical OOF folds | Establish a fair reference under the new protocol | Complete OOF predictions and per-label AUCs |
| EXP-012 | 2.5D adjacent-slice windows | Local through-plane context improves relevant findings | Improvement over the same-fold reference, stable across folds/labels |
| EXP-013 | Multi-plane/sequence ablations | Complementary series add information | Ablations show which plane/sequence helps, not just a larger model score |
| EXP-014 | LLM-assisted report-label audit | Soft/structured labels can expand usable training signal without excessive noise | Manual-audit precision/coverage by label and uncertainty state |
| EXP-015 | Weak-supervised vision training with strict gold-fold isolation | More report-labelled studies improve image predictions | OOF gain over image-only baseline without gold/report leakage |
| EXP-016 | Clinical-state auxiliary heads | Intermediate states help distinguish ambiguous findings | OOF gain with a predeclared state schema and mapping |
| EXP-017 | Verified medical-image foundation encoder | Domain pretraining improves transferable features | Fair OOF comparison with verified model/licence/provenance |
| EXP-018 | Complementary ensemble | Diverse errors improve macro AUC | OOF ensemble gain over each constituent; full runtime remains within limit |

### Gate rules

- Do not begin expensive fine-tuning before EXP-009/010 establish valid data handling and evaluation.
- A model is not an improvement unless it beats the reference on comparable OOF evidence.
- If a change alters slice selection, labels, split, or metric, assign a new experiment/configuration and document it.
- Stop adding complexity when it does not produce reproducible gains.
- Before final submission, run the complete offline Kaggle notebook against the real scoring environment and verify the full row count, exact columns, finite predictions, fallbacks and total runtime.

## 7. Immediate next actions

1. Implement the DICOM geometry/series audit (EXP-009) and tests; do not change the current baseline silently.
2. Implement and test deterministic OOF folds and metric handling (EXP-010).
3. Recompute reference baselines on the new folds (EXP-011).
4. Only then choose whether multi-plane 2.5D modelling or LLM-derived soft labels is the highest-value next experiment.

## 8. Evidence policy

Every experiment must record its hypothesis, exact code commit, data/label versions, fold IDs, seed, preprocessing, model/weights provenance, runtime, metrics, per-label support, failures, and decision. This document contains a research plan; none of EXP-009 through EXP-018 is claimed complete by this file.
