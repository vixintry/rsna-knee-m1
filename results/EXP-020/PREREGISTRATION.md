# EXP-020 — Rule-based report-label parser audit

- **Status:** PLANNED; not run
- **Parent finding:** EXP-007 / EXP-019 pre-flight weak-label audit
- **Purpose:** determine whether conservative rule-based report parsing can produce defensible negative labels for Medial OA, Lateral OA, PF OA, and Synovitis without treating silence as absence.
- **Scope:** parser analysis and local, human-reviewed validation only. No MRI feature extraction, model training, or model-performance claim.
- **Data:** official `train.csv` reports only. The 58 studies with any official target label remain excluded from weak-label generation. Do not inspect or use test reports.
- **Privacy:** keep report text in the local Kaggle session. Do not upload raw reports or send them to hosted third-party APIs.

## Motivation from the completed pre-flight

The current parser audit covered 4,349 unlabelled studies. Its registered floor of at least 20 positive and 20 negative weak labels per target failed for:
- Medial OA: 55 positive, 0 negative
- Lateral OA: 24 positive, 0 negative
- PF OA: 70 positive, 0 negative
- Synovitis: 381 positive, 12 negative

The other eight targets passed the count floor, which does not establish label accuracy.

## Hypothesis

The OA negative rules may be too narrow to recognise some explicit negative statements. This is only a hypothesis: the zero-negative counts do not prove that suitable negative statements exist in the reports.

## Procedure

1. Freeze and hash the current parser as the EXP-020 starting point. Do not overwrite it.
2. On the Kaggle machine, produce a local candidate-cue report for OA and synovitis. Keep only rule IDs, cue snippets needed for review, study IDs, and status locally; do not commit raw report text.
3. Review a pre-specified random sample of candidate positives, candidate negatives, unknowns, uncertain cases, and conflicts. Include reports where the cue is negated, uncertain, historical, or refers to a different compartment.
4. Have a human assign a reference label per target as positive, negative, uncertain, or not stated. The reviewer must see enough local context to interpret negation and compartment, but the raw report must not leave the Kaggle environment.
5. Freeze candidate regex changes before measuring their performance on a separate held-out reviewed sample. Do not tune on the held-out sample.
6. Report precision and recall for positive and negative calls separately, coverage, uncertainty/conflict rates, and a confusion matrix per target. If the sample is too small or reference labels are ambiguous, report the uncertainty and stop.
7. Re-run class-count coverage only after the quality audit. Meeting the 20/20 floor alone is not sufficient to proceed to training.

## Decision gates

- **G1 — Rules:** a human checks the full official competition rules and confirms that this report-processing method is permitted. Until then, no full experiment.
- **G2 — Review protocol:** sample selection, annotation labels, and acceptance thresholds are written and frozen before reviewing the held-out sample.
- **G3 — Label quality:** held-out precision/recall and uncertainty criteria meet the thresholds specified in a follow-up preregistration before any weak labels are used for training.
- **G4 — Coverage:** every one of the 12 targets has at least 20 positive and 20 negative weak labels, with adequate quality evidence. No target may be silently omitted from the registered primary metric.
- **G5 — Reproducibility:** parser version/hash, sample IDs, cue audit, annotation decisions, and outputs are saved without committing raw report text.

## Stop conditions

- No explicit negative evidence for a target: leave it unknown; do not infer a negative from silence.
- Conflicting or uncertain report language: do not force a binary label.
- Human-review evidence or official rule permission is unavailable: stop.
- Any target fails the quality or coverage gate: no EXP-020 model training and no claim that weak labels improve the model.

## Non-goals

This experiment does not change EXP-019's thresholds or result, does not edit the production notebook, and does not train a model. Any later parser change and any training experiment must be separately versioned and preregistered.
