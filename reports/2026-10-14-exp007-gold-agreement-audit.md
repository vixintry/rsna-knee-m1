# EXP-007 — Gold-set agreement audit

## Purpose
Apply the existing conservative report parser to the 58 fully labelled studies and compare its explicit binary outputs with the official labels. This is a parser diagnostic, not model training and not a clinical validation.

## Run in Kaggle
Run from the repository root after the repository files are available.

- Cell type: Code
- Persistence: one-off diagnostic run
- GPU: OFF (CPU-only)
- Markdown before it: recommended, with this section as the explanation
- Model behaviour: unchanged; no model is loaded or trained
- Writes: `/kaggle/working/exp007_gold_agreement/gold_agreement_detail.csv`, `gold_agreement_summary.csv`, and `gold_agreement_report.json`

Command:
```bash
python scripts/audit_exp007_gold_agreement.py
```

The script fails closed unless exactly 58 fully labelled studies are present. If the official input version legitimately differs, investigate and explicitly pass `--expected-gold-studies N`; do not silently change the expected count.

## Interpretation
- Coverage = share of gold study-target pairs receiving an explicit 0 or 1 from the parser.
- Precision, recall, specificity, accuracy, and F1 are computed only over covered pairs.
- Unknown, uncertain, and conflict cases remain uncovered; they are never recoded as negatives.
- Review the detail CSV's matched cues for false positives/negatives, particularly negation, history, and compartment-specific OA.
- Results are diagnostic only. The 58 gold studies must not be used as weak-label training examples.
- Do not proceed to weak-supervision training solely because aggregate agreement looks promising. Inspect per-target counts and error cases first.
