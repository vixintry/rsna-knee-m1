# EXP-007 — Experimental parser v2 diagnostic

## Hypothesis
The baseline regex parser has a small number of high-confidence errors caused by negation scope and by treating any mention of an abnormality as meeting the competition's severity threshold. A conservative wrapper may correct some of these outputs, but may also reduce coverage. This is a parser diagnostic only; no claim of model-performance improvement is made.

## Changes isolated from baseline
- `scripts/derive_report_weak_labels_v2.py` wraps the existing parser without editing it.
- Explicit “meniscus without surfacing tear” language is treated as negative unless a separate clause explicitly reports a tear.
- ACL low-grade/grade 1–2/mild findings without qualifying high-grade/full-thickness evidence are treated as below threshold.
- MCL low-grade/chronic findings are negative; generic ungraded sprain language is unknown; a high-grade/complete tear is not made positive without acute wording.
- Explicit small/trace effusion and small/trace Baker's cyst are negative under the official target thresholds.
- Regression tests cover each guard and two important positive-preservation cases.
- `scripts/audit_exp007_parser_v2.py` compares baseline and v2 on complete-label studies, writing detail, summary, and JSON outputs. Gold rows are evaluation-only and never exported as weak-training examples.

## Kaggle run instructions

### 1. Regression tests
- Cell type: Code
- Persistence: one-off diagnostic
- GPU: OFF
- Markdown before it: not required; this report records the hypothesis and method
- Model behaviour: unchanged; parser unit tests only
- Writes: no files expected

From the repository root:
```bash
python -m unittest tests.test_derive_report_weak_labels_v2 -v
```

### 2. Gold-set comparison
- Cell type: Code
- Persistence: one-off diagnostic
- GPU: OFF
- Markdown before it: not required; this report records the hypothesis and method
- Model behaviour: unchanged; no model is loaded or trained
- Writes:
  - `/kaggle/working/exp007_parser_v2_comparison/parser_v2_gold_detail.csv`
  - `/kaggle/working/exp007_parser_v2_comparison/parser_v2_gold_summary.csv`
  - `/kaggle/working/exp007_parser_v2_comparison/parser_v2_gold_report.json`

Run from the repository root:
```bash
python scripts/audit_exp007_parser_v2.py
```

The audit expects 58 complete-label studies by default and fails closed if the dataset differs. Inspect the input version before overriding this expectation.

## Decision rule
Do not use weak labels for training based on these results alone. Inspect every changed prediction and compare per-target coverage, precision, and recall. Because the set has only 58 studies and parser coverage is incomplete, metrics are conditional and highly uncertain. Keep the baseline as the control. Accept no change solely because it improves an aggregate metric; the changed examples must align with official criteria and regression tests.

## Execution status
Code and tests have been added to the experimental branch. Runtime test and Kaggle gold-set comparison results have **not yet been observed**; no result or improvement is claimed here.
