# EXP-007 — Gold-set agreement audit

## Purpose
Apply the existing report parser to the 58 fully labelled studies and compare its explicit binary outputs with the official labels. This is a parser diagnostic, not model training or clinical validation.

## Run in Kaggle
Run from the repository root after the repository files are available.

- Cell type: Code
- Persistence: one-off diagnostic run
- GPU: OFF (CPU-only)
- Markdown before it: not required for the one-off execution; this report is the durable record
- Model behaviour: unchanged; no model is loaded or trained
- Writes: `/kaggle/working/exp007_gold_agreement/gold_agreement_detail.csv`, `gold_agreement_summary.csv`, and `gold_agreement_report.json`

Command:
```bash
python scripts/audit_exp007_gold_agreement.py
```

The script fails closed unless exactly 58 fully labelled studies are found. If the official input version differs, investigate and explicitly pass `--expected-gold-studies N`; do not silently change the expected count.

## Results observed in Kaggle
The audit ran on 58 complete-label studies. Agreement metrics are conditional on parser coverage; unknown, uncertain, and conflict cases are excluded rather than treated as negatives.

| Target | Covered | Coverage | TP | FP | TN | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACL | 27 | 46.6% | 13 | 5 | 9 | 0 | 0.722 | 1.000 |
| MCL | 20 | 34.5% | 5 | 4 | 11 | 0 | 0.556 | 1.000 |
| Medial Meniscus | 18 | 31.0% | 9 | 3 | 6 | 0 | 0.750 | 1.000 |
| Lateral Meniscus | 18 | 31.0% | 8 | 3 | 7 | 0 | 0.727 | 1.000 |
| Medial OA | 1 | 1.7% | 1 | 0 | 0 | 0 | Not estimable | Not estimable |
| Lateral OA | 0 | 0.0% | 0 | 0 | 0 | 0 | Not estimable | Not estimable |
| PF OA | 0 | 0.0% | 0 | 0 | 0 | 0 | Not estimable | Not estimable |
| Effusion | 32 | 55.2% | 17 | 10 | 5 | 0 | 0.630 | 1.000 |
| Synovitis | 13 | 22.4% | 9 | 4 | 0 | 0 | 0.692 | Not estimable |
| Baker's | 14 | 24.1% | 4 | 2 | 8 | 0 | 0.667 | 1.000 |
| Contusion | 13 | 22.4% | 6 | 2 | 4 | 1 | 0.750 | 0.857 |
| Fracture | 13 | 22.4% | 6 | 1 | 6 | 0 | 0.857 | 1.000 |

These counts are preliminary and small. They do not establish parser quality or model benefit.

## Important interpretation from the official target criteria
The competition's label criteria are narrower than simple mention detection. In particular:
- ACL: high-grade partial or full-thickness tear; mild signal change, degeneration, or thickening without discontinuity is negative.
- MCL: high-grade partial or complete acute tear; low-grade sprains and chronic/remote stress changes are negative.
- Meniscus: a tear must meet the specified surface-contact or morphologic criteria; intrasubstance degeneration alone is negative.
- OA: sufficiently large area of high-grade cartilage loss in the specified compartment.
- Effusion and Baker's cyst: moderate or large amounts/collections.
- Contusion: bone marrow edema-like signal without a discrete fracture line.
- Acute fracture: acute cortical break or fracture line.
- Ambiguous/borderline findings are graded negative.

Therefore a report can explicitly mention an abnormality while the corresponding gold label is 0. Parser positives for phrases like “grade 1 ACL injury,” “grade 1/2 MCL sprain,” “small/trace effusion,” “trace Baker's cyst,” or cartilage defects that do not meet the OA threshold are potential *severity/definition false positives*, not necessarily annotation errors. Conversely, patterns such as “lateral meniscus without surfacing tear” being matched as a positive tear are genuine candidate negation/pattern errors.

## Findings and limits
- The parser has low coverage, especially for all three OA targets. This is a coverage limitation; it is not evidence that the OA targets are absent, as the 58-study gold set contains both 0 and 1 values for every target.
- High conditional recall alongside low coverage is not evidence of robust sensitivity on the full dataset.
- Several disagreement examples describe findings that may fall below the competition's severity threshold. Do not call them gold-label errors without applying the official criteria.
- Some disagreements are likely parser errors, particularly negation scope and uncertain language. They need separate, target-specific error review.
- Reports may be multilingual, internally inconsistent, or use grading conventions that a regex parser does not understand.
- The 58 gold studies are an evaluation set for this audit and must not be used as weak-label training examples.
- Weak labels remain audit-only. No weak-supervision training is justified by this audit alone.

## Next actions
1. Fix or test obvious negation-scope errors in a separate parser version; retain the current parser as the baseline.
2. Add severity-aware rules only where explicit wording and official criteria support them. Keep borderline/ambiguous cases unknown rather than forcing a positive or negative.
3. Re-run the audit on the same 58 studies and compare coverage and conditional metrics to this baseline, with per-target error examples.
4. Do not use the gold studies as weak-label training data, and do not claim any model performance gain until a separate controlled experiment supports it.
