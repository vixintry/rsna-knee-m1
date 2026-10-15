# EXP-007 parser v2 — case review (2026-10-15)

## Scope and provenance

This report records the first runtime comparison and source-report review for the experimental parser v2 on branch `research/exp007-parser-v2-diagnostics`.

- Dataset input: Kaggle competition `train.csv`.
- Gold subset: 58 complete-label studies.
- Compared parser decisions: baseline wrapper versus v2.
- No model training was performed.
- Gold studies are evaluation-only and must not be exported as weak-label training examples.
- Results are diagnostic and conditional on parser coverage; the sample is small and is not clinical validation.

## Runtime validation observed

The nine parser v2 regression tests passed in the earlier Kaggle run. After the Kaggle session restarted, the audit was rerun successfully with:

```bash
PYTHONPATH=. python scripts/audit_exp007_parser_v2.py \
  --output-dir /kaggle/working/exp007_parser_v2_comparison \
  --expected-gold-studies 58
```

The audit produced 696 target-study rows, of which 31 target predictions changed across 21 studies. Outputs were written to `/kaggle/working/exp007_parser_v2_comparison/`:
- `parser_v2_gold_detail.csv`
- `parser_v2_gold_summary.csv`
- `parser_v2_gold_report.json`

These are runtime artifacts, not committed dataset files.

## Summary metrics

Metrics below are calculated only where the parser returned a covered label. They must not be interpreted as full-cohort performance estimates.

| Target | Baseline coverage | V2 coverage | Baseline precision | V2 precision | Baseline recall | V2 recall |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 27 | 27 | 0.722 | 0.917 | 1.000 | 0.846 |
| MCL | 20 | 9 | 0.556 | 0.500 | 1.000 | 0.333 |
| Medial Meniscus | 18 | 18 | 0.750 | 0.750 | 1.000 | 1.000 |
| Lateral Meniscus | 18 | 18 | 0.727 | 0.800 | 1.000 | 1.000 |
| Medial OA | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| Lateral OA | 0 | 0 | N/A | N/A | N/A | N/A |
| PF OA | 0 | 0 | N/A | N/A | N/A | N/A |
| Effusion | 32 | 33 | 0.630 | 0.650 | 1.000 | 0.722 |
| Synovitis | 13 | 13 | 0.692 | 0.692 | 1.000 | 1.000 |
| Baker's | 14 | 14 | 0.667 | 0.800 | 1.000 | 1.000 |
| Contusion | 13 | 13 | 0.750 | 0.750 | 0.857 | 0.857 |
| Fracture | 13 | 13 | 0.857 | 0.857 | 1.000 | 1.000 |

V2 changes 31 target predictions: ACL 6, MCL 15, Effusion 8, Lateral Meniscus 1, Baker's 1. Improvements in some conditional precision metrics are accompanied by reduced recall and/or coverage, especially for ACL, MCL and Effusion.

## Source-report review: confirmed implementation concerns

### 1. MCL explicit normal/intact statements become unknown

Several reports explicitly state variants of:
- “The MCL is intact.”
- “MCL normal.”
- “Medial collateral ligament (MCL): Normal.”
- “The medial collateral ligament complexes are intact.”

V2 changes these baseline negatives to unknown because the MCL guard detects the word `thickening`, `injury`, `sprain`, or `tear` somewhere in the report without adequate target-specific scope. Some of these reports describe unrelated injuries or abnormalities elsewhere. This is a clear loss of negative coverage and warrants a targeted fix.

### 2. ACL severity guard is report-wide, not finding-scoped

The ACL guard searches for low-grade language and qualifying tear language across the whole report. It can therefore let a low-grade ACL cue override a separate qualifying ACL tear, or otherwise incorrectly scope the decision. Reviewed examples include reports with explicit “ACL tear” / “C/W ACL tear” wording that v2 returns as negative after finding low-grade language. Another report explicitly says “ACL: No tear”; that negative should remain negative. The guard must distinguish negation and severity attached to the ACL finding itself, not merely co-occurring words in the report.

### 3. MCL guard rejects explicit complete/high-grade tears without local acute wording

Reviewed reports include “Complete rupture of the MCL at the level of the medial jointline” and “Complete tear of the superficial portion” / “High-grade tear of the medial collateral ligament.” V2 returns these as unknown because it requires the word acute/recent somewhere in the report, despite an explicit complete/high-grade tear finding. The exact official target definition still governs positive classification; however, the current report-wide acute gate is overbroad and loses clearly described qualifying findings. Separate high-grade/complete findings from grade 1–2 and ambiguous sprain wording.

### 4. Grade 2 MCL wording needs careful threshold handling

A reviewed report says “Grade 2 injury” / “Grade 2 medial collateral ligament (MCL) sprain.” V2 marks it negative. Other grade 2 sprain reports have gold labels of both 0 and 1 in this small subset. Do not use gold labels alone to force a positive. Recheck the official target threshold and distinguish grade 2 partial injury from explicit high-grade/complete acute tear language; retain uncertainty where the report does not support the threshold.

### 5. Effusion small/minimal rule is overbroad across the report

The rule detects small/minimal effusion language anywhere and only checks for moderate/large wording in a narrow local text window. Reports can contain more than one effusion statement, and the audit includes gold-positive cases with “small joint effusion” or “minimal amount” wording as well as reports with unrelated moderate effusion wording. The report/gold disagreements need to remain visible. Refine the scope to the effusion finding and test contradictory statements; do not blindly make every case positive because gold is 1 or negative because a small finding is mentioned.

### 6. Baker's cyst size rule needs finding-level scope

One reviewed report describes a “moderate-sized popliteal cyst” and also a “small amount” of fluid/edema related to minimal partial rupture. The current rule may treat nearby small wording as cyst size evidence even when the cyst itself is moderate. Scope size qualifiers to the cyst finding itself.

### 7. Meniscus negation guard corrected one apparent false positive

The report “Degenerative signal throughout the lateral meniscus without surfacing tear” also describes a complete tear of the *medial* meniscus. V2 correctly changes the lateral meniscus label from positive to negative; this is consistent with the target-specific report evidence. Keep a regression test for this case pattern.

## Decision and next actions

- Do not merge or promote v2 to the production/weak-label pipeline yet.
- Keep the baseline parser unchanged as the control.
- Patch only the confirmed rule-scope defects in the experimental wrapper.
- Add regression tests for explicit intact/normal MCL despite unrelated report findings; qualifying ACL tear alongside a separate low-grade phrase; explicit ACL negation; explicit complete/high-grade MCL tear; moderate Baker's cyst with small rupture-related fluid; and reports with conflicting effusion size statements.
- Rerun all unit tests and the full 58-study audit after patching.
- Inspect every changed prediction again and report both coverage and conditional metrics.
- Gold labels remain evaluation-only; do not train on these 58 studies.

No model-performance improvement is claimed from this parser diagnostic.
