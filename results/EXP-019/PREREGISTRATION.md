# EXP-019: Frozen-DINOv2 probe trained on report-derived weak labels, tested on the 58 gold studies

**Status: PRE-REGISTERED (v2, revised before any run). NOT RUN. No result in this file is measured.**
Results go in `results/EXP-019/README.md` after the run, never here (Rule 1).

## Identifiers and provenance (only what was verified)

| Item | Value | How known |
|---|---|---|
| Experiment ID | **EXP-019** | v1 of this plan used EXP-012, which `EXPERIMENTS.md` already assigns to "2.5D adjacent-slice windows". EXP-019 is the next unused ID in that register. |
| v1 files | `scripts/EXP-012.md`, `scripts/analyse_exp012.py`, `scripts/extract_features.py` (commit `4a89fef`, 2026-10-17) | Superseded by this revision. They remain in git history. |
| Production notebook | `rsna-submission.ipynb`; SHA-256 `91fc7e589cb9473d39162fb2662b1e42a0e95f4aedd9038819f6f1ad82bd4ef6` | Content identical at commit `a5468a6a5111ac3d474192ca8d78c8fc8dfb9dcb` and at `main` when fetched on 2026-10-18. The file itself was last modified in `b9d4051`; `a5468a6` changed another file. |
| Weak-label parser | `scripts/derive_report_weak_labels.py`, blob `841d47fc3f932d3626595ed79a782b640318c0d4`, commit `56fdf6c` | GitHub API |
| Frozen gold folds (descriptive baseline only) | `results/EXP-010/exp010_folds.csv`, SHA-256 `5694a719f79a45c343ea6348f369b4e9154b3ff33bcc1f37e1f137fb96c161b6` | DEC-006 |
| Code commit of the EXP-019 scripts | recorded by the extractor at run time (`git_head_at_start` in `manifest.json`) | cannot be written here without a circular commit |
| Competition rules / metric | **NOT VERIFIED** (see Gates) | Official pages were not readable by the assistant |

Related register entries: EXP-007 (parser audit; its counts are not reproduced from the committed script), EXP-013 (multi-plane ablations; the extractor already stores plane slots for it), EXP-014 (report-label audit; may satisfy the quality gate), EXP-015 (full weak-supervised training; this experiment is a cheap frozen-feature pilot of the same question).

## Hypothesis

A logistic-regression probe on frozen DINOv2-Base features, trained on report-derived labels for the ~4,349 non-gold studies, ranks the 58 gold studies better than the same probe trained on the gold studies by cross-validation (macro OOF AUC 0.615784, notebook cell 15). Every earlier decision was tuned at n=58, where the signal is near the noise floor.

## Experiment record (repo template)

```
Experiment ID:            EXP-019
Date:                     pre-registered 2026-10-18 (v2); v1 2026-10-17
Hypothesis:               see above
Change from baseline:     TRAINING DATA ONLY. Series selection, 9-slice sampling, per-slice scaling, DINOv2
                          CLS, mean pooling, StandardScaler + LogisticRegression(balanced) are the production
                          pipeline (startup check against the notebook, below). C is re-chosen on a weak-label
                          validation split because the optimal C depends on n.
Data version:             competition train.csv / train_series.csv, DINOv2-Base (metaresearch/dinov2 pytorch/base/1);
                          weak labels from the parser blob above. Hashes of train.csv and weak_labels.csv are
                          written into exp019_results.json.
Validation split:         train = weak-labelled studies (gold removed); C selection = 80/20 stratified split of the
                          weak set (seed 0, weak labels as truth); test = the 58 gold studies, fixed holdout.
Model and configuration:  make_pipeline(StandardScaler(), LogisticRegression(C, class_weight="balanced",
                          max_iter=2000, random_state=42)); C in {0.001, 0.01, 0.1, 1.0}.
Random seed:              model 42; weak split 0; bootstrap 0 (B=2000); baseline CV seeds 42/123/2026.
Environment:              Kaggle, GPU for extraction. Library versions are not pinned and may differ from the
                          environment of the recorded 0.615784; the reproduction gate checks this.
Runtime / peak memory:    NOT MEASURED. The smoke test prints an extrapolated ESTIMATE; replace it with the
                          measured full-run time afterwards.
Primary metric:           macro ROC AUC over ALL 12 targets on the 58 gold studies (official metric NOT yet verified).
Per-target metrics:       [blank until run]
Failure modes / caveats:  [blank until run]
Conclusion:               [blank until run]
Decision:                 [blank until run]
Code commit:              see manifest.json
```

## Registered decision rule (unchanged from v1)

- **KEEP**: weak-trained macro AUC over 12 targets >= **0.68** AND the 95% bootstrap CI of the paired difference (weak-trained minus gold-only CV) has lower bound > 0.
- **INVESTIGATE**: point estimate above gold-only CV and mean difference > 0, but KEEP not met.
- **REJECT**: otherwise.
- Binding variant: `production` (the series production would choose). `sagittal_only` is a sensitivity check and issues no verdict.
- 0.68 is a judgement call (about +0.065 over 0.616, roughly two *estimated*, unmeasured standard errors of 0.03-0.04). It is fixed.

**Validity preconditions (added in v2, before any result exists; they do not change the thresholds).** The rule above is applied only if ALL hold; otherwise the verdict is **WITHHELD** and KEEP is impossible:
1. all 12 targets are modelled (each has at least `MIN_WEAK_PER_CLASS = 20` weak positives AND 20 weak negatives, gold removed) and evaluated on gold;
2. all 58 gold studies have features, and the gold-only baseline covers all 12 targets (36 evaluations);
3. the bootstrap macro is taken over all 12 targets in every resample, and at most 1% of resamples are discarded because a target had a single class;
4. the official metric is verified to be macro ROC AUC (gates file);
5. all gates below passed. There is no bypass option in either script.

**Minimum class-count rule.** A target with fewer than 20 weak positives or fewer than 20 weak negatives is not modelled. This floor was fixed before the weak-label counts were computed. It must not be changed after the counts or any results are seen. If it blocks a target, the experiment's verdict is WITHHELD, not adjusted.

## Series selection: equivalence with production

| Aspect | Status |
|---|---|
| `ranked_series`, `ordered_dicom_paths`, `dicom_to_image`, `sample_indices` | Copied verbatim. At every start the extractor compares their source text and the constants `N_SLICES`, `SLICE_RANGE`, `MAX_SERIES_TRIES` with `rsna-submission.ipynb` and aborts on any difference. |
| Production series slot (`prod_*`) | First success among the first 4 candidates of the notebook's own ladder, including the `any_series` rung and retry semantics. Replaces the v1 approximation. |
| Selftest (synthetic DICOMs, fake encoder) | Matches the notebook's own `extract_study` exactly on: normal studies, a short series, a failing study, a retry (rank 1), a study with no sagittal series and an `any_series` study. Real encoder and real DICOMs are untested. |
| Series gate | For the 58 gold studies, the stored series, rung and rank must equal the first candidate of the notebook's own `ranked_series` (executed from the notebook source). Any gold study needing a retry fails the gate. Result in `series_check.json`; the analysis re-checks hashes. |
| Plane slots (`sagittal/coronal/axial_*`) | Extra, 3 tries per plane, NOT part of production, unused by EXP-019. |
| Remaining differences | Library/GPU numerical differences; the extractor also stores the mean patch token. |

## Gates (in order; stop at the first failure)

**G0. Weak-label pre-flight (CPU, before any GPU time).** Run the parser, then `analyse_exp019.py --audit-weak-labels`. If any target is below the floor, STOP and resolve Blocker 1.
**G1. Gates file.** Complete `exp019_gates.json` from the OFFICIAL pages: rules permit report-derived labels from the provided training reports, the metric is confirmed, and the weak-label quality policy is satisfied (audit done, or a recorded `DEC-nnn` waiver). Enforced by both scripts; a full extraction is refused without it.
**G2. Smoke test** (`--smoke`: 30 random gold + 20 random non-gold studies, seed 0, separate output directory). The report lists both groups separately. Operational checks: all gold studies produce a production slot; none needs a retry; non-gold production failure rate <= 10%. The smoke test cannot satisfy G4 and cannot be analysed.
**G3. Full extraction** (resumable). Save the output directory as a Kaggle dataset; record its size and SHA-256.
**G4. Series gate** (automatic at the end of extraction; `--verify-selection` to re-run).
**G5. Reproduction gate** (analysis): gold-only CV from the stored features must give 0.615784 within 0.005.
**G6. Analysis**, run once. Record the output verbatim in `results/EXP-019/README.md`. Any change is a new experiment ID.

Evidence gathered so far for G1 (secondary sources only, NOT official, so the gates file stays NOT_VERIFIED): press coverage and one third-party repository README state the metric is macro ROC AUC, that it is a code competition (<= 9 h, internet off), that the final submission date is 2026-10-22, and that the dataset pairs every study with its original radiology report (reports in 12 languages). Confirm every point on the competition pages.

## Blockers (open)

1. **The registered rule may be unreachable.** `EXPERIMENTS.md` (EXP-007) records parser counts with zero weak negatives for Medial OA (55 positive), Lateral OA (24) and PF OA (70), and only 12 negatives for Synovitis (381 positive). If those counts reproduce, 4 of 12 targets fall below the floor and the verdict will be WITHHELD. These counts have not been reproduced from the committed script. Options (needs your decision BEFORE extraction): (a) improve the labels for those targets first (EXP-014), (b) register a new experiment ID with a different, pre-declared rule (for example a hybrid arm, or a rule on the modelled targets only), (c) run EXP-019 knowing the verdict will be WITHHELD and the result exploratory.
2. **G1 is a human task.** Rules and metric must be confirmed from the official pages.
3. **Weak-label quality.** EXP-007's recorded decision and `docs/COMPETITIVE_RESEARCH_PLAN.md` ask for a manual audit before training on report-derived labels. Do the audit, or record a waiver as a new DEC entry.
4. **Baseline protocol.** The gold-only baseline uses the notebook's repeated stratified CV (the number recorded as 0.615784). DEC-004/DEC-006 prescribe the frozen EXP-010 folds for new comparisons. The analysis can add a descriptive pooled-OOF baseline on the frozen folds (`--frozen-folds`, SHA-256 checked), but the registered rule uses the notebook baseline. Decide whether that is acceptable.
5. **Language coverage.** Reports come in 12 languages; the parser's regexes cover English, Spanish, French, Dutch and German as written. Weak-label call rates will differ by language/site. The call-rate table is printed; interpret gold AUC with that in mind.
6. **Deadline.** Secondary sources give 2026-10-22 as the final submission date. Check it.
7. **Nothing has run on Kaggle.** The real encoder path, the real DICOMs and `train_series.csv` columns are untested; the smoke test is the first check.
8. **Register not updated.** Suggested row for `EXPERIMENTS.md`: `| EXP-019 | Frozen-DINOv2 probe on report-derived weak labels, tested on the 58 gold studies (pre-registered) | Planned; not run |`.
9. **Repeated patients.** No patient identifier exists, so overlap between gold and weak studies cannot be excluded (DEC-004 groups by study only).

## Known confounds

Label-definition drift between parser labels and gold annotations (for example the parser treats any mention of "effusion" as positive); the parser was reviewed against gold reports in EXP-007 (this does not touch model fitting, but parser quality on non-gold studies is probably lower than agreement on gold suggests); gold enrichment (the 58 studies show fracture 31%, effusion 60%, ACL 41%); one holdout of 58 gives a wide CI; baseline C=0.01 was tuned on the same 58 studies (conservative for KEEP); not leaderboard evidence (Gate C).

## How to run (Kaggle notebook with internet ON for the git clone, GPU on)

```
!git clone --branch research/exp019-weak-labels https://github.com/vixintry/rsna-knee-m1.git
%cd rsna-knee-m1
!python scripts/derive_report_weak_labels.py --output-dir /kaggle/working/exp007_weak_labels      # G0
!python scripts/exp019/analyse_exp019.py --audit-weak-labels --weak-labels /kaggle/working/exp007_weak_labels/weak_labels.csv
!cp scripts/exp019/exp019_gates.template.json scripts/exp019/exp019_gates.json                      # G1: now EDIT it
!python scripts/exp019/extract_features.py --smoke                                                  # G2
!python scripts/exp019/extract_features.py                                                          # G3, G4
!python scripts/exp019/analyse_exp019.py --features /kaggle/working/exp019_features \
    --weak-labels /kaggle/working/exp007_weak_labels/weak_labels.csv \
    --frozen-folds results/EXP-010/exp010_folds.csv                                                 # G5, G6
```
Self-tests (CPU, synthetic only): `python scripts/exp019/extract_features.py --selftest` and `python scripts/exp019/analyse_exp019.py --selftest`.

## Revision history

- **v1 (2026-10-17, committed in `4a89fef` as EXP-012):** first plan; superseded.
- **v2 (2026-10-18, before any run):** ID changed to EXP-019; production series slot and notebook-equivalence check; gold+non-gold smoke test with separate reporting; gates file (rules, metric, weak-label policy); validity preconditions and WITHHELD verdict; part-file shard writing and a strict shard reader; documented class floor; bypass flag removed; pre-flight weak-label audit; optional frozen-fold descriptive baseline. Decision thresholds unchanged.
