# Project Journal — RSNA Knee Abnormality Detection

**Project:** Kaggle RSNA Knee Abnormality Detection  
**Repository:** https://github.com/vixintry/rsna-knee-m1  
**Journal created:** 2026-10-10  
**Purpose:** A detailed, newcomer-readable record of the project, what has actually been done, why decisions were made, what remains unknown, research ideas, and the next steps.

> **Status rule:** This document distinguishes completed work, observations from the Kaggle audit, plans, and hypotheses. A plan is not a result. A code test on synthetic data is not a test on the official dataset. No model score or competition submission should be claimed unless an actual run supports it.

---

## 1. Project in one minute

The goal is to build a reproducible machine-learning pipeline that predicts probabilities for 12 knee abnormalities from MRI studies and achieves the strongest legitimate performance possible in the RSNA Knee Abnormality Detection competition.

The challenge is not simply choosing a neural network. The dataset is very large, only a small subset of studies has complete expert labels, each study may contain several MRI series and many DICOM slices, and the official test-time inputs do not include radiology reports. This makes data integrity, weak supervision, MRI preprocessing, evaluation design, and compute constraints central research problems.

The project is being developed in a local VS Code repository on Windows, with Git/GitHub for source control and Kaggle-hosted notebooks/data for competition-scale work. The full MRI dataset should not be downloaded locally without a compelling reason and a verified storage plan.

### Current high-level status

- [x] Repository and core governance documents exist.
- [x] Python virtual environment and core data libraries are installed locally.
- [x] The 12 target names are centralised in code.
- [x] Basic label-summary and CSV-loader behaviour has been tested with synthetic data.
- [x] A Kaggle data-audit notebook has been run and produced the findings recorded below.
- [ ] Competition eligibility/rules acceptance and any required identity verification: user must confirm completion in Kaggle.
- [ ] Research public solutions and papers systematically and verify licences/rules for candidate methods.
- [ ] Establish and test a defensible study-level validation split.
- [ ] Build an initial valid baseline and measure its performance.
- [ ] Train and evaluate an MRI model.
- [ ] Create and verify a final competition submission.

**No model has been trained and no validated model score or verified submission is recorded as of this journal's creation.**

---

## 2. Competition task and inputs

The task is to predict 12 target labels for each test study:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

The expected output is a probability for every target and study, in the competition's required submission schema. The official scoring definition and exact submission format must be checked against the current competition documentation before implementing final evaluation or submission code.

### Dataset files described in the competition materials

- `train.csv`: one row per training study; includes `StudyInstanceUID`, `Report`, and target columns.
- `test.csv`: test study identifiers; the reports are unavailable for test-time prediction.
- `train_series.csv`: series-level metadata, including study and series identifiers, fluid-sensitive/fat-suppression flags, and anatomical plane.
- `test_series.csv`: corresponding test series metadata.
- DICOM series: MRI slices stored in a study → series → slice hierarchy.
- `sample_submission.csv`: template for the expected submission format.

The dataset description supplied for this project estimates approximately **569.76 GB and around 819,000 files**. This is a scale estimate from the supplied competition materials, not a fresh measurement of local files.

### Known data constraints

- MRI series can contain different numbers of slices; typical series were described as roughly 20–45 slices, with a long tail of much larger series.
- DICOM images may vary in orientation, intensity encoding, resolution, and metadata.
- Series may represent sagittal, coronal, or axial planes and different sequences.
- The test set does not provide reports at inference. Reports may be used for permitted training-time weak supervision, but the final predictor must work with the inputs allowed by the competition at test time.
- Study, series, and slice identifiers must be joined correctly. All series and slices from a study must remain together during train/validation splitting.

---

## 3. Data audit: findings already observed in Kaggle

The following observations came from a Kaggle audit run earlier in this project. They should be reproduced from the official files before being relied on in final experiments; this journal records the observed results rather than asserting that the audit is already packaged as a fully automated, reproducible pipeline.

### 3.1 Training labels

- `train.csv` shape: **4,407 rows × 14 columns**.
- **58 of 4,407 studies** had all 12 target labels populated.
- The other **4,349 rows** had all 12 target fields missing in the audit output.
- Within the 58 fully labelled studies, there were **456 observed zeros and 240 observed ones**, for 696 observed labels total.
- Every one of those 58 studies had at least one positive target; the largest number of positive targets in a study was 9.

Positive counts among the 58 fully labelled studies:

| Target | Positive studies |
|---|---:|
| Effusion | 35 |
| Synovitis | 27 |
| Medial Meniscus | 26 |
| ACL | 24 |
| Lateral Meniscus | 23 |
| PF OA | 21 |
| Contusion | 19 |
| Fracture | 18 |
| Medial OA | 15 |
| Baker's | 12 |
| Lateral OA | 11 |
| MCL | 9 |

**Interpretation:** 58 complete studies is an extremely small supervised sample for a 12-target image task. It creates risks of high-variance validation, unstable per-target AUC, and overfitting. It also means missing target values must not be silently converted to negatives. The report text may offer a way to construct additional *weak* labels, but report-derived labels are noisy and must be evaluated separately from the expert-provided labels.

The audit result that the other 4,349 rows have all targets missing is important and must be verified carefully against the official dataset version and competition documentation before designing training logic.

### 3.2 Series metadata

- `train_series.csv` shape: **24,371 rows × 5 columns**.
- Anatomical plane counts:
  - Sagittal: 9,864
  - Coronal: 8,609
  - Axial: 5,898
- The `Fluid_Sensitive` and `Fat_Suppression` counts observed in the audit were identical:
  - value 1: 14,010
  - value 0: 10,361
- The 58 fully labelled studies were associated with **336 series**:
  - sagittal: 136
  - coronal: 119
  - axial: 81
- Series per fully labelled study: minimum 4, median 5, mean 5.79, maximum 9.

These statistics suggest that a study-level model may need to select or aggregate multiple series and planes. They do not prove that all planes are equally useful or that the metadata flags are predictive.

### 3.3 DICOM inspection

A sample labelled study was inspected in Kaggle:

- Study identifier: `1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149`
- One inspected series identifier: `1.2.826.0.1.3680043.8.498.12451507871565834868863189854063726441`
- The study had five series with observed slice counts of 30, 36, 30, 32, and 36.
- The inspected series had 30 slices.
- `pydicom` version observed in Kaggle: 3.0.2.
- One inspected slice was 384 × 384 pixels, `MONOCHROME2`, Modality `MR`, `uint16` pixel array, BitsStored 12, rescale slope 1 and intercept 0.
- Its pixel values ranged from 0 to 1100, with mean about 202.47 and standard deviation about 224.77.
- Image orientation and pixel-spacing metadata were present.

This was a spot check, not a complete DICOM compatibility audit. A robust pipeline still needs to use DICOM geometry to order slices and handle different series consistently.

---

## 4. Development environment

### Local computer reported by the user

- OS/workflow: Windows, VS Code, PowerShell.
- CPU: Intel Core i5-12400F.
- RAM: 16 GB.
- GPU: AMD Radeon RX 6600 with 8 GB VRAM.
- Storage: 1 TB NVMe SSD.
- Python 3.12.10 installed for this project; Python 3.14.4 also exists on the system.
- Local virtual environment: `.venv` in the repository root.
- Selected VS Code interpreter: `.venv\Scripts\python.exe`.
- Installed core packages include NumPy 2.5.4 and pandas 3.0.6, with pinned dependencies in `requirements.txt`.

The exact versions should be checked against `requirements.txt` and the environment when reproducing a run. Package availability does not guarantee compatibility with every deep-learning framework or GPU backend. The AMD GPU should not be assumed to accelerate a particular framework on Windows without a compatibility test.

### Compute plan

Use the local machine for source code, CSV-scale tests, documentation, and small diagnostics. Use Kaggle-hosted compute for official dataset access and potentially for GPU training, subject to the competition's runtime and internet restrictions. The competition materials reviewed earlier specified a notebook execution limit of up to 9 hours and internet disabled for submission execution; recheck the live rules before the final run.

Do not download the full MRI dataset locally by default. Do not commit datasets, model weights, predictions, or large generated artifacts to Git.

---

## 5. Repository state and implementation history

The canonical repository is `vixintry/rsna-knee-m1`, on branch `main`.

### Governance and tracking files already in the repository

- `README.md`: short project overview, principles, current status, and links to governance files.
- `PROJECT_SPEC.md`: objectives, project rules, data integrity, evaluation design, research stages, and coding standards.
- `DECISIONS.md`: approved decisions and open questions.
- `RESEARCH_LOG.md`: recorded observations and sources.
- `EXPERIMENTS.md`: experiment index and record template.
- `notebooks/00_data_audit.ipynb`: Kaggle data-audit notebook.

This new `PROJECT_JOURNAL.md` is intended to be the most detailed newcomer-oriented overview and chronological status record. The more specialised files above remain useful: decisions belong in `DECISIONS.md`, observations in `RESEARCH_LOG.md`, and actual model experiments/results in `EXPERIMENTS.md`.

### Python modules

#### `src/labels.py`

Defines the target list `TARGETS` in the competition's intended order and exposes `get_targets()`, which returns a copy of that list.

A terminal test successfully reported 12 targets and confirmed that `get_targets() is not TARGETS`.

#### `src/validation.py`

The committed version contains:
- `validate_label_columns(df)`: checks that all 12 target columns exist.
- `get_label_summary(df)`: counts observed positive values, negative values, and missing values for each target.

A synthetic test successfully produced one positive, one negative, and one missing value for every target.

**Local working-copy note at journal creation:** the user then added a `validate_label_values(df)` function to reject non-binary values other than 0, 1, or missing, and created `test_validation.py`. The test passed locally: valid labels were accepted and an ACL value of 2 was rejected with a `ValueError`. These changes had **not yet been committed or pushed** when this journal was written, so they may exist only in the user's local working tree. Check `git status` before syncing or editing further. Do not assume the GitHub copy already contains them.

#### `src/data.py`

Defines `load_train_csv(path)`, which checks that the path exists, loads a CSV, verifies required identifier/report/target columns, rejects missing study IDs, and rejects duplicate study IDs.

A synthetic CSV test passed and returned shape `(1, 14)` with all target columns present. This was not a test against the official training CSV on the local computer.

### Relevant local Git history reported by the user

- `1480403`: data-audit notebook commit (reported earlier in the conversation).
- `fe1a319`, `c4ed760`, `4bc4688`: dependency-file and encoding-fix commits (reported earlier).
- `e532567`: `Add central target definitions`.
- `9dacdbb`: `Add label validation and summary utilities`.
- `dddd0b2`: `Add training CSV loader and validation`.

The latest confirmed push before this journal was `dddd0b2`. These abbreviated hashes are a conversation record; use `git log --oneline` to confirm the exact current history.

### Tests performed so far

All tests below used synthetic/local test inputs unless explicitly noted:

1. Target count and copy semantics: passed.
2. Label summary with values 1, 0, and missing: passed.
3. Training CSV loader with a temporary one-row CSV: passed.
4. Label-value validation with valid values and an intentionally invalid ACL value of 2: passed locally; code/test file not yet committed at the time of writing.

The Kaggle data-audit notebook was run separately in Kaggle and produced the audit observations in Section 3.

### Known unfinished engineering work

- Turn manual tests into a proper automated test suite (for example, `tests/test_labels.py`, `tests/test_validation.py`, and `tests/test_data.py`).
- Finish the validation code change and decide whether to keep `test_validation.py` at the root or move tests into a `tests/` directory.
- Add tests for malformed CSVs, missing required columns, duplicate IDs, and invalid labels.
- Add loaders for the series metadata and submission template only when needed.
- Keep module responsibilities small; do not build abstractions without a research need.

---

## 6. Project rules and scientific integrity

These rules are mandatory for all work.

1. **No fabricated results.** Never invent a metric, claim an experiment ran when it did not, or describe a submission as successful without evidence.
2. **No silent methodology changes.** Record changes to labels, data splits, preprocessing, model selection, or evaluation.
3. **No leakage.** A study and all of its series/slices belong in only one partition. Learned transformations must be fit using training data only.
4. **No leaderboard overfitting.** Use local validation as the main model-development signal. Public leaderboard scores are supplementary.
5. **Traceability.** Record code commit, data version, split, seed, configuration, environment, runtime, metrics, and conclusion for each meaningful experiment.
6. **Fair comparisons.** Compare candidate models on the same split and metric unless a protocol change is justified and recorded.
7. **Preserve history.** Keep previous results/configurations; do not overwrite the only copy of a useful run.
8. **Explain decisions.** Write down the hypothesis, expected benefit, risks, implementation, and success criterion before running a meaningful experiment.
9. **Respect compute constraints.** Begin with small diagnostics and estimate memory/runtime before expensive jobs.
10. **Follow competition rules.** Confirm current rules, permitted external data/weights, inference constraints, eligibility, and deadlines.
11. **Distinguish label sources.** Expert labels, report-derived weak labels, pseudo-labels, and model predictions must not be mixed without tracking their provenance.
12. **Do not assume missing means negative.** Missing target labels must remain missing unless the official data specification provides a justified interpretation.
13. **Respect third-party licences.** Reuse published ideas and code only in accordance with the relevant licence and attribution requirements. Never misrepresent someone else's work as original.
14. **Do not use test reports at inference.** The supplied competition description says they are not available for the test set.

---

## 7. Research strategy: how we will learn from public solutions and papers

The user proposed studying other competitors' public projects and papers rather than inventing every technique independently. This is an accepted research direction. It is not a decision to copy a solution blindly.

For each promising public notebook, repository, paper, or discussion, create a source record with:
- Title and URL.
- Author(s) and publication/date if available.
- Source type: official rule, paper, repository, notebook, or discussion.
- Licence and reuse restrictions, where applicable.
- Exact technique being considered.
- What problem it is intended to solve in this competition.
- Evidence reported by the source and what validation protocol produced it.
- Limitations, possible leakage, and compute cost.
- Whether we independently reproduced it.
- Decision: adopt, adapt, investigate, or reject.

### Research questions to investigate

#### A. Report-derived weak supervision
Can radiology reports provide useful additional training labels for studies with missing expert targets?

Possible methods:
- Rule-based extraction using medical terms, negation, uncertainty, and section/context cues.
- A local multilingual LLM that converts each report into structured findings and confidence/uncertainty fields.
- A hybrid system where rules and the LLM are compared and disagreements are reviewed.

Required safeguards:
- Check competition rules and the model's licence before use.
- Do not call LLM outputs ground truth.
- Keep expert labels and weak labels distinct.
- Evaluate agreement on the 58 complete studies, but recognise that this is a small sample and may not represent all reports.
- Inspect target-wise false positives/false negatives and multilingual performance.
- Check for distribution differences between fully labelled studies and unlabelled-report studies.
- Do not use reports as test-time features when they are unavailable for test studies.

An LLM is a candidate tool, not a foregone conclusion. We should compare it against a simpler rule-based baseline and retain it only if evidence supports the extra complexity.

#### B. MRI preprocessing and sampling
Research how to:
- Read DICOM pixel arrays and relevant metadata robustly.
- Order slices using physical image position/orientation, not just filename order.
- Identify relevant MRI series and planes.
- Select a manageable number of slices per series/study.
- Resize/normalise images consistently without destroying important signal.
- Aggregate multiple slices and series into a study-level prediction.

Start with a small subset and profile RAM, VRAM, throughput, and runtime before scaling up.

#### C. Pretrained image models
Investigate appropriate pretrained medical-imaging or general computer-vision encoders, permitted weights, input-channel strategies, and multi-slice methods. Compare them using the same fixed split. The local AMD GPU's compatibility with the chosen training framework must be measured, not assumed.

#### D. Small-sample evaluation
Only 58 fully labelled studies makes validation difficult. Investigate:
- Stratification or iterative multilabel stratification where feasible.
- Repeated or grouped cross-validation, if compute and label counts support it.
- Per-target class counts in each fold.
- How the official metric handles targets with only one class.
- Uncertainty in performance estimates.

All slices/series from the same study must stay together. Avoid choosing a split solely because it gives an attractive score.

#### E. Metadata baselines and ensembling
A metadata-only model may be useful as a diagnostic, but series descriptors should not be assumed to predict pathology. Ensembling is a later option only if independently validated models make complementary errors.

### Literature/source review workflow

1. Start with official competition rules and the official evaluation definition.
2. Search the competition's public notebooks/discussions and public code repositories.
3. Search papers for methods that solve the specific identified bottleneck (weak supervision, MRI series selection, multi-instance/multi-slice learning, small-sample evaluation).
4. Record sources and licences before implementing.
5. Select one promising, affordable idea.
6. State the hypothesis and success criterion before coding.
7. Reproduce a baseline first, then change one major factor at a time.
8. Keep negative results in the log.

Do not rely on a source's public leaderboard rank alone. Check the validation method, possible leakage, date, compute budget, and whether its assumptions match this dataset.

---

## 8. Planned roadmap and stage gates

The dates below are a prioritisation plan, not a guarantee of completion. The competition deadline is close, so research should be focused and time-boxed.

### Stage 0 — Entry and rules
**Status:** Needs user confirmation.

- Confirm competition entry/rules acceptance in Kaggle.
- Complete any identity-verification prompt personally if required.
- Recheck the live rules, evaluation metric, notebook runtime, internet restriction, external data/weights policy, and submission schema.
- Record any restrictions relevant to LLM use or report processing.

**Gate:** Do not invest heavily in a method until its compliance is clear.

### Stage 1 — Research triage
**Next immediate project task.**

- Review a small number of relevant public baselines and papers.
- Focus on the two most consequential questions: how to learn from sparse labels and how to turn a multi-series DICOM study into a model input.
- Record sources, techniques, licences, reported validation, risks, and compute cost.
- Decide which ideas are worth a small experiment.

**Deliverable:** A ranked research shortlist and a documented baseline plan, not merely a collection of links.

### Stage 2 — Data and validation integrity
- Re-run/verify the CSV audit on the official Kaggle inputs.
- Confirm exact missingness semantics and which studies/targets can be used for supervised loss.
- Verify ID relationships across train/test and series tables.
- Define a deterministic study-level split, check class counts, and freeze its identifier.
- Add tests that ensure there is no study overlap across partitions.
- Confirm the official metric implementation.

**Gate:** No model comparison until the split and metric are defined and tested.

### Stage 3 — Baseline
- Create a valid sample-submission pipeline.
- Determine what can be learned directly from expert labels.
- Consider a deliberately simple baseline before a complex model.
- Record per-target metrics and any undefined AUC cases.
- Save exact code/configuration/data split information.

**Gate:** No claim of model improvement without a comparable measured result.

### Stage 4 — Weak-label experiment
- Compare rule-based report extraction with a local LLM or hybrid extraction, if permitted and feasible.
- Keep output states distinct: positive, negative, uncertain, and not mentioned.
- Evaluate against expert labels and inspect errors.
- Decide whether weak labels improve training data quality enough to justify use.

**Gate:** Do not treat generated labels as truth or let the model see unavailable test reports.

### Stage 5 — MRI baseline and iteration
- Build a DICOM reader and deterministic series/slice selection.
- Profile a small sample before processing large volumes.
- Train a feasible pretrained image model.
- Add multi-plane/multi-series aggregation only when a simpler model is understood.
- Test one major change at a time and retain all results.

### Stage 6 — Finalisation
- Select using the predeclared local validation procedure.
- Verify the notebook works with internet disabled and within the runtime limit.
- Check exact study IDs, output columns/order, row count, finite probabilities, and submission filename.
- Submit only after an end-to-end test.
- Record the submission identifier and actual outcome.

### Priority order under time pressure

1. Competition entry/rules.
2. Correctly understood labels and validation.
3. A reproducible baseline and valid submission path.
4. Highest-value weak-label and MRI experiments.
5. Additional complexity only if time and evidence justify it.

---

## 9. Experiment tracking template

Every meaningful experiment belongs in `EXPERIMENTS.md`. Use a unique identifier such as `EXP-001`.

Record:
- **Question/hypothesis:** What are we testing?
- **Why it might work:** What evidence or mechanism motivates it?
- **Change from baseline:** Exactly one main change where possible.
- **Data and labels:** Version, source, label provenance, filters.
- **Split:** Frozen split ID and seed.
- **Model/configuration:** Architecture, preprocessing, hyperparameters.
- **Environment:** Kaggle/local, framework/library versions, hardware.
- **Runtime/memory:** Actual measurements.
- **Primary metric:** Official metric or clearly named diagnostic.
- **Per-target results:** Include class counts and undefined metrics.
- **Errors/caveats:** Failures, uncertainty, suspected leakage.
- **Conclusion:** What did the evidence show?
- **Decision:** Adopt, reject, or investigate further.
- **Code commit:** Git hash associated with the run.

A failed run is still informative. Record why it failed instead of quietly removing it.

---

## 10. Open questions and risks

These remain unresolved and must not be treated as settled facts:

1. Has the user completed Kaggle's required entry/identity steps?
2. What exact current official metric and edge-case aggregation does the evaluation code use?
3. What does the official specification say about missing target labels, and can the 58 fully labelled studies be confirmed against the current data version?
4. Is there a robust validation split with enough positives and negatives per target?
5. Which report language(s) occur, and how reliable are candidate extraction methods across them?
6. Which public solutions are methodologically sound, licenced appropriately, and reproducible under the competition rules?
7. Which pretrained weights and external resources are permitted and feasible?
8. Can the intended training framework use the local AMD GPU, or should all GPU training happen on Kaggle?
9. What MRI sampling strategy retains sufficient signal within the runtime limit?
10. Can a meaningful local baseline be trained with the expert-labelled sample size, and how uncertain will its score be?
11. What final submission format and offline-execution requirements are currently enforced by Kaggle?

### Main risks

- **Sparse labels:** overfitting and unreliable validation.
- **Label noise:** report-derived labels may conflict with the competition's target definitions.
- **Leakage:** related series/slices from the same study could cross split boundaries.
- **Compute:** the dataset is far too large to casually download and process on the local machine.
- **Time:** research can expand indefinitely; each research task must have a concrete deliverable.
- **Public code:** solutions may contain leakage, undocumented assumptions, restrictive licences, or results that cannot be reproduced.
- **Metric mismatch:** a locally calculated score is not meaningful unless it matches the official evaluation.
- **Stale documentation:** always distinguish what was observed in an earlier audit from what has been rechecked on the current dataset/version.

---

## 11. How a new contributor should start

1. Read this journal to understand the whole project.
2. Read `README.md` for the short overview.
3. Read `PROJECT_SPEC.md` for non-negotiable rules.
4. Read `DECISIONS.md` before changing methodology.
5. Read `RESEARCH_LOG.md` for dated observations and their provenance.
6. Read `EXPERIMENTS.md` before starting a model run.
7. Inspect `git status` and `git log --oneline` before modifying files.
8. Use the documented Python virtual environment and run the tests.
9. Do not assume local files are synced to GitHub; check the current branch and working tree.
10. Before any expensive run, state the hypothesis, resource estimate, split, metric, and success criterion.

### Where each kind of information belongs

- **Project-wide context and current roadmap:** `PROJECT_JOURNAL.md` (this file).
- **Rules and technical standards:** `PROJECT_SPEC.md`.
- **Approved decisions and methodology changes:** `DECISIONS.md`.
- **Audited facts, observations, and research notes:** `RESEARCH_LOG.md`.
- **Actual experiment configurations and results:** `EXPERIMENTS.md`.
- **How to install/run the code:** `README.md`.
- **Implementation:** `src/`, `tests/`, `configs/`, and `notebooks/` as the project grows.

---

## 12. Immediate next actions

1. **Sync local and GitHub state carefully.** The label-value validator and `test_validation.py` passed locally but were not yet committed/pushed when this journal was created. Check `git status`; preserve these changes before pulling the new journal commit from GitHub.
2. Confirm competition entry and current rules in Kaggle.
3. Research and record a short list of public baselines/papers, checking licences and validation quality.
4. Choose the first experiment based on the audit findings and deadline, rather than adding more generic infrastructure.
5. Update this journal whenever the project status or roadmap materially changes. Add dates and preserve historical findings instead of silently rewriting them.

**Last status assertion:** The repository's last confirmed push before this journal was commit `dddd0b2`. At that point, the basic modules had been added and synthetic tests had passed. The local label-value validation change had also passed a test but was still uncommitted. No model had been trained, no score had been measured, and no submission had been verified.
