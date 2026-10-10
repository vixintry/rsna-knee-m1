# ACL report-label prototype

This prototype extracts **candidate** ACL labels from report text. It is an exploratory weak-supervision tool, not a source of ground truth.

## Output states
- `positive`: explicit ACL injury/tear language without detected uncertainty or negation.
- `negative`: explicit intact/normal or no-tear language.
- `uncertain`: ambiguous, historical, conflicting, or insufficiently clear ACL mention.
- `not_mentioned`: no ACL mention or empty report.

## Scope and limitations
- English-only rules. Multilingual reports are not translated or automatically labelled.
- Simple sentence splitting and regex rules; complex clinical language can be misread.
- This is not a complete clinical NLP system. A negative result must not be inferred from `not_mentioned`.
- Report-derived candidates must never overwrite official labels.
- Test-time reports are unavailable, so this is for training-time weak-supervision research only.
- Validate against the 58 studies with official labels and manually inspect false positives/negatives before using candidates for training.

## Run tests
From the repository root:
```powershell
python -m unittest discover -s tests -v
```

No validation performance is claimed until the prototype is run against the competition data.
