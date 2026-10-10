"""Conservative, English-only ACL weak-label extraction from radiology reports.

This module produces candidates for review; it does not create ground truth.
Unsupported languages and ambiguous contexts should remain uncertain/not mentioned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReportLabel:
    """A candidate label and the text fragment that triggered it."""

    status: str  # "positive", "negative", "uncertain", or "not_mentioned"
    evidence: str | None
    reason: str


_ACL_TERM = re.compile(
    r"\b(?:acl|anterior cruciate ligament)\b",
    flags=re.IGNORECASE,
)
_POSITIVE = re.compile(
    r"\b(?:tear(?:ed|s)?|ruptur(?:e|ed|es)|disrupt(?:ion|ed)|"
    r"sprain(?:ed)?|injur(?:y|ed)|partial[- ]thickness tear)\b",
    flags=re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(?:intact|normal|unremarkable|preserved|"
    r"no evidence of (?:a )?(?:tear|rupture|injury)|"
    r"without (?:evidence of )?(?:a )?(?:tear|rupture|injury)|"
    r"no (?:acl|anterior cruciate ligament) (?:tear|rupture|injury))\b",
    flags=re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|probable|questionable|疑|cannot exclude|"
    r"may represent|may reflect|could represent|suspicious for|"
    r"indeterminate|limited evaluation|not well seen)\b",
    flags=re.IGNORECASE,
)
_HISTORICAL = re.compile(
    r"\b(?:history of|prior|previous|previously|status post|old)\b",
    flags=re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    """Split roughly by sentence/line boundaries without external NLP packages."""
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if part.strip()
    ]


def extract_acl_candidate(report: str | None) -> ReportLabel:
    """Return a conservative ACL label candidate from an English report.

    Only explicit, local ACL context is considered. If the report is missing,
    the ACL is not mentioned, or evidence is ambiguous/conflicting, no binary
    label is forced. This intentionally does not support multilingual extraction.
    """
    if not isinstance(report, str) or not report.strip():
        return ReportLabel("not_mentioned", None, "empty_report")

    matches: list[ReportLabel] = []
    for sentence in _sentences(report):
        acl = _ACL_TERM.search(sentence)
        if not acl:
            continue

        evidence = sentence[:500]
        # If there are multiple claims about the ACL, prefer uncertainty over
        # forcing a binary decision from a potentially conflicting sentence.
        if _UNCERTAIN.search(sentence) or _HISTORICAL.search(sentence):
            matches.append(
                ReportLabel("uncertain", evidence, "uncertain_or_historical_context")
            )
            continue

        # Evaluate a small context around the ACL term and its sentence.
        # Conservative: explicit intact/normal or explicit no-tear language
        # supports a negative candidate; a tear/rupture/injury term supports
        # a positive candidate only when not negated.
        if _NEGATIVE.search(sentence):
            matches.append(ReportLabel("negative", evidence, "explicit_negative"))
        elif _POSITIVE.search(sentence):
            # Catch common local negation preceding a positive keyword.
            positive_match = _POSITIVE.search(sentence)
            prefix = sentence[max(0, positive_match.start() - 35):positive_match.start()]
            if re.search(r"\b(?:no|not|without|denies|negative for)\b", prefix, re.I):
                matches.append(
                    ReportLabel("negative", evidence, "negated_positive_term")
                )
            else:
                matches.append(ReportLabel("positive", evidence, "explicit_injury_term"))
        elif re.search(r"\b(?:intact|normal|unremarkable|preserved)\b", sentence, re.I):
            matches.append(ReportLabel("negative", evidence, "normal_acl_description"))
        else:
            matches.append(ReportLabel("uncertain", evidence, "acl_mentioned_without_clear_status"))

    if not matches:
        return ReportLabel("not_mentioned", None, "acl_not_mentioned")

    statuses = {item.status for item in matches}
    if "uncertain" in statuses or len(statuses) > 1:
        evidence = " | ".join(item.evidence or "" for item in matches)[:1000]
        return ReportLabel("uncertain", evidence, "ambiguous_or_conflicting_mentions")

    # Same status across mentions: return the first matching evidence, but retain
    # the fact that this is only a rule-based candidate.
    return matches[0]
