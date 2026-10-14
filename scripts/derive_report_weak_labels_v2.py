"""Experimental conservative parser wrapper for EXP-007.

This is deliberately separate from derive_report_weak_labels.py. It keeps the
baseline parser untouched and adds only high-confidence guards motivated by
the official target definitions and observed parser errors. It is diagnostic,
not clinically validated.
"""
import re

import numpy as np

from scripts.derive_report_weak_labels import classify_report as classify_baseline


def _has(pattern, text):
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def classify_report_v2(report, target):
    """Return the baseline tuple unless a conservative, documented guard applies."""
    raw = str(report or "")
    label, status, positive, negative = classify_baseline(raw, target)
    text = re.sub(r"\s+", " ", raw).strip()

    # An explicit "without surfacing tear" statement must not be turned into a
    # positive merely because the word "tear" appears near the meniscus.
    if target in ("Medial Meniscus", "Lateral Meniscus"):
        side = "medial" if target == "Medial Meniscus" else "lateral"
        negated_surface = rf"\b{side}\s+menisc\w*\b[^.;!?]{{0,100}}\bwithout\s+(?:a\s+)?surfacing\s+tear\b"
        if _has(negated_surface, text):
            # Preserve a positive only if a separate clause explicitly reports
            # a tear for the same meniscus.
            clauses = re.split(r"(?<=[.;!?])\s+|\n+", text)
            other_positive = any(
                (
                    _has(rf"\b{side}\s+menisc\w*\b[^.;!?]{{0,50}}\b(?:tear|torn|rupture|flap)\b", clause)
                    or _has(rf"\b(?:tear|torn|rupture|flap)\b[^.;!?]{{0,50}}\b{side}\s+menisc\w*\b", clause)
                )
                and not _has(r"\bwithout\s+(?:a\s+)?surfacing\s+tear\b", clause)
                for clause in clauses
            )
            if not other_positive:
                return 0.0, "negative", "", "without surfacing tear"

    # Competition ACL labels require a high-grade partial or full-thickness
    # tear. Low-grade/grade 1-2 injury or sprain wording alone is insufficient.
    if target == "ACL":
        acl_mention = r"\b(?:acl|anterior cruciate ligament)\b"
        low_grade = (
            _has(r"\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|mild)\b[^.;!?]{0,50}\b(?:tear|sprain|injury|signal|acl|anterior cruciate ligament)\b", text)
            or _has(r"\b(?:acl|anterior cruciate ligament)\b[^.;!?]{0,50}\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|mild)\b", text)
        )
        explicit_no_discontinuity = _has(
            r"\bwithout\s+(?:evidence\s+of\s+)?(?:fiber\s+)?discontinuity\b|"
            r"\bno\s+(?:fiber\s+)?discontinuity\b|"
            r"\bintact\b[^.;!?]{0,40}\b(?:acl|anterior cruciate ligament)\b",
            text,
        )
        qualifying = _has(
            r"\b(?:high[- ]grade|complete|full[- ]thickness|full thickness|"
            r"discontinuity|disrupted fibers|rupture|ruptured)\b",
            text,
        ) and not explicit_no_discontinuity
        if low_grade and (not qualifying or explicit_no_discontinuity) and _has(acl_mention, text):
            return 0.0, "negative", "", "low-grade ACL finding below target threshold"

    # MCL label threshold is high-grade partial or complete acute tear. Explicit
    # low-grade/chronic findings are negative; generic sprain language is left
    # unknown rather than promoted to a positive.
    if target == "MCL" and _has(r"\b(?:mcl|medial collateral ligament)\b", text):
        mcl_finding = _has(r"\b(?:tear|torn|rupture|sprain|injury|thickening)\b", text)
        low_or_chronic = _has(
            r"\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|chronic|remote|old)\b",
            text,
        )
        qualifying = _has(
            r"\b(?:high[- ]grade|complete|full[- ]thickness|full thickness)\b",
            text,
        ) and _has(r"\b(?:tear|torn|rupture|ruptured)\b", text)
        acute = _has(r"\b(?:acute|recent)\b", text)
        if mcl_finding and low_or_chronic and not qualifying:
            return 0.0, "negative", "", "low-grade or chronic MCL finding below target threshold"
        if mcl_finding and not qualifying:
            return float("nan"), "unknown", "", ""

        if qualifying and not acute:
            # The official criterion specifies acute injury; absent acute
            # wording is not enough evidence to assert a positive.
            return float("nan"), "unknown", "", ""

    # Only moderate/large effusions and Baker's cysts meet the label threshold.
    # Explicit small/trace wording alone is therefore negative.
    if target == "Effusion" and _has(r"\b(?:small|trace|minimal|tiny)\b[^.;!?]{0,30}\b(?:joint |knee |articular )?effusion\b", text):
        if not _has(r"\b(?:moderate|large|marked|substantial)\b[^.;!?]{0,30}\beffusion\b", text):
            return 0.0, "negative", "", "small/trace effusion below target threshold"
    if target == "Baker's" and _has(r"\b(?:small|trace|tiny)\b[^.;!?]{0,35}\b(?:baker['’]?s? cyst|popliteal cyst)\b", text):
        if not _has(r"\b(?:moderate|large|marked)\b[^.;!?]{0,35}\b(?:baker['’]?s? cyst|popliteal cyst)\b", text):
            return 0.0, "negative", "", "small/trace cyst below target threshold"

    return label, status, positive, negative
