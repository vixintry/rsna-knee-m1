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


def _clauses(text):
    """Split report text into sentence-like clauses for local cue matching."""
    return [part.strip() for part in re.split(r"(?<=[.;!?])\s+|\n+", text) if part.strip()]


def classify_report_v2(report, target):
    """Return the baseline tuple unless a conservative, documented guard applies."""
    raw = str(report or "")
    label, status, positive, negative = classify_baseline(raw, target)
    text = re.sub(r"\s+", " ", raw).strip()
    clauses = _clauses(text)

    # An explicit "without surfacing tear" statement must not be turned into a
    # positive merely because the word "tear" appears near the meniscus.
    if target in ("Medial Meniscus", "Lateral Meniscus"):
        side = "medial" if target == "Medial Meniscus" else "lateral"
        negated_surface = rf"\b{side}\s+menisc\w*\b[^.;!?]{{0,100}}\bwithout\s+(?:a\s+)?surfacing\s+tear\b"
        if _has(negated_surface, text):
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

    # Severity/negation cues must refer to the ACL finding, not another
    # ligament elsewhere in the report.
    if target == "ACL":
        acl_mention = r"\b(?:acl|anterior cruciate ligament)\b"
        acl_clauses = [c for c in clauses if _has(acl_mention, c)]
        explicit_acl_negative = any(
            _has(r"\b(?:acl|anterior cruciate ligament)\b[^.;!?]{0,50}\b(?:intact|normal|no tear|without tear|no evidence of tear)\b", c)
            or _has(r"\b(?:intact|normal|no tear|without tear|no evidence of tear)\b[^.;!?]{0,50}\b(?:acl|anterior cruciate ligament)\b", c)
            for c in acl_clauses
        )
        if explicit_acl_negative:
            return 0.0, "negative", "", "explicit ACL normal/intact/no-tear statement"

        low_grade_acl = any(
            _has(r"\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|mild)\b[^.;!?]{0,50}\b(?:tear|sprain|injury|signal|acl|anterior cruciate ligament)\b", c)
            or _has(r"\b(?:acl|anterior cruciate ligament)\b[^.;!?]{0,50}\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|mild)\b", c)
            for c in acl_clauses
        )
        qualifying_acl = any(
            _has(r"\b(?:high[- ]grade|complete|full[- ]thickness|full thickness|discontinuity|disrupted fibers|rupture|ruptured|tear|torn)\b", c)
            and not _has(r"\b(?:no|without|negative for)\s+(?:evidence of\s+)?(?:a\s+)?(?:tear|rupture|discontinuity)\b", c)
            for c in acl_clauses
        )
        if low_grade_acl and not qualifying_acl:
            return 0.0, "negative", "", "low-grade ACL finding below target threshold"

    # MCL decisions are based on clauses mentioning the MCL. Preserve explicit
    # normal/intact negatives even when another structure is abnormal.
    if target == "MCL":
        mcl_mention = r"\b(?:mcl|medial collateral ligament(?: complexes?)?)\b"
        mcl_clauses = [c for c in clauses if _has(mcl_mention, c)]
        if mcl_clauses:
            explicit_mcl_negative = any(
                _has(r"\b(?:intact|normal|unremarkable|no tear|without tear|no evidence of tear)\b", c)
                for c in mcl_clauses
            )
            if explicit_mcl_negative:
                return 0.0, "negative", "", "explicit MCL normal/intact/no-tear statement"

            mcl_finding = any(_has(r"\b(?:tear|torn|rupture|sprain|injury|thickening)\b", c) for c in mcl_clauses)
            low_or_chronic = any(
                _has(r"\b(?:low[- ]grade|grade\s*[12]|grade\s+[iI]{1,2}|chronic|remote|old)\b", c)
                for c in mcl_clauses
            )
            qualifying_mcl = any(
                _has(r"\b(?:high[- ]grade|complete|full[- ]thickness|full thickness)\b", c)
                and _has(r"\b(?:tear|torn|rupture|ruptured)\b", c)
                and not _has(r"\b(?:no|without|negative for)\s+(?:evidence of\s+)?(?:a\s+)?(?:tear|rupture)\b", c)
                for c in mcl_clauses
            )
            if mcl_finding and low_or_chronic and not qualifying_mcl:
                return 0.0, "negative", "", "low-grade or chronic MCL finding below target threshold"
            if mcl_finding and not qualifying_mcl:
                return float("nan"), "unknown", "", ""
            # Do not require the word "acute" elsewhere in the report to
            # validate a target-specific complete/high-grade MCL tear. The
            # official criterion still governs whether a positive is warranted.

    # Size qualifiers must be tied to the same effusion/cyst finding. Conflicting
    # small and moderate descriptions are left to the baseline rather than
    # resolved by a report-wide word match.
    if target == "Effusion":
        effusion_clauses = [c for c in clauses if _has(r"\beffusion\b", c)]
        small = any(_has(r"\b(?:small|trace|minimal|tiny)\b", c) for c in effusion_clauses)
        moderate = any(_has(r"\b(?:moderate|large|marked|substantial)\b", c) for c in effusion_clauses)
        if small and not moderate:
            return 0.0, "negative", "", "small/trace effusion below target threshold"

    if target == "Baker's":
        cyst_clauses = [c for c in clauses if _has(r"\b(?:baker['’]?s? cyst|popliteal cyst)\b", c)]
        small = any(_has(r"\b(?:small|trace|tiny)\b", c) for c in cyst_clauses)
        moderate = any(_has(r"\b(?:moderate|large|marked)\b", c) for c in cyst_clauses)
        if small and not moderate:
            return 0.0, "negative", "", "small/trace cyst below target threshold"

    return label, status, positive, negative
