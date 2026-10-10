
"""Target definitions for the RSNA Knee Abnormality Detection project."""

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def get_targets() -> list[str]:
    """Return the target names in the official project order."""
    return TARGETS.copy()

