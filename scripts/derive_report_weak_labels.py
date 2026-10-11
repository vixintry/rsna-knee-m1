#!/usr/bin/env python3
"""Conservative rule-based weak labels from knee MRI reports.

Only studies with every official target missing are processed. Explicit positive
or negative cues may yield 1 or 0; no cue, uncertainty, or conflicting cues yield
NaN. This is a candidate generator, not validated clinical NLP.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Keep these rules conservative. Silence is never a negative label.
RULES = {
"ACL": {
"positive": [
r"\b(?:acl|anterior cruciate ligament)\b.{0,40}\b(?:tear|torn|rupture|ruptured|sprain|injury)\b",
r"\b(?:tear|rupture|sprain|injury)\b.{0,40}\b(?:acl|anterior cruciate ligament)\b",
r"\b(?:rotura|ruptura)\b.{0,40}\bligamento cruzado anterior\b",
r"\b(?:rupture|déchirure)\b.{0,40}\bligament croisé antérieur\b",
r"\bvoorste kruisband\b.{0,30}\b(?:scheur|ruptuur|letsel)\b"],
"negative": [
r"\b(?:acl|anterior cruciate ligament)\b.{0,25}\b(?:intact|normal|unremarkable)\b",
r"\b(?:no|without|absence of)\b.{0,40}\b(?:acl|anterior cruciate ligament)\b.{0,25}\b(?:tear|rupture|injury)\b",
r"\bligamento cruzado anterior\b.{0,30}\b(?:íntegro|integro|normal)\b",
r"\bligament croisé antérieur\b.{0,30}\b(?:intact|normal)\b"]},
"MCL": {
"positive": [
r"\b(?:mcl|medial collateral ligament)\b.{0,40}\b(?:tear|rupture|sprain|injury|thickening)\b",
r"\b(?:tear|rupture|sprain|injury)\b.{0,40}\b(?:mcl|medial collateral ligament)\b",
r"\b(?:esguince|rotura|ruptura)\b.{0,35}\bligamento colateral medial\b",
r"\b(?:entorse|rupture|déchirure)\b.{0,35}\bligament collatéral médial\b"],
"negative": [
r"\b(?:mcl|medial collateral ligament)\b.{0,25}\b(?:intact|normal|unremarkable)\b",
r"\b(?:no|without|absence of)\b.{0,40}\b(?:mcl|medial collateral ligament)\b.{0,25}\b(?:tear|rupture|injury)\b",
r"\bligamento colateral medial\b.{0,25}\b(?:íntegro|integro|normal)\b"]},
"Medial Meniscus": {
"positive": [
r"\bmedial menisc(?:us|al)\b.{0,40}\b(?:tear|torn|rupture|flap)\b",
r"\b(?:tear|torn|rupture)\b.{0,40}\bmedial menisc(?:us|al)\b",
r"\b(?:rotura|ruptura)\b.{0,35}\bmenisco (?:medial|interno)\b",
r"\b(?:déchirure|rupture)\b.{0,35}\b(?:ménisque médial|ménisque interne)\b",
r"\b(?:mediale meniscus|binnenmeniscus)\b.{0,30}\b(?:scheur|ruptuur)\b"],
"negative": [
r"\bmedial menisc(?:us|al)\b.{0,30}\b(?:intact|normal|not torn|no tear|without tear)\b",
r"\b(?:no|without|absence of|no evidence of|no signs of)\b.{0,35}\b(?:tear|rupture)\b.{0,30}\bmedial menisc(?:us|al)\b",
r"\bmenisco (?:medial|interno)\b.{0,30}\b(?:sin rotura|normal|conservado)\b",
r"\b(?:ménisque médial|ménisque interne)\b.{0,30}\b(?:intact|normal|sans déchirure)\b"]},
"Lateral Meniscus": {
"positive": [
r"\blateral menisc(?:us|al)\b.{0,40}\b(?:tear|torn|rupture|flap)\b",
r"\b(?:tear|torn|rupture)\b.{0,40}\blateral menisc(?:us|al)\b",
r"\b(?:rotura|ruptura)\b.{0,35}\bmenisco lateral\b",
r"\b(?:déchirure|rupture)\b.{0,35}\b(?:ménisque latéral|ménisque externe)\b",
r"\b(?:laterale meniscus|buitenmeniscus)\b.{0,30}\b(?:scheur|ruptuur)\b"],
"negative": [
r"\blateral menisc(?:us|al)\b.{0,30}\b(?:intact|normal|not torn|no tear|without tear)\b",
r"\b(?:no|without|absence of|no evidence of|no signs of)\b.{0,35}\b(?:tear|rupture)\b.{0,30}\blateral menisc(?:us|al)\b",
r"\bmenisco lateral\b.{0,30}\b(?:sin rotura|normal|conservado)\b",
r"\b(?:ménisque latéral|ménisque externe)\b.{0,30}\b(?:intact|normal|sans déchirure)\b"]},
"Medial OA": {
"positive": [
r"\bmedial\b.{0,35}\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose|gonarthrosis)\b",
r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose|gonarthrosis)\b.{0,35}\bmedial\b",
r"\bartrosis femorotibial medial\b",
r"\bcompartiment médial\b.{0,35}\b(?:arthrose|ostéoarthrose)\b"],
"negative": [
r"\bno\b.{0,25}\bmedial (?:compartment )?(?:osteoarthritis|osteoarthrosis|arthrosis)\b",
r"\bpas d['’]arthrose\b.{0,35}\b(?:compartiment médial|médial)\b"]},
"Lateral OA": {
"positive": [
r"\blateral\b.{0,35}\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose|gonarthrosis)\b",
r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose|gonarthrosis)\b.{0,35}\blateral\b",
r"\bartrosis femorotibial lateral\b",
r"\bcompartiment latéral\b.{0,35}\b(?:arthrose|ostéoarthrose)\b"],
"negative": [
r"\bno\b.{0,25}\blateral (?:compartment )?(?:osteoarthritis|osteoarthrosis|arthrosis)\b",
r"\bpas d['’]arthrose\b.{0,35}\b(?:compartiment latéral|latéral)\b"]},
"PF OA": {
"positive": [
r"\bpatellofemoral\b.{0,35}\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose)\b",
r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|artrosis|arthrose)\b.{0,35}\bpatellofemoral\b",
r"\b(?:retropatellar|patellofemoral)\b.{0,35}\b(?:arthrose|artrosis|osteoarthritis)\b"],
"negative": [
r"\bno\b.{0,25}\bpatellofemoral (?:osteoarthritis|osteoarthrosis|arthrosis)\b",
r"\bpas d['’]arthrose\b.{0,35}\b(?:patellofémoral|rétropatellaire)\b"]},
"Effusion": {
"positive": [r"\b(?:joint |knee |articular )?effusion\b",
r"\b(?:derrame articular|derrame de rodilla|épanchement articulaire|épanchement du genou|gewrichtseffusie)\b"],
"negative": [r"\b(?:no|without|absence of|no significant)\b.{0,25}\b(?:joint |knee |articular )?effusion\b",
r"\bno hay (?:evidente )?derrame(?: articular)?\b",
r"\bpas d['’]épanchement(?: articulaire)?\b", r"\bgeen (?:gewrichts)?effusie\b", r"\bkein gelenkerguss\b"]},
"Synovitis": {
"positive": [r"\bsynovitis\b", r"\bsinovitis\b", r"\bsynovite\b", r"\bsynoviale ontsteking\b"],
"negative": [r"\b(?:no|without|absence of)\b.{0,20}\bsynovitis\b", r"\bno hay (?:signos de )?sinovitis\b",
r"\bsin sinovitis\b", r"\bpas de synovite\b", r"\bgeen synovitis\b"]},
"Baker's": {
"positive": [r"\bbaker['’]?s? cyst\b", r"\bpopliteal cyst\b", r"\bquiste de baker\b", r"\bkyste poplité\b", r"\bcyste poplité\b", r"\bbakerzyste\b"],
"negative": [r"\b(?:no|without|absence of)\b.{0,25}\b(?:baker['’]?s? cyst|popliteal cyst)\b",
r"\bno hay (?:quiste de baker|quistes poplíteos?)\b", r"\bpas de kyste poplité\b"]},
"Contusion": {
"positive": [r"\bbone contusion\b", r"\bbone bruise\b", r"\bcontusión ósea\b", r"\bcontusion osseuse\b", r"\bknochenprellung\b"],
"negative": [r"\b(?:no|without|absence of)\b.{0,25}\b(?:bone contusion|bone bruise)\b",
r"\bno hay contusión ósea\b", r"\bpas de contusion osseuse\b", r"\bkeine knochenprellung\b"]},
"Fracture": {
"positive": [r"\bfracture\b", r"\bfractura\b", r"\bfractuur\b", r"\bfraktur\b"],
"negative": [r"\b(?:no|without|absence of|no evidence of)\b.{0,25}\bfracture\b",
r"\b(?:fractures?|fracturas?)\s*:\s*(?:none|no|aucune|ninguna)\b",
r"\bno hay (?:evidencia de )?fractura\b", r"\bpas de fracture\b", r"\bkeine fraktur\b", r"\bgeen fractuur\b"]},
}

UNCERTAINTY = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspected|suspicious for|cannot exclude|"
    r"may represent|might represent|questionable|indeterminate|posible|posiblemente|"
    r"sospecha de|no se puede descartar|suspect|à confirmer|mogelijk|verdacht voor)\b",
    re.IGNORECASE,
)
COMPILED = {
    t: {p: [re.compile(x, re.IGNORECASE) for x in pats] for p, pats in groups.items()}
    for t, groups in RULES.items()
}

def classify_report(report, target):
    text = re.sub(r"\s+", " ", str(report or ""))
    spans = {"positive": [], "negative": []}
    uncertain = False
    for polarity in ("positive", "negative"):
        for pattern in COMPILED[target][polarity]:
            for match in pattern.finditer(text):
                context = text[max(0, match.start()-80):min(len(text), match.end()+80)]
                if UNCERTAINTY.search(context):
                    uncertain = True
                    continue
                cue = match.group(0).strip()
                spans[polarity].append((match.start(), match.end(), cue))

    # A positive keyword can also occur inside an explicit negative phrase
    # (e.g. "no knee effusion"). Suppress only overlapping positive matches;
    # separate positive and negative evidence elsewhere remains a conflict.
    negative_spans = [(a, b) for a, b, _ in spans["negative"]]
    positive = [
        cue for a, b, cue in spans["positive"]
        if not any(a < neg_b and neg_a < b for neg_a, neg_b in negative_spans)
    ]
    negative = [cue for _, _, cue in spans["negative"]]
    positive = list(dict.fromkeys(positive))
    negative = list(dict.fromkeys(negative))

    if positive and negative:
        return np.nan, "conflict", " | ".join(positive), " | ".join(negative)
    if positive:
        return 1.0, "positive", " | ".join(positive), ""
    if negative:
        return 0.0, "negative", "", " | ".join(negative)
    if uncertain:
        return np.nan, "uncertain", "", ""
    return np.nan, "unknown", "", ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--output-dir", default="/kaggle/working/exp007_weak_labels")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"StudyInstanceUID", "Report", *TARGETS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if df["Report"].isna().any():
        raise ValueError("Report contains missing values; inspect source first")

    # Exclude every study with any official label. Do not touch gold labels.
    has_gold = df[TARGETS].notna().any(axis=1)
    weak_source = df.loc[~has_gold, ["StudyInstanceUID", "Report"]]
    print(f"Total studies: {len(df)}")
    print(f"Studies with official labels excluded: {int(has_gold.sum())}")
    print(f"Unlabelled studies to process: {len(weak_source)}")

    wide, audit = [], []
    for study_id, report in weak_source[["StudyInstanceUID", "Report"]].itertuples(index=False, name=None):
        output = {"StudyInstanceUID": study_id}
        for target in TARGETS:
            label, status, positive_cues, negative_cues = classify_report(report, target)
            output[target] = label
            audit.append({
                "StudyInstanceUID": study_id, "target": target, "weak_label": label,
                "status": status, "positive_cues": positive_cues, "negative_cues": negative_cues,
            })
        wide.append(output)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    weak_labels = pd.DataFrame(wide, columns=["StudyInstanceUID", *TARGETS])
    audit_df = pd.DataFrame(audit)
    weak_labels.to_csv(outdir / "weak_labels.csv", index=False)
    audit_df.to_csv(outdir / "weak_label_audit.csv", index=False)
    summary = audit_df.groupby(["target", "status"], dropna=False).size().rename("studies").reset_index()
    summary.to_csv(outdir / "weak_label_summary.csv", index=False)

    coverage = audit_df.groupby("target").agg(
        known=("weak_label", lambda s: int(s.notna().sum())),
        positive=("weak_label", lambda s: int((s == 1).sum())),
        negative=("weak_label", lambda s: int((s == 0).sum())),
        unknown=("status", lambda s: int((s == "unknown").sum())),
        uncertain=("status", lambda s: int((s == "uncertain").sum())),
        conflict=("status", lambda s: int((s == "conflict").sum())),
    )
    print("\nWeak-label coverage by target:")
    print(coverage.to_string())
    print(f"\nSaved outputs to {outdir}")
    print("Audit samples and error rates before using these labels for training.")

if __name__ == "__main__":
    main()
