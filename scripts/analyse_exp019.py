#!/usr/bin/env python3
"""EXP-019 analysis: does a probe trained on report-derived (weak) labels beat gold-only training?

    # 0. before spending GPU time: can the weak labels support all 12 targets at all? (CPU, no features needed)
    python analyse_exp019.py --audit-weak-labels --weak-labels weak_labels.csv

    # 1. after a full extraction + series check:
    python analyse_exp019.py --features /kaggle/working/exp019_features --weak-labels weak_labels.csv \
        [--frozen-folds results/EXP-010/exp010_folds.csv] --out /kaggle/working/exp019_results

weak_labels.csv is the output of scripts/derive_report_weak_labels.py (StudyInstanceUID + 12 targets,
1.0 / 0.0 / NaN). NaN means no usable call (unknown, uncertain or conflict) and is NEVER treated as negative.

Protocol (registered in results/EXP-019/PREREGISTRATION.md; there is NO option to bypass a gate):
  Gates (any failure -> STOPPED, no statistics are computed):
    G1 gates file verified by a human from the official pages (rules + metric + weak-label quality policy)
    G2 features come from a complete full-mode extraction (manifest, strict shard reader, order hash)
    G3 series gate: stored gold series == the notebook's own ranked_series, hashes consistent
    G4 reproduction gate: gold-only CV from the stored features reproduces 0.615784 within 0.005
  Train  : per target, weak-labelled studies with a call for that target, gold removed.
           A target needs >= MIN_WEAK_PER_CLASS positives AND negatives, otherwise it is NOT modelled.
  Select : C on a weak-label validation split only. Gold is never used to choose anything.
  Test   : the 58 gold studies, fixed holdout.
  Compare: paired bootstrap over the 58 studies; macro over ALL 12 targets in every resample.
  Verdict: only if the evaluation is valid (all 12 targets modelled and evaluated, all gates passed,
           metric = macro ROC AUC, at most 1% of bootstrap resamples discarded) is the registered rule applied.
           Otherwise the verdict is WITHHELD and says why. KEEP is impossible in that case.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_ID = "EXP-019"
SCRIPT = Path(__file__).resolve()
DEFAULT_GATES = SCRIPT.with_name("exp019_gates.json")
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
EMB_DIM = 768
SHARD_RE = re.compile(r"^shard_(\d{4})\.npz$")
GATES_URL_PREFIX = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"
SELFTEST = False   # set only by --selftest; marks every output as not-a-verdict

# ---- registered constants (PREREGISTRATION.md). Changing any of them after results exist is a protocol deviation. ----
GOLD_CV_SEEDS = (42, 123, 2026)
BASELINE_C = 0.01
REFERENCE_GOLD_CV_MACRO = 0.615784   # notebook cell 15: C=0.01 balanced, 36 target/seed evaluations
REPRO_TOL = 0.005
C_GRID = (0.001, 0.01, 0.1, 1.0)
WEAK_VAL_FRACTION = 0.2
WEAK_VAL_SEED = 0
MIN_WEAK_PER_CLASS = 20              # per target: positives AND negatives among weak calls (gold removed)
N_BOOT = 2000
BOOT_SEED = 0
MAX_DISCARDED_BOOT_FRACTION = 0.01   # validity condition, not a decision criterion
PRIMARY_MIN_MACRO_AUC = 0.68         # KEEP needs macro AUC (all 12 targets) >= this ...
# ... AND the 95% bootstrap CI of (weak-trained - gold-CV) must have a lower bound > 0.
DEC006_FOLDS_SHA256 = "5694a719f79a45c343ea6348f369b4e9154b3ff33bcc1f37e1f137fb96c161b6"


class GateError(Exception):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42))


# ----------------------------------------------------------------------------- gates file (same text as extract_features.py)
def load_gates(path):
    """Validate the human-completed gates file. Returns (ok, problems, data)."""
    p = Path(path)
    if not p.exists():
        return False, [f"{p} does not exist (copy exp019_gates.template.json, complete it from the OFFICIAL "
                       "competition pages, and set status to VERIFIED)"], {}
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"{p} is not valid JSON: {e}"], {}
    problems = []
    if g.get("status") != "VERIFIED":
        problems.append("status must be exactly 'VERIFIED'")
    for k in ("checked_by", "rules_source_url", "metric_source_url", "metric_name",
              "report_labels_evidence", "metric_evidence"):
        if not isinstance(g.get(k), str) or len(g[k].strip()) < 5:
            problems.append(f"{k} missing or too short")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(g.get("date_checked", ""))):
        problems.append("date_checked must be YYYY-MM-DD")
    for k in ("rules_source_url", "metric_source_url"):
        if not str(g.get(k, "")).startswith(GATES_URL_PREFIX):
            problems.append(f"{k} must point at {GATES_URL_PREFIX}...")
    if g.get("report_derived_labels_permitted") is not True:
        problems.append("report_derived_labels_permitted must be true (training on labels derived from the "
                        "provided train reports must be permitted by the official rules)")
    if not isinstance(g.get("metric_is_macro_roc_auc"), bool):
        problems.append("metric_is_macro_roc_auc must be true or false")
    wa = g.get("weak_label_quality_gate", {})
    if not (isinstance(wa, dict) and (
            (wa.get("manual_audit_done") is True and len(str(wa.get("audit_reference", "")).strip()) >= 5)
            or (wa.get("waived") is True and re.fullmatch(r"DEC-\d{3}", str(wa.get("waiver_decision_id", ""))))
    )):
        problems.append("weak_label_quality_gate needs manual_audit_done=true + audit_reference, or "
                        "waived=true + waiver_decision_id 'DEC-nnn' recorded in DECISIONS.md")
    return (not problems), problems, g


# ----------------------------------------------------------------------------- weak labels
def read_weak_labels(weak_csv, gold_ids):
    weak = pd.read_csv(weak_csv)
    missing = [c for c in ["StudyInstanceUID"] + TARGETS if c not in weak.columns]
    if missing:
        raise ValueError(f"weak labels csv lacks columns {missing}")
    weak["StudyInstanceUID"] = weak["StudyInstanceUID"].astype(str)
    if weak["StudyInstanceUID"].duplicated().any():
        raise ValueError("weak labels csv has duplicate StudyInstanceUID rows")
    vals = weak[TARGETS].to_numpy(dtype=float)
    if not np.isin(vals[np.isfinite(vals)], (0.0, 1.0)).all():
        raise ValueError("weak labels must be 0, 1 or NaN (NaN = no usable call)")
    n0 = len(weak)
    weak = weak[~weak["StudyInstanceUID"].isin(gold_ids)].reset_index(drop=True)
    assert not (set(weak["StudyInstanceUID"]) & set(gold_ids))          # DEC-004: held-out studies never train
    return weak, n0 - len(weak)


def weak_label_counts(weak):
    rows = []
    for t in TARGETS:
        y = weak[t].to_numpy(dtype=float)
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        rows.append({"target": t, "positives": pos, "negatives": neg, "no_call": int(np.isnan(y).sum()),
                     "modelled": bool(min(pos, neg) >= MIN_WEAK_PER_CLASS)})
    return rows


def audit_weak_labels(weak_csv, train_csv, out_path=None):
    """CPU-only pre-flight: can the weak labels support every target under the registered floor?"""
    labels = pd.read_csv(train_csv)
    gold_ids = set(labels.dropna(subset=TARGETS)["StudyInstanceUID"].astype(str))
    weak, n_removed = read_weak_labels(weak_csv, gold_ids)
    rows = weak_label_counts(weak)
    print(f"weak-label audit: {len(weak)} studies ({n_removed} gold rows removed); floor = {MIN_WEAK_PER_CLASS} "
          "positives AND negatives per target")
    for r in rows:
        print(f"  {r['target']:<17} pos {r['positives']:>5} | neg {r['negatives']:>5} | no call {r['no_call']:>5} | "
              f"{'OK' if r['modelled'] else 'BELOW FLOOR -> not modelled'}")
    short = [r["target"] for r in rows if not r["modelled"]]
    result = {"n_weak_studies": len(weak), "floor": MIN_WEAK_PER_CLASS, "targets": rows,
              "targets_below_floor": short, "all_12_can_be_modelled": not short,
              "weak_csv_sha256": sha256_file(weak_csv)}
    if short:
        print(f"\nRESULT: {len(short)} of 12 targets are below the floor: {short}.\nUnder the registered rule the "
              "verdict for EXP-019 will be WITHHELD and KEEP is unreachable. Resolve this BEFORE extraction "
              "(see PREREGISTRATION.md, 'Blockers'). Do not change the floor after seeing outcomes.")
    else:
        print("\nRESULT: all 12 targets meet the floor.")
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


# ----------------------------------------------------------------------------- features
def load_features(feat_dir):
    """Strict loader. Only complete, full-mode extractions are accepted."""
    d = Path(feat_dir)
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise GateError(f"{mpath} missing")
    man = json.loads(mpath.read_text())
    cfg = man["config"]
    if cfg.get("mode") != "full":
        raise GateError(f"features directory is a {cfg.get('mode')!r} run, not a full extraction")
    files = sorted(p for p in d.iterdir() if SHARD_RE.match(p.name))
    nums = [int(SHARD_RE.match(p.name).group(1)) for p in files]
    if nums != list(range(len(nums))):
        raise GateError(f"shard numbering is not consecutive from 0: {nums[:10]}")
    ids, X, ok, rung = [], [], [], []
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            ids.extend(z["ids"].astype(str).tolist())
            X.append(z["prod_cls"].astype(np.float32).mean(axis=1))      # mean pooling, as production
            ok.append(z["prod_ok"].astype(bool))
            rung.append(z["prod_rung"].astype(str))
        except Exception as e:
            raise GateError(f"shard {f.name} unreadable: {e!r}")
    ids = np.array(ids)
    if len(set(ids.tolist())) != len(ids):
        raise GateError("duplicate study ids across shards")
    expected_shards = math.ceil(cfg["n_studies"] / cfg["shard_size"])
    if len(files) != expected_shards or len(ids) != cfg["n_studies"]:
        raise GateError(f"incomplete extraction: {len(files)}/{expected_shards} shards, {len(ids)}/{cfg['n_studies']} studies")
    if hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest() != cfg["order_sha256"]:
        raise GateError("study order/ids do not match the manifest")
    return ids, np.concatenate(X), np.concatenate(ok), np.concatenate(rung), cfg


def check_series_gate(feat_dir, cfg):
    p = Path(feat_dir) / "series_check.json"
    if not p.exists():
        raise GateError("series_check.json missing (run extract_features.py --verify-selection)")
    s = json.loads(p.read_text())
    problems = []
    if s.get("passed") is not True:
        problems.append(f"series check did not pass (checked {s.get('n_gold_checked')}/{s.get('n_gold_total')}, "
                        f"mismatches {s.get('n_mismatches')}, mode {s.get('mode')})")
    for k in ("extractor_sha256", "notebook_sha256"):
        if s.get(k) != cfg.get(k):
            problems.append(f"{k} in series_check.json differs from the manifest")
    if problems:
        raise GateError("; ".join(problems))
    return s


# ----------------------------------------------------------------------------- statistics
def gold_cv_oof(X, Y, seeds=GOLD_CV_SEEDS, C=BASELINE_C):
    """Replicates notebook cell 14/15: per target StratifiedKFold(min(5, minority)), pooled OOF, balanced LR."""
    n = len(X)
    oof = {s: np.full((n, len(TARGETS)), np.nan) for s in seeds}
    aucs = []
    for j in range(len(TARGETS)):
        y = Y[:, j].astype(int)
        minority = int(np.bincount(y, minlength=2).min())
        if len(np.unique(y)) != 2 or minority < 2:
            continue
        for s in seeds:
            cv = StratifiedKFold(n_splits=min(5, minority), shuffle=True, random_state=s)
            for tr, va in cv.split(np.zeros(n), y):
                oof[s][va, j] = make_model(C).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y, oof[s][:, j]))
    return oof, float(np.mean(aucs)), len(aucs)


def frozen_fold_baseline(X, Y, g_ids, folds_csv):
    """DESCRIPTIVE ONLY. Gold-only pooled OOF on the DEC-006 frozen folds (comparable to EXP-011)."""
    sha = sha256_file(folds_csv)
    if sha != DEC006_FOLDS_SHA256:
        return {"status": "HASH_MISMATCH_NOT_REPORTED", "sha256": sha, "expected": DEC006_FOLDS_SHA256}
    fo = pd.read_csv(folds_csv).assign(StudyInstanceUID=lambda d: d["StudyInstanceUID"].astype(str))
    fmap = dict(zip(fo["StudyInstanceUID"], fo["fold"].astype(int)))
    if set(g_ids) != set(fmap):
        return {"status": "ID_MISMATCH_NOT_REPORTED", "sha256": sha}
    folds = np.array([fmap[s] for s in g_ids])
    per = {}
    for j, t in enumerate(TARGETS):
        y = Y[:, j].astype(int)
        if len(np.unique(y)) != 2:
            per[t] = None
            continue
        oof = np.full(len(y), np.nan)
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            if len(np.unique(y[tr])) == 2:
                oof[va] = make_model(BASELINE_C).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        per[t] = float(roc_auc_score(y, oof)) if np.isfinite(oof).all() else None
    vals = [v for v in per.values() if v is not None]
    return {"status": "OK", "sha256": sha, "pooled_oof_macro_auc": float(np.mean(vals)), "n_targets": len(vals),
            "per_target": per, "note": "descriptive; not part of the registered rule"}


def fit_weak(Xw, Yw, C):
    models, counts = {}, []
    for j, t in enumerate(TARGETS):
        keep = np.isfinite(Yw[:, j])
        y = Yw[keep, j].astype(int)
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        counts.append({"target": t, "n_labelled": int(keep.sum()), "positives": pos, "negatives": neg,
                       "modelled": bool(min(neg, pos) >= MIN_WEAK_PER_CLASS)})
        models[j] = make_model(C).fit(Xw[keep], y) if min(neg, pos) >= MIN_WEAK_PER_CLASS else None
    return models, counts


def select_c_on_weak(Xw, Yw):
    """C by macro AUC on a held-out slice of the WEAK data (weak labels as truth). Gold is not touched."""
    per_c = {C: [] for C in C_GRID}
    for j in range(len(TARGETS)):
        keep = np.where(np.isfinite(Yw[:, j]))[0]
        y = Yw[keep, j].astype(int)
        if min((y == 0).sum(), (y == 1).sum()) < MIN_WEAK_PER_CLASS:
            continue
        tr, va = train_test_split(keep, test_size=WEAK_VAL_FRACTION, stratify=y, random_state=WEAK_VAL_SEED)
        for C in C_GRID:
            m = make_model(C).fit(Xw[tr], Yw[tr, j].astype(int))
            per_c[C].append(roc_auc_score(Yw[va, j].astype(int), m.predict_proba(Xw[va])[:, 1]))
    macro = {C: (float(np.mean(v)) if v else float("nan")) for C, v in per_c.items()}
    best = max(C_GRID, key=lambda C: (macro[C] if np.isfinite(macro[C]) else -1, -C))      # ties -> smaller C
    return best, macro


def per_target_auc(Y, S):
    out = []
    for j in range(len(TARGETS)):
        y, s = Y[:, j].astype(int), S[:, j]
        out.append(float(roc_auc_score(y, s)) if (len(np.unique(y)) == 2 and np.isfinite(s).all()) else float("nan"))
    return np.array(out)


def paired_bootstrap(Y, S_weak, oof_by_seed, targets, n_boot=None, seed=None):
    """Resample studies. Per resample the macro is taken over exactly `targets`; a resample in which any of
    them is undefined (single class) is DISCARDED for both arms and counted."""
    if not targets:
        return None
    n_boot = N_BOOT if n_boot is None else n_boot          # read at call time so the selftest can shrink it
    seed = BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    a_all, b_all, discarded = [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a_t, b_t, bad = [], [], False
        for j in targets:
            y = Y[idx, j].astype(int)
            if len(np.unique(y)) < 2:
                bad = True
                break
            a_t.append(roc_auc_score(y, S_weak[idx, j]))
            b_t.append(np.mean([roc_auc_score(y, oof_by_seed[s][idx, j]) for s in oof_by_seed]))
        if bad:
            discarded += 1
            continue
        a_all.append(np.mean(a_t))
        b_all.append(np.mean(b_t))
    if not a_all:
        return {"n_boot_used": 0, "n_boot_discarded": discarded, "discarded_fraction": 1.0}
    a, b = np.array(a_all), np.array(b_all)
    d = a - b
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n_targets": len(targets), "n_boot_used": int(len(a)), "n_boot_discarded": discarded,
            "discarded_fraction": discarded / n_boot, "weak_macro_ci95": q(a), "goldcv_macro_ci95": q(b),
            "delta_mean": float(d.mean()), "delta_ci95": q(d), "p_delta_le_0": float((d <= 0).mean())}


# ----------------------------------------------------------------------------- one variant
def run_variant(variant, ids, X_all, prod_ok, prod_rung, gold, weak, gates, frozen_folds, check_repro):
    valid = prod_ok.copy()
    if variant == "sagittal_only":
        valid &= (prod_rung == "sagittal_fluid")
    idmap = {s: i for i, s in enumerate(ids)}
    res = {"variant": variant}

    g_ids = [s for s in gold["StudyInstanceUID"].astype(str) if s in idmap and valid[idmap[s]]]
    res["n_gold_with_features"], res["n_gold_total"] = len(g_ids), len(gold)
    gsel = gold.set_index(gold["StudyInstanceUID"].astype(str)).loc[g_ids]
    Xg = X_all[[idmap[s] for s in g_ids]]
    Yg = gsel[TARGETS].to_numpy(dtype=float).astype(int)

    oof, cv_macro, n_cv_evals = gold_cv_oof(Xg, Yg)
    res["gold_cv_macro_auc"], res["gold_cv_n_evaluations"] = cv_macro, n_cv_evals
    res["gold_cv_per_target"] = dict(zip(TARGETS, np.round(np.nanmean(
        [per_target_auc(Yg, oof[s]) for s in GOLD_CV_SEEDS], axis=0), 4).tolist()))
    if check_repro:
        full = len(g_ids) == len(gold)
        diff = abs(cv_macro - REFERENCE_GOLD_CV_MACRO)
        res["repro_gate"] = {"reference": REFERENCE_GOLD_CV_MACRO, "recomputed": cv_macro, "abs_diff": diff,
                             "tolerance": REPRO_TOL, "all_gold_present": full, "passed": bool(full and diff <= REPRO_TOL)}
        if not res["repro_gate"]["passed"]:
            res["status"] = "STOPPED_REPRODUCTION_GATE"
            return res

    w_valid = weak["StudyInstanceUID"].map(lambda s: s in idmap and bool(valid[idmap[s]]))
    w = weak[w_valid]
    w_rows = [idmap[s] for s in w["StudyInstanceUID"]]
    Xw, Yw = X_all[w_rows], w[TARGETS].to_numpy(dtype=float)
    res["n_weak_train_studies"] = int(len(w))
    res["n_weak_without_features"] = int((~w_valid).sum())
    res["rung_counts_weak"] = pd.Series(prod_rung[w_rows]).value_counts().to_dict()
    res["rung_counts_gold"] = pd.Series(prod_rung[[idmap[s] for s in g_ids]]).value_counts().to_dict()

    best_c, weak_val = select_c_on_weak(Xw, Yw)
    res["selected_C"], res["weak_val_macro_auc_by_C"] = best_c, {str(k): v for k, v in weak_val.items()}
    res["c_at_grid_edge"] = bool(best_c in (min(C_GRID), max(C_GRID)))

    def gold_scores(C):
        models, counts = fit_weak(Xw, Yw, C)
        S = np.full((len(Xg), len(TARGETS)), np.nan)
        for j, m in models.items():
            if m is not None:
                S[:, j] = m.predict_proba(Xg)[:, 1]
        return S, counts

    S_best, counts = gold_scores(best_c)
    res["weak_training_counts"] = counts
    auc_t = per_target_auc(Yg, S_best)
    modelled = [j for j in range(len(TARGETS)) if counts[j]["modelled"] and np.isfinite(auc_t[j])]
    res["targets_modelled"] = len(modelled)
    res["targets_not_modelled"] = [TARGETS[j] for j in range(len(TARGETS)) if j not in modelled]
    res["weak_per_target_gold_auc"] = dict(zip(TARGETS, [None if not np.isfinite(v) else round(float(v), 4) for v in auc_t]))
    res["weak_macro_gold_auc_all12"] = float(np.mean(auc_t)) if len(modelled) == len(TARGETS) else None
    res["descriptive_gold_auc_by_C_modelled_targets"] = {
        str(C): float(np.nanmean(per_target_auc(Yg, gold_scores(C)[0])[modelled])) if modelled else None for C in C_GRID}

    # ---- primary bootstrap: all 12 targets or nothing ----
    res["bootstrap_primary_all12"] = (paired_bootstrap(Yg, S_best, oof, list(range(len(TARGETS))))
                                      if len(modelled) == len(TARGETS) else None)
    # ---- exploratory bootstrap on the modelled subset, in BOTH arms; never used for the verdict ----
    res["EXPLORATORY_modelled_subset"] = {
        "targets": [TARGETS[j] for j in modelled],
        "weak_macro_auc": float(np.mean(auc_t[modelled])) if modelled else None,
        "goldcv_macro_auc_same_targets": float(np.mean([np.nanmean([per_target_auc(Yg, oof[s])[j] for s in GOLD_CV_SEEDS])
                                                        for j in modelled])) if modelled else None,
        "bootstrap": paired_bootstrap(Yg, S_best, oof, modelled) if modelled else None,
        "note": "NOT the registered metric; explanatory only"}

    rows = []
    for j, t in enumerate(TARGETS):
        known = np.isfinite(Yw[:, j])
        rows.append({"target": t, "gold_prevalence": float(Yg[:, j].mean()),
                     "weak_prevalence": float(Yw[known, j].mean()) if known.any() else None,
                     "weak_call_rate": float(known.mean()) if len(Yw) else None, "weak_n_known": int(known.sum())})
    res["prevalence"] = rows
    if frozen_folds and variant == "production":
        res["frozen_fold_descriptive"] = frozen_fold_baseline(Xg, Yg, g_ids, frozen_folds)

    # ---- verdict ----
    if variant != "production":
        res["verdict"] = "SENSITIVITY ONLY: no verdict is issued for this variant"
    else:
        b = res["bootstrap_primary_all12"]
        elig = {
            "all_12_targets_modelled_and_evaluated": len(modelled) == len(TARGETS),
            "all_gold_studies_have_features": len(g_ids) == len(gold),
            "gold_cv_baseline_covers_all_12_targets": n_cv_evals == len(TARGETS) * len(GOLD_CV_SEEDS),
            "bootstrap_discard_fraction_ok": bool(b and b.get("n_boot_used", 0) > 0
                                                  and b["discarded_fraction"] <= MAX_DISCARDED_BOOT_FRACTION),
            "official_metric_is_macro_roc_auc": gates.get("metric_is_macro_roc_auc") is True,
        }
        res["eligibility"] = elig
        if not all(elig.values()):
            failed = [k for k, v in elig.items() if not v]
            res["verdict"] = "WITHHELD (evaluation not valid under the registered rule): " + "; ".join(failed)
            res["valid_for_decision"] = False
        else:
            macro_ok = res["weak_macro_gold_auc_all12"] >= PRIMARY_MIN_MACRO_AUC
            ci_ok = b["delta_ci95"][0] > 0
            if macro_ok and ci_ok:
                v = "KEEP: weak labels clearly help"
            elif res["weak_macro_gold_auc_all12"] > cv_macro and b["delta_mean"] > 0:
                v = "INVESTIGATE: point estimate improves but the pre-specified bar is not met"
            else:
                v = "REJECT: no improvement over gold-only training"
            res["rule"] = {"min_macro_auc": PRIMARY_MIN_MACRO_AUC, "macro_met": bool(macro_ok),
                           "delta_ci_excludes_zero": bool(ci_ok)}
            res["verdict"] = v
            res["valid_for_decision"] = True
        if SELFTEST:
            res["verdict"] = "SELFTEST ONLY, NOT A VERDICT: " + res["verdict"]
            res["valid_for_decision"] = False
    res["status"] = "OK"
    return res


def print_report(r):
    print("\n" + "=" * 78)
    print(f"VARIANT: {r['variant']}   status: {r['status']}")
    if "repro_gate" in r:
        g = r["repro_gate"]
        print(f"reproduction gate: reference {g['reference']:.6f} | recomputed {g['recomputed']:.6f} | "
              f"|diff| {g['abs_diff']:.6f} (tol {g['tolerance']}) | all gold present {g['all_gold_present']} | passed {g['passed']}")
    if r["status"] != "OK":
        print(">>> STOP. No further statistics are computed or interpretable. <<<")
        return
    print(f"gold used {r['n_gold_with_features']}/{r['n_gold_total']} | weak training studies {r['n_weak_train_studies']} "
          f"| targets modelled {r['targets_modelled']}/12 | not modelled: {r['targets_not_modelled']}")
    print(f"rungs gold {r['rung_counts_gold']} | rungs weak {r['rung_counts_weak']}")
    print(f"C chosen on weak validation: {r['selected_C']} {({k: round(v, 4) for k, v in r['weak_val_macro_auc_by_C'].items()})}")
    if r["c_at_grid_edge"]:
        print("WARNING: selected C is at the edge of C_GRID. Do not widen the grid silently; log it in DECISIONS.md and "
              "register a new experiment.")
    print(f"gold-CV baseline macro AUC (12 targets, {r['gold_cv_n_evaluations']} evaluations): {r['gold_cv_macro_auc']:.4f}")
    b = r["bootstrap_primary_all12"]
    if b and b.get("n_boot_used"):
        print(f"weak-trained macro AUC on gold (12 targets): {r['weak_macro_gold_auc_all12']:.4f}  "
              f"CI95 {b['weak_macro_ci95'][0]:.3f}-{b['weak_macro_ci95'][1]:.3f}")
        print(f"paired difference: {b['delta_mean']:+.4f}  CI95 [{b['delta_ci95'][0]:+.4f}, {b['delta_ci95'][1]:+.4f}]  "
              f"discarded resamples {b['discarded_fraction']:.3%}")
    else:
        print("primary 12-target comparison not available (not all targets modelled).")
    ex = r["EXPLORATORY_modelled_subset"]
    if ex["weak_macro_auc"] is not None:
        print(f"[EXPLORATORY, not the registered metric] modelled subset {ex['targets']}: weak {ex['weak_macro_auc']:.4f} "
              f"vs gold-CV {ex['goldcv_macro_auc_same_targets']:.4f}")
    if "frozen_fold_descriptive" in r:
        f = r["frozen_fold_descriptive"]
        print("[descriptive] gold-only pooled OOF on DEC-006 frozen folds:",
              f.get("pooled_oof_macro_auc") if f["status"] == "OK" else f["status"])
    print("\nper-target gold AUC   weak-trained | gold-CV")
    for t in TARGETS:
        w = r["weak_per_target_gold_auc"][t]
        print(f"  {t:<17} {('n/a (not modelled)' if w is None else f'{w:.3f}'):>18} | {r['gold_cv_per_target'][t]:.3f}")
    print("\nprevalence / call rate (gold vs weak)")
    for row in r["prevalence"]:
        wp = "n/a" if row["weak_prevalence"] is None else f"{row['weak_prevalence']:.2f}"
        print(f"  {row['target']:<17} gold {row['gold_prevalence']:.2f} | weak {wp} | call rate {row['weak_call_rate']:.2f} (n={row['weak_n_known']})")
    print(f"\nVERDICT: {r['verdict']}")


# ----------------------------------------------------------------------------- driver
def run(feat_dir, weak_csv, train_csv, out_dir, gates_path=DEFAULT_GATES, frozen_folds=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = {"experiment": EXPERIMENT_ID, "analysis_sha256": sha256_file(SCRIPT), "selftest": SELFTEST,
            "weak_csv_sha256": sha256_file(weak_csv), "train_csv_sha256": sha256_file(train_csv)}

    def stop(name, problems):
        msg = {"status": f"STOPPED_{name}", "problems": problems, "valid_for_decision": False, "provenance": prov}
        (out / "exp019_results.json").write_text(json.dumps(msg, indent=2, default=str))
        print(f"\n>>> STOPPED at gate {name}:")
        for p in problems:
            print("    -", p)
        print(">>> No statistics were computed. There is no bypass.")
        return [msg]

    ok, problems, gates = load_gates(gates_path)
    if not ok:
        return stop("G1_GATES_FILE", problems)
    prov["gates"] = gates
    try:
        ids, X, prod_ok, prod_rung, cfg = load_features(feat_dir)
        prov["manifest_config"] = cfg
        prov["series_check"] = check_series_gate(feat_dir, cfg)
    except GateError as e:
        return stop("G2_G3_FEATURES_OR_SERIES", [str(e)])

    labels = pd.read_csv(train_csv)
    gold = labels.dropna(subset=TARGETS).sort_values("StudyInstanceUID").reset_index(drop=True)
    gold_ids = set(gold["StudyInstanceUID"].astype(str))
    weak, n_removed = read_weak_labels(weak_csv, gold_ids)
    print(f"weak labels: {len(weak)} usable studies ({n_removed} gold rows removed)")
    prov["weak_label_counts"] = weak_label_counts(weak)

    results = []
    prim = run_variant("production", ids, X, prod_ok, prod_rung, gold, weak, gates, frozen_folds, check_repro=True)
    results.append(prim)
    if prim["status"] == "OK":
        results.append(run_variant("sagittal_only", ids, X, prod_ok, prod_rung, gold, weak, gates, None, check_repro=False))
    for r in results:
        print_report(r)
    doc = {"provenance": prov, "results": results, "valid_for_decision": bool(results[0].get("valid_for_decision", False))}
    (out / "exp019_results.json").write_text(json.dumps(doc, indent=2, default=str))
    print(f"\nwrote {out / 'exp019_results.json'}")
    print("The decision is the PRIMARY ('production') variant's verdict only. Record it verbatim; no re-runs with tweaks.")
    return results


# ----------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic end-to-end checks of plumbing and of every gate/validity rule. Says nothing about the science."""
    global SELFTEST, REFERENCE_GOLD_CV_MACRO, N_BOOT
    SELFTEST = True
    real_n_boot, N_BOOT = N_BOOT, 60                        # speed only; the registered value is restored below
    rng = np.random.default_rng(1)
    n_gold, n_weak = 58, 700
    n = n_gold + n_weak
    ids = np.array([f"S{i:04d}" for i in range(n)])
    Z = rng.normal(size=(n, len(TARGETS)))
    dirs = rng.normal(size=(len(TARGETS), EMB_DIM))
    feats = (Z @ dirs) * 0.12 + rng.normal(size=(n, EMB_DIM))
    truth = (Z > np.quantile(Z, 0.6, axis=0)).astype(float)

    # the two gates-file validators must be textually identical in both scripts
    other = SCRIPT.with_name("extract_features.py")
    if other.exists():
        def src(p):
            t = ast.parse(p.read_text(encoding="utf-8"))
            return next(ast.get_source_segment(p.read_text(encoding="utf-8"), x) for x in t.body
                        if isinstance(x, ast.FunctionDef) and x.name == "load_gates")
        assert src(SCRIPT) == src(other), "load_gates differs between extract_features.py and analyse_exp019.py"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shard_size = 250
        feat = td / "feat"
        feat.mkdir()
        for k, lo in enumerate(range(0, n, shard_size)):
            hi = min(n, lo + shard_size)
            m = hi - lo
            d = {"ids": ids[lo:hi], "prod_ok": np.ones(m, bool), "prod_series": np.array(["s"] * m),
                 "prod_rung": np.array(["sagittal_fluid" if rng.random() < 0.9 else "coronal_fluid" for _ in range(m)]),
                 "prod_rank": np.zeros(m, np.int8),
                 "prod_cls": (feats[lo:hi][:, None, :] + rng.normal(scale=0.2, size=(m, 9, EMB_DIM))).astype(np.float32)}
            np.savez(feat / f"shard_{k:04d}.npz", **d)
        np.savez(feat / "shard_0099.tmp.npz", junk=np.zeros(1))        # must be ignored by the strict reader
        (feat / "part_shard_0003.part").write_bytes(b"junk")
        cfg = {"experiment": EXPERIMENT_ID, "mode": "full", "n_studies": n, "shard_size": shard_size,
               "order_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
               "notebook_sha256": "nb" * 32, "extractor_sha256": "ex" * 32}
        (feat / "manifest.json").write_text(json.dumps({"config": cfg}))
        (feat / "series_check.json").write_text(json.dumps({"passed": True, "mode": "full", "n_gold_checked": n_gold,
                                                            "n_gold_total": n_gold, "n_mismatches": 0,
                                                            "notebook_sha256": cfg["notebook_sha256"],
                                                            "extractor_sha256": cfg["extractor_sha256"]}))
        gold_df = pd.DataFrame(truth[:n_gold], columns=TARGETS)
        gold_df.insert(0, "StudyInstanceUID", ids[:n_gold])
        rest = pd.DataFrame(np.full((n - n_gold, len(TARGETS)), np.nan), columns=TARGETS)
        rest.insert(0, "StudyInstanceUID", ids[n_gold:])
        pd.concat([gold_df, rest]).assign(Report="x").to_csv(td / "train.csv", index=False)

        def write_weak(name, mutate=None):
            wk = truth[n_gold:].copy()
            wk[rng.random(wk.shape) < 0.15] = np.nan
            flip = rng.random(wk.shape) < 0.10
            wk[flip & np.isfinite(wk)] = 1 - wk[flip & np.isfinite(wk)]
            if mutate:
                mutate(wk)
            w = pd.DataFrame(wk, columns=TARGETS)
            w.insert(0, "StudyInstanceUID", ids[n_gold:])
            leak = pd.DataFrame(truth[:n_gold], columns=TARGETS)
            leak.insert(0, "StudyInstanceUID", ids[:n_gold])
            pd.concat([w, leak]).to_csv(td / name, index=False)         # gold rows included on purpose
            return td / name

        good_gates = {"status": "VERIFIED", "checked_by": "selftest", "date_checked": "2026-01-01",
                      "rules_source_url": GATES_URL_PREFIX + "/rules", "metric_source_url": GATES_URL_PREFIX + "/overview",
                      "metric_name": "selftest", "metric_is_macro_roc_auc": True, "report_derived_labels_permitted": True,
                      "report_labels_evidence": "selftest only", "metric_evidence": "selftest only",
                      "weak_label_quality_gate": {"waived": True, "waiver_decision_id": "DEC-999"}}
        gates = td / "gates.json"
        gates.write_text(json.dumps(good_gates))
        weak = write_weak("weak_ok.csv")
        # calibrate the reproduction reference to the synthetic data (the real constant belongs to the real notebook)
        _ids, _X, _ok, _rg, _ = load_features(feat)
        gold_s = pd.read_csv(td / "train.csv").dropna(subset=TARGETS).sort_values("StudyInstanceUID")
        Xg = _X[[list(_ids).index(s) for s in gold_s["StudyInstanceUID"]]]
        _, REFERENCE_GOLD_CV_MACRO, _ = gold_cv_oof(Xg, gold_s[TARGETS].to_numpy(float).astype(int))

        # A. happy path: valid evaluation, gold rows dropped, shard strays ignored, verdict marked as selftest
        res = run(feat, weak, td / "train.csv", td / "outA", gates)
        p = res[0]
        assert p["status"] == "OK" and p["targets_modelled"] == 12 and p["valid_for_decision"] is False  # selftest never valid
        assert p["verdict"].startswith("SELFTEST ONLY, NOT A VERDICT: ") and "KEEP" in p["verdict"]
        assert p["eligibility"] and all(p["eligibility"].values())
        assert p["n_weak_train_studies"] == n - n_gold
        # B. a target without negatives -> NOT modelled -> verdict withheld, no KEEP possible
        def kill_oa(wk):
            wk[:, TARGETS.index("Medial OA")] = np.where(np.isfinite(wk[:, TARGETS.index("Medial OA")]), 1.0, np.nan)
        res = run(feat, write_weak("weak_oa.csv", kill_oa), td / "train.csv", td / "outB", gates)
        assert "WITHHELD" in res[0]["verdict"] and "KEEP" not in res[0]["verdict"], res[0]["verdict"]
        assert res[0]["targets_not_modelled"] == ["Medial OA"] and res[0]["bootstrap_primary_all12"] is None
        # C. metric mismatch -> withheld
        (td / "gates_metric.json").write_text(json.dumps({**good_gates, "metric_is_macro_roc_auc": False}))
        res = run(feat, weak, td / "train.csv", td / "outC", td / "gates_metric.json")
        assert "official_metric_is_macro_roc_auc" in res[0]["verdict"] and "WITHHELD" in res[0]["verdict"]
        # D. gates file not verified / waiver without DEC id -> STOPPED, no statistics
        for name, g in (("g_unver", {**good_gates, "status": "NOT_VERIFIED"}),
                        ("g_waiver", {**good_gates, "weak_label_quality_gate": {"waived": True}}),
                        ("g_perm", {**good_gates, "report_derived_labels_permitted": None})):
            (td / f"{name}.json").write_text(json.dumps(g))
            res = run(feat, weak, td / "train.csv", td / f"out_{name}", td / f"{name}.json")
            assert res[0]["status"] == "STOPPED_G1_GATES_FILE", name
        # E. series gate missing/failed -> STOPPED
        sc = json.loads((feat / "series_check.json").read_text())
        (feat / "series_check.json").write_text(json.dumps({**sc, "passed": False}))
        res = run(feat, weak, td / "train.csv", td / "outE", gates)
        assert res[0]["status"] == "STOPPED_G2_G3_FEATURES_OR_SERIES"
        (feat / "series_check.json").write_text(json.dumps(sc))
        # F. smoke-mode manifest rejected
        (feat / "manifest.json").write_text(json.dumps({"config": {**cfg, "mode": "smoke"}}))
        res = run(feat, weak, td / "train.csv", td / "outF", gates)
        assert res[0]["status"] == "STOPPED_G2_G3_FEATURES_OR_SERIES"
        (feat / "manifest.json").write_text(json.dumps({"config": cfg}))
        # G. reproduction gate: a wrong reference must stop the run
        REFERENCE_GOLD_CV_MACRO += 0.05
        res = run(feat, weak, td / "train.csv", td / "outG", gates)
        assert res[0]["status"] == "STOPPED_REPRODUCTION_GATE" and len(res) == 1
        REFERENCE_GOLD_CV_MACRO -= 0.05
        # H. pre-flight audit reports the floor
        audit = audit_weak_labels(write_weak("weak_oa2.csv", kill_oa), td / "train.csv")
        assert audit["targets_below_floor"] == ["Medial OA"] and not audit["all_12_can_be_modelled"]
        # I. the CLI exposes no bypass option
        assert "skip" not in " ".join(build_parser().format_help().lower().split())
    SELFTEST, N_BOOT = False, real_n_boot
    print("\nANALYSIS SELFTEST PASSED (synthetic data only; says nothing about the real experiment)")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--weak-labels")
    ap.add_argument("--train-csv", default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    ap.add_argument("--out", default="/kaggle/working/exp019_results")
    ap.add_argument("--gates", default=str(DEFAULT_GATES))
    ap.add_argument("--frozen-folds", default=None, help="results/EXP-010/exp010_folds.csv (descriptive baseline)")
    ap.add_argument("--audit-weak-labels", action="store_true", help="CPU-only floor check; needs no features")
    ap.add_argument("--selftest", action="store_true")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if a.audit_weak_labels:
        if not a.weak_labels:
            sys.exit("--weak-labels is required")
        r = audit_weak_labels(a.weak_labels, a.train_csv, Path(a.out) / "weak_label_audit.json"
                              if Path(a.out).exists() else None)
        sys.exit(0 if r["all_12_can_be_modelled"] else 1)
    if not a.features or not a.weak_labels:
        build_parser().error("--features and --weak-labels are required")
    run(a.features, a.weak_labels, a.train_csv, a.out, Path(a.gates), a.frozen_folds)
