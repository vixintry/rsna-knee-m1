#!/usr/bin/env python3
"""EXP-012 analysis: does training on report-derived (weak) labels beat training on the 58 gold studies?

    python analyse_exp012.py --features /kaggle/working/exp012_features \
                             --weak-labels /kaggle/working/weak_labels.csv \
                             --out /kaggle/working/exp012_results

weak_labels.csv: one row per study, columns = StudyInstanceUID + the 12 targets, values 0 / 1 / NaN.
NaN means the parser made no call. NaN is NEVER treated as negative. Produce it with the repo's
report parser (the one audited in EXP-007); this script does not parse reports.

Protocol (pre-specified in EXP-012.md; do not change after seeing results without logging it):
  Gate     : re-derive the gold-only CV baseline from the stored features and require it to match the
             recorded notebook result (macro OOF AUC 0.615784, seeds 42/123/2026, C=0.01, balanced)
             within REPRO_TOL. If it fails, STOP: the features differ from production.
  Train    : per target, all weak-labelled studies with a call for that target, gold studies removed.
  Select C : on a weak-label validation split only. Gold is never used to choose anything.
  Test     : the 58 gold studies, fixed holdout, no CV.
  Compare  : paired bootstrap over the 58 studies of  macro AUC(weak-trained) - macro AUC(gold-CV OOF),
             averaged over the three recorded CV seeds.
  Variants : 'production' (primary; production's series fallback ladder) and
             'sagittal_only' (sensitivity; sagittal fluid-sensitive series only). The decision uses
             the primary variant only.
"""
from __future__ import annotations

import argparse
import json
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

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = ("sagittal", "coronal", "axial")
EMB_DIM = 768

# ---- pre-specified constants (EXP-012.md) ----
GOLD_CV_SEEDS = (42, 123, 2026)
BASELINE_C = 0.01
REFERENCE_GOLD_CV_MACRO = 0.615784   # cell 15 output, C=0.01 balanced, 36 target/seed evaluations
REPRO_TOL = 0.005
C_GRID = (0.001, 0.01, 0.1, 1.0)
WEAK_VAL_FRACTION = 0.2
WEAK_VAL_SEED = 0
MIN_WEAK_PER_CLASS = 20              # per target, in the weak training set
N_BOOT = 2000
BOOT_SEED = 0
PRIMARY_MIN_MACRO_AUC = 0.68         # point estimate must reach this ...
# ... AND the 95% bootstrap CI of (weak-trained - gold-CV) must exclude 0 (lower bound > 0).


def make_model(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42))


# ----------------------------------------------------------------------------- features
def load_features(feat_dir):
    """Concatenate shards -> ids, per-plane mean-pooled CLS (N,768), ok flags, fluid flags."""
    files = sorted(Path(feat_dir).glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shard_*.npz in {feat_dir}")
    ids, cm, ok, fl = [], {p: [] for p in PLANES}, {p: [] for p in PLANES}, {p: [] for p in PLANES}
    for f in files:
        z = np.load(f, allow_pickle=False)
        ids.extend(z["ids"].astype(str).tolist())
        for p in PLANES:
            cm[p].append(z[f"{p}_cls"].astype(np.float32).mean(axis=1))   # mean pooling, as production
            ok[p].append(z[f"{p}_ok"].astype(bool))
            fl[p].append(z[f"{p}_fluid"].astype(bool))
    ids = np.array(ids)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate study ids across shards")
    return (ids, {p: np.concatenate(cm[p]) for p in PLANES},
            {p: np.concatenate(ok[p]) for p in PLANES}, {p: np.concatenate(fl[p]) for p in PLANES})


def choose_features(cm, ok, fl, variant):
    """Production's ladder: sagittal_fluid > coronal_fluid > sagittal_any > coronal_any > axial.
    'sagittal_only' keeps only studies with a sagittal fluid-sensitive series.
    Note: production's last rung ('any series') is approximated by axial because only three planes are stored."""
    n = len(ok["sagittal"])
    X = np.zeros((n, EMB_DIM), np.float32)
    rung = np.array(["none"] * n, dtype=object)
    for i in range(n):
        if ok["sagittal"][i] and fl["sagittal"][i]:
            p, r = "sagittal", "sagittal_fluid"
        elif variant == "production":
            if ok["coronal"][i] and fl["coronal"][i]:
                p, r = "coronal", "coronal_fluid"
            elif ok["sagittal"][i]:
                p, r = "sagittal", "sagittal_any"
            elif ok["coronal"][i]:
                p, r = "coronal", "coronal_any"
            elif ok["axial"][i]:
                p, r = "axial", ("axial_fluid" if fl["axial"][i] else "axial_any")
            else:
                continue
        else:
            continue
        X[i] = cm[p][i]
        rung[i] = r
    return X, rung, rung != "none"


# ----------------------------------------------------------------------------- statistics
def gold_cv_oof(X, Y, seeds=GOLD_CV_SEEDS, C=BASELINE_C):
    """Replicates the notebook: per target, StratifiedKFold(min(5, minority)), pooled OOF, C=0.01 balanced.
    Returns oof[seed] -> (n, 12) array (NaN where a target was skipped) and the mean AUC over target/seed."""
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
                m = make_model(C).fit(X[tr], y[tr])
                oof[s][va, j] = m.predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y, oof[s][:, j]))
    return oof, float(np.mean(aucs))


def fit_weak(Xw, Yw, C):
    """Per-target fit on the studies with a weak call for that target. Returns {j: model or None}, counts."""
    models, counts = {}, []
    for j, t in enumerate(TARGETS):
        keep = np.isfinite(Yw[:, j])
        y = Yw[keep, j].astype(int)
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        counts.append({"target": t, "n_labelled": int(keep.sum()), "positives": pos, "negatives": neg})
        if min(neg, pos) < MIN_WEAK_PER_CLASS:
            models[j] = None
            continue
        models[j] = make_model(C).fit(Xw[keep], y)
    return models, counts


def select_c_on_weak(Xw, Yw):
    """Pick C by macro AUC on a held-out slice of the WEAK data (weak labels as truth). Gold is not touched."""
    per_c = {C: [] for C in C_GRID}
    for j, t in enumerate(TARGETS):
        keep = np.where(np.isfinite(Yw[:, j]))[0]
        y = Yw[keep, j].astype(int)
        if min((y == 0).sum(), (y == 1).sum()) < MIN_WEAK_PER_CLASS:
            continue
        tr, va = train_test_split(keep, test_size=WEAK_VAL_FRACTION, stratify=y, random_state=WEAK_VAL_SEED)
        for C in C_GRID:
            m = make_model(C).fit(Xw[tr], Yw[tr, j].astype(int))
            per_c[C].append(roc_auc_score(Yw[va, j].astype(int), m.predict_proba(Xw[va])[:, 1]))
    macro = {C: (float(np.mean(v)) if v else float("nan")) for C, v in per_c.items()}
    best = max(C_GRID, key=lambda C: (macro[C] if np.isfinite(macro[C]) else -1, -C))   # ties -> smaller C
    return best, macro


def per_target_auc(Y, S):
    out = []
    for j in range(len(TARGETS)):
        y = Y[:, j].astype(int)
        s = S[:, j]
        out.append(float(roc_auc_score(y, s)) if (len(np.unique(y)) == 2 and np.isfinite(s).all()) else float("nan"))
    return np.array(out)


def paired_bootstrap(Y, S_weak, oof_by_seed, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample studies. Per resample: macro AUC of the weak-trained scores, macro AUC of the gold-CV OOF
    scores (mean over the CV seeds), and their difference. Targets with a single class in a resample, or
    undefined in either arm, are dropped from BOTH arms for that resample."""
    rng = np.random.default_rng(seed)
    n, T = Y.shape
    seeds = list(oof_by_seed)
    a_all, b_all = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = Y[idx]
        a_t, b_t = [], []
        for j in range(T):
            y = yb[:, j].astype(int)
            if len(np.unique(y)) < 2:
                continue
            sw = S_weak[idx, j]
            if not np.isfinite(sw).all():
                continue
            bs = []
            for s in seeds:
                o = oof_by_seed[s][idx, j]
                if np.isfinite(o).all():
                    bs.append(roc_auc_score(y, o))
            if not bs:
                continue
            a_t.append(roc_auc_score(y, sw))
            b_t.append(np.mean(bs))
        if a_t:
            a_all.append(np.mean(a_t))
            b_all.append(np.mean(b_t))
    a, b = np.array(a_all), np.array(b_all)
    d = a - b
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n_boot_used": int(len(a)), "weak_macro_ci95": q(a), "goldcv_macro_ci95": q(b),
            "delta_mean": float(d.mean()), "delta_ci95": q(d), "p_delta_le_0": float((d <= 0).mean())}


# ----------------------------------------------------------------------------- main experiment
def run_variant(variant, ids, cm, ok, fl, gold, weak, skip_gate):
    X, rung, valid = choose_features(cm, ok, fl, variant)
    idmap = {s: i for i, s in enumerate(ids)}
    res = {"variant": variant}

    # ---- gold set, in the notebook's order (sorted by StudyInstanceUID) ----
    g_ids = [s for s in gold["StudyInstanceUID"].astype(str) if s in idmap and valid[idmap[s]]]
    res["n_gold_with_features"] = len(g_ids)
    res["n_gold_total"] = len(gold)
    gsel = gold.set_index(gold["StudyInstanceUID"].astype(str)).loc[g_ids]
    Xg = X[[idmap[s] for s in g_ids]]
    Yg = gsel[TARGETS].to_numpy(dtype=float).astype(int)

    oof, cv_macro = gold_cv_oof(Xg, Yg)
    res["gold_cv_macro_auc"] = cv_macro
    res["gold_cv_per_target"] = dict(zip(TARGETS, np.round(np.nanmean(
        [per_target_auc(Yg, oof[s]) for s in GOLD_CV_SEEDS], axis=0), 4).tolist()))
    if variant == "production":
        full = (len(g_ids) == len(gold))
        res["repro_gate"] = {"reference": REFERENCE_GOLD_CV_MACRO, "recomputed": cv_macro,
                             "abs_diff": abs(cv_macro - REFERENCE_GOLD_CV_MACRO),
                             "all_gold_present": full,
                             "passed": bool(full and abs(cv_macro - REFERENCE_GOLD_CV_MACRO) <= REPRO_TOL)}
        if not res["repro_gate"]["passed"] and not skip_gate:
            res["status"] = "STOPPED_REPRODUCTION_GATE"
            return res

    # ---- weak training set ----
    w = weak[weak["StudyInstanceUID"].astype(str).map(lambda s: s in idmap and bool(valid[idmap[s]]))]
    w_rows = [idmap[s] for s in w["StudyInstanceUID"].astype(str)]
    Xw = X[w_rows]
    Yw = w[TARGETS].to_numpy(dtype=float)
    res["n_weak_train_studies"] = int(len(w))
    res["rung_counts_weak"] = pd.Series(rung[w_rows]).value_counts().to_dict()
    res["rung_counts_gold"] = pd.Series(rung[[idmap[s] for s in g_ids]]).value_counts().to_dict()

    best_c, weak_val = select_c_on_weak(Xw, Yw)
    res["selected_C"] = best_c
    res["c_at_grid_edge"] = bool(best_c in (min(C_GRID), max(C_GRID)))
    res["weak_val_macro_auc_by_C"] = {str(k): v for k, v in weak_val.items()}

    # ---- test on gold (descriptive for every C; the decision uses only the C chosen on weak data) ----
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
    res["weak_per_target_gold_auc"] = dict(zip(TARGETS, np.round(auc_t, 4).tolist()))
    res["targets_modelled"] = int(np.isfinite(auc_t).sum())
    res["weak_macro_gold_auc"] = float(np.nanmean(auc_t))
    res["descriptive_gold_auc_by_C"] = {str(C): float(np.nanmean(per_target_auc(Yg, gold_scores(C)[0])))
                                        for C in C_GRID}

    # ---- paired bootstrap against gold-CV, on the targets the weak model could fit ----
    res["bootstrap"] = paired_bootstrap(Yg, S_best, oof)

    # ---- prevalence / coverage diagnostics ----
    rows = []
    for j, t in enumerate(TARGETS):
        known = np.isfinite(Yw[:, j])
        rows.append({"target": t, "gold_prevalence": float(Yg[:, j].mean()),
                     "weak_prevalence": float(Yw[known, j].mean()) if known.any() else float("nan"),
                     "weak_known_fraction": float(known.mean()) if len(Yw) else float("nan"),
                     "weak_n_known": int(known.sum())})
    res["prevalence"] = rows

    # ---- pre-specified decision rule (primary variant only is binding) ----
    b = res["bootstrap"]
    macro_ok = res["weak_macro_gold_auc"] >= PRIMARY_MIN_MACRO_AUC
    ci_ok = b["delta_ci95"][0] > 0
    if macro_ok and ci_ok:
        verdict = "KEEP: weak labels clearly help"
    elif res["weak_macro_gold_auc"] > cv_macro and b["delta_mean"] > 0:
        verdict = "INVESTIGATE: point estimate improves but pre-specified bar not met"
    else:
        verdict = "REJECT: no improvement over gold-only training"
    res["rule"] = {"min_macro_auc": PRIMARY_MIN_MACRO_AUC, "macro_met": bool(macro_ok),
                   "delta_ci_excludes_zero": bool(ci_ok), "verdict": verdict}
    res["status"] = "OK"
    return res


def print_report(r):
    print("\n" + "=" * 78)
    print(f"VARIANT: {r['variant']}   status: {r['status']}")
    if "repro_gate" in r:
        g = r["repro_gate"]
        print(f"reproduction gate: reference {g['reference']:.6f} | recomputed {g['recomputed']:.6f} | "
              f"|diff| {g['abs_diff']:.6f} | all 58 gold present: {g['all_gold_present']} | passed: {g['passed']}")
    if r["status"] != "OK":
        print(">>> STOP. Features do not reproduce the recorded baseline; do not interpret anything below. <<<")
        return
    b = r["bootstrap"]
    print(f"gold studies used: {r['n_gold_with_features']}/{r['n_gold_total']} | weak training studies: "
          f"{r['n_weak_train_studies']} | targets modelled: {r['targets_modelled']}/12")
    print(f"rungs gold: {r['rung_counts_gold']} | rungs weak: {r['rung_counts_weak']}")
    print(f"C chosen on weak validation: {r['selected_C']}  (weak-val macro AUC by C: "
          f"{ {k: round(v, 4) for k, v in r['weak_val_macro_auc_by_C'].items()} })")
    print(f"gold-CV baseline macro AUC: {r['gold_cv_macro_auc']:.4f}  "
          f"(bootstrap 95% CI {b['goldcv_macro_ci95'][0]:.3f}-{b['goldcv_macro_ci95'][1]:.3f})")
    print(f"weak-trained macro AUC on gold: {r['weak_macro_gold_auc']:.4f}  "
          f"(bootstrap 95% CI {b['weak_macro_ci95'][0]:.3f}-{b['weak_macro_ci95'][1]:.3f})")
    print(f"paired difference (weak - goldCV): {b['delta_mean']:+.4f}  95% CI "
          f"[{b['delta_ci95'][0]:+.4f}, {b['delta_ci95'][1]:+.4f}]  P(delta<=0)={b['p_delta_le_0']:.3f}")
    if r.get("c_at_grid_edge"):
        print("WARNING: selected C is at the edge of C_GRID, so the optimum may lie outside it. Do not silently "
              "widen the grid: log the change in DECISIONS.md and rerun (Rule 2).")
    print("descriptive gold AUC by C (NOT used for selection):",
          {k: round(v, 4) for k, v in r["descriptive_gold_auc_by_C"].items()})
    print("\nper-target gold AUC   weak-trained | gold-CV")
    for t in TARGETS:
        print(f"  {t:<17} {r['weak_per_target_gold_auc'][t]:>8.3f}     | {r['gold_cv_per_target'][t]:.3f}")
    print("\nprevalence / coverage (gold vs weak)")
    for row in r["prevalence"]:
        print(f"  {row['target']:<17} gold {row['gold_prevalence']:.2f} | weak {row['weak_prevalence']:.2f} | "
              f"parser call rate {row['weak_known_fraction']:.2f} (n={row['weak_n_known']})")
    print(f"\nPRE-SPECIFIED RULE -> {r['rule']['verdict']}")
    print(f"  macro AUC >= {r['rule']['min_macro_auc']}: {r['rule']['macro_met']} | "
          f"CI of delta excludes 0: {r['rule']['delta_ci_excludes_zero']}")


def run(feat_dir, weak_csv, train_csv, out_dir, skip_gate=False):
    ids, cm, ok, fl = load_features(feat_dir)
    labels = pd.read_csv(train_csv)
    gold = labels.dropna(subset=TARGETS).sort_values("StudyInstanceUID").reset_index(drop=True)
    gold_ids = set(gold["StudyInstanceUID"].astype(str))
    weak = pd.read_csv(weak_csv)
    missing = [c for c in ["StudyInstanceUID"] + TARGETS if c not in weak.columns]
    if missing:
        raise ValueError(f"weak labels csv lacks columns {missing}")
    n0 = len(weak)
    weak = weak[~weak["StudyInstanceUID"].astype(str).isin(gold_ids)].reset_index(drop=True)
    print(f"weak labels: {n0} rows, {n0 - len(weak)} gold studies removed -> {len(weak)} usable")
    assert not (set(weak["StudyInstanceUID"].astype(str)) & gold_ids)   # Rule 3 / EXP-007: gold is eval-only

    results = [run_variant(v, ids, cm, ok, fl, gold, weak, skip_gate) for v in ("production", "sagittal_only")]
    for r in results:
        print_report(r)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp012_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out / 'exp012_results.json'}")
    print("The decision is the PRIMARY ('production') variant's rule. 'sagittal_only' is a sensitivity check.")
    return results


# ----------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic end-to-end run with planted signal. Verifies the plumbing, not the science."""
    rng = np.random.default_rng(1)
    n_gold, n_weak = 58, 700
    n = n_gold + n_weak
    ids = np.array([f"S{i:04d}" for i in range(n)])
    Z = rng.normal(size=(n, len(TARGETS)))                 # latent severity per target
    dirs = rng.normal(size=(len(TARGETS), EMB_DIM))
    feats = (Z @ dirs) * 0.12 + rng.normal(size=(n, EMB_DIM))
    truth = (Z > np.quantile(Z, 0.6, axis=0)).astype(float)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # shards: gold first (ids S0000..S0057), sagittal fluid for 90%, coronal fluid for the rest
        for k, lo in enumerate(range(0, n, 250)):
            hi = min(n, lo + 250)
            m = hi - lo
            d = {"ids": ids[lo:hi]}
            for p in PLANES:
                d[f"{p}_cls"] = np.zeros((m, 9, EMB_DIM), np.float32)
                d[f"{p}_patch"] = np.zeros((m, 9, EMB_DIM), np.float16)
                d[f"{p}_ok"] = np.zeros(m, bool)
                d[f"{p}_fluid"] = np.zeros(m, bool)
                d[f"{p}_series"] = np.array([""] * m)
                d[f"{p}_nslices"] = np.zeros(m, np.int32)
                d[f"{p}_unreadable"] = np.zeros(m, np.int16)
            for i in range(m):
                plane = "sagittal" if rng.random() < 0.9 else "coronal"
                d[f"{plane}_cls"][i] = feats[lo + i][None, :] + rng.normal(scale=0.2, size=(9, EMB_DIM))
                d[f"{plane}_ok"][i] = True
                d[f"{plane}_fluid"][i] = True
            np.savez(td / f"shard_{k:04d}.npz", **d)
        gold_df = pd.DataFrame(truth[:n_gold], columns=TARGETS)
        gold_df.insert(0, "StudyInstanceUID", ids[:n_gold])
        rest = pd.DataFrame(np.full((n - n_gold, len(TARGETS)), np.nan), columns=TARGETS)
        rest.insert(0, "StudyInstanceUID", ids[n_gold:])
        pd.concat([gold_df, rest]).assign(Report="x").to_csv(td / "train.csv", index=False)
        wk = truth[n_gold:].copy()
        flip = rng.random(wk.shape) < 0.15                  # 15% label noise
        wk[flip] = 1 - wk[flip]
        wk[rng.random(wk.shape) < 0.2] = np.nan             # 20% "no call"
        weak_df = pd.DataFrame(wk, columns=TARGETS)
        weak_df.insert(0, "StudyInstanceUID", ids[n_gold:])
        # include the gold studies in the weak csv on purpose: the script must drop them
        leak = pd.DataFrame(truth[:n_gold], columns=TARGETS)
        leak.insert(0, "StudyInstanceUID", ids[:n_gold])
        pd.concat([weak_df, leak]).to_csv(td / "weak.csv", index=False)

        res = run(td, td / "weak.csv", td / "train.csv", td / "out", skip_gate=True)
        prim = res[0]
        assert prim["status"] == "OK"
        assert prim["n_gold_with_features"] == n_gold
        assert prim["n_weak_train_studies"] == n - n_gold, prim["n_weak_train_studies"]   # gold rows removed
        assert prim["targets_modelled"] == 12
        assert prim["weak_macro_gold_auc"] > prim["gold_cv_macro_auc"], "planted signal: more data should win"
        assert prim["bootstrap"]["delta_ci95"][0] < prim["bootstrap"]["delta_ci95"][1]
        assert res[1]["n_gold_with_features"] < n_gold or res[1]["n_gold_with_features"] == n_gold
        # the real gate must fail on synthetic data (reference number is from the real notebook)
        res_gate = run_variant("production", *load_features(td), pd.read_csv(td / "train.csv").dropna(subset=TARGETS)
                               .sort_values("StudyInstanceUID").reset_index(drop=True),
                               pd.read_csv(td / "weak.csv").iloc[:n - n_gold], skip_gate=False)
        assert res_gate["status"] == "STOPPED_REPRODUCTION_GATE"
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--weak-labels")
    ap.add_argument("--train-csv", default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    ap.add_argument("--out", default="/kaggle/working/exp012_results")
    ap.add_argument("--skip-gate", action="store_true", help="debug only; the gate is part of the protocol")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.features or not a.weak_labels:
        ap.error("--features and --weak-labels are required")
    run(a.features, a.weak_labels, a.train_csv, a.out, skip_gate=a.skip_gate)
