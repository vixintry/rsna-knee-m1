#!/usr/bin/env python3
"""EXP-019 feature extraction: DINOv2-Base embeddings for all train studies.

Run on Kaggle from a clone of the repo (GPU on; competition + metaresearch/dinov2 base attached):

    python scripts/exp019/extract_features.py --smoke          # 30 gold + 20 non-gold studies, own output dir
    python scripts/exp019/extract_features.py                  # full run (resumable; needs gates file, see below)
    python scripts/exp019/extract_features.py --verify-selection   # re-run the series check on a finished run
    python scripts/exp019/extract_features.py --selftest       # synthetic DICOMs + fake encoder; no GPU, no real data

What is stored per study
  * prod_*   : the series production would choose (verbatim ranked_series, first success within
               the first 4 candidates, exactly like extract_study in rsna-submission.ipynb).
               This is the only slot the EXP-019 analysis uses.
  * <plane>_*: up to three further slots (sagittal/coronal/axial; fluid first, <=3 tries per plane),
               stored for later multi-plane work (EXP-013). Not used by EXP-019. Disable with --no-planes.
  CLS is float32 (the production feature); the mean patch token is float16 and unused here.

Safety properties
  * Startup check: ordered_dicom_paths, dicom_to_image, sample_indices, ranked_series are compared
    (AST source text) with the production notebook, as are N_SLICES, SLICE_RANGE and MAX_SERIES_TRIES.
    Any difference aborts the run. There is no bypass flag.
  * Completed shards are written as part files and atomically renamed; part files never match the
    shard pattern, and stale part files are deleted at startup.
  * The manifest records the code hash, notebook hash and git HEAD, and a run refuses to resume if
    its configuration changed.
  * Full run requires a verified gates file (exp019_gates.json). Smoke runs do not.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

EXPERIMENT_ID = "EXP-019"
DATA = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
MODEL_PATH = "/kaggle/input/models/metaresearch/dinov2/pytorch/base/1"
OUT_DIR = Path("/kaggle/working/exp019_features")
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]                       # scripts/exp019/extract_features.py -> repo root
DEFAULT_NOTEBOOK = REPO_ROOT / "rsna-submission.ipynb"
DEFAULT_GATES = SCRIPT.with_name("exp019_gates.json")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = ("sagittal", "coronal", "axial")
N_SLICES = 9
SLICE_RANGE = (0.0, 1.0)
EMB_DIM = 768
PROD_MAX_SERIES_TRIES = 4      # must equal MAX_SERIES_TRIES in the production notebook (checked at startup)
PLANE_MAX_SERIES_TRIES = 3     # extra plane slots only; not part of production behaviour
SHARD_SIZE = 250
TIME_BUDGET_SECONDS = 8.0 * 3600
SMOKE_SEED, SMOKE_N_GOLD, SMOKE_N_NONGOLD = 0, 30, 20
SMOKE_MAX_NONGOLD_FAIL_RATE = 0.10
SHARD_RE = re.compile(r"^shard_(\d{4})\.npz$")

START = time.time()
LOG: list[dict] = []


def log(event, **kw):
    LOG.append({"event": event, "t": round(time.time() - START, 1), **kw})


# ======================================================================
# VERBATIM from rsna-submission.ipynb (verified against the notebook at every start; do not edit)
# ======================================================================
def ranked_series(series_df, study_id):
    """Ordered list of (series_id, rung). The first entry matches the EXP-004 choice when a fluid sagittal exists."""
    c = series_df[series_df["StudyInstanceUID"].astype(str) == str(study_id)]
    if c.empty:
        return []
    plane = (c["Anatomical_Plane"].astype(str).str.strip().str.lower()
             if "Anatomical_Plane" in c else pd.Series("", index=c.index))
    fluid = ((pd.to_numeric(c["Fluid_Sensitive"], errors="coerce") == 1)
             if "Fluid_Sensitive" in c else pd.Series(False, index=c.index))
    everything = pd.Series(True, index=c.index)
    rungs = [
        ("sagittal_fluid", (plane == "sagittal") & fluid),
        ("coronal_fluid", (plane == "coronal") & fluid),
        ("sagittal_any", plane == "sagittal"),
        ("coronal_any", plane == "coronal"),
        ("axial_fluid", (plane == "axial") & fluid),
        ("any_series", everything),
    ]
    out, seen = [], set()
    for name, mask in rungs:
        for sid in sorted(c.loc[mask, "SeriesInstanceUID"].astype(str)):
            if sid not in seen:
                seen.add(sid)
                out.append((sid, name))
    return out


def ordered_dicom_paths(folder):
    """Sort slices by InstanceNumber; fall back to position along the slice normal, then to file name."""
    paths = sorted(Path(folder).glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM files in {folder}")
    heads = []
    for p in paths:
        try:
            heads.append((p, pydicom.dcmread(p, stop_before_pixels=True)))
        except Exception:
            log("header_unreadable", file=str(p))
    if not heads:
        raise ValueError(f"No readable DICOM headers in {folder}")

    inst = [getattr(ds, "InstanceNumber", None) for _, ds in heads]
    if all(i is not None for i in inst) and len({int(i) for i in inst}) == len(inst):
        order = sorted(range(len(heads)), key=lambda k: (int(inst[k]), heads[k][0].name))
        return [heads[k][0] for k in order], "instance_number"
    try:
        iop = np.array(heads[0][1].ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        pos = [float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float))) for _, ds in heads]
        order = sorted(range(len(heads)), key=lambda k: (pos[k], heads[k][0].name))
        return [heads[k][0] for k in order], "image_position"
    except Exception:
        return sorted(h[0] for h in heads), "filename"


def dicom_to_image(path):
    """Identical preprocessing to EXP-004."""
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Unsupported pixel array shape {pixels.shape}")
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1))
    pixels += float(getattr(ds, "RescaleIntercept", 0))
    lo, hi = np.percentile(pixels, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid intensity range: {path}")
    pixels = np.clip((pixels - lo) / (hi - lo), 0, 1)
    pixels = (pixels * 255).astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255 - pixels
    return Image.fromarray(pixels).convert("RGB")


def sample_indices(n):
    lo, hi = SLICE_RANGE
    return np.rint(np.linspace(lo * (n - 1), hi * (n - 1), N_SLICES)).astype(int)
# ======================================================================
VERBATIM_FUNCS = ("ranked_series", "ordered_dicom_paths", "dicom_to_image", "sample_indices")
VERBATIM_CONSTS = ("N_SLICES", "SLICE_RANGE", "MAX_SERIES_TRIES")


# ----------------------------------------------------------------------------- provenance
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _defs_from_source(src: str, names, const_names):
    funcs, consts = {}, {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return funcs, consts
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.setdefault(node.name, []).append(ast.get_source_segment(src, node))
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in const_names):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return funcs, consts


def notebook_definitions(nb_path):
    """All top-level function sources and selected constants from the notebook's code cells."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    funcs, consts = {}, {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        f, c = _defs_from_source("".join(cell["source"]), None, VERBATIM_CONSTS)
        for k, v in f.items():
            funcs.setdefault(k, []).extend(v)
        consts.update(c)
    return funcs, consts


def verify_against_notebook(nb_path) -> dict:
    """Abort unless the copied functions and constants equal the production notebook's."""
    nb_path = Path(nb_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Production notebook not found at {nb_path}. Run from a clone of the repo or pass "
                                "--notebook. The production-equivalence check cannot be skipped.")
    nb_funcs, nb_consts = notebook_definitions(nb_path)
    my_funcs, _ = _defs_from_source(SCRIPT.read_text(encoding="utf-8"), None, ())
    report, problems = {}, []
    for name in VERBATIM_FUNCS:
        theirs, mine = nb_funcs.get(name, []), my_funcs.get(name, [])
        ok = len(theirs) == 1 and len(mine) == 1 and theirs[0] == mine[0]
        report[name] = bool(ok)
        if not ok:
            problems.append(f"{name}: notebook definitions={len(theirs)}, local={len(mine)}, identical={ok}")
    mine_consts = {"N_SLICES": N_SLICES, "SLICE_RANGE": SLICE_RANGE, "MAX_SERIES_TRIES": PROD_MAX_SERIES_TRIES}
    for k in VERBATIM_CONSTS:
        ok = nb_consts.get(k) == mine_consts[k]
        report[k] = bool(ok)
        if not ok:
            problems.append(f"constant {k}: notebook={nb_consts.get(k)!r}, local={mine_consts[k]!r}")
    if problems:
        raise RuntimeError("Extractor differs from the production notebook:\n  " + "\n  ".join(problems))
    return {"notebook_path": str(nb_path), "notebook_sha256": sha256_file(nb_path), "checked": report,
            "ranked_series_sha256": hashlib.sha256(nb_funcs["ranked_series"][0].encode()).hexdigest()}


def notebook_ranked_series(nb_path):
    """The notebook's OWN ranked_series, executed from its source (independent of the local copy)."""
    funcs, _ = notebook_definitions(nb_path)
    ns = {"pd": pd, "np": np}
    exec(funcs["ranked_series"][0], ns)
    return ns["ranked_series"]


# ----------------------------------------------------------------------------- gates file
GATES_URL_PREFIX = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"


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


# ----------------------------------------------------------------------------- extraction
def load_encoder():
    """Same model and processor as production; additionally returns the mean patch token."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device).eval()
    assert model.config.hidden_size == EMB_DIM, "expected DINOv2-Base (768-d)"
    assert getattr(model.config, "num_register_tokens", 0) == 0, "register tokens would shift the patch slice"

    def encode(images):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            h = model(**inputs).last_hidden_state
        return h[:, 0, :].float().cpu().numpy(), h[:, 1:, :].mean(dim=1).float().cpu().numpy()

    return encode, str(device)


def plane_candidates(series_by_study, study_id, plane):
    """[(series_id, is_fluid)] for one plane: fluid series first, then non-fluid, each sorted by UID."""
    c = series_by_study.get(str(study_id))
    if c is None or c.empty:
        return []
    p = c["Anatomical_Plane"].astype(str).str.strip().str.lower()
    fl = pd.to_numeric(c["Fluid_Sensitive"], errors="coerce") == 1
    in_plane = p == plane
    out = []
    for flag, mask in ((True, in_plane & fl), (False, in_plane & ~fl)):
        for sid in sorted(c.loc[mask, "SeriesInstanceUID"].astype(str)):
            out.append((sid, flag))
    return out


def extract_series(encode, root, study_id, series_id):
    paths, order_mode = ordered_dicom_paths(Path(root) / str(study_id) / series_id)
    idx = sample_indices(len(paths)).tolist()
    images, failed = {}, 0
    for k in sorted(set(idx)):
        try:
            images[k] = dicom_to_image(paths[k])
        except Exception as e:
            failed += 1
            log("slice_failed", study=str(study_id), series=series_id, slice=k, error=repr(e)[:160])
    if not images:
        raise RuntimeError("no readable sampled slices")
    readable = sorted(images)
    batch = [images[min(readable, key=lambda j: abs(j - k))] for k in idx]   # nearest readable neighbour
    cls, patch = encode(batch)
    cls = np.asarray(cls, dtype=np.float32)
    patch = np.asarray(patch, dtype=np.float32)
    if cls.shape != (N_SLICES, EMB_DIM) or not np.isfinite(cls).all() or not np.isfinite(patch).all():
        raise RuntimeError(f"invalid features {cls.shape}")
    return cls, patch, {"n_slices": len(paths), "unreadable": failed, "order_mode": order_mode}


def new_shard(n):
    d = {"ids": []}
    slots = ("prod",) + PLANES
    for s in slots:
        d[f"{s}_cls"] = np.zeros((n, N_SLICES, EMB_DIM), np.float32)
        d[f"{s}_patch"] = np.zeros((n, N_SLICES, EMB_DIM), np.float16)
        d[f"{s}_ok"] = np.zeros(n, bool)
        d[f"{s}_series"] = np.array([""] * n, dtype=object)
        d[f"{s}_nslices"] = np.zeros(n, np.int32)
        d[f"{s}_unreadable"] = np.zeros(n, np.int16)
    d["prod_rung"] = np.array([""] * n, dtype=object)
    d["prod_rank"] = np.full(n, -1, np.int8)
    for p in PLANES:
        d[f"{p}_fluid"] = np.zeros(n, bool)
    return d


def save_shard(path: Path, d):
    out = {"ids": np.array(d["ids"], dtype=str)}
    for k, v in d.items():
        if k == "ids":
            continue
        out[k] = np.array(v, dtype=str) if (isinstance(v, np.ndarray) and v.dtype == object) else v
    part = path.with_name("part_" + path.name[:-4] + ".part")     # never matches shard_NNNN.npz
    with open(part, "wb") as f:                                   # file handle: numpy must not append '.npz'
        np.savez(f, **out)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def put_slot(d, slot, i, cls, patch, info, series_id):
    d[f"{slot}_cls"][i] = cls
    d[f"{slot}_patch"][i] = patch.astype(np.float16)
    d[f"{slot}_ok"][i] = True
    d[f"{slot}_series"][i] = series_id
    d[f"{slot}_nslices"][i] = info["n_slices"]
    d[f"{slot}_unreadable"][i] = info["unreadable"]


def extract_all_slots(encode, root, series_df, series_by_study, sid, d, i, do_planes):
    """Fill the production slot and (optionally) the plane slots for study index i. Returns a record dict."""
    cache = {}

    def get(series_id):
        if series_id not in cache:
            try:
                cache[series_id] = extract_series(encode, root, sid, series_id)
            except Exception as e:
                log("series_failed", study=sid, series=series_id, error=repr(e)[:200])
                cache[series_id] = None
        return cache[series_id]

    rec = {"id": sid, "prod_ok": False, "prod_rung": "", "prod_rank": -1, "planes": {}}
    cands = ranked_series(series_df, sid)[:PROD_MAX_SERIES_TRIES]     # same call and slice as extract_study
    if not cands:
        log("prod_no_candidates", study=sid)
    for rank, (series_id, rung) in enumerate(cands):
        got = get(series_id)
        if got is not None:
            cls, patch, info = got
            put_slot(d, "prod", i, cls, patch, info, series_id)
            d["prod_rung"][i], d["prod_rank"][i] = rung, rank
            rec.update(prod_ok=True, prod_rung=rung, prod_rank=rank)
            break
    else:
        if cands:
            log("prod_study_failed", study=sid, tried=len(cands))
    if do_planes:
        for plane in PLANES:
            state = "none"
            for series_id, is_fluid in plane_candidates(series_by_study, sid, plane)[:PLANE_MAX_SERIES_TRIES]:
                got = get(series_id)
                if got is None:
                    continue
                cls, patch, info = got
                put_slot(d, plane, i, cls, patch, info, series_id)
                d[f"{plane}_fluid"][i] = is_fluid
                state = "fluid" if is_fluid else "nonfluid"
                break
            rec["planes"][plane] = state
    return rec


def build_groups(labels):
    ids = labels["StudyInstanceUID"].astype(str)
    gold = sorted(labels.dropna(subset=TARGETS)["StudyInstanceUID"].astype(str))
    gold_set = set(gold)
    rest = sorted(s for s in ids if s not in gold_set)
    return gold, rest


def read_shards(out_dir):
    """Strict shard reader: only shard_NNNN.npz files, consecutive numbering, every file must load."""
    out_dir = Path(out_dir)
    files = sorted(p for p in out_dir.iterdir() if SHARD_RE.match(p.name))
    nums = [int(SHARD_RE.match(p.name).group(1)) for p in files]
    if nums != list(range(len(nums))):
        raise RuntimeError(f"shard numbering is not consecutive from 0: {nums[:10]}...")
    return files


def verify_selection(out_dir, data, notebook, mode):
    """Compare the series stored in the production slot of every GOLD study with the notebook's own
    ranked_series (executed from the notebook source). Writes series_check.json."""
    out_dir, data = Path(out_dir), Path(data)
    nb_rs = notebook_ranked_series(notebook)
    labels = pd.read_csv(data / "train.csv")
    series_df = pd.read_csv(data / "train_series.csv")
    series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].astype(str)
    gold = set(labels.dropna(subset=TARGETS)["StudyInstanceUID"].astype(str))
    stored, rung_all = {}, Counter()
    for f in read_shards(out_dir):
        z = np.load(f, allow_pickle=False)
        for i, sid in enumerate(z["ids"].astype(str).tolist()):
            ok = bool(z["prod_ok"][i])
            stored[sid] = (ok, str(z["prod_series"][i]), str(z["prod_rung"][i]), int(z["prod_rank"][i]))
            rung_all[str(z["prod_rung"][i]) if ok else "NONE"] += 1
    checked, mismatches, rank_gt0, rung_gold = 0, [], [], Counter()
    for sid in sorted(gold):
        if sid not in stored:
            continue
        checked += 1
        ok, series_id, rung, rank = stored[sid]
        cands = nb_rs(series_df, sid)[:PROD_MAX_SERIES_TRIES]
        top = cands[0] if cands else None
        rung_gold[rung if ok else "NONE"] += 1
        if not ok or top is None or rank != 0 or (series_id, rung) != top:
            mismatches.append({"study": sid, "stored": [series_id, rung, rank, ok], "notebook_top": top})
        if ok and rank > 0:
            rank_gt0.append(sid)
    res = {"experiment": EXPERIMENT_ID, "mode": mode, "n_gold_total": len(gold), "n_gold_checked": checked,
           "n_mismatches": len(mismatches), "mismatches": mismatches[:50], "n_gold_rank_gt0": len(rank_gt0),
           "rung_counts_gold": dict(rung_gold), "rung_counts_all_stored": dict(rung_all),
           "all_checked_match": len(mismatches) == 0 and checked > 0,
           "covers_all_gold": checked == len(gold),
           "notebook_sha256": sha256_file(notebook), "extractor_sha256": sha256_file(SCRIPT),
           "git_head": git_head()}
    res["passed"] = bool(res["all_checked_match"] and res["covers_all_gold"] and mode == "full")
    (out_dir / "series_check.json").write_text(json.dumps(res, indent=2))
    print(f"series check: {checked}/{len(gold)} gold studies checked | mismatches {len(mismatches)} | "
          f"gold with rank>0 {len(rank_gt0)} | passed={res['passed']} (smoke runs can never pass)")
    return res


# ----------------------------------------------------------------------------- smoke report
def smoke_report(records, durations, gold_set, out_dir):
    groups = {"gold": [r for r in records if r["id"] in gold_set],
              "non_gold": [r for r in records if r["id"] not in gold_set]}
    report = {"experiment": EXPERIMENT_ID, "note": "Smoke subsets are random samples (seed "
              f"{SMOKE_SEED}); per-group numbers are small-sample diagnostics, not estimates."}
    for name, recs in groups.items():
        ids = {r["id"] for r in recs}
        ev = Counter()
        for e in LOG:
            sid = e.get("study") or next((i for i in ids if e.get("file") and f"/{i}/" in e["file"]), None)
            if sid in ids:
                ev[e["event"]] += 1
        secs = [durations[r["id"]] for r in recs]
        planes = {p: dict(Counter(r["planes"].get(p, "n/a") for r in recs)) for p in PLANES}
        report[name] = {
            "n_studies": len(recs), "prod_ok": sum(r["prod_ok"] for r in recs),
            "prod_failed": sum(not r["prod_ok"] for r in recs),
            "prod_rung_counts": dict(Counter(r["prod_rung"] or "NONE" for r in recs)),
            "prod_rank_gt0": sum(r["prod_rank"] > 0 for r in recs),
            "plane_availability": planes, "events": dict(ev),
            "seconds_total": round(sum(secs), 1),
            "seconds_per_study_mean": round(float(np.mean(secs)), 3) if secs else None,
            "seconds_per_study_max": round(float(np.max(secs)), 3) if secs else None}
    g, n = report["gold"], report["non_gold"]
    report["operational_checks"] = {
        "gold_all_prod_ok": g["n_studies"] > 0 and g["prod_failed"] == 0,
        "gold_prod_rank_zero": g["prod_rank_gt0"] == 0,
        "non_gold_fail_rate_le_10pct": (n["n_studies"] > 0
                                        and n["prod_failed"] / n["n_studies"] <= SMOKE_MAX_NONGOLD_FAIL_RATE),
    }
    if g["seconds_per_study_mean"] is not None and n["seconds_per_study_mean"] is not None:
        full = 58 * g["seconds_per_study_mean"] + 4349 * n["seconds_per_study_mean"]
        report["ESTIMATE_full_run_hours"] = round(full / 3600, 2)
        report["ESTIMATE_note"] = "extrapolated from the smoke sample; not a measurement of the full run"
    (Path(out_dir) / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== SMOKE REPORT (gold and non-gold shown separately) ===")
    for name in ("gold", "non_gold"):
        r = report[name]
        print(f"[{name}] n={r['n_studies']} prod_ok={r['prod_ok']} failed={r['prod_failed']} "
              f"rank>0={r['prod_rank_gt0']} s/study mean={r['seconds_per_study_mean']} max={r['seconds_per_study_max']}")
        print(f"        rungs={r['prod_rung_counts']} events={r['events']}")
        print(f"        planes={r['plane_availability']}")
    print("operational checks:", report["operational_checks"])
    if "ESTIMATE_full_run_hours" in report:
        print(f"ESTIMATE (not measured) full-run hours: {report['ESTIMATE_full_run_hours']}")
    return report


# ----------------------------------------------------------------------------- main
def main(argv=None, encoder_factory=load_encoder, data=None, out_dir=None, notebook=None, gates=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_N_GOLD} gold + {SMOKE_N_NONGOLD} non-gold studies")
    ap.add_argument("--no-planes", action="store_true", help="skip the extra plane slots")
    ap.add_argument("--verify-selection", action="store_true", help="only (re)run the series check")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--notebook", default=None)
    ap.add_argument("--gates", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    mode = "smoke" if args.smoke else "full"
    data = Path(data or DATA)
    out_dir = Path(out_dir or (OUT_DIR.with_name(OUT_DIR.name + "_smoke") if args.smoke else OUT_DIR))
    notebook = Path(notebook or args.notebook or DEFAULT_NOTEBOOK)
    gates = Path(gates or args.gates or DEFAULT_GATES)

    nb_info = verify_against_notebook(notebook)                      # aborts on any mismatch
    print("production-equivalence check passed:", json.dumps(nb_info["checked"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_selection:
        m = json.loads((out_dir / "manifest.json").read_text())["config"]
        return verify_selection(out_dir, data, notebook, m["mode"])

    if mode == "full":
        ok, problems, _ = load_gates(gates)
        if not ok:
            raise SystemExit("GATES NOT SATISFIED (full run refused):\n  - " + "\n  - ".join(problems))

    labels = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    for col in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"):
        if col not in series.columns:
            raise ValueError(f"train_series.csv lacks column {col!r}: {series.columns.tolist()}")
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series_by_study = {k: g for k, g in series.groupby("StudyInstanceUID")}

    gold, rest = build_groups(labels)
    gold_set = set(gold)
    if mode == "smoke":
        rng = random.Random(SMOKE_SEED)
        order = rng.sample(gold, min(SMOKE_N_GOLD, len(gold))) + rng.sample(rest, min(SMOKE_N_NONGOLD, len(rest)))
    else:
        order = gold + rest
    order_hash = hashlib.sha256("\n".join(order).encode()).hexdigest()
    config = {"experiment": EXPERIMENT_ID, "mode": mode, "n_slices": N_SLICES, "slice_range": list(SLICE_RANGE),
              "model_path": MODEL_PATH, "planes": [] if args.no_planes else list(PLANES),
              "prod_max_series_tries": PROD_MAX_SERIES_TRIES, "plane_max_series_tries": PLANE_MAX_SERIES_TRIES,
              "shard_size": args.shard_size, "order_sha256": order_hash, "n_studies": len(order),
              "n_gold_in_order": sum(s in gold_set for s in order), "smoke_seed": SMOKE_SEED if mode == "smoke" else None,
              "notebook_sha256": nb_info["notebook_sha256"], "extractor_sha256": sha256_file(SCRIPT)}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())["config"]
        if old != config:
            diff = {k: (old.get(k), config.get(k)) for k in set(old) | set(config) if old.get(k) != config.get(k)}
            raise RuntimeError(f"Existing manifest has a different config; refusing to mix runs (Rule 2). "
                               f"Use a new output directory. Differences: {diff}")
    else:
        manifest_path.write_text(json.dumps({"config": config, "git_head_at_start": git_head()}, indent=2))

    for stale in out_dir.glob("*.part"):                              # unfinished writes from a killed run
        stale.unlink()
        log("stale_part_file_removed", file=str(stale))

    encode, device = encoder_factory()
    print(f"mode={mode} device={device} studies={len(order)} (gold {config['n_gold_in_order']}) out={out_dir}")

    n_shards = (len(order) + args.shard_size - 1) // args.shard_size
    records, durations = [], {}
    t_run = time.time()
    for k in range(n_shards):
        shard_path = out_dir / f"shard_{k:04d}.npz"
        if shard_path.exists():
            continue
        if time.time() - START > TIME_BUDGET_SECONDS:
            log("time_budget_exceeded", next_shard=k)
            print(f"time budget reached before shard {k}; rerun the same command to resume")
            break
        ids = order[k * args.shard_size:(k + 1) * args.shard_size]
        d = new_shard(len(ids))
        d["ids"] = list(ids)
        t0 = time.time()
        for i, sid in enumerate(ids):
            ts = time.perf_counter()
            rec = extract_all_slots(encode, data / "train_series", series, series_by_study, sid, d, i,
                                    not args.no_planes)
            durations[sid] = time.perf_counter() - ts
            records.append(rec)
        save_shard(shard_path, d)
        print(f"shard {k + 1}/{n_shards}: {len(ids)} studies in {time.time() - t0:.0f}s")

    summary = {"mode": mode, "device": device, "elapsed_seconds": round(time.time() - t_run, 1),
               "event_counts": dict(Counter(e["event"] for e in LOG)),
               "shards_present": len([p for p in out_dir.iterdir() if SHARD_RE.match(p.name)]),
               "shards_expected": n_shards, "first_events": LOG[:100]}
    (out_dir / f"extract_log_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(summary, indent=2, default=str))

    if mode == "smoke" and records:
        smoke_report(records, durations, gold_set, out_dir)
    if summary["shards_present"] == n_shards:
        verify_selection(out_dir, data, notebook, mode)
    else:
        print("extraction incomplete: rerun to resume; the series check runs when all shards exist")
    print(json.dumps({k: v for k, v in summary.items() if k != "first_events"}, indent=2))
    return summary


# ----------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic DICOMs + a fake encoder. Checks plumbing and equivalence with the notebook's own
    extract_study on cases including a short series, a failing study and a study without sagittal."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    nb = DEFAULT_NOTEBOOK
    if not nb.exists():
        sys.exit(f"selftest needs the production notebook at {nb}")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "train_series").mkdir()

        def write_dcm(path, arr, inst):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.Rows, ds.Columns = arr.shape
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.InstanceNumber = inst
            ds.PixelData = arr.astype(np.uint16).tobytes()
            ds.save_as(path)

        specs = {
            "g0": [("sagittal", 1, 12), ("coronal", 1, 10), ("axial", 0, 8)],
            "g1": [("sagittal", 1, 15), ("sagittal", 1, 15), ("coronal", 0, 10)],
            "n0": [("coronal", 1, 10), ("axial", 1, 9)],
            "n1": [("sagittal", 0, 5)],
            "n2": [("sagittal", 1, 0)],                              # empty folder -> production slot fails
            "n3": [("sagittal", 1, 0), ("sagittal", 1, 9), ("coronal", 1, 9)],   # first candidate fails, rank 1 works
            "n4": [("oblique", 0, 9)],                               # only the 'any_series' rung applies
        }
        srows, lab = [], []
        for sid, ser in specs.items():
            for j, (pl, fl, n) in enumerate(ser):
                suid = f"{sid}_ser{j}"
                d = td / "train_series" / sid / suid
                d.mkdir(parents=True)
                for k in range(n):
                    write_dcm(d / f"{k}.dcm", rng.integers(0, 4000, (32, 32)), k + 1)
                srows.append(dict(StudyInstanceUID=sid, SeriesInstanceUID=suid, Anatomical_Plane=pl, Fluid_Sensitive=fl))
            row = {"StudyInstanceUID": sid, "Report": "x"}
            for t in TARGETS:
                row[t] = int(rng.integers(0, 2)) if sid.startswith("g") else np.nan
            lab.append(row)
        pd.DataFrame(lab).to_csv(td / "train.csv", index=False)
        pd.DataFrame(srows).to_csv(td / "train_series.csv", index=False)

        def fake_factory():
            def encode(images):
                a = np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1, 2)) for im in images])
                cls = np.tile(a[:, None], (1, EMB_DIM)).astype(np.float32)
                return cls, cls * 0.5
            return encode, "fake"

        gates_ok = {"status": "VERIFIED", "checked_by": "selftest", "date_checked": "2026-01-01",
                    "rules_source_url": GATES_URL_PREFIX + "/rules", "metric_source_url": GATES_URL_PREFIX + "/overview",
                    "metric_name": "selftest", "metric_is_macro_roc_auc": True, "report_derived_labels_permitted": True,
                    "report_labels_evidence": "selftest only", "metric_evidence": "selftest only",
                    "weak_label_quality_gate": {"waived": True, "waiver_decision_id": "DEC-999"}}
        (td / "gates.json").write_text(json.dumps(gates_ok))
        out = td / "out"
        # 1. a full run with a bad gates file is refused
        (td / "gates_bad.json").write_text(json.dumps({**gates_ok, "status": "NOT_VERIFIED"}))
        try:
            main(["--shard-size", "3"], fake_factory, td, td / "out_refused", nb, td / "gates_bad.json")
            raise AssertionError("full run with unverified gates was not refused")
        except SystemExit as e:
            assert "GATES NOT SATISFIED" in str(e)
        # 2. a full run with good gates works; stale part file is removed; strict shard reader ignores strays
        out.mkdir()
        (out / "part_shard_0009.part").write_bytes(b"junk")
        (out / "shard_0009.tmp.npz").write_bytes(b"junk")          # the old v1 temp name must never be read
        main(["--shard-size", "3"], fake_factory, td, out, nb, td / "gates.json")
        assert not list(out.glob("*.part")), "stale part file not removed"
        # shard_0009.tmp.npz is junk that does not match the strict pattern: reader must skip it
        files = read_shards(out)
        assert [p.name for p in files] == [f"shard_{k:04d}.npz" for k in range(len(files))]
        chk = json.loads((out / "series_check.json").read_text())
        assert chk["covers_all_gold"] and chk["all_checked_match"] and chk["passed"], chk
        # 3. equivalence with the notebook's own extract_study on every study
        funcs, consts = notebook_definitions(nb)
        ns = {"np": np, "pd": pd, "pydicom": pydicom, "Path": Path, "Image": Image, "N_SLICES": 9,
              "SLICE_RANGE": (0.0, 1.0), "MAX_SERIES_TRIES": 4, "log": lambda *a, **k: None, "time": time}
        for name in ("ranked_series", "ordered_dicom_paths", "dicom_to_image", "sample_indices", "extract_study"):
            exec(funcs[name][0], ns)
        enc_cls_only = lambda images: fake_factory()[0](images)[0]
        series_df = pd.read_csv(td / "train_series.csv")
        stored = {}
        for f in files:
            z = np.load(f, allow_pickle=False)
            for i, sid in enumerate(z["ids"].astype(str).tolist()):
                stored[sid] = (bool(z["prod_ok"][i]), z["prod_cls"][i], str(z["prod_series"][i]), int(z["prod_rank"][i]))
        for sid in specs:
            try:
                feats, info = ns["extract_study"](enc_cls_only, series_df, td / "train_series", sid)
                expected = (True, feats, info["series_id"], info["rung_rank"])
            except Exception:
                expected = (False, None, "", -1)
            ok, cls, series_id, rank = stored[sid]
            assert ok == expected[0], f"{sid}: ok {ok} != notebook {expected[0]}"
            if ok:
                assert np.allclose(cls, expected[1], atol=0, rtol=0), f"{sid}: features differ from notebook"
                assert (series_id, rank) == (expected[2], expected[3]), f"{sid}: series/rank differ"
        assert stored["n3"][3] == 1, "retry rank not reproduced"
        # 4. smoke mode: separate groups, own directory, can never pass the full series gate
        smoke_out = td / "out_smoke"
        main(["--smoke"], fake_factory, td, smoke_out, nb, td / "gates.json")
        rep = json.loads((smoke_out / "smoke_report.json").read_text())
        assert rep["gold"]["n_studies"] == 2 and rep["non_gold"]["n_studies"] == 5, rep
        assert json.loads((smoke_out / "series_check.json").read_text())["passed"] is False
        # 5. config change on resume is refused
        try:
            main(["--shard-size", "4"], fake_factory, td, out, nb, td / "gates.json")
            raise AssertionError("config change was not refused")
        except RuntimeError as e:
            assert "different config" in str(e)
        # 6. a tampered verbatim function must abort the startup check
        bad = json.loads(Path(nb).read_text())
        for cell in bad["cells"]:
            if cell["cell_type"] == "code" and "def sample_indices" in "".join(cell["source"]):
                cell["source"] = ["".join(cell["source"]).replace("np.rint(", "np.floor(")]
        (td / "bad_nb.ipynb").write_text(json.dumps(bad))
        try:
            verify_against_notebook(td / "bad_nb.ipynb")
            raise AssertionError("tampered notebook was not detected")
        except RuntimeError:
            pass
    print("\nEXTRACTOR SELFTEST PASSED (synthetic data only; real encoder and real DICOMs untested)")


if __name__ == "__main__":
    main()
