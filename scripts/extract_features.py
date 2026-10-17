#!/usr/bin/env python3
"""EXP-012 feature extraction: DINOv2-Base embeddings for ALL train studies, per plane.

Run on Kaggle (GPU on, internet off, inputs = competition + metaresearch/dinov2 base):

    python extract_features.py                  # full run, resumable
    python extract_features.py --limit 20       # smoke test on 20 studies (writes to a separate dir)

Design rules (see EXP-012.md):
  * Preprocessing is copied VERBATIM from rsna-submission.ipynb @ a5468a6
    (ordered_dicom_paths, dicom_to_image, sample_indices; 9 slices, SLICE_RANGE (0, 1),
    per-slice 1-99 percentile scaling, HF AutoImageProcessor, CLS token). Do not "improve"
    them here: EXP-012 changes the TRAINING DATA only (Rule 2 / Rule 6).
  * Per study we store up to three planes (sagittal, coronal, axial). Within a plane the series
    is the first fluid-sensitive series by sorted SeriesInstanceUID (same order as production),
    else the first non-fluid series. The analysis script reconstructs production's fallback
    ladder from these, so no re-extraction is needed for plane experiments.
  * Stored per slice: CLS (float32, the production feature) and mean patch token (float16,
    NOT used by the primary analysis; kept so a later CLS+patch experiment needs no re-extraction).
  * Gold (fully labelled) studies are processed first. Output is sharded and resumable.
  * Refuses to resume if the stored config differs from the current one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

DATA = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
MODEL_PATH = "/kaggle/input/models/metaresearch/dinov2/pytorch/base/1"
OUT_DIR = Path("/kaggle/working/exp012_features")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = ("sagittal", "coronal", "axial")
N_SLICES = 9
SLICE_RANGE = (0.0, 1.0)
EMB_DIM = 768
MAX_SERIES_TRIES = 3          # candidate series attempted per plane before giving up on that plane
SHARD_SIZE = 250
TIME_BUDGET_SECONDS = 8.0 * 3600   # stop cleanly before Kaggle's 9 h limit; rerun resumes

START = time.time()
LOG: list[dict] = []


def log(event, **kw):
    LOG.append({"event": event, "t": round(time.time() - START, 1), **kw})


# ======================================================================
# VERBATIM from rsna-submission.ipynb @ a5468a6 (do not edit)
# ======================================================================
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
        cls = h[:, 0, :].float().cpu().numpy()
        patch = h[:, 1:, :].mean(dim=1).float().cpu().numpy()
        return cls, patch

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
    for p in PLANES:
        d[f"{p}_cls"] = np.zeros((n, N_SLICES, EMB_DIM), np.float32)
        d[f"{p}_patch"] = np.zeros((n, N_SLICES, EMB_DIM), np.float16)
        d[f"{p}_ok"] = np.zeros(n, bool)
        d[f"{p}_fluid"] = np.zeros(n, bool)
        d[f"{p}_series"] = np.array([""] * n, dtype=object)
        d[f"{p}_nslices"] = np.zeros(n, np.int32)
        d[f"{p}_unreadable"] = np.zeros(n, np.int16)
    return d


def save_shard(path, d):
    out = {"ids": np.array(d["ids"], dtype=str)}
    for p in PLANES:
        for k in ("cls", "patch", "ok", "fluid", "nslices", "unreadable"):
            out[f"{p}_{k}"] = d[f"{p}_{k}"]
        out[f"{p}_series"] = np.array(d[f"{p}_series"], dtype=str)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)


def build_order(labels):
    ids = labels["StudyInstanceUID"].astype(str)
    gold = sorted(labels.dropna(subset=TARGETS)["StudyInstanceUID"].astype(str))
    gold_set = set(gold)
    rest = sorted(s for s in ids if s not in gold_set)
    return gold + rest, len(gold)


def main(argv=None, encoder_factory=load_encoder, data=None, out_dir=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N studies (smoke test)")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = ap.parse_args(argv)

    data = Path(data or DATA)
    out_dir = Path(out_dir or (OUT_DIR if not args.limit else OUT_DIR.with_name(OUT_DIR.name + "_smoke")))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    for col in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"):
        if col not in series.columns:
            raise ValueError(f"train_series.csv lacks column {col!r}: {series.columns.tolist()}")
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series_by_study = {k: g for k, g in series.groupby("StudyInstanceUID")}

    order, n_gold = build_order(labels)
    if args.limit:
        order = order[: args.limit]
    order_hash = hashlib.sha256("\n".join(order).encode()).hexdigest()
    config = {"n_slices": N_SLICES, "slice_range": list(SLICE_RANGE), "model_path": MODEL_PATH,
              "planes": list(PLANES), "max_series_tries": MAX_SERIES_TRIES,
              "shard_size": args.shard_size, "order_sha256": order_hash, "n_studies": len(order),
              "n_gold_first": min(n_gold, len(order)),
              "preprocessing_source": "rsna-submission.ipynb@a5468a6"}

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("config") != config:
            raise RuntimeError("Existing manifest has a different config; refusing to mix runs "
                               "(Rule 2). Use a new output directory.\n"
                               f"old={old.get('config')}\nnew={config}")
    else:
        manifest_path.write_text(json.dumps({"config": config}, indent=2))

    encode, device = encoder_factory()
    print(f"device={device} | studies={len(order)} (gold first: {config['n_gold_first']}) | out={out_dir}")

    n_shards = (len(order) + args.shard_size - 1) // args.shard_size
    stats = Counter()
    t_run = time.time()
    for k in range(n_shards):
        shard_path = out_dir / f"shard_{k:04d}.npz"
        if shard_path.exists():
            continue
        if time.time() - START > TIME_BUDGET_SECONDS:
            log("time_budget_exceeded", next_shard=k)
            print(f"time budget reached before shard {k}; rerun to resume")
            break
        ids = order[k * args.shard_size:(k + 1) * args.shard_size]
        d = new_shard(len(ids))
        d["ids"] = list(ids)
        t0 = time.time()
        for i, sid in enumerate(ids):
            for plane in PLANES:
                got = False
                for series_id, is_fluid in plane_candidates(series_by_study, sid, plane)[:MAX_SERIES_TRIES]:
                    try:
                        cls, patch, info = extract_series(encode, data / "train_series", sid, series_id)
                    except Exception as e:
                        log("series_failed", study=sid, plane=plane, series=series_id, error=repr(e)[:200])
                        continue
                    d[f"{plane}_cls"][i] = cls
                    d[f"{plane}_patch"][i] = patch.astype(np.float16)
                    d[f"{plane}_ok"][i] = True
                    d[f"{plane}_fluid"][i] = is_fluid
                    d[f"{plane}_series"][i] = series_id
                    d[f"{plane}_nslices"][i] = info["n_slices"]
                    d[f"{plane}_unreadable"][i] = info["unreadable"]
                    stats[f"{plane}_{'fluid' if is_fluid else 'nonfluid'}"] += 1
                    got = True
                    break
                if not got:
                    stats[f"{plane}_none"] += 1
        save_shard(shard_path, d)
        print(f"shard {k + 1}/{n_shards} done: {len(ids)} studies in {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / len(ids):.2f} s/study)")

    summary = {"device": device, "plane_counts_this_run": dict(stats),
               "event_counts": dict(Counter(e["event"] for e in LOG)),
               "elapsed_seconds": round(time.time() - t_run, 1),
               "shards_present": len(list(out_dir.glob("shard_*.npz"))), "shards_expected": n_shards,
               "first_events": LOG[:100]}
    (out_dir / f"extract_log_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "first_events"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
