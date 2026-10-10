# DICOM Spatial Ordering Audit — 2026-10-10

## Objective
Check whether sorting DICOM slices by `InstanceNumber`, as used in the initial MRI experiments, agrees with physical slice position. Spatial position was computed as the dot product of `ImagePositionPatient` with the normalized slice normal derived from `ImageOrientationPatient`.

## Method
- Used the notebook's existing series-selection metadata and labelled study IDs.
- Audited 30 sagittal series and 30 coronal series from labelled studies separately.
- For each readable series, checked completeness of `InstanceNumber` and spatial metadata, consistency of orientation, and monotonicity of spatial position after sorting by `InstanceNumber`.
- Compared adjacent spatial-coordinate direction and reported direction concordance.

## Results

| Plane | Series audited | Ordering consistent | Instance number complete | Spatial coordinates complete | Orientation consistent | Direction concordance |
|---|---:|---:|---:|---:|---:|---:|
| Sagittal | 30 | 30/30 | 30/30 | 30/30 | 30/30 | 1.0 for every series |
| Coronal | 30 | 30/30 | 30/30 | 30/30 | 30/30 | 1.0 for every series |

All 60 audited series were marked `ORDERING_CONSISTENT`; no `CHECK` rows were reported. Coronal series in the audit had 16–42 readable slices per series.

## Interpretation
There was no detected mismatch between `InstanceNumber` order and physical slice-position order in this 60-series sample. This reduces concern that the current baseline's weak scores are caused by slice ordering errors in these sampled series.

## Limitations
- This was a sample of 60 series, not an exhaustive audit of the full dataset.
- The checks do not establish that the selected slices contain sufficient diagnostic information, that series selection is optimal, or that the labels are noise-free.
- The audit does not validate other planes or series-selection edge cases outside the sampled series.

## Follow-up
Continue treating the existing slice ordering as provisionally acceptable, while keeping the audit scope explicit. Prioritize validation reliability, limited labelled data, label/report inconsistencies, and model/representation choices when investigating baseline performance. Do not claim that slice ordering is universally verified.
