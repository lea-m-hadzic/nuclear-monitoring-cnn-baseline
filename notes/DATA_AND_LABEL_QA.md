# Data and label QA (CNN baseline)

One place for every data and label check done on the CNN baseline. The detailed records are linked in each section.

**Bottom line:**
- No check changed a label, manifest or frozen result.
- Positive labels are confirmed.
- The targeted negative review is pending with Yanliang.
- The extraction and preprocessing audit found one disclosed Saratov extraction-mode difference, and its sensitivity showed no material effect.

## 1. Positive labels (external training positives): CONFIRMED

- **Coverage:** all 287 Yanliang-provided positives were reviewed visually: a 475 m crop at the point, and the full 1.9 km tile for the 54 doubtful ones.
- **Flagged:** 14 positives.
  - 4 had the point likely off-site: Kashiwazaki-Kariwa, Shimane (val), SAFARI-1, Heysham.
  - 10 had no secure perimeter visible at this resolution.
- **2026-10-21: Yanliang reviewed all 14 and confirmed them valid.** No coordinate, label, split or weight was changed, and the frozen manifests and results stand. The 10 "no visible perimeter" cases are confirmed identities that are harder examples of the target, not label noise.
- **Out-of-sample check:** 3 of 56 val positives average below 0.5 over the 9 frozen runs (Plant 272, RWTDF, Maišiagala). All three were graded plausible; they are small, low-contrast sites.
- **Co-location:** 36 positive pairs overlap (< 1.9 km), all within one split group, so there is no leakage.

Records: `results/qa/positive_review/label_qa_report.md` and `positive_proposed_corrections.csv`. Script: `cnn_benchmark/external_data/label_qa.py`. The private per-positive sheets are local only (see `PRIVATE_DATA_NOTES.md`).

## 2. Negative labels: TARGETED REVIEW PENDING (no label changed)

- **Package:** 41 of the 382 ablation negatives were selected for Yanliang's Google Earth review, using **metadata and QA heuristics only, no model scores**:

  | Rule | Tiles | Meaning |
  |---|---|---|
  | P | 23 | Nearby negatives in the closest 10% to the parent fence point (≤ 3.25 km) |
  | X | 5 | Parent is a multi-point complex (≥ 3 points) and the tile is closer than the median (4.43 km) |
  | B | 12 | Top 5% edge density among nearby negatives (built-up or industrial scenes) |
  | H | 3 | Hard negatives pre-flagged at screening or earlier review |

  Two tiles meet two rules, giving 38 nearby and 3 hard negatives.
- **Status:** sent; review pending. **No negative label is changed** until his answers come back and are approved. If contamination is found, its effect on the frozen results has to be assessed then; it is not assumed to be negligible.

Record: `results/qa/negative_review/README.md`. Script: `cnn_benchmark/external_data/review_package_negatives.py`. The KML and CSV contain coordinates and are local only.

## 3. Benchmark facility map (provisional)

The checks found the map clean for every denominator facility (24 Saratov, 1 Natanz, 1 Fordow): all footprint tiles have imagery and are human-positive. Flagged items for author confirmation:
- the Fordow footprint is probably incomplete;
- 5 Saratov facilities have a z17 label in a neighbouring z16 tile;
- 2 centroids fall outside every parent tile;
- 106 of the 146 human-positive tiles lie outside every footprint. These are non-nuclear layered-fence sites, so tile AUROC measures the fence signature, not nuclear identity.

**Impact on the frozen results:** none until the authors confirm. Any change would mean re-scoring the existing checkpoints, not retraining.

Records: `results/qa/benchmark_facility_checks/` and `cnn_benchmark/outputs/facility_mapping_questions.md` (open questions A1–A6). Script: `label_qa.py benchmark`.

## 4. Extraction and preprocessing audit (2026-10-22)

13 tiles were re-extracted with the intended `--auto_mosaic --output_px 2560`, into a fresh directory with no cache reuse.

| Tiles | Re-extraction result |
|---|---|
| Training (external) | Reproduce **byte-identically** after decode (Hamaoka 95% of pixels equal) |
| Natanz | Identical, including the earlier "tile already exists" tiles |
| Fordow | Identical |
| **Saratov** | **Legacy z16 2×2 mosaics.** Same geometry (1.89–1.92 km, 0.74 m/px), different pixels |

**Preprocessing:** 2560 → 1280 px by an exact 2×2 box reduction, giving an effective model input of about **1.48 m/px**. There is no other resize.

**Follow-up:** the Saratov extraction-mode sensitivity (`ROBUSTNESS.md` §6) showed no material change, and the legacy imagery stays the reported benchmark.

Record: `results/qa/extraction_audit/extraction_audit_note.md`. Script: `cnn_benchmark/external_data/extraction_audit.py`. The per-tile table has latitudes and is local only.

## 5. Imagery artefacts and integrity

- **Watermarks and mosaic seams differ by AOI** and are strongly AOI-identifying (AUC up to 1.0). This motivated the per-tile geometric watermark mask used in all frozen runs. The mask-off robustness runs (`ROBUSTNESS.md` §2) showed no material effect.
  - Record: `results/qa/artifact_audit/`. Script: `cnn_benchmark/src/artifact_audit.py`.
- **Image integrity:** all 2,940 benchmark parent tiles resolve to 2560×2560 RGB PNGs at 1.9 km. Natanz uses Qiqi's complete 776-tile re-upload.
  - Record: `cnn_benchmark/outputs/image_integrity_report.md`. Earlier notes are in `notes/archive/`.
