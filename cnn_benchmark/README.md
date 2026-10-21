# CNN baseline benchmark: facility-level evaluation

## Purpose

This is a traditional supervised CNN baseline for the broad-area-search paper *"Catching Proliferation at the Fence"*. The baseline answers a reviewer's request for a CNN comparison. Its headline evaluation is **facility retrieval versus human-review burden**, not tile accuracy.

## Current status

The infrastructure is complete, and the full-resolution imagery is complete for all three AOIs: Saratov
1,387/1,387, Fordow 777/777, and Natanz 776/776 (from Qiqi's canonical re-upload). The 2,940-tile decoded cache is
built and verified, and `src/train.py --dry-run` passes for both split types. The real benchmark is technically
ready.

**Natanz imagery: RESOLVED (2026-10-09).** The first shared Natanz folder held only 246 of 776 benchmark tiles plus
261 tiles from a grid about 1° further south. Qiqi's re-upload (`NEW_natanz/`, configured under `image_sources`)
contains all 776. The 246 overlapping tiles are byte-identical, and the old folder is no longer used.
The provenance list of the 530 then-missing tiles is in `outputs/natanz_missing_tiles.md`.

Still open (not data):
- **Author confirmation of the facility mapping.** See `outputs/facility_mapping_questions.md`, split into "must
  decide before final training" and "can defer".
- **Method decisions** listed in `outputs/meeting_summary.md`, e.g. the watermark mask and the training schedule.

Meeting status: `outputs/meeting_summary.md`. Saratov full-res sanity check (plumbing/runtime only):
`outputs/fullres_sanity_check_summary.md`. Watermark/seam audit: `outputs/artifact_audit/artifact_audit.md`.

The code already does the following:
- Builds the canonical parent-tile dataset and the provisional facility map from the workbook.
- Reproduces **all** facility-level numbers in Manuscript Tables 2, 3 and 4, plus the counts quoted in the results text: 96 of 97 checks. The one miss is a narrative claim that the labels do not support (see `outputs/manuscript_reproduction.csv`).
- Evaluates **any** score file under a common facility-footprint rule.
- Writes the candidate leakage-safe splits.
- Has a ResNet-50 training and inference pipeline, tested on thumbnails for plumbing only.

## Benchmark design

| | |
|---|---|
| **Training target** | The human-reviewed layered-fencing signature on a z16 parent tile (1.9 km). It is positive if the human label is TP or FN, aggregated from quadrant labels when the VLM zoomed in. It is **never** derived from VLM scores. |
| **Model** | torchvision ResNet-50, ImageNet-pretrained, single-logit head, BCE weighted by n_neg/n_pos, random 90° rotations and flips. Input 2560 → 1280 px by default (configurable). |
| **Evaluation unit** | A known target facility. A facility is surfaced once any z16 parent tile in its footprint has been reviewed. |
| **Primary metric** | Facility recall vs. area reviewed (km² and % of AOI) as tiles are reviewed in descending score order (Saratov: 24 facilities). |
| **Secondary metrics** | Area needed for 50/75/90/100% recall; per-facility first-hit rank and area (the only meaningful metric for Natanz and Fordo, which have one facility each); workload reduction (1 − area/AOI). |
| **Scope** | z16 only: Saratov (1,387 parents), Natanz (776), Fordo (777). z17 is used only for manuscript regression. |

### Two surfacing rules, kept separate

- **Rule A, `manuscript_reasoning_validated`** (`src/manuscript_rule.py`). This is the manuscript's own rule. Units are VLM output images (parents, quadrants, zoom-out blocks). A facility is surfaced if a unit scored ≥ t carries a human label naming it. The reasoning check is already folded into the TP labels: tiles with a rejected rationale were labelled FP and lost the facility name. Rule A is used only to reproduce the paper.
- **Rule B, `footprint_parent_tiles`** (`src/evaluation.py`). This is the common rule for comparing the CNN and the VLM. Units are z16 parent tiles, each counting as 3.61 km² of review. A facility is surfaced when any parent in its footprint is reviewed. It needs no reasoning trace, so it applies equally to CNN and VLM scores. VLM parent scores come in two variants: single pass, and "adaptive max" (the max over the final outputs inside the parent).

### Parent-label aggregation (`src/build_parent_table.py`)

| Parent stage | `human_target` | Flags |
|---|---|---|
| Not followed up (`Original output`) | Its own label is TP or FN | — |
| Replaced by quadrants | **Any quadrant** is TP or FN (quadrants tile the parent exactly) | If the parent's own label disagrees: `label_conflict` → `exclude_from_training`, `needs_author_confirmation` (4 Natanz tiles) |
| Replaced by zoom-out blocks only | The parent's own label | `zoomout_only_followup`, `needs_author_confirmation` (3 tiles). Blocks are offset 1.9 km views and are never aggregated. |

Labels containing "cut off", "former" or "under renovation" are kept but flagged `ambiguous_label`, for sensitivity runs. Tiles excluded from training are still scored and evaluated.

### Facility reconstruction (`src/facility_mapping.py`, PROVISIONAL)

The workbook has no facility IDs.
- **Natanz and Fordo:** all rows whose label names the site.
- **Saratov:** rows naming a target type (silo, warhead storage, transfer point, waste repository), grouped by single-linkage clustering of their centres at 2.5 km, per zoom. z17 clusters are matched to z16 IDs.

This reproduces Tables 2 and 3 exactly, but it remains a reconstruction:
- **21 z16 silo locations vs 20 in the paper.** The unmatched one (`SAR-SILO-01`, `r2_c17`) is kept with status `candidate_extra_silo_not_in_manuscript_count` and is excluded from the default denominator. Use `--include-candidate-facilities` to include it.
- **The waste repository** is kept as a non-target, as the manuscript specifies.

### Tie policy (`src/evaluation.py`)

VLM scores are integers with large ties.
- **Default `group`:** all units with the same score are reviewed together. This is equivalent to thresholding at score ≥ t. First-hit metrics are reported at the end of the group, which is the conservative choice.
- **`optimistic` / `pessimistic`:** bounds that review footprint units first or last within each tie.
- **Expected rank under random tie-breaking:** always reported, (g+1)/(k+1) within the group.

Continuous CNN scores rarely tie, so the policies coincide for them.

### Candidate splits (`src/splits.py`, NOT FINAL)

- **`saratov_spatial_cv`:** 5 k-means blocks on the tile grid. Every facility footprint is kept in one block. In each fold, test = block k and val = block k+1; tiles within 1 tile (Chebyshev distance) of a test or val tile become buffer and are excluded from training.
- **`loao`:** test on one whole AOI and train on the other two. Each training AOI holds out one validation block plus buffer, chosen to lose the fewest positives.

Automatic checks cover: a facility split across roles, train tiles next to held-out tiles, non-z16 IDs (z17 or quadrant), duplicate assignments, duplicate images, and each Saratov tile appearing in test exactly once.

## Important caveats

- **The human labels are not independent of the VLM.** They were made while reviewing VLM outputs and reasoning, and are recorded as TP/FP/TN/FN relative to VLM score ≥ 5. VLM-flagged tiles were inspected more closely (at quadrant level); tiles with real fencing but wrong VLM reasoning may be labelled FP.
- **Facility-level evaluation is the headline.** The facility list comes from outside sources and is not defined by VLM outputs, but its tile footprint is reconstructed from those labels.
- **The CNN is supervised; the VLM is zero-shot.** Always report the labels used next to any comparison, and never present within-Saratov spatial CV as cross-region generalisation.
- **z17 is excluded** from the CNN experiment (same ground as z16; never mix them).
- **The facility map is provisional** (`needs_author_confirmation`). Natanz and Fordo have one facility each, so report first-hit area rather than recall percentages.
- **The 320 px thumbnails are for `PIPELINE_SMOKE_TEST_ONLY`.** Those runs are written to `outputs/_PIPELINE_SMOKE_TEST_ONLY/` (git-ignored), and the evaluator routes any score file carrying that label there.
- **Source data are read-only inputs** at the repository root and are never modified.

## How to run

```bash
cd cnn_benchmark
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate

# 1. canonical dataset + provisional facility map
python src/facility_mapping.py        # facilities_provisional.csv, facility_mapping_questions.md, memberships
python src/build_parent_table.py      # outputs/parent_tiles.csv (+ summary / inconsistencies)

# 2. reproduce manuscript facility metrics + VLM baselines and plot
python src/reproduce_manuscript.py    # manuscript_reproduction.csv, vlm_*.csv, vlm_facility_recall_vs_review_area.png

# 3. candidate splits (+ leakage checks)
python src/splits.py                  # outputs/splits/{saratov_spatial_cv,loao}.csv

# 4. tests
python -m pytest -q

# 5. verify the image cache, then train (blocked until every benchmark parent resolves)
python src/image_index.py
python src/train.py --split saratov_spatial_cv --fold 0 --decoded-cache --dry-run   # startup check only
python src/decoded_cache.py build      # once; resumable
for f in 0 1 2 3 4; do python src/train.py --split saratov_spatial_cv --fold $f --decoded-cache; done
for f in 0 1 2;     do python src/train.py --split loao --fold $f --decoded-cache; done

# 6. assemble out-of-fold scores (each tile scored once) or score tiles with one checkpoint
python src/predict.py assemble --split saratov_spatial_cv --out outputs/cnn_scores_saratov_cv.csv
python src/predict.py assemble --split loao --out outputs/cnn_scores_loao.csv
python src/predict.py score --checkpoint outputs/runs/loao_fold2/best.pt --regions Saratov --out scores.csv

# 7. evaluate ANY score file (columns: region, tile_id, <score>) under the footprint rule
python src/evaluate_scores.py --scores outputs/cnn_scores_saratov_cv.csv --score-col cnn_score --regions Saratov --name cnn_saratov_cv
python src/evaluate_scores.py --scores outputs/vlm_parent_scores.csv --score-col vlm_score_single_pass --name vlm_single_pass
```

## Full-resolution image cache

**Pointing the benchmark at the cache.** `image_root` in `configs/default.yaml` defaults to `data`, which is
relative to the repository root. Override it with `$CNN_BENCH_IMAGE_ROOT=/path/to/cache` or with
`--image-root` on `train.py`, `predict.py score` or `image_index.py`. No machine-specific path is stored.

**Expected structure.** The cache can be used as Qiqi's Google-Drive download arrives; nothing needs renaming
or copying:

```
<image_root>/
  saratov_40km_tiles_z16_mosaic/tiles/region_r001_c021_51.878829_45.972539_tile1.90km.png
  saratov_40km_tiles_z16_mosaic 2/tiles/...      # extracted download parts ("<scan_dir> N")
  fordow_30km_tiles_auto_mosaic/tiles/...
  natanz_30km_tiles_auto_mosaic/tiles/...         # old mixed Natanz folder: NOT used (see image_sources)
  zips/saratov_40km_tiles_z16_mosaic-...-007.zip # zip parts; members are read directly if not extracted
<repo>/NEW_natanz/natanz_30km_tiles_auto_mosaic_reupload-...-00N.zip   # canonical Natanz (git-ignored)
```

**Canonical per-AOI sources.** `image_sources` in `configs/default.yaml` overrides `image_root` for one AOI:

```yaml
image_sources:
  Natanz: {root: NEW_natanz, scan_dir: natanz_30km_tiles_auto_mosaic_reupload, exclusive: true}
```

- `root` is relative to the repo root, or absolute.
- `scan_dir` is the folder name inside the source, an alias for the benchmark scan name.
- `exclusive: true` ignores that AOI's copies under `image_root`.
- Matching is still by exact filename. `outputs/image_index.csv` records each tile's source.

Each parent tile is matched on (scan folder, **exact original filename**) from the VLM CSV `image_path`, so row,
column and centre coordinates must all agree. This matters because the shared folders also contain tiles from
other grids, which a row/column match alone would accept. Extracted files take precedence over zip members.

**Verified geometry.** Every resolved tile is a 2560×2560 RGB PNG covering 1.9 km, i.e. **0.742 m/px** in all
three AOIs. Fordow and Natanz tiles are resampled 4×4 mosaics of zoom-17 imagery; Saratov tiles come from zoom-16
imagery. The watermark grids and sharpness therefore differ by AOI; see `outputs/fullres_qa_notes.md`.

**Recommended input resolution: 1280 px** (exact 2× box reduction, about 1.5 m/px). Double fences remain
separable at 1280 px but merge at 640 px; native resolution costs about 4× more compute.
Measured on an Apple M5 Max (MPS), ResNet-50 at 1280 px:

| | per image |
|---|---|
| Training | about 0.31 s (batch 4 uses 13.6 GB; batch 8 uses 26.5 GB) |
| Inference | about 0.08 s |
| PNG decode | about 0.11 s per worker |

**Pre-decoded 1280 px cache (recommended for training).** `python src/decoded_cache.py build` decodes every
resolved tile once with the same exact 2× box reduction the loader uses. It writes
`cache/decoded_1280px/<region>__<tile_id>.npy` (uint8, git-ignored) plus `cache_manifest.csv` (AOI, tile ID, source
filename/locator, source and cache SHA-256, shape, dtype), then verifies everything. Unresolved tiles are skipped,
never substituted. Train with `--decoded-cache`; cached tensors are bit-identical to the PNG path.

| Cache stat | Value |
|---|---|
| Build + verify | about 50 s for 2,410 tiles, plus 20 s for the 530 added Natanz tiles (2026-10-09) |
| Size | 14.45 GB for all 2,940 tiles (Saratov 1,387, Fordow 777, Natanz 776) |
| Throughput on M5 Max (cache, 0 workers) | 0.33 s/img end-to-end, GPU-bound (vs 0.535 s/img from PNGs with 6 workers) |

`python src/throughput.py --source png|cache` reproduces the timing. `decoded_cache.py build` is incremental. It
adds new tiles. When a tile's source moves, it re-points the entry if the source SHA-256 is unchanged (this is
how the 246 old Natanz entries were re-pointed to the canonical re-upload) and rebuilds the tile otherwise.

**Startup validation without training:**
`python src/train.py --split loao --fold 0 --decoded-cache --dry-run` checks the guards, image and cache
resolution, sample counts, the dataloaders (one batch each) and a forward pass, then exits without writing outputs.

**Integrity check before training:**

```bash
python src/image_index.py          # resolve every parent + header/dimension check (~5 s)
python src/image_index.py --deep   # + full decode of every image (a few minutes)
```

These write `outputs/image_integrity_report.md`, plus `outputs/image_index.csv` (local, git-ignored). The
command exits non-zero if any parent tile is missing, ambiguous, unreadable, or not 2560×2560.

Smoke test (plumbing only, no pretrained download):
`python src/train.py --split saratov_spatial_cv --fold 0 --smoke-test-thumbnails --epochs 1 --max-batches 2 --input-size 224 --no-pretrained --num-workers 0`

## Layout

```
configs/default.yaml        paths, geometry, AOI areas, split + training defaults
src/common.py               read-only loaders for the workbook / VLM CSVs
src/image_index.py          full-resolution cache resolver (files + zip members) and integrity report
src/artifact_audit.py       non-destructive watermark / seam audit by AOI (outputs/artifact_audit/)
src/decoded_cache.py        build / verify the pre-decoded 1280 px cache (cache/, git-ignored)
src/throughput.py           loader and end-to-end training throughput, PNG vs cache
src/facility_mapping.py     provisional facility reconstruction + questions file
src/build_parent_table.py   canonical z16 parent table (human target, flags, provenance)
src/evaluation.py           model-agnostic facility-level review evaluator (+ footprint rule)
src/manuscript_rule.py      the manuscript's rule (rule A) on VLM output images
src/reproduce_manuscript.py manuscript regression + VLM baseline outputs/plot
src/splits.py               candidate spatial-CV / LOAO manifests + leakage checks
src/dataset.py, model.py, train.py, predict.py, evaluate_scores.py   CNN pipeline
tests/                      regression + invariant tests
outputs/                    generated artifacts (runs/ and smoke tests are git-ignored)
```
