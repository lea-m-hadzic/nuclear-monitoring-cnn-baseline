# cnn_benchmark: code guide

This folder holds the **code** and the pipeline inputs/intermediates (`outputs/`) for the CNN baseline. For what the baseline is, the final results and the QA record, start at the repository `README.md`. Methods and caveats: `../notes/METHODOLOGY.md`.

- `src/`: benchmark infrastructure. Parent-tile table, provisional facility map, facility-level evaluator (`evaluation.py`), manuscript regression, image index and decoded cache, generic training plumbing.
- `external_data/`: external training data, the A/B/C ablation (manifests, training, evaluation, aggregation), robustness tables, QA and review packages, extraction audit and the Saratov auto_mosaic re-scoring. Each script's docstring says what it writes. The result-to-script index is `../results/README.md`.
- `configs/default.yaml`, `external_data/ablation_config.yaml`, `external_data/extraction_config.yaml`: configuration.
- `outputs/`: inputs and intermediates the code reads or regenerates:
  - parent table, facility map, splits, VLM scores and evaluation;
  - ablation manifests and freeze records.
  - Paper-facing results are in `../results/`; figures are in `../figures/`.
- `tests/`: offline tests (`python -m pytest -q` from this folder).

Commands below are the original development instructions. The **frozen** experiments must not be re-run; they are documented for provenance and reproduction.

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
