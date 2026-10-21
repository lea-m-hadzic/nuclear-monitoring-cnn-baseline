# External layered-fence training imagery

The goal is to train the CNN on **external** layered-fence examples, then evaluate it unchanged on Saratov / Natanz / Fordow.
Imagery comes from Qiqi's extraction script, used **unmodified** and pinned to a commit. Nothing here changes
labels, splits, training or evaluation.

**Status (2026-10-13):** the tooling is ready, but nothing has been extracted. `$GOOGLE_MAPS_STATIC_API_KEY` is not
visible to the process that runs these tools, so the one-tile reproduction check has not run yet. We are also waiting for Yanliang's coordinate list.

## Setup

```bash
git clone https://github.com/QiqiNerd/IMINT-Analysis-Longview.git vendor/IMINT-Analysis-Longview   # from repo root
git -C vendor/IMINT-Analysis-Longview checkout a5aec12f2805e14379eb62184a0ebc45440382a1
cnn_benchmark/.venv/bin/pip install -r cnn_benchmark/requirements.txt                          # adds python-dotenv
```

`vendor/` and `external_imagery/` are git-ignored. The key is read only from the environment of the shell that
runs the command. It is passed to the child process through its environment, never on a command line. Manifests are
key-redacted after every run, and every text output is scanned for leftover keys.

## What the vendor script does (`scripts/scan_tiles_overscan_crop.py` @ a5aec12)

| flag | meaning |
|---|---|
| `--mosaic2x2` | legacy mode: fetch 2x2 Static Maps images around each tile centre, stitch, save the full mosaic **without crop** |
| `--mosaic_zoom 16` | zoom for legacy mode (default is also 16) |
| `--size 640 --scale 2` | each Static Maps image is 640 logical px at scale 2, so **1280x1280** returned px |
| `--tile_km 1.9` | spacing of tile centres. In legacy mode it is **not** the saved footprint |
| `--mosaic_overlap_m 0` | sub-images are placed edge-to-edge (spacing = 1280 x m/px) |
| `--maptype satellite` | imagery only, no labels |
| `--skip_raw` | do not save a separate raw image (no effect in legacy mode beyond not creating `raw/`) |
| `--auto_mosaic --output_px 2560` | choose the lowest zoom with m/px <= 1900/2560, stitch NxN, centre-crop exactly 1.9 km, resize to 2560 |

- **Output:** a 2560x2560 RGB PNG in both modes. Legacy is 2x1280 stitched. Auto is a crop resized to 2560.
- **Filenames:** `<output>/tiles/region_r{row:03d}_c{col:03d}_{lat:.6f}_{lon:.6f}_tile1.90km.png`. Row and col index
  the scan grid around `--lat/--lon`, so a single centred tile is `r001_c001`. The name pattern is identical to the benchmark's.
- **Metadata files:** `manifest.csv` has one row per tile (centre, zoom, m/px, mosaic width, grid, crop, sizes, status,
  **url incl. the API key**) and `preview.html`. The wrapper adds `point.json`, `run.log` and `extraction_record.json`.
- **Requests per tile** (from the code; cached tiles cost 0):
  - Legacy: **5**. One probe image at the tile centre is fetched to learn the returned size and then discarded, plus 4 sub-images.
  - Auto: **2 + grid_n²**, because two probes are fetched. That is 11 for a 3x3 grid and 18 for a 4x4 grid (z17 at 35°N).
- Hidden behaviour that matters:
  1. **Legacy footprint depends on latitude:** 2 x 1280 x 0.5 x 156543.03 x cos(lat) / 2^16 m. That is 1.90 km at Saratov (51.5°N),
     2.54 km at Natanz, 2.51 km at Fordow, 2.50 km at 35°N, 1.51 km at 60°N and 3.06 km at the equator. The m/px changes with it.
  2. **The benchmark is not all legacy.** The manifests show:
     - Saratov is legacy z16 (`message=ok_mosaic2x2`, 1.90 km, 0.743 m/px).
     - **Natanz and Fordow are `auto_mosaic`**: z17, 3x3 / 4x4 grid, cropped to 1.9 km, resized to 2560, 0.742 m/px.

     All three AOIs therefore share ~1.9 km and ~0.742 m/px. Only `auto_mosaic_1p9km` reproduces that at arbitrary
     latitudes. Legacy z16 does so only near 51.5°.
  3. The vendor `--dry_run` **still makes every request**, because it only skips saving. The wrapper never passes it.
  4. On an HTTP error, the error text includes the full URL with the key. It is printed and written to the manifest
     `message` column. The wrapper redacts both.
  5. There are no retries. A failed fetch leaves the tile missing (`status=error`). Re-running the wrapper re-fetches only missing
     tiles, because the vendor skips existing PNGs.
  6. Without `$GOOGLE_MAPS_STATIC_API_KEY`, the vendor falls back to a `.env` file (`load_dotenv`). The wrapper refuses to
     run without the environment variable, so the key never comes from a file.

## Presets (`extraction_config.yaml`), deliberately with no default

| preset | flags | footprint | m/px | requests/tile |
|---|---|---|---|---|
| `legacy_mosaic2x2_z16` | Qiqi's stated command | 1.9 x cos(lat)/cos(51.52°) km | varies | 5 |
| `auto_mosaic_1p9km` | as in Natanz/Fordow manifests | 1.9 km | 0.742 | 11 (3x3) or 18 (4x4), depending on latitude |

**Decision needed (Qiqi/Lea):** I recommend `auto_mosaic_1p9km` for external data. It is the only preset that matches the footprint and GSD of all three
evaluation AOIs wherever the external sites are.

## Tools

```bash
cd cnn_benchmark/external_data
python estimate_cost.py --n-coords 50 --radius-km 0 --lat 40                 # no API calls
python extract_external_tiles.py --csv external_facilities.csv --preset auto_mosaic_1p9km --dry-run
python extract_external_tiles.py --csv external_facilities.csv --preset auto_mosaic_1p9km --max-facilities 2
python validate_external_tiles.py --preset auto_mosaic_1p9km                 # no API calls
python validate_reproduction.py                                              # 1 tile, 5 requests (needs key)
python validate_reproduction.py --tile fordow_r008_c012                      # auto path, 18 requests (optional)
```

- `extract_external_tiles.py`:
  - Writes to `external_imagery/<preset>/<facility_id>/<point_id>/` and skips points already complete.
  - `--dry-run` prints the exact commands and the estimate. `--max-facilities N` limits the run.
  - Every point is logged in `external_imagery/extraction_log.jsonl`.
- `validate_external_tiles.py` checks:
  - full PNG decode, 2560x2560, RGB, and the benchmark filename pattern;
  - a manifest row with an ok status;
  - footprint 1.9 km ±3%, with zoom and grid as the preset predicts;
  - tile centre equal to the intake coordinate;
  - unique ids, and no coordinate extracted twice;
  - `point.json` and `extraction_record.json` present, and no key in any text file.

  Self-test on the benchmark: `--benchmark --sample 8 --scan-dirs ../../data/fordow_30km_tiles_auto_mosaic "../../data/saratov_40km_tiles_z16_mosaic 4"`.
  All tiles pass, and it correctly reports the API key that the benchmark manifests contain.
- `validate_reproduction.py` re-extracts benchmark tile Saratov r3_c20 and compares it with the original:
  - size, mode, filename and manifest columns;
  - footprint and alignment (phase correlation, whole tile and centre);
  - pixel similarity, sharpness, and watermark/attribution corners.

  It also writes a side-by-side figure. Everything goes to the git-ignored `_validation/`.

## Intake format: `external_facilities.csv` (template: `external_facilities.template.csv`)

There is one row per point: `point_id, facility_id, facility_name, country, latitude, longitude, coordinate_type, label_source, notes`.
`coordinate_type` is `fence_segment` when the point is centred ON a layered-fence line, otherwise `facility_centroid` or `other`.
`facility_id` groups several fence points of one facility and is the split/leakage unit. The full schema is in `intake.py`.
Real lists stay local: `external_data/*.csv` is git-ignored except the template.

## How much imagery per coordinate

Example: 50 fence points at ~40°N, `auto_mosaic_1p9km` (18 requests/tile). Cost assumes the 10k/month free cap and $2/1000 after it,
as checked 2026-10-13. Run `estimate_cost.py` with the real list.

| strategy | tiles / point | 50 points | requests | cost | PNG |
|---|---|---|---|---|---|
| A. one 1.9 km tile centred on each fence point (`--radius-km 0`) | 1 | 50 | 900 | $0 | 0.7 GB |
| B. small local grid (`--radius-km 2`: centre + 4 edge neighbours; `2.7`: 3x3) | 5 / 9 | 250 / 450 | 4.5k / 8.1k | $0 | 3 / 6 GB |
| C. benchmark-style AOI scan (`--radius-km 30`) | 777 | 38,850 | ~700k | ~$1,060 | ~500 GB |

| strategy | positives | negatives | leakage risk | labelling burden | similarity to benchmark |
|---|---|---|---|---|---|
| **A** | 1 per point (fence at centre by construction) | none: must come from elsewhere | low if split by `facility_id`; nearby fence points can give overlapping tiles (dedupe points < ~1 km apart) | ~none (coordinate = label) | same footprint/GSD; **but the fence is always centred**, whereas benchmark tiles sit on a fixed grid with fences anywhere/partial |
| **B** | centre + some neighbours (facilities often span > 1.9 km) | most neighbours, as "hard" negatives from the same context | neighbours share a facility, so split by `facility_id` | every neighbour must be labelled (4-8 per point) | adds off-centre/partial fences and same-region negatives |
| **C** | a handful per site | hundreds per site (benchmark-like imbalance) | as B, at scale | very high (hundreds of tiles per site) | closest to the evaluation distribution |

**Recommended minimal strategy for December:** **A** for positives, deduplicated per facility.

Negatives must come from outside the three evaluation AOIs, so they cannot reuse benchmark tiles. Qiqi's repo has
`random_coordinates.py` and `generate_russian_negatives.py` for this. A modest set from similar terrain, plus some non-nuclear industrial sites, at
~3-5 negatives per positive, keeps the cost within the free tier.

Two decisions before extraction:
1. Use the preset above.
2. Decide whether to jitter positive tile centres by up to ±0.5 tile, so the fence is not always centred. This is cheap and needs no
   extra labelling if the jitter stays well inside the fence. The wrapper does not do this yet.

Upgrade to B only if A-trained models overfit to the centred fence.

## External dataset pipeline (2026-10-14; `auto_mosaic_1p9km`, confirmed by Qiqi)

```bash
cd cnn_benchmark/external_data
python dataset_plan.py --source "../../Reviewed fence list.csv"     # ingest + jitter + negatives + split + checks (no API)
python dataset_plan.py --source "../../Reviewed fence list.csv" --no-jitter   # ablation plan, separate files
python extract_external_tiles.py --plan ../../external_imagery/dataset/tile_plan.csv --preset auto_mosaic_1p9km \
    --max-requests 21600 --workers 6                                 # resumable; stops on budget/geometry surprises
python dataset_audit.py                                              # validation, QA, hashes, watermark, review sheets
python write_audit_report.py                                         # -> ../outputs/external_data/external_dataset_audit.md
```

- Requests are **metered**: `vendor_runner.py` runs Qiqi's unmodified script via runpy and counts
  `requests.Session.get` calls to the Static Maps endpoint. The wrapper stops on: request budget, a tile using more
  requests than planned, a footprint/zoom/grid other than planned, or a key found in any output file.
- Design parameters (jitter, annulus, exclusion, split) live in `extraction_config.yaml -> dataset`.
- Coordinate-bearing outputs (`positive_audit.csv`, `tile_plan*.csv`, `external_tiles_metadata.csv`,
  `candidate_negative_review.csv`, `near_duplicate_pairs.csv`, contact sheets) are in the git-ignored
  `external_imagery/dataset/`. Only the aggregate report and plot in `outputs/external_data/` are tracked.
- `candidate_negative_review.csv`: fill `review_decision` with `confirmed_negative`, `reject_fence_or_facility_visible`,
  `reject_unusable` or `uncertain`. Re-running `dataset_audit.py` keeps entered decisions.
