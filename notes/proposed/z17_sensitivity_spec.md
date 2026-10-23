# z17 resolution sensitivity: specification (NOT RUN; needs explicit approval)

**Sole hypothesis:** does finer spatial resolution materially improve broad-area retrieval of the layered-fence signature?

Everything else is held at the frozen main condition C.

**Status: FROZEN 2026-10-21 as a proposed optional sensitivity. NOT RUN.**
- Nothing has been extracted, cached or trained, and no API call has been made.
- Running it needs explicit approval. It also waits for Yanliang's targeted negative review (41 tiles): if that review changes any negative label, the condition-C manifest this spec is pinned to would change, and the spec must be revisited first.
- Until it runs, finer-resolution performance is **untested**. Paper text describes the frozen CNN input as about 1.5 m/px effective model input resolution.

## 1. What actually changes

The frozen pipeline does not feed the CNN "z16 imagery":
- Every tile, training or benchmark, is a **1.9 km footprint at 0.742 m/px (2560 px)**.
  - External and Natanz/Fordow tiles: `auto_mosaic`, which already uses z17 sources for 80% of condition-C tiles (353 of 444) (all of them below about 51.5°) and z16 above that.
  - Saratov: legacy 2x2 z16, 0.743 m/px.
- The decoded cache then **box-reduces 2x to 1280 px**, so the **model input is 1.48 m/px** everywhere.

So "z16 vs z17" is really a comparison of **model-input ground sample distance (GSD)**.

| | Frozen C (reference) | z17 arm |
|---|---|---|
| Source zoom | 16/17 (latitude-adaptive) | **17 for every tile** |
| Footprint | 1.9 km | 1.9 km (same tile centres) |
| Model input | 1280 px, 1.48 m/px | **3168 px, 0.60 m/px (2.47x finer)** |

The 3168 px size (0.5997 m/px) was chosen so that **no tile is upsampled**:
- the coarsest native z17 GSD in the tile set is 0.591 m/px (lowest-latitude tile);
- at Saratov z17 is 0.372 m/px, at Natanz 0.497 and at Fordow 0.490.

So every tile is a downsample of real z17 pixels to one common GSD. This mirrors how the frozen design standardises GSD.

**What stays identical:**
- Condition C manifest (SHA `fe5b7a2c…`): the same 444 tiles, 357 train / 87 val, the same tile IDs and labels, and the split groups.
- The shared 118-tile validation set (SHA `ade28778…`).
- Model and training: ResNet-50 (IMAGENET1K_V2), weighted BCE, AdamW 1e-4 with weight decay, cosine schedule, batch 4, rot90 and flip augmentation.
- Stopping rule: up to 20 epochs, minimum 5, patience 5, selection on shared-val PR-AUC.
- Masking: geometric watermark mask, recomputed for the z17 mosaic layouts. The masked area is 2.3–2.7% vs 1.9–2.8% now, so it is not a confound.
- Seeds 0, 1, 2.
- Evaluation: `src/evaluation.py`, the provisional facility map, the footprint rule and the review-budget curves.

**Parent-tile comparability:** benchmark tiles are re-extracted at the **same 1.9 km centres** as the frozen Saratov/Natanz/Fordow grids (same `rNN_cNN` IDs). Each z17 tile therefore *is* its z16 parent tile, scored 1:1. No aggregation is needed and the facility metrics are directly comparable. (The VLM's Saratov z17 scan used 0.95 km tiles; it is not reused.)

**Reference arm:** the frozen C MPS runs, not re-run. The z17 arm runs on the same Mac MPS backend. If it has to move to CUDA, the comparison is instead against the existing C CUDA control (0.744 pooled AUROC), never across backends.

**Unavoidable coupled change:** the receptive field covers about 2.5x less ground at 0.60 m/px. This is inherent in "finer resolution at a fixed architecture" and is reported, not corrected.

## 2. Pre-registered readout

**Primary:**
- Saratov facility recall at a 20% review budget, and the area needed to reach 50% Saratov recall (seed mean and per seed).
- Reference values: C MPS 0.54 / 18.5%; VLM 50% recall at 9.4%.

**Secondary:**
- Natanz/Fordow first-hit rank.
- Pooled tile AUROC / PR-AUC.
- Shared-val PR-AUC.

**"Material improvement"** means **both**:
1. seed-mean Saratov recall at 20% improves by ≥ 0.10 (twice the 0.05 MPS-vs-CUDA gap for identical C) **or** the area to 50% recall drops by ≥ 5 percentage points of AOI;
2. the direction holds in at least 2 of 3 seeds.

Anything smaller is reported as "no material effect". No other variant (other zooms, condition A/B, unmasked, FULL_MIXED) is in scope.

## 3. Cost and effort estimates

**Google Static Maps requests:** computed with `geometry.py` and the pinned vendor functions; no API call was made. External tiles are condition C plus the shared val, 475 tiles.

| Tile set | z17 grid | Requests |
|---|---|---|
| External (475) | 3x3:114, 4x4:265, 5x5:89, 6x6:6, 7x7:1 | 8,706 |
| Saratov (1,387) | 4x4 | 24,966 |
| Natanz (776) | 3x3 | 8,536 |
| Fordow (777) | 4x4 | 13,986 |
| **Total** | | **56,194** |

That is about **$92** at the configured list price (pricing checked 2026-10-13, 10k free per month). External + Saratov only would be 33,672 requests, about $47.

**Storage** (1.3 TB free):

| Item | Size |
|---|---|
| PNGs (about 20 MB each at 3168 px) | about 69 GB |
| Decoded uint8 caches (30 MB per tile): external | 14 GB |
| Decoded uint8 caches: benchmark | 89 GB |
| **Total** | **about 170 GB** |

The benchmark cache is optional: the benchmark can be decoded on the fly at evaluation time.

**Compute:**
- Pixel count is 6.1x the frozen input.
- Frozen C on MPS took about 173 s train + 33 s val per epoch, at 13.7 GB peak.
- Expected: about **21 min per epoch**. With the same epoch counts as before (20/15/14) that is about 17 h of training.
- Benchmark scoring (2,940 tiles × 3 seeds): about 4 h.
- **Total: about 21 h of Mac MPS time.**
- Peak memory at batch 4 is estimated at 80–90 GB, which should fit in 128 GB of unified memory but is not guaranteed. If it runs out of memory, the fix is **gradient checkpointing** (same mathematics), **not** a smaller batch, which would change BatchNorm and the optimizer.

**People-time:** about 1–1.5 days of implementation and QA, plus about 1 day of wall clock for extraction and caching.

## 4. Implementation changes required

1. **Extraction preset** `fixed_z17_1p9km`, in `extraction_config.yaml` and `geometry.py`:
   - force zoom 17, 1.9 km crop, output 3168 px;
   - use the existing tile centres for external tiles (metadata `center_lat/lon`) and for the benchmark (existing manifests);
   - check whether the pinned vendor script accepts a fixed zoom with `--auto_mosaic`. If it doesn't, add a thin wrapper that calls its functions; the vendor code stays unmodified.
   - The Google key is read only from the environment, as now.
2. **Decoded cache** (`src/decoded_cache.py`): parameterise `SIZE` and `reduce_factor` (3168, no reduction) and write to a new cache directory with its own manifest. The 1280 px caches are untouched.
3. **Training and evaluation** (`train_ablation.py`, `evaluate_ablation.py`, `ablation_config.yaml`):
   - replace the hard-coded `(1280, 1280, 3)` check with `input_size` from config;
   - add a `--resolution z17` switch that selects the z17 caches and the z17 watermark geometry;
   - write outputs to `outputs/ablation/runs_z17_<backend>/`;
   - add optional gradient checkpointing.
4. **Watermark geometry** (`watermark_geometry.py`): add z17 rows (grid 3–7, `crop_px = 1900 / native_mpp`) and masks at n = 3168, then regenerate the mask QA overlays for a visual check.
5. **Freeze record:** before launch, write `freeze_z17.json` with checksums of the manifest, the shared val, both z17 cache manifests, the geometry and the git commit.
6. **Aggregation:** a short `aggregate_z17.py` that compares z17 against frozen C MPS with the readout in §2. It reuses the `aggregate_robustness.py` pattern.
