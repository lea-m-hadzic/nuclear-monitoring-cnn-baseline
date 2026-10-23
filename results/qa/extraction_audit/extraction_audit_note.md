# Extraction and preprocessing audit (2026-10-22)

This was a read-only audit. No label, manifest, cache, checkpoint or frozen result was changed. Nothing was retrained, and the z17 sensitivity was not run.

**What was re-extracted**
- Path: Qiqi's intended extraction, `--auto_mosaic --output_px 2560`, using preset `auto_mosaic_1p9km`. All other flags were unchanged: `--size 640 --scale 2 --tile_km 1.9 --mosaic_overlap_m 0 --maptype satellite --skip_raw`, with the vendor script at its pinned commit.
- Tiles: 13, at the exact centres of the existing tiles:
  - Saratov ×2;
  - Natanz ×3, including 2 of the 173 tiles that the original Natanz run had marked "tile already exists";
  - Fordow ×2;
  - 6 external positives at latitudes 8°, 35°, 41°, 50°, 56° and 67°.
- Output: a fresh root, `external_imagery/extraction_audit_2026-10-22/` (git-ignored). The script refuses to run if that root already exists. All 13 vendor manifest rows say `downloaded/ok`; none says "tile already exists". So no cached tile could have been reused.
- Cost: 191 Static Maps requests.
- Key handling: the key was never printed, and 0 output files contain it.

Code: `external_data/extraction_audit.py`. Table: `extraction_audit_table.csv`.

## Results

| Tile | Lat | Fresh zoom / grid | Fresh px | Fresh footprint (km) | Fresh GSD (m/px) | Original mode / zoom | Original footprint (km) | Fresh vs cached (1280 px) |
|---|---|---|---|---|---|---|---|---|
| Saratov r21_c21 | 51.54 | 17 / 4 | 2560² | 1.900 | 0.742 | legacy 2×2 / z16 | 1.902 | **different pixels** (mean abs diff 16.7). Same footprint: best scale 1.00, NCC 0.97 |
| Saratov r1_c21 | 51.88 | 16 / 3 | 2560² | 1.900 | 0.742 | legacy 2×2 / z16 | 1.888 | **different pixels** (mean abs diff 9.2). Best scale 0.99, NCC 0.93 |
| Natanz r1_c12 | 33.98 | 17 / 3 | 2560² | 1.900 | 0.742 | auto_mosaic, "already exists" | not recorded | **identical** |
| Natanz r5_c16 | 33.91 | 17 / 3 | 2560² | 1.900 | 0.742 | auto_mosaic, "already exists" | not recorded | **identical** |
| Natanz r20_c20 | 33.65 | 17 / 3 | 2560² | 1.900 | 0.742 | auto_mosaic / z17 | 1.900 | **identical** |
| Fordow ×2 | 34.7–34.9 | 17 / 4 | 2560² | 1.900 | 0.742 | auto_mosaic / z17 | 1.900 | **identical** |
| External ×6 | 8–67 | z17 below about 51.5°, z16 above; grid 3–4 | 2560² | 1.900 | 0.742 | auto_mosaic, same zoom and grid | 1.900 | **5 identical**. Hamaoka: 95% of pixels equal, mean abs diff 0.03 |

"Identical" means the fresh tile, decoded with the same `decode_reduce`, equals the cached 1280 px array byte for byte.

## CNN preprocessing trace

1. **Cache build.** `src/decoded_cache.decode_reduce` handles the benchmark cache, and `external_data/external_cache.py` the external one.
   - Each PNG is converted to RGB.
   - It must be exactly 2560×2560, or the build raises an error. Every one of the 2,940 benchmark and 669 external cache entries passed.
   - It is then reduced 2560 → 1280 with PIL's exact 2×2 box filter (`Image.reduce(2)`).
2. **Training and evaluation.** `train_ablation.NpyTileDataset` loads the 1280×1280×3 arrays and errors on any other shape. Both training and evaluation use this loader. Augmentation is rot90 and flips only, and there is no further resize or crop.
3. **Effective model input.** 1.9 km / 1280 px = **1.484 m/px**. For Saratov's legacy tiles the range is 1.475–1.498 m/px.

## Interpretation

**Training inputs and the Natanz/Fordow benchmark are consistent with the intended path.**
- All 669 cached external tiles (positives, nearby and hard negatives) came from fresh `auto_mosaic_1p9km` downloads: all 1,633 vendor manifest rows in that output root say `downloaded`, none say cached.
- Re-extraction reproduces them byte for byte (Hamaoka differs only trivially, probably a minor change in the provider's imagery).
- Natanz and Fordow re-extract identically too.
- That includes 2 of the 173 Natanz tiles from the earlier "tile already exists" run, so that earlier run used the same mode. The other 171 were not re-extracted.

**Saratov is the exception: a different extraction mode, at the same geometry.**
- The frozen Saratov benchmark tiles are Qiqi's legacy `--mosaic2x2` z16 tiles. These are also the tiles the VLM baseline used.
- Their geometry is equivalent: same centres, footprint 1.888–1.917 km, GSD 0.737–0.749 m/px, scale match 0.99–1.00.
- But their pixels differ. Under auto_mosaic, 841 of the 1,389 Saratov tiles would come from z17 sources and 548 from z16, with different mosaicking and resampling.
- So the CNN was trained on auto_mosaic imagery and tested on Saratov legacy-z16 imagery. This is a real preprocessing difference between training and the Saratov test set.
- It is not a cache collision, and it is not a footprint or GSD error.

## Recommendation

**No re-extraction or retraining of the training set is needed.** The training inputs are the intended ones and reproduce exactly.

The frozen results can be called final **as evaluated on the benchmark imagery that the VLM baseline used**, with one disclosure: the Saratov tiles use Qiqi's legacy z16 2×2 mosaic, at equivalent geometry, while training, Natanz and Fordow use auto_mosaic.

Whether Saratov should *also* be evaluated on auto_mosaic tiles is Qiqi's call. If yes, the smallest clean option is an **evaluation-only** sensitivity:
- re-extract the 1,389 Saratov tiles (about 21,000 requests, about $22);
- build a separate cache;
- re-score the existing frozen checkpoints with the same evaluation code.

That needs no retraining, and the frozen results stay as they are. Not run; it needs approval.
