# Saratov auto_mosaic rescoring sensitivity (condition C, no retraining)

**Status:** sensitivity analysis only. The frozen legacy Saratov result is unchanged and stays the reported result.
- Nothing was retrained, fine-tuned, calibrated or re-thresholded.
- A/B and C_matched were not touched.
- No label, manifest or checkpoint was changed; Natanz, Fordow and the training data were not changed.

**Question.** The frozen Saratov benchmark tiles are Qiqi's legacy `--mosaic2x2` z16 tiles. Training, Natanz and Fordow use `auto_mosaic`. Does this preprocessing-mode mismatch materially change the facility-level CNN results?

**Code:** `external_data/saratov_auto_mosaic_rescore.py`, run as `extract`, `cache`, `score`, `report`.

## Method

**Extraction**
- 1,387 Saratov parent tiles, with the vendor script at its pinned commit and preset `auto_mosaic_1p9km`: `--auto_mosaic --output_px 2560`, other flags unchanged.
- Centres were taken from `parent_tiles.csv`. They are identical to the legacy extraction manifest, and every vendor-reported centre matched to within 5e-8°.
- The output went to a new root, `external_imagery/saratov_auto_mosaic_2026-10-22/` (git-ignored), which the script refuses to reuse unless it created it. The vendor reported `downloaded` for every tile and "tile already exists" for none, so no old tile could have been reused.

**Preprocessing:** the frozen `decoded_cache.decode_reduce`, unchanged:
- it requires a 2560×2560 RGB input (all 1,387 passed);
- it applies an exact 2×2 box reduction to 1280 px;
- output went to a separate cache, `cache/saratov_auto_mosaic_1280px/`, with its own manifest of source and cache SHA-256s.

**Mask:** the frozen geometric watermark-mask rule, with the same `geometry_mask` function, template (SHA checked against each checkpoint) and fill code path.
- It is evaluated with **each new tile's own mosaic geometry**: grid 3 or 4, and `crop_px` from the vendor manifest. This is exactly how the frozen evaluation masks the auto_mosaic tiles of Natanz and Fordow.
- Applying the legacy 2×2 mask to auto_mosaic tiles would mask the wrong pixels.
- Masked area is 2.37–2.79%, vs 2.57% for the legacy layout.

**Scoring and evaluation**
- Scored with the 3 frozen C checkpoints (`outputs/ablation/runs/C/seed{0,1,2}/best.pt`), on the same Apple MPS backend.
- Evaluated with the frozen `src/evaluation.py` footprint rule and the provisional facility map, using the same AOI area and tie policy.
- **Reproduction check:** re-scoring the *legacy* cache with the same script reproduces each seed's frozen `scores.csv`, to a maximum absolute difference of 1.1e-16 over 1,387 tiles. The metric code also reproduces the frozen T3/T4 values. So every difference below comes from the imagery alone.

## Extraction summary

**Outcome:** 1,387 of 1,387 tiles were extracted, with 0 failures. 0 files contain the API key.

**Source zoom under auto_mosaic:**
- **839 tiles changed from z16 to z17** (4×4 grid; the southern part of the AOI, below about 51.6°).
- 548 stayed at z16 but changed from a 2×2 to a 3×3 mosaic, with a different crop and resampling. So every tile's pixels changed.

**Footprint and GSD:**

| | Footprint | Output GSD | Model input |
|---|---|---|---|
| auto_mosaic | 1.8996–1.9002 km (median 1.8999) | 0.7420–0.7423 m/px | 1.484–1.485 m/px |
| legacy | 1.888–1.917 km | 0.737–0.749 m/px | — |

**Cost and storage:**
- 21,130 Static Maps requests, about $22 at the configured list price if this month's 10k free requests are unused, or about $42 if they are used.
- Storage: PNGs 17.5 GB (16 GB on disk); new cache 6.4 GB.

## Results (condition C, n = 3 seeds; mean ± SD)

The per-seed values are in `saratov_auto_mosaic_rescore_seed_level.csv`, and all columns, including the per-seed differences, are in `saratov_auto_mosaic_rescore_summary.csv`.

| Metric | Legacy (frozen) | auto_mosaic | Δ mean | Δ per seed (0 / 1 / 2) |
|---|---|---|---|---|
| Tile AUROC | 0.734 ± 0.048 | 0.721 ± 0.049 | −0.013 | −0.015 / −0.014 / −0.009 |
| Tile PR-AUC | 0.324 ± 0.048 | 0.310 ± 0.034 | −0.014 | −0.031 / −0.016 / +0.005 |
| Facility recall @10% area | 0.319 ± 0.064 | 0.319 ± 0.064 | 0.000 | +0.042 / −0.125 / +0.083 |
| **Facility recall @20% area** | **0.542 ± 0.042** | **0.514 ± 0.064** | **−0.028** | **0.000 / −0.125 / +0.042** |
| Facility recall @30% area | 0.625 ± 0.072 | 0.639 ± 0.096 | +0.014 | +0.042 / −0.083 / +0.083 |
| Area for 25% recall (% AOI) | 7.3 ± 2.6 | 6.4 ± 3.6 | −0.9 | −1.9 / +1.5 / −2.2 |
| **Area for 50% recall** | **18.5 ± 1.3** | **18.6 ± 4.1** | **+0.05** | **−1.2 / +4.7 / −3.4** |
| Area for 75% recall | 44.3 ± 5.1 | 46.0 ± 15.5 | +1.7 | +6.1 / +10.4 / −11.4 |
| Area for 90% recall | 65.9 ± 17.2 | 69.6 ± 19.2 | +3.7 | +5.7 / +3.8 / +1.6 |
| First hit SAR-WHS-01 (% AOI) | 3.4 ± 4.8 | 3.9 ± 4.9 | +0.5 | +1.0 / +0.6 / 0.0 |
| First hit SAR-WHS-02 | 3.4 ± 4.3 | 3.7 ± 4.4 | +0.2 | +0.1 / +0.2 / +0.3 |
| First hit SAR-WHS-03 | 4.1 ± 4.2 | 12.0 ± 13.8 | +8.0 | +1.8 / **+19.1** / +3.0 |

**Agreement between legacy and auto_mosaic tile rankings** (`saratov_auto_mosaic_rank_comparison.csv`):

| Seed | Spearman (scores) | Top 1% overlap | Top 5% | Top 10% | Top 20% |
|---|---|---|---|---|---|
| 0 | 0.950 | 10/14 (0.71) | 57/70 (0.81) | 119/139 (0.86) | 241/278 (0.87) |
| 1 | 0.968 | 11/14 (0.79) | 61/70 (0.87) | 120/139 (0.86) | 245/278 (0.88) |
| 2 | 0.955 | 13/14 (0.93) | 62/70 (0.89) | 113/139 (0.81) | 249/278 (0.90) |

Figure: `fig_saratov_legacy_vs_auto_mosaic_recall.png`, with facility recall vs % of AOI reviewed per seed and the two imagery modes overlaid.

## Interpretation (pre-specified rule)

The mismatch would count as **material** only if facility recall at 20% area changed by at least 0.10, or the area to reach 50% recall changed by at least 5 percentage points, with the direction holding in at least 2 of 3 seeds.

- **Recall at 20% area:** the mean change is −0.028. Only seed 1 reaches 0.10 (−0.125). Seed 0 is unchanged and seed 2 goes the other way (+0.042). **Not met.**
- **Area for 50% recall:** the mean change is +0.05 pp, with per-seed changes of −1.2, +4.7 and −3.4. No seed reaches 5 pp and the directions are mixed. **Not met.**

**Conclusion: the Saratov preprocessing-mode mismatch does not materially affect the CNN baseline.**
- Tile scores are highly concordant across the two imagery modes (Spearman 0.95–0.97; 81–90% overlap of the top 5–20%).
- Tile AUROC drops slightly but consistently on auto_mosaic (−0.009 to −0.015 in every seed). This is well within the seed spread (SD 0.05) and practically negligible.

**Two secondary signals to report, but not material by the rule:**
- The upper recall targets (75% and 90%) need somewhat more area on auto_mosaic, at +1.7 and +3.7 pp. The 75% target is not consistent in direction across seeds; the 90% target is consistent (+1.6 to +5.7) but small relative to the seed spread.
- **SAR-WHS-03** is found later in seed 1 (27.9% vs 8.8% of the AOI). Its other seeds move by only +1.8 and +3.0 pp. Single-site first hits are the noisiest metric here.

**What this does not change:**
- The frozen legacy Saratov result stays the reported benchmark. It is also the imagery the VLM baseline was scored on, so the CNN-vs-VLM comparison stays like-for-like.
- This sensitivity can be cited as showing that the reported C result is robust to the Saratov extraction mode.
