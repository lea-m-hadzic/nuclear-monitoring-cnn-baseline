# Full-resolution (2560 px) input sensitivity: result

**Date:** 2026-10-25. **Pre-registration:** `FULLRES_2560_SPEC.md` (frozen 2026-10-23, `fullres_2560_prereg_freeze.json`). **Script:** `cnn_benchmark/external_data/fullres_2560_compare.py` (reads outputs only). **Run records:** `provenance/run_records/fullres_2560/` (no checkpoints).

## Verdict

**Material, and in the worse direction.** Both pre-registered criteria are met, in all 3 of 3 seeds:

| Criterion (threshold) | Seed 0 | Seed 1 | Seed 2 | Rule met |
|---|---|---|---|---|
| Δ Saratov facility recall @20% area (≥ 0.10) | −0.46 | −0.33 | −0.29 | Yes, 3/3 worse |
| Δ area to 50% Saratov recall (≥ 5 pp) | +24.9 pp | +24.2 pp | +12.9 pp | Yes, 3/3 worse |

Δ = 2560 minus frozen 1280 of the same seed. **Following the spec, the frozen 1280 baseline is kept.** Higher input resolution did not help: with the same recipe, it made broad-area retrieval clearly worse. No hyperparameter was tuned on these results, and no follow-up variant was run.

## Side by side (mean ± SD over 3 seeds)

| Metric | C-1280 (frozen) | C-2560 | Δ mean |
|---|---|---|---|
| Pooled tile AUROC | 0.745 ± 0.047 | 0.678 ± 0.067 | −0.067 |
| Pooled tile PR-AUC | 0.237 ± 0.034 | 0.133 ± 0.056 | −0.104 |
| Saratov recall @10% area | 0.32 ± 0.06 | 0.13 ± 0.08 | −0.19 |
| **Saratov recall @20% area** | **0.54 ± 0.04** | **0.18 ± 0.12** | **−0.36** |
| Saratov recall @30% area | 0.63 ± 0.07 | 0.31 ± 0.09 | −0.32 |
| Area to 25% Saratov recall (% AOI) | 7.3 ± 2.6 | 22.8 ± 7.8 | +15.5 |
| **Area to 50% Saratov recall (% AOI)** | **18.5 ± 1.3** | **39.2 ± 5.6** | **+20.7** |
| Area to 75% Saratov recall | 44.3 ± 5.1 | 64.6 ± 9.9 | +20.3 |
| Area to 90% Saratov recall | 65.9 ± 17.2 | 78.9 ± 5.2 | +13.1 |
| First hit SAR-WHS-01 / -02 / -03 (% AOI) | 3.4 / 3.4 / 4.1 | 5.7 / 12.3 / 37.5 | worse |
| Natanz first hit (% AOI) | 1.7 ± 2.8 | 11.7 ± 17.0 | +9.9 |
| Fordow first hit (% AOI) | 2.8 ± 3.2 | 4.9 ± 4.2 | +2.1 |
| Best epoch / epochs run | 11.0 / 16.3 | 8.7 / 14.7 | |
| Shared-val PR-AUC / AUROC (best epoch) | 0.939 / 0.947 | 0.908 / 0.910 | −0.031 / −0.036 |
| Train loss at best epoch | 0.092 | 0.175 | +0.083 |

At 2560, Saratov recall at 20% area (0.18) is at or below random review (0.20). Natanz and Fordow first hits are single facilities, so their deltas are not interpreted.

**Score agreement with the 1280 model of the same seed** (`rank_overlap.csv`) is low:
- Spearman 0.39–0.49 on all 2,940 tiles, and 0.33–0.57 on Saratov.
- Top-10% overlap 0.38–0.47.
- Top-1% overlap 0.03–0.38.

The 2560 model ranks tiles substantially differently; it is not a mild perturbation of the 1280 ranking.

**Per-AOI tile AUROC** (descriptive, from each run's `summary.json`):

| Seed | Saratov 1280 → 2560 | Natanz 1280 → 2560 | Fordow 1280 → 2560 |
|---|---|---|---|
| 0 | 0.68 → 0.65 | 0.60 → 0.37 | 0.75 → 0.72 |
| 1 | 0.77 → 0.65 | 0.75 → 0.42 | 0.78 → 0.58 |
| 2 | 0.75 → 0.75 | 0.63 → 0.77 | 0.67 → 0.73 |

The drop is not confined to Saratov.

## Reading (not a tested explanation)

- **Weaker in-domain fit.** Shared-validation PR-AUC and AUROC are lower and the training loss at the best epoch is higher at 2560. With batch 4, 357 training tiles and ≤ 20 epochs, the larger input appears to be harder to fit in the frozen budget. The degradation is therefore not purely a transfer effect.
- **Saratov imagery.** Saratov uses legacy z16 2×2 mosaics (README, extraction audit). At 2560 the model sees those at full nominal size, so a source-resolution mismatch with the training imagery may add to the Saratov drop. Natanz and Fordow (same extraction as training) also degrade in 2 of 3 seeds, so this cannot be the whole story.
- **Variance.** Seed 2 is close to 1280 on tile AUROC but still loses 0.29 recall @20% on Saratov.

Testing any of these would need new runs, which the spec doesn't include. No further CNN experiments will be run.

**Paper wording:** *Increasing the CNN input resolution from 1280 to 2560 pixels, while holding the training protocol fixed, materially degraded broad-area retrieval across all three seeds.*

## Run record and deviations

- **Configuration:** all three runs used condition C (same manifest, SHA fe5b7a2c…), 357 train / 118 shared-val tiles, batch 4, AdamW 1e-4, wd 1e-4, Apple MPS. The only recorded difference from frozen C is `input_size` 2560, plus the mask fill σ 12 px as pre-registered.
- **Seed 2 interruption.** The first seed-2 run was killed after 12 epochs when the controlling session ended (no summary written). It was not evaluated and is not kept in this repo. Seed 2 was restarted **from scratch with the identical recipe** on 2026-10-25 03:45 UTC. The reported seed 2 is the restarted run. The interrupted run was not evaluated.
- **Compute contention.** Seed 2 shared the machine with other jobs. This affects wall-clock only, not the recipe or the data order.
- **Facility results were not inspected between seeds.** All evaluation ran after all three seeds finished training.
- **Frozen 1280 outputs unchanged:** `provenance/run_records/runs/C/` was read only.
- **Area to 25% recall** is not among the evaluator's stored recall targets. It is read off the same facility curve, and the script checks that this reading reproduces the evaluator's stored 50/75/90% values exactly.

## Files

- `seed_level.csv`: all metrics, both resolutions × 3 seeds.
- `summary.csv`: mean and SD per resolution.
- `comparison_1280_vs_2560.csv`: means, SDs and per-seed Δ.
- `rank_overlap.csv`: Spearman and top 1/5/10/20% overlap, all tiles and Saratov.
- `materiality.csv`: the rule applied as written.
- Figure: `figures/fig_robustness_fullres_2560_recall_vs_review_area.png`, Saratov facility recall vs % AOI reviewed. The copy written next to these tables is git-ignored, like the other aggregation PNGs.
