# Full-resolution (2560 px) input sensitivity: pre-registration

**Written before any 2560 training or evaluation result existed (2026-10-23).**

**Question:** does giving the CNN the full ≈ 0.74 m/px input materially improve broad-area facility retrieval, compared with the frozen ≈ 1.48 m/px (1280 px) baseline?

This is a **benchmarking sensitivity only**. The frozen 1280 px baseline (`results/main/`) is not altered, overwritten or replaced.

## Fixed (identical to frozen C-1280)

| | |
|---|---|
| Condition | **C only** (manifest `outputs/ablation/manifests/C.csv`); same 287 positives, same 157 negatives (126 train / 31 val), same splits and labels |
| Footprint and imagery | Same 1.9 km tiles; **same extracted PNGs**, with no new extraction and no z17. The source SHA-256 of every tile equals the frozen 1280 cache manifest |
| Validation and stopping | Shared 118-tile validation set; up to 20 epochs, minimum 5, patience 5, selected on shared-validation PR-AUC |
| Model | ResNet-50, ImageNet (IMAGENET1K_V2) weights, single logit; **no architecture change** |
| Loss | BCE with pos_weight = n_neg / n_pos |
| Optimizer and schedule | AdamW (lr 1e-4, wd 1e-4), cosine over 20 epochs |
| **Batch size** | **4, unchanged.** A one-batch probe at 2560 used about 55 GB of the Mac's 128 GB unified memory, so **no batch-size deviation is needed** |
| Augmentation | Random 90° rotation plus horizontal/vertical flips; **no crops or resizing** |
| Sampling and data order | Same seeded shuffling (seeds 0, 1, 2); data-loader workers 0, as in the frozen runs |
| Backend | Apple MPS (M5 Max), fp32, the same backend as frozen C |
| Evaluation | Same parent-tile scoring of all 2,940 benchmark tiles; same `src/evaluation.py` footprint rule, provisional facility map and tie policy. **Saratov uses the same legacy imagery** as frozen C and the VLM |
| Freeze check | Every run re-checks manifests, shared val, watermark template and geometry table against `outputs/ablation/freeze.json` |

## Changed (the only intended change)

1. **Model input resolution: 2560 × 2560 instead of 1280 × 1280.** Tiles are decoded from the same PNGs without the 2× box reduction. For **every** tile (475 external, 2,940 benchmark), the exact 2× reduction of the 2560 array reproduces the frozen 1280 cache array byte for byte (`cache/*_2560px/cache_manifest.csv`, `reduce_equals_frozen_1280`).
2. **Watermark mask, same physical rule at the new pixel size.**
   - The frozen geometric rule (`watermark_geometry.geometry_mask`) is rendered at 2560, so it covers the same physical area.
   - The normalised-convolution fill uses **σ = 12 px**, the frozen 6 px at 1280 at the same physical scale (≈ 8.9 m).
   - Implementation detail: the fill is deterministic and precedes augmentation, so it is applied once and stored (`cache/*_2560px_filled/`) instead of on the fly. The dataset tensors were checked to be **identical** to on-the-fly filling on sample tiles. This only saves CPU time (≈ 2.4 s per tile per epoch).

Code: `train_ablation.py --input-px 2560` and `evaluate_ablation.py` (reads `input_px` from the checkpoint). Defaults are unchanged, and the frozen 1280 path re-scores frozen C to 9.7e-17. Caches: `fullres_cache.py`. Outputs: `outputs/ablation/runs_fullres_2560/C/seed{0,1,2}/` (local) and `results/robustness/fullres_2560/` (tables, figure, memo).

## Reported, side by side with frozen C-1280 (mean ± SD over 3 seeds, plus per-seed values)

- **Tiles:** pooled tile AUROC and PR-AUC.
- **Saratov:**
  - facility recall at 10, 20 and 30% of the area reviewed;
  - area to 25, 50, 75 and 90% recall;
  - first-hit area for SAR-WHS-01, -02 and -03.
- **Natanz and Fordow:** first-hit area.
- **Score agreement** with the 1280 model of the same seed: Spearman on the 2,940 tiles (and on Saratov), plus top 1/5/10/20% overlap.
- **Training and validation:** best epoch, epochs run, shared-validation PR-AUC and AUROC, train loss. These help tell transfer apart from overfitting.

## Materiality rule (fixed now)

The 2560 change is **material** only if at least one holds, **with the same direction in at least 2 of 3 seeds** (per-seed differences: 2560 minus the frozen 1280 of the same seed):
- an absolute change of **≥ 0.10 in Saratov facility recall at 20%** review area; or
- a change of **≥ 5 percentage points in the area needed for 50% Saratov facility recall**.

**Interpretation:**
- **If material and better:** report as a stronger-resolution CNN sensitivity and discuss what it means for benchmarking. The frozen baseline is **not** replaced automatically.
- **Otherwise:** keep the frozen baseline and state that input resolution was not the main driver.

No hyperparameter is tuned on these results.

## Pre-launch verification (2026-10-23, before any training)

- **Same imagery:** 2560 caches built from the frozen manifests' source files (source SHA-256 equal for every tile). The 2× box reduction equals the frozen 1280 array for **475/475** external and **2,940/2,940** benchmark tiles.
- **Filled cache = on-the-fly fill:** dataset tensors were **identical for 7/7 and then 60/60** sampled tiles (25 external train/val, 35 benchmark across all three AOIs).
- **Memory:** batch 4 at 2560 used ≈ 55 GB, so no batch-size change.
- **Frozen 1280 path unchanged:** the edited code re-scores frozen C seed 0 with max |Δ| 9.7e-17; tests 113 passed.
- **Freeze record:** `fullres_2560_prereg_freeze.json` (SHA-256 of this spec, the code, the cache manifests and the frozen inputs).
