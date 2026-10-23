# CNN baseline: robustness record (definitive)

**Status: complete and frozen.** Robustness runs were frozen on 2026-10-21; the Saratov auto_mosaic re-scoring (no retraining) was added on 2026-10-22. No further CNN experiments are planned.
- No more backend controls, mask variants or nuisance-matching variants.
- FULL_MIXED stays off (`proposed/full_mixed_spec.md`, never run).
- z17 / finer resolution: specified and frozen as a **proposed optional sensitivity, not run** (`proposed/z17_sensitivity_spec.md`).
- The targeted negative-label review (41 tiles) is with Yanliang. No negative label has been changed. See `DATA_AND_LABEL_QA.md`.

**Scope of the claims:**
- Every condition has **n = 3 seeds** (0, 1, 2). Values are mean ± sample SD.
- **No significance claims are made from n = 3.** Differences smaller than the seed spread are described as "within seed variability".
- Facility-level broad-area-search metrics are primary. Tile AUROC and PR-AUC are secondary.
- The facility map is provisional (`cnn_benchmark/outputs/facility_mapping_questions.md`).

**Tables** (generated from raw per-run outputs by `cnn_benchmark/external_data/robustness_tables.py`):
- `results/robustness/paper_tables/table_nuisance_matching.md`
- `results/robustness/paper_tables/table_mask_robustness.md`
- `results/robustness/paper_tables/seed_level_metrics.csv`

## 1. Main ablation (frozen; Apple MPS, M5 Max)

- **Setup:** frozen at commit `8742c46` (`cnn_benchmark/outputs/ablation/freeze.json`). ResNet-50 trained on external layered-fence tiles only; evaluated on Saratov, Natanz and Fordow, which never contribute training data. Same 287 positives and 157 negatives per condition (126 train / 31 val).
- **Shared recipe:** geometric watermark mask; checkpoint selection on one shared validation set (118 tiles; up to 20 epochs, minimum 5, patience 5). Tables: `results/main/abc_tables/`; summary: `results/main/abc_summary.csv`.

| Condition | Pooled tile AUROC | Pooled PR-AUC | Saratov recall @20% | Area to 50% Saratov recall | Natanz / Fordow first hit (% AOI) |
|---|---|---|---|---|---|
| A: nearby negatives | 0.763 ± 0.057 | 0.225 ± 0.126 | 0.32 ± 0.26 | 33.6 ± 21.2% | 0.15 / 2.09 |
| B: hard negatives | 0.545 ± 0.047 | 0.056 ± 0.008 | 0.19 ± 0.16 | 41.2 ± 10.8% | 42.5 / 16.5 |
| **C: mixed, natural sampling** | 0.745 ± 0.047 | 0.237 ± 0.034 | **0.54 ± 0.04** | **18.5 ± 1.3%** | 1.75 / 2.77 |

**Main result (unchanged):**
- **C** (natural nearby negatives mixed with confirmed hard negatives) is the main condition.
- A and C are comparable on tile discrimination. C is the most budget-efficient and most stable on Saratov facility retrieval.
- B fails to transfer.
- A finds the single Natanz plant faster (an AOI-specific reversal).

**VLM comparison at the same operating point:** the VLM reaches 50% Saratov facility recall at **9.4%** of the AOI, against 18.5% for frozen C. The VLM remains more review-efficient on Saratov at that point (`results/main/vlm_comparison.csv`; VLM evaluation in `cnn_benchmark/outputs/evaluations/vlm_single_pass__footprint_parent/`).

**Caveat:** the shared validation set has C's nearby/hard mixture, so C's high validation score is distributionally favourable to C and is not independent evidence.

## 2. Robustness to imagery masking (Apple MPS; backend-matched with the main ablation)

The mask-off runs use the identical recipe with the geometric watermark mask disabled. "Mask on" rows are the frozen runs.

| Condition | Pooled tile AUROC | Pooled PR-AUC | Saratov recall @20% | Area to 50% Saratov recall | Natanz / Fordow first hit (% AOI) |
|---|---|---|---|---|---|
| A, mask on (frozen) | 0.763 ± 0.057 | 0.225 ± 0.126 | 0.32 ± 0.26 | 33.6 ± 21.2% | 0.15 / 2.09 |
| A, mask off | 0.762 ± 0.065 | 0.234 ± 0.103 | 0.43 ± 0.20 | 22.5 ± 13.4% | 0.13 / 3.49 |
| C, mask on (frozen) | 0.745 ± 0.047 | 0.237 ± 0.034 | 0.54 ± 0.04 | 18.5 ± 1.3% | 1.75 / 2.77 |
| C, mask off | 0.715 ± 0.032 | 0.221 ± 0.066 | 0.49 ± 0.10 | 16.7 ± 6.3% | 0.47 / 3.66 |

All mask-on vs mask-off differences are within seed variability. The A-vs-C interpretation is unchanged:
- comparable tile discrimination;
- C at least as efficient on Saratov;
- A first at Natanz.

## 3. Robustness to nuisance matching (CUDA, NVIDIA A100 80GB PCIe, bend GPU 1; backend-controlled)

- **C_matched:** C's hard-negative half, plus a nearby half matched to the positives on contrast and water fraction (per-split nearest-neighbour matching in rank space).
- **Controlled comparison:** C [CUDA control] vs C_matched [CUDA]. Same backend, seeds and recipe; only the nearby-negative sampling differs.
- **Interpretation rule, fixed before the CUDA C results:**
  - if CUDA C stays close to frozen MPS C and clearly outperforms CUDA C_matched, conclude that nuisance matching materially hurts transfer;
  - if CUDA C also degrades substantially, treat the C_matched result as backend-confounded.

| Condition | Pooled tile AUROC | Pooled PR-AUC | Saratov recall @20% | Area to 50% Saratov recall | Natanz / Fordow first hit (% AOI) | Shared-val PR-AUC |
|---|---|---|---|---|---|---|
| *C [MPS, frozen]: reference only* | *0.745 ± 0.047* | *0.237 ± 0.034* | *0.54 ± 0.04* | *18.5 ± 1.3%* | *1.75 / 2.77* | *0.939 ± 0.023* |
| C [CUDA control] | 0.744 ± 0.006 | 0.228 ± 0.116 | 0.49 ± 0.13 | 20.4 ± 9.5% | 0.17 / 0.64 | 0.948 ± 0.024 |
| C_matched [CUDA] | 0.672 ± 0.026 | 0.107 ± 0.035 | 0.15 ± 0.05 | 40.4 ± 9.5% | 4.30 / 12.64 | 0.923 ± 0.010 |

**Outcome under the rule:**
- **The backend does not explain the drop.** CUDA C matches frozen MPS C within seed variability.
- **C_matched is clearly worse** than CUDA C on every benchmark metric. Its Saratov recall at 20% is below random (0.24), with no AOI-specific reversal.
- **The shared validation set does not show it:** C_matched keeps a high shared-validation PR-AUC (0.92).
- **Conclusion:** nuisance-matched nearby sampling materially reduces broad-area transfer.

## 4. Backend, device and code for every experiment

| Experiment | Runs | Device | Launch commit(s) |
|---|---|---|---|
| Main A/B/C (frozen) | 9 | Apple MPS (M5 Max), fp32 | `8742c46` |
| A, C mask off | 6 | Apple MPS (M5 Max), fp32 | `ee7e38b` (A seed 0); `83b43d7` (others)¹ |
| C [CUDA control] | 3 | CUDA, NVIDIA A100 80GB PCIe (bend GPU 1), fp32, TF32 off | `ee7e38b` |
| C_matched [CUDA] | 3 | CUDA, NVIDIA A100 80GB PCIe (bend GPU 1), fp32, TF32 off | `be0d861` (seed 0)²; `804329f` (seeds 1–2) |

¹ Commit `83b43d7` only added the aggregation script during the run chain; the training code is identical to `ee7e38b`.

² Trained at `be0d861`. Its evaluation was re-run at `804329f` (see section 5).

**Held fixed across all of these** (verified against `freeze.json` / `freeze_robustness.json` before every run):
- manifests and the shared validation set;
- watermark template and geometry table;
- model, loss, optimizer, schedule, batch size, augmentations, data-loader workers, stopping rule and seeds.

On CUDA, TF32 was explicitly disabled so the arithmetic stays fp32, as on MPS. Robustness outputs are kept by backend:
- `provenance/run_records/runs_robustness_mps/`
- `provenance/run_records/runs_robustness_cuda/`

The frozen main runs (`provenance/run_records/runs/`) were never modified. Checkpoints are not in this repo.

## 5. Technical failures and fixes

1. **C_matched seed 0: evaluation failed after normal training** (`UnpicklingError: Weights only load failed … TorchVersion`).
   - **Cause:** commit `be0d861` stored `torch.__version__` (a `TorchVersion` object) in the checkpoint metadata, and PyTorch's default safe loading rejects that class.
   - **Fix (`804329f`):** the version is stored as a plain string; evaluation keeps safe loading and allowlists only `TorchVersion`.
   - **Recovery:** the run was not retrained. Only its evaluation was re-run, and the checkpoint stayed byte-identical (SHA-256 checked).
2. **A mask-off seed 0 on bend: CUDA out of memory** on 2026-10-19 at 19:08, within 6 s of starting.
   - **Cause:** another user's jobs had taken about 69 GB of the same GPU. No training step completed.
   - **Action:**
     - The mask-off runs were moved to the Mac (MPS), which also keeps them backend-matched with the frozen mask-on runs.
     - Output folders were separated by backend (`ee7e38b`).
     - A CUDA C control was added to interpret C_matched.
     - That control started each seed only after GPU 1 had been idle on two checks 10 minutes apart.

## 6. Saratov extraction-mode sensitivity (2026-10-22; re-scoring only, no retraining)

**Why:** the extraction audit (`DATA_AND_LABEL_QA.md` §4) found that the frozen Saratov benchmark tiles are Qiqi's legacy z16 2×2 mosaics. Training, Natanz and Fordow use `auto_mosaic`. The geometry is equivalent; the pixels differ.

**What:**
- All 1,387 Saratov parent tiles were re-extracted with `--auto_mosaic --output_px 2560` at the frozen centres, into a fresh directory (0 failures). 839 tiles changed from z16 to z17 sources.
- They went through the frozen cache preprocessing and the frozen geometric-mask rule (each tile's own mosaic geometry), and were scored with the 3 frozen C checkpoints.
- **Check:** re-scoring the legacy cache with the same script reproduces the frozen scores to 1.1e-16.

| C, 3 seeds | Legacy (frozen, reported) | auto_mosaic | Δ per seed |
|---|---|---|---|
| Saratov tile AUROC | 0.734 ± 0.048 | 0.721 ± 0.049 | −0.015 / −0.014 / −0.009 |
| Facility recall @20% area | 0.54 ± 0.04 | 0.51 ± 0.06 | 0.00 / −0.125 / +0.04 |
| Area to 50% recall (% AOI) | 18.5 ± 1.3 | 18.6 ± 4.1 | −1.2 / +4.7 / −3.4 |
| Spearman legacy vs auto scores | | 0.95–0.97 | |

**Pre-specified rule:** material only if recall@20% changes by ≥ 0.10 or the area to 50% changes by ≥ 5 pp, with the same direction in at least 2 of 3 seeds. **Not met.** The legacy result stays the reported benchmark because it is the imagery the VLM was scored on.

Files: `results/robustness/saratov_auto_mosaic/` and `figures/fig_saratov_legacy_vs_auto_mosaic.png`. Script: `cnn_benchmark/external_data/saratov_auto_mosaic_rescore.py`.

## 7. Paper-ready text

**Input resolution.** All CNN results use 1.9 km tiles at an effective model input resolution of about 1.5 m/px. Imagery was extracted at about 0.74 m/px and box-reduced 2x to 1280 px. Performance at finer resolution has not been tested, and we make no claim about it.

**Robustness to imagery masking.** We removed provider watermarks with a per-tile geometric mask. To check that the mask did not drive the results, we retrained A and C with the mask disabled, using the same backend (Apple MPS), recipe and seeds. Removing the mask did not materially change benchmark performance:
- pooled tile AUROC 0.763 vs 0.762 for A, and 0.745 vs 0.715 for C;
- review area to reach 50% Saratov facility recall 33.6% vs 22.5% (A) and 18.5% vs 16.7% (C).

All differences were within seed variability (n = 3). The A-vs-C comparison was unchanged: comparable tile discrimination, with C at least as review-efficient on Saratov.

**Robustness to nuisance matching.** We tested a variant of C whose nearby negatives were matched to the positives on scene contrast and water fraction (C_matched). Nuisance matching substantially reduced broad-area transfer:
- pooled tile AUROC 0.672 ± 0.026 vs 0.744 ± 0.006;
- Saratov facility recall at 20% of the AOI reviewed 0.15 ± 0.05 vs 0.49 ± 0.13, below the random-ordering expectation of 0.24;
- review area to reach 50% Saratov recall 40.4% vs 20.4%.

Both conditions were trained on the same CUDA backend (NVIDIA A100). The CUDA control rules out the compute backend as the explanation: C [CUDA] closely matched the frozen C [MPS] (pooled AUROC 0.744 vs 0.745; 50% Saratov recall at 20.4% vs 18.5% of the AOI). C_matched kept a high shared-validation PR-AUC (0.92) despite its poor benchmark transfer, so the in-distribution validation set did not reveal the problem. We do not establish why nuisance matching hurts. The result is consistent with matching removing useful contextual variation from the negatives, or inducing a less transferable decision rule. Natural nearby sampling, as in C, is therefore retained.

**Saratov extraction mode.** The Saratov benchmark tiles were extracted with a legacy z16 2×2 mosaic, whereas training, Natanz and Fordow tiles used an adaptive-zoom mosaic, at equivalent footprint and resolution. Re-extracting all 1,387 Saratov tiles with the adaptive mosaic and re-scoring them with the frozen C checkpoints (no retraining) did not materially change facility-level performance: recall at 20% of the AOI 0.54 ± 0.04 vs 0.51 ± 0.06, and area to 50% recall 18.5 ± 1.3% vs 18.6 ± 4.1%, with tile rankings highly concordant (Spearman 0.95–0.97). We report the legacy Saratov results, which use the same imagery as the VLM baseline.
