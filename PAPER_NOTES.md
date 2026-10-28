# Paper notes: CNN baseline

Three things the README doesn't cover: how the results are evaluated and what to caveat, the open label questions, and drafted paper text.

## 1. Evaluation rule and caveats

**Benchmark:**

| AOI | Parent tiles (z16) | Target facilities |
|---|---|---|
| Saratov | 1,387 | 24 |
| Natanz | 776 | 1 |
| Fordow | 777 | 1 |

**Facility rule** (`cnn_benchmark/src/evaluation.py`, footprint rule):
- Parent tiles are reviewed in descending score order, each counting 3.61 km².
- A facility is surfaced once any tile in its footprint has been reviewed.
- Ties are handled as groups. First hits are expected values under random tie-breaking.
- The VLM scores use the same code.

**Metrics:**
- **Primary:** Saratov facility recall vs % of AOI reviewed; first-hit area for Natanz and Fordow.
- **Secondary:** pooled tile AUROC and PR-AUC against `human_target`.

**State with the results:**
- **The shared validation set favours C** (it has C's nearby/hard mix). The benchmark, not validation, is the neutral comparison.
- **n = 3 seeds; no significance claims.** A has large seed ranges and stops very early (best epoch 0–3).
- **Tile AUROC measures the fence signature, not nuclear identity:** 106 of 146 human-positive tiles are non-nuclear fenced sites. Facility retrieval is the headline.
- **Saratov uses Qiqi's legacy z16 2×2 mosaics,** the same imagery the VLM saw. Re-scoring on auto_mosaic changed nothing material (§3).
- **The human benchmark labels were made while reviewing VLM outputs,** so they are not independent of the VLM.
- **The CNN is supervised; the VLM is zero-shot.**

## 2. Label status (open items)

- **Positives:** all 287 were reviewed. The 14 flagged were **confirmed valid by Yanliang** (2026-10-21); nothing changed. Records: `results/qa/positive_review/`.
- **Negatives:**
  - A targeted 41-tile review is **pending with Yanliang**. Tiles were selected by metadata heuristics only, no model scores. Record: `results/qa/negative_review/`.
  - **No negative label is changed** until his answers are approved. If contamination is found, its effect on the frozen results must be assessed then.
- **Benchmark facility map: provisional.**
  - Open author questions: `cnn_benchmark/outputs/facility_mapping_questions.md`.
  - Flagged items: the Fordow footprint is probably incomplete; 5 Saratov facilities have a z17 label in a neighbouring tile; 2 centroids fall outside every parent tile.
  - Any correction means re-scoring the existing checkpoints, not retraining.

## 3. Paper-ready text

**Input resolution.** All CNN results use 1.9 km tiles at an effective model input resolution of about 1.5 m/px. Imagery was extracted at about 0.74 m/px and box-reduced 2x to 1280 px. Increasing the CNN input resolution from 1280 to 2560 pixels, while holding the training protocol fixed, materially degraded broad-area retrieval across all three seeds (Saratov facility recall at 20% of the AOI reviewed 0.54 ± 0.04 vs 0.18 ± 0.12; review area to reach 50% recall 18.5 ± 1.3% vs 39.2 ± 5.6%). We therefore report the 1280 px baseline.

**Robustness to imagery masking.** We removed provider watermarks with a per-tile geometric mask. To check that the mask did not drive the results, we retrained A and C with the mask disabled, using the same backend (Apple MPS), recipe and seeds. Removing the mask did not materially change benchmark performance (pooled tile AUROC 0.763 vs 0.762 for A and 0.745 vs 0.715 for C; review area to reach 50% Saratov facility recall 33.6% vs 22.5% for A and 18.5% vs 16.7% for C). All differences were within seed variability (n = 3), and the A-vs-C comparison was unchanged.

**Robustness to nuisance matching.** We tested a variant of C whose nearby negatives were matched to the positives on scene contrast and water fraction (C_matched). Nuisance matching substantially reduced broad-area transfer (pooled tile AUROC 0.672 ± 0.026 vs 0.744 ± 0.006; Saratov facility recall at 20% of the AOI 0.15 ± 0.05 vs 0.49 ± 0.13, below the random-ordering expectation of 0.24; review area to reach 50% Saratov recall 40.4% vs 20.4%). Both conditions were trained on the same CUDA backend, and C [CUDA] closely matched the frozen C [MPS] (pooled AUROC 0.744 vs 0.745), ruling out the backend as the explanation. C_matched kept a high shared-validation PR-AUC (0.92) despite its poor transfer, so the in-distribution validation set did not reveal the problem. We do not establish why matching hurts; natural nearby sampling, as in C, is retained.

**Saratov extraction mode.** The Saratov benchmark tiles were extracted with a legacy z16 2×2 mosaic, whereas training, Natanz and Fordow tiles used an adaptive-zoom mosaic, at equivalent footprint and resolution. Re-extracting all 1,387 Saratov tiles with the adaptive mosaic and re-scoring them with the frozen C checkpoints (no retraining) did not materially change facility-level performance (recall at 20% of the AOI 0.54 ± 0.04 vs 0.51 ± 0.06; area to 50% recall 18.5 ± 1.3% vs 18.6 ± 4.1%; Spearman 0.95–0.97). We report the legacy Saratov results, which use the same imagery as the VLM baseline.
