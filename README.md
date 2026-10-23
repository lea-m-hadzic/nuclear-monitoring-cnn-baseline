# CNN baseline for *Catching Proliferation at the Fence*

This is an internal research-results repository. 

**Private: see `PRIVATE_DATA_NOTES.md` before sharing anything.**

## 1. What this baseline tests

- **What it is:** a supervised CNN baseline for the paper's **broad-area-search benchmark**.
- **What it detects:** the **visible layered-fence / secure-facility signature** in 1.9 km tiles. **It is not a nuclear-facility classifier.** Many fenced non-nuclear sites carry the same signature.
- **How it is evaluated:** it scores every benchmark parent tile (Saratov, Natanz, Fordow), reviews tiles in descending score order, and measures **how many known target facilities are surfaced as review area grows**. That uses the same rule and code as the VLM.

## 2. Final experimental setup

| | |
|---|---|
| Model | ImageNet-pretrained **ResNet-50** (IMAGENET1K_V2), single logit |
| Input | **1280 px** (2560 px tiles reduced 2×); 1.9 km footprint, an effective ≈ 1.5 m/px |
| Loss / optimiser | **Weighted BCE** (n_neg/n_pos); AdamW 1e-4, cosine schedule, batch 4 |
| Augmentation | Random 90° rotations plus horizontal/vertical flips; per-tile geometric watermark mask |
| Stopping | Up to 20 epochs (minimum 5, patience 5) on a shared 118-tile validation set |
| Seeds | **3** (0, 1, 2); Apple MPS, fp32 |
| Training data | 287 external positives (Yanliang) plus 157 negatives per condition. Benchmark AOIs never used |

**Negative-composition conditions:**
- **A**: nearby negatives;
- **B**: hard negatives;
- **C**: mixed nearby and hard negatives. **C is the main reported condition.**

Details: `notes/METHODOLOGY.md`.

## 3. Main results

**Mean ± SD over 3 seeds.** Facility metrics are primary; tile AUROC and PR-AUC (all 2,940 tiles) are secondary. First-hit area is the expected % of the AOI reviewed before the single Natanz or Fordow facility is surfaced.

| Condition | Pooled tile AUROC | Pooled tile PR-AUC | Saratov facility recall @20% area | Area to 50% Saratov recall (% AOI) | Natanz first hit (% AOI) | Fordow first hit (% AOI) |
|---|---|---|---|---|---|---|
| A: nearby | 0.763 ± 0.057 | 0.225 ± 0.126 | 0.32 ± 0.26 | 33.6 ± 21.2 | 0.15 ± 0.04 | 2.09 ± 1.31 |
| B: hard | 0.545 ± 0.047 | 0.056 ± 0.008 | 0.19 ± 0.16 | 41.2 ± 10.8 | 42.5 ± 27.0 | 16.5 ± 3.8 |
| **C: mixed (main)** | **0.745 ± 0.047** | **0.237 ± 0.034** | **0.54 ± 0.04** | **18.5 ± 1.3** | **1.75 ± 2.80** | **2.77 ± 3.15** |

**Reading:**
- **C is the most budget-efficient and most stable** on Saratov facility retrieval.
- A and C are comparable on tile discrimination.
- B fails to transfer.
- A finds the single Natanz plant faster, an AOI-specific reversal.

Files: `results/main/abc_summary.csv`, `facility_metrics.csv` (per seed) and `abc_tables/`.

## 4. Comparison with the VLM (same facility rule, same Saratov imagery)

| Saratov | VLM (single pass) | CNN C (3 seeds) |
|---|---|---|
| **Area to reach 50% facility recall** | **≈ 9.4%** of the AOI | **18.5 ± 1.3%** |
| Facility recall @20% area | 0.71 | 0.54 ± 0.04 |

**On this comparison the VLM is roughly 2× more review-efficient** (1.97×) at reaching 50% Saratov facility recall.
- This is n = 3 seeds for the CNN, so no significance claim is made.
- The CNN is supervised and the VLM zero-shot.
- The human benchmark labels were made while reviewing VLM outputs.

For the single Natanz and Fordow facilities, both methods surface the plant within the first few percent of the AOI (VLM 1.6% and 4.3%; CNN C 1.7 ± 2.8% and 2.8 ± 3.2%). One facility per AOI is too few for a ranking claim. File: `results/main/vlm_comparison.csv`.

## 5. Robustness and QA

| Check | Question | Result | Changes main conclusion? |
|---|---|---|---|
| Watermark mask sensitivity | Does the geometric watermark mask drive the results? | A and C retrained with the mask off (MPS): all differences within seed variability (C: AUROC 0.745 → 0.715; area to 50% 18.5 → 16.7%) | **No** |
| CUDA backend reproduction | Is the result backend-specific? | C retrained on CUDA (A100) reproduces frozen MPS C (AUROC 0.744 vs 0.745; area to 50% 20.4 vs 18.5%) | **No** |
| Nuisance-matched C | Does matching nearby negatives to positives on contrast and water fraction help? | It **hurts** transfer: AUROC 0.672, Saratov recall @20% 0.15 (below random 0.24), area to 50% 40.4%. The shared validation set did not reveal it | **No.** Natural sampling (C) retained |
| Positive-label QA | Are the 287 external positives valid? | All reviewed; 14 flagged, **all 14 confirmed valid by Yanliang**; nothing changed | **No** |
| Negative-label QA | Could negatives contain layered fences? | Targeted 41-tile review (metadata heuristics, no model scores) sent to Yanliang; **pending; no label changed** | Not yet known |
| Extraction audit | Were the inputs produced consistently? | Training, Natanz and Fordow re-extract byte-identically. Saratov uses legacy z16 2×2 mosaics (same geometry, different pixels). Model input ≈ 1.48 m/px | **No.** Disclosed |
| Saratov auto_mosaic sensitivity | Does the Saratov extraction mode matter? | Re-scored with frozen C checkpoints (no retraining): recall @20% 0.54 → 0.51, area to 50% 18.5 → 18.6%; Spearman 0.95–0.97. Below the pre-set materiality rule | **No** |

Details: `notes/ROBUSTNESS.md` and `notes/DATA_AND_LABEL_QA.md`. Tables: `results/robustness/` and `results/qa/`.

## Repository map

| Path | Contents |
|---|---|
| `results/` | All paper-facing tables; **`results/README.md` maps every file to the script that generated it** |
| `figures/` | Paper-relevant figures, clearly named (index in `results/README.md`) |
| `notes/` | `METHODOLOGY.md`, `ROBUSTNESS.md` (incl. paper-ready text), `DATA_AND_LABEL_QA.md`; `proposed/` (not run); `archive/` (superseded notes) |
| `provenance/` | Origin, commits, the reorganisation record, and per-run records (configs, metrics, scores; no checkpoints) |
| `cnn_benchmark/` | Code (`src/`, `external_data/`), configs, tests, and the pipeline inputs/intermediates in `outputs/` (parent table, facility map, manifests, freeze records, VLM evaluation). Code guide: `cnn_benchmark/README.md` |
| `PRIVATE_DATA_NOTES.md` | What must not leave the collaborator group, and what is never in this repo |

## Tests

From `cnn_benchmark/`, run `python -m pytest -q`.
- Tests that need the private workbook or imagery are skipped or fail without them.
- With the private root source data present: see the latest result in `provenance/README.md`.
