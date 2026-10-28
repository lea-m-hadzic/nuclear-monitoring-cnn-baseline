# Methodology: CNN baseline

Numbers and robustness checks are in `ROBUSTNESS.md`; data and label checks in `DATA_AND_LABEL_QA.md`.

## 1. What the baseline is

- **What it is:** a supervised CNN baseline for the broad-area-search benchmark of *Catching Proliferation at the Fence*, added at a reviewer's request.
- **What it detects:** the visible layered-fence / secure-facility signature in a 1.9 km tile. **It is not a nuclear-facility classifier.**
- **How it is evaluated:** it ranks the benchmark parent tiles, and we measure how many known target facilities are surfaced as review area increases. That is directly comparable to the VLM under the same rule.

## 2. Training data (external; the benchmark AOIs never contribute)

- **Positives:** 287 Yanliang-provided, fence-centred layered-fence sites. All were QA-reviewed, and the 14 flagged ones were confirmed valid (`DATA_AND_LABEL_QA.md` §1).
- **Negatives:**
  - *nearby*: tiles 3–6 km from a positive's fence point;
  - *hard*: 157 manually confirmed non-nuclear industrial, energy, logistics and water sites, reviewed under a "reject any visible target signature" rule.
- **Splits:** by facility and split group (single linkage at 15 km). No benchmark AOI is used for training or negatives (25 km buffer).
- **Imagery:**
  - Google Static Maps via Qiqi's pinned extraction script, `--auto_mosaic --output_px 2560`;
  - 1.9 km tiles at 0.742 m/px; source zoom 16/17 is chosen by latitude.

## 3. Model and training (identical across conditions)

| | |
|---|---|
| Model | torchvision ResNet-50, ImageNet (IMAGENET1K_V2) weights, single-logit head |
| Input | 2560 px PNG reduced **2× (exact box filter) to 1280 px**, an effective **≈ 1.5 m/px** |
| Loss | BCE weighted by n_neg / n_pos |
| Optimizer | AdamW 1e-4, cosine schedule, batch 4 |
| Augmentation | Random 90° rotations plus horizontal/vertical flips |
| Watermark handling | Per-tile geometric watermark mask, filled before augmentation (`DATA_AND_LABEL_QA.md` §5) |
| Stopping and selection | Up to 20 epochs, minimum 5, patience 5, on one **shared validation set** of 118 tiles (PR-AUC) |
| Seeds | 0, 1, 2 (n = 3; no significance claims) |
| Backend | Apple MPS (M5 Max), fp32 |
| Freeze | Inputs frozen at commit `8742c46` (`cnn_benchmark/outputs/ablation/freeze.json`) |

## 4. Negative-composition ablation

Every condition has the same 287 positives and 157 negatives (126 train / 31 val):

| Condition | Negatives |
|---|---|
| **A** | Nearby negatives only |
| **B** | Confirmed hard negatives only |
| **C** | Mixed: about 50/50 nearby (natural sampling) and hard. **This is the main reported condition** |

Robustness variants are C_matched (nuisance-matched nearby half) and A/C with the mask off; see `ROBUSTNESS.md`.

## 5. Benchmark and evaluation

- **AOIs:**
  - Saratov: 1,387 z16 parent tiles, 24 target facilities;
  - Natanz: 776 tiles, 1 facility;
  - Fordow: 777 tiles, 1 facility.
- **Facility rule** (`cnn_benchmark/src/evaluation.py`, the "footprint" rule B): parent tiles are reviewed in descending score order, each counting 3.61 km². A facility is surfaced once any parent tile in its footprint has been reviewed. The VLM single-pass scores are evaluated with the same code.
- **Primary metrics:**
  - Saratov facility recall vs % of AOI reviewed (recall at 20%; area to 50/75/90% recall);
  - first-hit area (% AOI) for Natanz and Fordow, which have one facility each;
  - ties are handled as groups, and first hits are expected values under random tie-breaking.
- **Secondary metrics:** pooled tile AUROC and PR-AUC over all 2,940 tiles, against `human_target` (the human-reviewed layered-fence label of the parent tile).
- **Facility map:** reconstructed from the workbook labels and **provisional** (open author questions: `cnn_benchmark/outputs/facility_mapping_questions.md`).
- **Manuscript rule:** the separate "rule A" (`src/manuscript_rule.py`) only reproduces the paper's own VLM tables (96 of 97 checks, `cnn_benchmark/outputs/manuscript_reproduction.csv`).

## 6. Caveats to state with the results

- **The shared validation set favours C by construction,** because it has C's nearby/hard mixture. C's high validation score is not independent evidence; the benchmark is the neutral comparison.
- **A stops very early** (best epoch 0–3) and behaves like a lightly fine-tuned model.
- **n = 3 seeds:** seed ranges are large (up to 0.27 SD in Saratov recall for A). We describe A vs C as comparable on discrimination and different in where they find facilities.
- **Tile label vs facility credit:** 106 of 146 human-positive benchmark tiles are non-nuclear fenced sites. Tile AUROC therefore measures the fence signature; facility retrieval is the headline.
- **Saratov imagery:**
  - the reported Saratov tiles are Qiqi's legacy z16 2×2 mosaics, the same imagery the VLM saw;
  - training, Natanz and Fordow use auto_mosaic;
  - the auto_mosaic re-scoring showed no material change (`ROBUSTNESS.md` §6).
- **Resolution:** all reported results use about 1.5 m/px effective input (1280 px). Doubling the input to 2560 px with the protocol fixed materially degraded retrieval on all three seeds (`ROBUSTNESS.md` §7).
- **The human benchmark labels are not independent of the VLM,** because they were made while reviewing VLM outputs. Always state the labels used next to any comparison.
- **The CNN is supervised; the VLM is zero-shot.**
