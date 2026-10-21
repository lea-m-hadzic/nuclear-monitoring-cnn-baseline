# Methodology (summary)

The full details and numbers are in `cnn_benchmark/outputs/ablation/results_robustness/robustness_memo.md` (the definitive record) and `cnn_benchmark/outputs/ablation/results/results_memo.md`.

**Training data**
- External positives: 287 Yanliang-provided, fence-centred layered-fence sites.
- Negatives: nearby negatives, plus 157 confirmed hard negatives, chosen by manual review of industrial and other non-nuclear sites.
- Splits are by facility and split group. Imagery is Google Static Maps, using Qiqi's extraction script (auto_mosaic).

**Benchmark**
- Saratov, Natanz and Fordow parent tiles. These never contribute training data.
- Facility retrieval is measured against review budget (`src/evaluation.py`, footprint rule). The facility map is provisional.

**Ablation**

All conditions use 287 positives and 157 negatives (126 train / 31 val):
- **A:** nearby negatives;
- **B:** hard negatives;
- **C:** mixed, natural sampling (about 50/50).

The shared recipe:
- ResNet-50 (ImageNet V2 weights), weighted BCE, AdamW with a cosine schedule, batch 4, rot90 and flip augmentation, 1280 px input;
- per-tile geometric watermark mask;
- checkpoint selection on one shared validation set of 118 tiles (up to 20 epochs, minimum 5, patience 5);
- seeds 0–2; no significance claims at n = 3.

The main condition is **C**.

**Robustness (complete)**
- **Masking:** A and C with the mask off (MPS). No material change.
- **Nuisance matching:** C_matched vs a C control, both on CUDA (A100). Matching substantially reduces transfer; CUDA C reproduces the frozen MPS C, which rules out the backend as the cause.
