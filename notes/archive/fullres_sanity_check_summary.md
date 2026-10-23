# Full-resolution sanity check: FULLRES_SANITY_CHECK_ONLY (not a result)

Purpose: confirm memory, resizing, loss behaviour, checkpointing, inference export and runtime on real
2560×2560 tiles. **Model accuracy is not interpreted.** Run artifacts are in
`outputs/_FULLRES_SANITY_CHECK_ONLY/saratov_spatial_cv_fold0/` (git-ignored).

| Setting | Value |
|---|---|
| Command | `python src/train.py --split saratov_spatial_cv --fold 0 --sanity-check --allow-incomplete-cache --epochs 3 --input-size 1280 --batch-size 4 --num-workers 6` |
| Hardware | Apple M5 Max, 128 GB unified memory, PyTorch 2.8 MPS |
| Model | torchvision ResNet-50, ImageNet IMAGENET1K_V2 weights (sha256 prefix 11ad3fa6 verified), 1-logit head |
| Input | 2560 → 1280 px (exact 2× box reduce), random rot90 + flips (train only) |
| Data | Saratov candidate fold 0: 778 train (47 positive; pos_weight 15.6), 276 val, 278 test, 55 buffer excluded |
| Optimiser | AdamW, lr 1e-4, wd 1e-4, cosine schedule, weighted BCE (unchanged defaults, no tuning) |

| Epoch | Train loss (mean) | First → last quarter | Val loss (weighted) | Train time | Val time | Peak MPS memory |
|---|---|---|---|---|---|---|
| 0 | 1.078 | 1.179 → 0.786 | 2.544 | 372 s* | 115 s* | 13.7 GB |
| 1 | 0.915 | 0.920 → 1.186 | 2.535 | 582 s | 130 s | 13.7 GB |
| 2 | 0.920 | 0.871 → 0.691 | 1.675 | 623 s | 82 s | 13.7 GB |

\* Epoch 0 ran while the artifact audit was also decoding images.

- **Loss behaviour:** training loss decreases from about 1.18 to about 0.7–0.9 (noisy at batch 4 with 47
  positives and pos_weight 15.6); validation loss decreases 2.54 → 1.68. The optimisation path works.
- **Total wall time:** 33.5 min (3 epochs + test inference). Process peak memory footprint 15.6 GB.
- **Checkpointing:** `best.pt` / `last.pt` (94 MB each) written each epoch; the best checkpoint (epoch 2) reloads.
- **Inference export:** `test_predictions.csv`, 278 rows (one per test tile, no NaN, labelled
  FULLRES_SANITY_CHECK_ONLY). `predict.py score` on all 777 Fordow tiles from the checkpoint also works.
  This revealed and fixed a bug where standalone predictions did not inherit the sanity label.

## Throughput diagnosis

| Configuration (1280 px, batch 4) | Time per image | Share of time in GPU step |
|---|---|---|
| GPU step alone (random tensors) | 0.31 s | 100% |
| Loader alone, 1 process (PNG decode + reduce + augment) | 0.11 s | — |
| Real end-to-end, 6 workers (as run) | 0.68–0.80 s | ~50% |
| 6 workers with 1 torch thread each, persistent | 0.70–0.76 s | 97–99% |
| 10 workers with 1 torch thread each, persistent | 0.82–1.23 s | 98–99% |

On this SoC, CPU-side PNG decoding and the GPU share memory bandwidth and power, so parallel decoding slows the
GPU step instead of overlapping with it. **Fix (not yet applied):** decode every tile once to a 1280 px uint8
cache (deterministic 2× box reduce; about 4.9 MB per tile, about 14 GB for all 2,940 tiles). The source imagery
is untouched, and the per-epoch cost should approach the 0.31 s/img GPU bound.

## Runtime estimate for the full benchmark (20 epochs, 1280 px, batch 4)

| | As-is on M5 Max (≈0.77 s/img train, ≈0.35 s/img val) | With 1280 px pre-decoded cache (≈0.31 / 0.08 s/img) |
|---|---|---|
| Saratov CV, 5 folds (≈730–780 train / ≈275 val each) | ≈ 19–20 h | ≈ 7–8 h |
| Leave-one-AOI-out, 3 runs (1,648 / 1,640 / 1,152 train) | ≈ 21 h | ≈ 8 h |
| **Total** | **≈ 40 h** | **≈ 15 h** |
| A100-class CUDA GPU (rough) | — | ≈ 2–3 GPU-hours (decode-bound unless pre-cached) |

- **Storage:** checkpoints ≈ 94 MB × 2 × 8 runs ≈ 1.5 GB; optional pre-decoded cache ≈ 14 GB; predictions are
  negligible.
- **Bottleneck:** PNG decoding competing with the GPU on the M5 Max; the GPU itself on CUDA hardware with a
  pre-cache.
- **Recommendation:** do **not** run the full benchmark before the meeting. Leave-one-AOI-out needs the complete
  Natanz cache, the facility mapping and the watermark decision. Saratov CV could start right after the meeting
  (≈ 7–8 h with the pre-cache).

## Update: decoded cache built (2026-10-08)

Built with `src/decoded_cache.py`: 2,410 tiles, 11.8 GB, 50 s including verification. On the same 160-tile
harness (`src/throughput.py`; results in `outputs/throughput.csv`), end-to-end training takes:

| | s/img |
|---|---|
| PNG, 6 workers (previous default) | 0.535 |
| PNG, 0 workers | 0.37 |
| Cache, 6 workers | 0.52 |
| **Cache, 0 workers (new default)** | **0.33 (GPU busy 98% of the time)** |

The real sanity-check epochs ran at about 0.75 s/img. **Updated full-benchmark estimate** (20 epochs, cache,
0 workers, M5 Max): Saratov CV about 1.5 h per fold, ≈ 7.6 h for 5 folds; leave-one-AOI-out ≈ 3.2 / 3.2 / 2.3 h,
≈ 8.7 h; **total ≈ 16 h**. Update 2026-10-09: the canonical Natanz imagery is in place and the cache now holds all
2,940 tiles (14.45 GB). Dry runs pass for Saratov CV fold 0 and for leave-one-AOI-out folds 0 (Natanz test) and
1 (Fordow test).
