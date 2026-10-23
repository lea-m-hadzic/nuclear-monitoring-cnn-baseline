# ResNet-50 negative-composition ablation: main matrix

Runs: 6 of 6.
Values: mean ± sample std over seeds (n = 3). Facility map PROVISIONAL (facility_mapping_questions.md).

## T1 Checkpoint selection (shared validation set, 118 tiles)

| condition | best_epoch | epochs_run | best_shared_val_pr_auc | best_shared_val_auroc | best_shared_val_loss |
|---|---|---|---|---|---|
| C [CUDA] | 10.667 ± 2.887 | 16.667 ± 2.887 | 0.948 ± 0.024 | 0.952 ± 0.017 | 0.306 ± 0.018 |
| C_matched [CUDA] | 7.667 ± 3.215 | 13.667 ± 3.215 | 0.923 ± 0.010 | 0.918 ± 0.005 | 0.367 ± 0.070 |

## T2 Benchmark tile discrimination (label: human_target)

| index | C [CUDA] | C_matched [CUDA] | random |
|---|---|---|---|
| Saratov_tile_auroc | 0.741 ± 0.025 | 0.605 ± 0.052 | 0.500 |
| Saratov_tile_pr_auc | 0.337 ± 0.147 | 0.159 ± 0.067 | 0.074 |
| Natanz_tile_auroc | 0.733 ± 0.072 | 0.570 ± 0.176 | 0.500 |
| Natanz_tile_pr_auc | 0.206 ± 0.103 | 0.059 ± 0.065 | 0.010 |
| Fordow_tile_auroc | 0.732 ± 0.025 | 0.670 ± 0.100 | 0.500 |
| Fordow_tile_pr_auc | 0.109 ± 0.024 | 0.080 ± 0.030 | 0.046 |
| pooled_tile_auroc | 0.744 ± 0.006 | 0.672 ± 0.026 | 0.500 |
| pooled_tile_pr_auc | 0.228 ± 0.116 | 0.107 ± 0.035 | 0.050 |
| mean_tile_auroc | 0.735 ± 0.026 | 0.615 ± 0.039 | 0.500 |
| mean_tile_pr_auc | 0.218 ± 0.087 | 0.099 ± 0.040 | 0.043 |

## T3 Facility recall at equal review budget (% of AOI area reviewed)

Known facilities in the primary denominator: Fordow 1, Natanz 1, Saratov 24, pooled 26.

| region | budget | C [CUDA] | C_matched [CUDA] | random (mean of 200) |
|---|---|---|---|---|
| Saratov | 1 | 0.042 ± 0.072 | 0.000 ± 0.000 | 0.012 |
| Saratov | 2 | 0.139 ± 0.146 | 0.000 ± 0.000 | 0.022 |
| Saratov | 5 | 0.236 ± 0.120 | 0.042 ± 0.042 | 0.065 |
| Saratov | 10 | 0.361 ± 0.146 | 0.111 ± 0.024 | 0.130 |
| Saratov | 20 | 0.486 ± 0.127 | 0.153 ± 0.048 | 0.241 |
| Saratov | 30 | 0.597 ± 0.087 | 0.347 ± 0.096 | 0.356 |
| Natanz | 1 | 1.000 ± 0.000 | 0.333 ± 0.577 | 0.035 |
| Natanz | 2 | 1.000 ± 0.000 | 0.333 ± 0.577 | 0.055 |
| Natanz | 5 | 1.000 ± 0.000 | 0.667 ± 0.577 | 0.135 |
| Natanz | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.275 |
| Natanz | 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.440 |
| Natanz | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.650 |
| Fordow | 1 | 0.667 ± 0.577 | 0.000 ± 0.000 | 0.025 |
| Fordow | 2 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.045 |
| Fordow | 5 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.110 |
| Fordow | 10 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.200 |
| Fordow | 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.450 |
| Fordow | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.620 |
| pooled | 1 | 0.103 ± 0.080 | 0.013 ± 0.022 | 0.013 |
| pooled | 2 | 0.205 ± 0.135 | 0.013 ± 0.022 | 0.024 |
| pooled | 5 | 0.295 ± 0.111 | 0.064 ± 0.044 | 0.070 |
| pooled | 10 | 0.410 ± 0.135 | 0.141 ± 0.022 | 0.139 |
| pooled | 20 | 0.526 ± 0.118 | 0.218 ± 0.044 | 0.257 |
| pooled | 30 | 0.628 ± 0.080 | 0.397 ± 0.089 | 0.378 |

## T4 Review area (% of AOI) needed to reach a facility-recall target

| region | recall_target | C [CUDA] | C_matched [CUDA] |
|---|---|---|---|
| Saratov | 0.5 | 20.419 ± 9.502 | 40.382 ± 9.468 |
| Saratov | 0.75 | 46.247 ± 14.814 | 63.051 ± 3.016 |
| Saratov | 0.9 | 75.786 ± 11.633 | 80.310 ± 6.745 |
| Saratov | 1.0 | 92.183 ± 4.717 | 94.912 ± 1.408 |
| Natanz | 0.5 | 0.170 ± 0.074 | 4.299 ± 4.352 |
| Natanz | 0.75 | 0.170 ± 0.074 | 4.299 ± 4.352 |
| Natanz | 0.9 | 0.170 ± 0.074 | 4.299 ± 4.352 |
| Natanz | 1.0 | 0.170 ± 0.074 | 4.299 ± 4.352 |
| Fordow | 0.5 | 0.638 ± 0.511 | 12.642 ± 2.171 |
| Fordow | 0.75 | 0.638 ± 0.511 | 12.642 ± 2.171 |
| Fordow | 0.9 | 0.638 ± 0.511 | 12.642 ± 2.171 |
| Fordow | 1.0 | 0.638 ± 0.511 | 12.642 ± 2.171 |

## T5 Review area (% of AOI, expected under ties) to surface each known facility

| region | facility | C [CUDA] | C_matched [CUDA] |
|---|---|---|---|
| Saratov | SAR-SILO-02 (icbm_silo) | 22.262 ± 24.322 | 32.459 ± 19.013 |
| Saratov | SAR-SILO-03 (icbm_silo) | 4.740 ± 4.384 | 63.937 ± 23.371 |
| Saratov | SAR-SILO-04 (icbm_silo) | 36.385 ± 25.745 | 62.692 ± 16.813 |
| Saratov | SAR-SILO-05 (icbm_silo) | 66.546 ± 19.214 | 92.542 ± 4.200 |
| Saratov | SAR-SILO-06 (icbm_silo) | 66.378 ± 35.596 | 45.553 ± 18.434 |
| Saratov | SAR-SILO-07 (icbm_silo) | 20.586 ± 27.208 | 52.950 ± 28.466 |
| Saratov | SAR-SILO-08 (icbm_silo) | 22.358 ± 6.261 | 67.839 ± 11.008 |
| Saratov | SAR-SILO-09 (icbm_silo) | 74.780 ± 17.302 | 29.180 ± 2.921 |
| Saratov | SAR-SILO-10 (icbm_silo) | 45.122 ± 37.550 | 34.183 ± 23.715 |
| Saratov | SAR-SILO-11 (icbm_silo) | 35.882 ± 8.108 | 32.507 ± 12.656 |
| Saratov | SAR-SILO-12 (icbm_silo) | 56.061 ± 34.329 | 33.105 ± 4.894 |
| Saratov | SAR-SILO-13 (icbm_silo) | 4.787 ± 6.312 | 39.473 ± 2.674 |
| Saratov | SAR-SILO-14 (icbm_silo) | 11.825 ± 14.871 | 19.677 ± 10.250 |
| Saratov | SAR-SILO-15 (icbm_silo) | 80.095 ± 5.785 | 60.897 ± 17.656 |
| Saratov | SAR-SILO-16 (icbm_silo) | 7.062 ± 3.027 | 74.302 ± 26.664 |
| Saratov | SAR-SILO-17 (icbm_silo) | 16.253 ± 11.485 | 52.471 ± 6.178 |
| Saratov | SAR-SILO-18 (icbm_silo) | 13.573 ± 8.509 | 76.049 ± 3.098 |
| Saratov | SAR-SILO-19 (icbm_silo) | 28.342 ± 19.988 | 46.391 ± 20.680 |
| Saratov | SAR-SILO-20 (icbm_silo) | 19.437 ± 3.144 | 38.324 ± 16.775 |
| Saratov | SAR-SILO-21 (icbm_silo) | 35.858 ± 26.296 | 37.127 ± 20.962 |
| Saratov | SAR-WHS-01 (warhead_storage) | 3.615 ± 3.167 | 13.261 ± 13.307 |
| Saratov | SAR-WHS-02 (warhead_storage) | 2.513 ± 1.499 | 7.923 ± 5.143 |
| Saratov | SAR-WHS-03 (warhead_storage) | 1.939 ± 1.944 | 5.147 ± 1.975 |
| Saratov | SAR-XFER-01 (warhead_rail_to_road_transfer) | 58.790 ± 8.221 | 67.097 ± 7.285 |
| Natanz | NAT-ENR-01 (enrichment_plant) | 0.170 ± 0.074 | 4.299 ± 4.352 |
| Fordow | FOR-ENR-01 (enrichment_plant) | 0.638 ± 0.511 | 12.642 ± 2.171 |

## Seed variability (max − min over seeds)

| condition | best_shared_val_pr_auc | pooled_tile_auroc | pooled_tile_pr_auc | Saratov_tile_auroc | Natanz_tile_auroc | Fordow_tile_auroc |
|---|---|---|---|---|---|---|
| C [CUDA] | 0.047 | 0.011 | 0.211 | 0.044 | 0.144 | 0.048 |
| C_matched [CUDA] | 0.02 | 0.052 | 0.07 | 0.097 | 0.337 | 0.199 |

## AOI-specific reversals (condition ranking differs from pooled)

None.
