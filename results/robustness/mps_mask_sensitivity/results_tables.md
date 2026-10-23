# ResNet-50 negative-composition ablation: main matrix

Runs: 12 of 12.
Values: mean ± sample std over seeds (n = 3). Facility map PROVISIONAL (facility_mapping_questions.md).

## T1 Checkpoint selection (shared validation set, 118 tiles)

| condition | best_epoch | epochs_run | best_shared_val_pr_auc | best_shared_val_auroc | best_shared_val_loss |
|---|---|---|---|---|---|
| A [MPS, mask off] | 6.667 ± 7.024 | 12.667 ± 7.024 | 0.806 ± 0.030 | 0.798 ± 0.053 | 1.819 ± 2.373 |
| A [MPS, mask on] | 1.667 ± 1.528 | 7.667 ± 1.528 | 0.723 ± 0.022 | 0.755 ± 0.010 | 2.263 ± 1.673 |
| C [MPS, mask off] | 10.000 ± 6.083 | 16.000 ± 6.083 | 0.961 ± 0.008 | 0.958 ± 0.004 | 0.256 ± 0.010 |
| C [MPS, mask on] | 11.000 ± 4.359 | 16.333 ± 3.215 | 0.939 ± 0.023 | 0.947 ± 0.006 | 0.276 ± 0.061 |

## T2 Benchmark tile discrimination (label: human_target)

| index | A [MPS, mask off] | A [MPS, mask on] | C [MPS, mask off] | C [MPS, mask on] | random |
|---|---|---|---|---|---|
| Saratov_tile_auroc | 0.763 ± 0.075 | 0.774 ± 0.075 | 0.695 ± 0.016 | 0.734 ± 0.048 | 0.500 |
| Saratov_tile_pr_auc | 0.286 ± 0.115 | 0.295 ± 0.153 | 0.288 ± 0.097 | 0.324 ± 0.048 | 0.074 |
| Natanz_tile_auroc | 0.841 ± 0.042 | 0.912 ± 0.042 | 0.742 ± 0.146 | 0.660 ± 0.081 | 0.500 |
| Natanz_tile_pr_auc | 0.342 ± 0.145 | 0.310 ± 0.217 | 0.274 ± 0.197 | 0.168 ± 0.142 | 0.010 |
| Fordow_tile_auroc | 0.714 ± 0.069 | 0.682 ± 0.033 | 0.726 ± 0.060 | 0.733 ± 0.059 | 0.500 |
| Fordow_tile_pr_auc | 0.126 ± 0.050 | 0.102 ± 0.013 | 0.111 ± 0.037 | 0.120 ± 0.029 | 0.046 |
| pooled_tile_auroc | 0.762 ± 0.065 | 0.763 ± 0.057 | 0.715 ± 0.032 | 0.745 ± 0.047 | 0.500 |
| pooled_tile_pr_auc | 0.234 ± 0.103 | 0.225 ± 0.126 | 0.221 ± 0.066 | 0.237 ± 0.034 | 0.050 |
| mean_tile_auroc | 0.773 ± 0.060 | 0.789 ± 0.041 | 0.721 ± 0.031 | 0.709 ± 0.050 | 0.500 |
| mean_tile_pr_auc | 0.251 ± 0.092 | 0.236 ± 0.128 | 0.224 ± 0.101 | 0.204 ± 0.069 | 0.043 |

## T3 Facility recall at equal review budget (% of AOI area reviewed)

Known facilities in the primary denominator: Fordow 1, Natanz 1, Saratov 24, pooled 26.

| region | budget | A [MPS, mask off] | A [MPS, mask on] | C [MPS, mask off] | C [MPS, mask on] | random (mean of 200) |
|---|---|---|---|---|---|---|
| Saratov | 1 | 0.014 ± 0.024 | 0.000 ± 0.000 | 0.056 ± 0.064 | 0.056 ± 0.064 | 0.012 |
| Saratov | 2 | 0.014 ± 0.024 | 0.014 ± 0.024 | 0.153 ± 0.146 | 0.097 ± 0.024 | 0.022 |
| Saratov | 5 | 0.097 ± 0.024 | 0.083 ± 0.144 | 0.264 ± 0.120 | 0.194 ± 0.087 | 0.065 |
| Saratov | 10 | 0.222 ± 0.146 | 0.222 ± 0.277 | 0.361 ± 0.096 | 0.319 ± 0.064 | 0.130 |
| Saratov | 20 | 0.431 ± 0.197 | 0.319 ± 0.265 | 0.486 ± 0.105 | 0.542 ± 0.042 | 0.241 |
| Saratov | 30 | 0.597 ± 0.192 | 0.375 ± 0.217 | 0.611 ± 0.024 | 0.625 ± 0.072 | 0.356 |
| Natanz | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.577 | 0.667 ± 0.577 | 0.035 |
| Natanz | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.577 | 0.055 |
| Natanz | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.135 |
| Natanz | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.275 |
| Natanz | 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.440 |
| Natanz | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.650 |
| Fordow | 1 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.025 |
| Fordow | 2 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.333 ± 0.577 | 0.667 ± 0.577 | 0.045 |
| Fordow | 5 | 0.667 ± 0.577 | 1.000 ± 0.000 | 0.667 ± 0.577 | 0.667 ± 0.577 | 0.110 |
| Fordow | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.200 |
| Fordow | 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.450 |
| Fordow | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.620 |
| pooled | 1 | 0.064 ± 0.022 | 0.051 ± 0.022 | 0.090 ± 0.097 | 0.090 ± 0.089 | 0.013 |
| pooled | 2 | 0.064 ± 0.022 | 0.064 ± 0.044 | 0.192 ± 0.139 | 0.141 ± 0.044 | 0.024 |
| pooled | 5 | 0.154 ± 0.038 | 0.154 ± 0.133 | 0.308 ± 0.102 | 0.244 ± 0.089 | 0.070 |
| pooled | 10 | 0.282 ± 0.135 | 0.282 ± 0.256 | 0.410 ± 0.089 | 0.372 ± 0.059 | 0.139 |
| pooled | 20 | 0.474 ± 0.182 | 0.372 ± 0.244 | 0.526 ± 0.097 | 0.577 ± 0.038 | 0.257 |
| pooled | 30 | 0.628 ± 0.178 | 0.423 ± 0.200 | 0.641 ± 0.022 | 0.654 ± 0.067 | 0.378 |

## T4 Review area (% of AOI) needed to reach a facility-recall target

| region | recall_target | A [MPS, mask off] | A [MPS, mask on] | C [MPS, mask off] | C [MPS, mask on] |
|---|---|---|---|---|---|
| Saratov | 0.5 | 22.453 ± 13.441 | 33.560 ± 21.188 | 16.660 ± 6.318 | 18.504 ± 1.329 |
| Saratov | 0.75 | 48.497 ± 19.498 | 61.759 ± 15.555 | 51.777 ± 13.541 | 44.308 ± 5.051 |
| Saratov | 0.9 | 71.022 ± 8.473 | 77.677 ± 8.088 | 81.986 ± 7.681 | 65.852 ± 17.168 |
| Saratov | 1.0 | 88.497 ± 6.426 | 86.629 ± 9.208 | 97.090 ± 2.116 | 89.861 ± 4.615 |
| Natanz | 0.5 | 0.128 ± 0.000 | 0.170 ± 0.074 | 0.468 ± 0.590 | 1.745 ± 2.802 |
| Natanz | 0.75 | 0.128 ± 0.000 | 0.170 ± 0.074 | 0.468 ± 0.590 | 1.745 ± 2.802 |
| Natanz | 0.9 | 0.128 ± 0.000 | 0.170 ± 0.074 | 0.468 ± 0.590 | 1.745 ± 2.802 |
| Natanz | 1.0 | 0.128 ± 0.000 | 0.170 ± 0.074 | 0.468 ± 0.590 | 1.745 ± 2.802 |
| Fordow | 0.5 | 3.490 ± 4.274 | 2.086 ± 1.311 | 3.661 ± 3.866 | 2.767 ± 3.152 |
| Fordow | 0.75 | 3.490 ± 4.274 | 2.086 ± 1.311 | 3.661 ± 3.866 | 2.767 ± 3.152 |
| Fordow | 0.9 | 3.490 ± 4.274 | 2.086 ± 1.311 | 3.661 ± 3.866 | 2.767 ± 3.152 |
| Fordow | 1.0 | 3.490 ± 4.274 | 2.086 ± 1.311 | 3.661 ± 3.866 | 2.767 ± 3.152 |

## T5 Review area (% of AOI, expected under ties) to surface each known facility

| region | facility | A [MPS, mask off] | A [MPS, mask on] | C [MPS, mask off] | C [MPS, mask on] |
|---|---|---|---|---|---|
| Saratov | SAR-SILO-02 (icbm_silo) | 25.302 ± 25.566 | 40.598 ± 23.916 | 6.463 ± 6.406 | 20.706 ± 19.385 |
| Saratov | SAR-SILO-03 (icbm_silo) | 11.705 ± 6.962 | 24.512 ± 19.838 | 2.992 ± 1.305 | 7.109 ± 1.631 |
| Saratov | SAR-SILO-04 (icbm_silo) | 39.473 ± 32.843 | 52.614 ± 40.077 | 38.396 ± 29.568 | 37.127 ± 20.290 |
| Saratov | SAR-SILO-05 (icbm_silo) | 62.142 ± 18.252 | 68.174 ± 7.256 | 86.677 ± 7.900 | 51.537 ± 24.153 |
| Saratov | SAR-SILO-06 (icbm_silo) | 36.193 ± 20.092 | 34.757 ± 18.167 | 74.421 ± 25.923 | 52.950 ± 37.318 |
| Saratov | SAR-SILO-07 (icbm_silo) | 32.363 ± 34.861 | 49.670 ± 38.756 | 8.234 ± 2.500 | 36.816 ± 14.679 |
| Saratov | SAR-SILO-08 (icbm_silo) | 37.845 ± 37.922 | 67.216 ± 25.240 | 7.540 ± 3.247 | 10.844 ± 7.274 |
| Saratov | SAR-SILO-09 (icbm_silo) | 59.820 ± 33.129 | 64.128 ± 6.245 | 77.342 ± 35.848 | 63.410 ± 32.816 |
| Saratov | SAR-SILO-10 (icbm_silo) | 28.605 ± 14.439 | 62.716 ± 15.248 | 28.103 ± 12.491 | 49.718 ± 13.312 |
| Saratov | SAR-SILO-11 (icbm_silo) | 43.710 ± 7.256 | 58.790 ± 22.976 | 42.393 ± 22.787 | 29.970 ± 17.081 |
| Saratov | SAR-SILO-12 (icbm_silo) | 59.796 ± 14.413 | 82.943 ± 11.074 | 33.680 ± 18.911 | 58.838 ± 22.496 |
| Saratov | SAR-SILO-13 (icbm_silo) | 6.200 ± 7.320 | 12.136 ± 6.593 | 8.665 ± 6.572 | 2.035 ± 2.321 |
| Saratov | SAR-SILO-14 (icbm_silo) | 18.815 ± 16.445 | 35.882 ± 28.585 | 8.354 ± 11.988 | 7.038 ± 4.107 |
| Saratov | SAR-SILO-15 (icbm_silo) | 83.853 ± 10.580 | 54.362 ± 27.596 | 71.764 ± 31.202 | 71.262 ± 17.777 |
| Saratov | SAR-SILO-16 (icbm_silo) | 27.456 ± 28.391 | 20.873 ± 15.947 | 16.732 ± 9.052 | 9.240 ± 6.323 |
| Saratov | SAR-SILO-17 (icbm_silo) | 11.945 ± 4.430 | 21.496 ± 15.820 | 28.366 ± 41.728 | 28.486 ± 15.907 |
| Saratov | SAR-SILO-18 (icbm_silo) | 39.186 ± 28.027 | 53.787 ± 41.239 | 17.091 ± 12.003 | 11.083 ± 8.320 |
| Saratov | SAR-SILO-19 (icbm_silo) | 38.826 ± 32.646 | 54.865 ± 40.233 | 44.404 ± 28.397 | 29.802 ± 17.165 |
| Saratov | SAR-SILO-20 (icbm_silo) | 34.015 ± 6.082 | 33.105 ± 12.972 | 25.637 ± 0.658 | 17.307 ± 1.370 |
| Saratov | SAR-SILO-21 (icbm_silo) | 48.114 ± 18.077 | 65.469 ± 3.088 | 54.314 ± 23.756 | 34.733 ± 19.860 |
| Saratov | SAR-WHS-01 (warhead_storage) | 3.974 ± 0.467 | 10.197 ± 11.401 | 1.580 ± 0.757 | 3.351 ± 4.812 |
| Saratov | SAR-WHS-02 (warhead_storage) | 15.679 ± 12.175 | 16.780 ± 14.591 | 1.556 ± 1.669 | 3.447 ± 4.317 |
| Saratov | SAR-WHS-03 (warhead_storage) | 17.594 ± 16.145 | 28.198 ± 28.655 | 2.849 ± 3.155 | 4.069 ± 4.170 |
| Saratov | SAR-XFER-01 (warhead_rail_to_road_transfer) | 9.838 ± 10.132 | 6.176 ± 2.709 | 63.290 ± 19.699 | 60.633 ± 36.466 |
| Natanz | NAT-ENR-01 (enrichment_plant) | 0.128 ± 0.000 | 0.149 ± 0.037 | 0.468 ± 0.590 | 1.745 ± 2.802 |
| Fordow | FOR-ENR-01 (enrichment_plant) | 3.490 ± 4.274 | 2.086 ± 1.311 | 3.661 ± 3.866 | 2.767 ± 3.152 |

## Seed variability (max − min over seeds)

| condition | best_shared_val_pr_auc | pooled_tile_auroc | pooled_tile_pr_auc | Saratov_tile_auroc | Natanz_tile_auroc | Fordow_tile_auroc |
|---|---|---|---|---|---|---|
| A [MPS, mask off] | 0.059 | 0.126 | 0.202 | 0.136 | 0.077 | 0.139 |
| A [MPS, mask on] | 0.04 | 0.104 | 0.227 | 0.142 | 0.083 | 0.066 |
| C [MPS, mask off] | 0.015 | 0.06 | 0.126 | 0.028 | 0.27 | 0.106 |
| C [MPS, mask on] | 0.045 | 0.094 | 0.065 | 0.089 | 0.153 | 0.112 |

## AOI-specific reversals (condition ranking differs from pooled)

| metric | region | aoi_order | pooled_order | aoi_spread | mean_seed_std |
|---|---|---|---|---|---|
| tile_auroc | Natanz | A [MPS, mask on] > A [MPS, mask off] > C [MPS, mask off] > C [MPS, mask on] | A [MPS, mask on] > A [MPS, mask off] > C [MPS, mask on] > C [MPS, mask off] | 0.252 | 0.078 |
| tile_auroc | Fordow | C [MPS, mask on] > C [MPS, mask off] > A [MPS, mask off] > A [MPS, mask on] | A [MPS, mask on] > A [MPS, mask off] > C [MPS, mask on] > C [MPS, mask off] | 0.051 | 0.055 |
| tile_pr_auc | Saratov | C [MPS, mask on] > A [MPS, mask on] > C [MPS, mask off] > A [MPS, mask off] | C [MPS, mask on] > A [MPS, mask off] > A [MPS, mask on] > C [MPS, mask off] | 0.038 | 0.103 |
| tile_pr_auc | Natanz | A [MPS, mask off] > A [MPS, mask on] > C [MPS, mask off] > C [MPS, mask on] | C [MPS, mask on] > A [MPS, mask off] > A [MPS, mask on] > C [MPS, mask off] | 0.174 | 0.175 |
| tile_pr_auc | Fordow | A [MPS, mask off] > C [MPS, mask on] > C [MPS, mask off] > A [MPS, mask on] | C [MPS, mask on] > A [MPS, mask off] > A [MPS, mask on] > C [MPS, mask off] | 0.024 | 0.032 |
| facility_recall@5% | Fordow | A [MPS, mask on] > A [MPS, mask off] > C [MPS, mask off] > C [MPS, mask on] | C [MPS, mask off] > C [MPS, mask on] > A [MPS, mask off] > A [MPS, mask on] | 0.333 | 0.433 |
