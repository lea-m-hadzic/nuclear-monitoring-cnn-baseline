# ResNet-50 negative-composition ablation: main matrix

Runs: 9 of 9.
Values: mean ± sample std over seeds (n = 3). Facility map PROVISIONAL (facility_mapping_questions.md).

## T1 Checkpoint selection (shared validation set, 118 tiles)

| condition | best_epoch | epochs_run | best_shared_val_pr_auc | best_shared_val_auroc | best_shared_val_loss |
|---|---|---|---|---|---|
| A | 1.667 ± 1.528 | 7.667 ± 1.528 | 0.723 ± 0.022 | 0.755 ± 0.010 | 2.263 ± 1.673 |
| B | 9.333 ± 0.577 | 15.333 ± 0.577 | 0.788 ± 0.048 | 0.810 ± 0.036 | 0.749 ± 0.214 |
| C | 11.000 ± 4.359 | 16.333 ± 3.215 | 0.939 ± 0.023 | 0.947 ± 0.006 | 0.276 ± 0.061 |

## T2 Benchmark tile discrimination (label: human_target)

| index | A | B | C | random |
|---|---|---|---|---|
| Saratov_tile_auroc | 0.774 ± 0.075 | 0.485 ± 0.042 | 0.734 ± 0.048 | 0.500 |
| Saratov_tile_pr_auc | 0.295 ± 0.153 | 0.072 ± 0.008 | 0.324 ± 0.048 | 0.074 |
| Natanz_tile_auroc | 0.912 ± 0.042 | 0.266 ± 0.053 | 0.660 ± 0.081 | 0.500 |
| Natanz_tile_pr_auc | 0.310 ± 0.217 | 0.008 ± 0.000 | 0.168 ± 0.142 | 0.010 |
| Fordow_tile_auroc | 0.682 ± 0.033 | 0.660 ± 0.090 | 0.733 ± 0.059 | 0.500 |
| Fordow_tile_pr_auc | 0.102 ± 0.013 | 0.087 ± 0.040 | 0.120 ± 0.029 | 0.046 |
| pooled_tile_auroc | 0.763 ± 0.057 | 0.545 ± 0.047 | 0.745 ± 0.047 | 0.500 |
| pooled_tile_pr_auc | 0.225 ± 0.126 | 0.056 ± 0.008 | 0.237 ± 0.034 | 0.050 |
| mean_tile_auroc | 0.789 ± 0.041 | 0.470 ± 0.052 | 0.709 ± 0.050 | 0.500 |
| mean_tile_pr_auc | 0.236 ± 0.128 | 0.056 ± 0.014 | 0.204 ± 0.069 | 0.043 |

## T3 Facility recall at equal review budget (% of AOI area reviewed)

Known facilities in the primary denominator: Fordow 1, Natanz 1, Saratov 24, pooled 26.

| region | budget | A | B | C | random (mean of 200) |
|---|---|---|---|---|---|
| Saratov | 1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.056 ± 0.064 | 0.012 |
| Saratov | 2 | 0.014 ± 0.024 | 0.000 ± 0.000 | 0.097 ± 0.024 | 0.022 |
| Saratov | 5 | 0.083 ± 0.144 | 0.000 ± 0.000 | 0.194 ± 0.087 | 0.065 |
| Saratov | 10 | 0.222 ± 0.277 | 0.042 ± 0.072 | 0.319 ± 0.064 | 0.130 |
| Saratov | 20 | 0.319 ± 0.265 | 0.194 ± 0.158 | 0.542 ± 0.042 | 0.241 |
| Saratov | 30 | 0.375 ± 0.217 | 0.389 ± 0.105 | 0.625 ± 0.072 | 0.356 |
| Natanz | 1 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.577 | 0.035 |
| Natanz | 2 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.577 | 0.055 |
| Natanz | 5 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.135 |
| Natanz | 10 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.275 |
| Natanz | 20 | 1.000 ± 0.000 | 0.333 ± 0.577 | 1.000 ± 0.000 | 0.440 |
| Natanz | 30 | 1.000 ± 0.000 | 0.333 ± 0.577 | 1.000 ± 0.000 | 0.650 |
| Fordow | 1 | 0.333 ± 0.577 | 0.000 ± 0.000 | 0.333 ± 0.577 | 0.025 |
| Fordow | 2 | 0.333 ± 0.577 | 0.000 ± 0.000 | 0.667 ± 0.577 | 0.045 |
| Fordow | 5 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.577 | 0.110 |
| Fordow | 10 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.200 |
| Fordow | 20 | 1.000 ± 0.000 | 0.667 ± 0.577 | 1.000 ± 0.000 | 0.450 |
| Fordow | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.620 |
| pooled | 1 | 0.051 ± 0.022 | 0.000 ± 0.000 | 0.090 ± 0.089 | 0.013 |
| pooled | 2 | 0.064 ± 0.044 | 0.000 ± 0.000 | 0.141 ± 0.044 | 0.024 |
| pooled | 5 | 0.154 ± 0.133 | 0.000 ± 0.000 | 0.244 ± 0.089 | 0.070 |
| pooled | 10 | 0.282 ± 0.256 | 0.038 ± 0.067 | 0.372 ± 0.059 | 0.139 |
| pooled | 20 | 0.372 ± 0.244 | 0.218 ± 0.118 | 0.577 ± 0.038 | 0.257 |
| pooled | 30 | 0.423 ± 0.200 | 0.410 ± 0.080 | 0.654 ± 0.067 | 0.378 |

## T4 Review area (% of AOI) needed to reach a facility-recall target

| region | recall_target | A | B | C |
|---|---|---|---|---|
| Saratov | 0.5 | 33.560 ± 21.188 | 41.220 ± 10.766 | 18.504 ± 1.329 |
| Saratov | 0.75 | 61.759 ± 15.555 | 57.234 ± 12.767 | 44.308 ± 5.051 |
| Saratov | 0.9 | 77.677 ± 8.088 | 72.889 ± 9.829 | 65.852 ± 17.168 |
| Saratov | 1.0 | 86.629 ± 9.208 | 91.824 ± 1.517 | 89.861 ± 4.615 |
| Natanz | 0.5 | 0.170 ± 0.074 | 42.523 ± 26.962 | 1.745 ± 2.802 |
| Natanz | 0.75 | 0.170 ± 0.074 | 42.523 ± 26.962 | 1.745 ± 2.802 |
| Natanz | 0.9 | 0.170 ± 0.074 | 42.523 ± 26.962 | 1.745 ± 2.802 |
| Natanz | 1.0 | 0.170 ± 0.074 | 42.523 ± 26.962 | 1.745 ± 2.802 |
| Fordow | 0.5 | 2.086 ± 1.311 | 16.473 ± 3.846 | 2.767 ± 3.152 |
| Fordow | 0.75 | 2.086 ± 1.311 | 16.473 ± 3.846 | 2.767 ± 3.152 |
| Fordow | 0.9 | 2.086 ± 1.311 | 16.473 ± 3.846 | 2.767 ± 3.152 |
| Fordow | 1.0 | 2.086 ± 1.311 | 16.473 ± 3.846 | 2.767 ± 3.152 |

## T5 Review area (% of AOI, expected under ties) to surface each known facility

| region | facility | A | B | C |
|---|---|---|---|---|
| Saratov | SAR-SILO-02 (icbm_silo) | 40.598 ± 23.916 | 15.081 ± 9.026 | 20.706 ± 19.385 |
| Saratov | SAR-SILO-03 (icbm_silo) | 24.512 ± 19.838 | 63.817 ± 22.282 | 7.109 ± 1.631 |
| Saratov | SAR-SILO-04 (icbm_silo) | 52.614 ± 40.077 | 79.688 ± 6.849 | 37.127 ± 20.290 |
| Saratov | SAR-SILO-05 (icbm_silo) | 68.174 ± 7.256 | 50.029 ± 17.632 | 51.537 ± 24.153 |
| Saratov | SAR-SILO-06 (icbm_silo) | 34.757 ± 18.167 | 34.853 ± 26.672 | 52.950 ± 37.318 |
| Saratov | SAR-SILO-07 (icbm_silo) | 49.670 ± 38.756 | 54.003 ± 28.310 | 36.816 ± 14.679 |
| Saratov | SAR-SILO-08 (icbm_silo) | 67.216 ± 25.240 | 38.970 ± 16.658 | 10.844 ± 7.274 |
| Saratov | SAR-SILO-09 (icbm_silo) | 64.128 ± 6.245 | 46.654 ± 31.309 | 63.410 ± 32.816 |
| Saratov | SAR-SILO-10 (icbm_silo) | 62.716 ± 15.248 | 24.656 ± 12.592 | 49.718 ± 13.312 |
| Saratov | SAR-SILO-11 (icbm_silo) | 58.790 ± 22.976 | 29.275 ± 6.426 | 29.970 ± 17.081 |
| Saratov | SAR-SILO-12 (icbm_silo) | 82.943 ± 11.074 | 36.935 ± 14.924 | 58.838 ± 22.496 |
| Saratov | SAR-SILO-13 (icbm_silo) | 12.136 ± 6.593 | 56.899 ± 19.549 | 2.035 ± 2.321 |
| Saratov | SAR-SILO-14 (icbm_silo) | 35.882 ± 28.585 | 27.456 ± 13.933 | 7.038 ± 4.107 |
| Saratov | SAR-SILO-15 (icbm_silo) | 54.362 ± 27.596 | 52.854 ± 9.480 | 71.262 ± 17.777 |
| Saratov | SAR-SILO-16 (icbm_silo) | 20.873 ± 15.947 | 48.066 ± 20.403 | 9.240 ± 6.323 |
| Saratov | SAR-SILO-17 (icbm_silo) | 21.496 ± 15.820 | 51.178 ± 5.760 | 28.486 ± 15.907 |
| Saratov | SAR-SILO-18 (icbm_silo) | 53.787 ± 41.239 | 56.444 ± 12.231 | 11.083 ± 8.320 |
| Saratov | SAR-SILO-19 (icbm_silo) | 54.865 ± 40.233 | 43.207 ± 10.390 | 29.802 ± 17.165 |
| Saratov | SAR-SILO-20 (icbm_silo) | 33.105 ± 12.972 | 19.150 ± 9.227 | 17.307 ± 1.370 |
| Saratov | SAR-SILO-21 (icbm_silo) | 65.469 ± 3.088 | 45.457 ± 17.166 | 34.733 ± 19.860 |
| Saratov | SAR-WHS-01 (warhead_storage) | 10.197 ± 11.401 | 12.926 ± 6.300 | 3.351 ± 4.812 |
| Saratov | SAR-WHS-02 (warhead_storage) | 16.780 ± 14.591 | 18.934 ± 6.700 | 3.447 ± 4.317 |
| Saratov | SAR-WHS-03 (warhead_storage) | 28.198 ± 28.655 | 28.581 ± 21.871 | 4.069 ± 4.170 |
| Saratov | SAR-XFER-01 (warhead_rail_to_road_transfer) | 6.176 ± 2.709 | 91.824 ± 1.517 | 60.633 ± 36.466 |
| Natanz | NAT-ENR-01 (enrichment_plant) | 0.149 ± 0.037 | 42.523 ± 26.962 | 1.745 ± 2.802 |
| Fordow | FOR-ENR-01 (enrichment_plant) | 2.086 ± 1.311 | 16.473 ± 3.846 | 2.767 ± 3.152 |

## Seed variability (max − min over seeds)

| condition | best_shared_val_pr_auc | pooled_tile_auroc | pooled_tile_pr_auc | Saratov_tile_auroc | Natanz_tile_auroc | Fordow_tile_auroc |
|---|---|---|---|---|---|---|
| A | 0.04 | 0.104 | 0.227 | 0.142 | 0.083 | 0.066 |
| B | 0.096 | 0.093 | 0.016 | 0.081 | 0.102 | 0.179 |
| C | 0.045 | 0.094 | 0.065 | 0.089 | 0.153 | 0.112 |

## AOI-specific reversals (condition ranking differs from pooled)

| metric | region | aoi_order | pooled_order | aoi_spread | mean_seed_std |
|---|---|---|---|---|---|
| tile_auroc | Fordow | C > A > B | A > C > B | 0.073 | 0.061 |
| tile_pr_auc | Natanz | A > C > B | C > A > B | 0.302 | 0.12 |
| facility_recall@5% | Natanz | A > C > B | C > A > B | 1.0 | 0.0 |
| facility_recall@5% | Fordow | A > C > B | C > A > B | 1.0 | 0.192 |
| facility_recall@10% | Natanz | A > C > B | C > A > B | 1.0 | 0.0 |
| facility_recall@10% | Fordow | A > C > B | C > A > B | 1.0 | 0.0 |
| facility_recall@20% | Natanz | A > C > B | C > A > B | 0.667 | 0.192 |
| facility_recall@20% | Fordow | A > C > B | C > A > B | 0.333 | 0.192 |
