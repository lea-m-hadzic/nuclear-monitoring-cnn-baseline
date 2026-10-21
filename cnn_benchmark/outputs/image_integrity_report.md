# Full-resolution image cache integrity report

Image root: `data` (check level: header + full decode)

| AOI | parents | resolved | from extracted files | from zip only | missing | ambiguous | unreadable | dimensions | m/px (1.9 km / px) |
|---|---|---|---|---|---|---|---|---|---|
| Saratov | 1387 | 1387 | 1174 | 213 | 0 | 0 | 0 | 2560x2560 (1387) | 0.742 |
| Natanz | 776 | 776 | 0 | 776 | 0 | 0 | 0 | 2560x2560 (776) | 0.742 |
| Fordow | 777 | 777 | 777 | 0 | 0 | 0 | 0 | 2560x2560 (777) | 0.742 |

## Orphan images in benchmark scan folders (not used)

- `fordow_30km_tiles_auto_mosaic`: 1 x duplicate_download_copy; lat 35.068..35.068, e.g. `region_r005_c014_35.067958_50.958342_tile1.90km(1).png`
- `saratov_40km_tiles_z16_mosaic`: 13 x not_a_benchmark_parent (other grid); lat 51.366..51.605, e.g. `region_r031_c041_51.366216_46.521750_tile1.90km.png`
