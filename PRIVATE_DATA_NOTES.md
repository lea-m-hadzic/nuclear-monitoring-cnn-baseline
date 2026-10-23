# Private and sensitive material: do not share externally

This repository is **internal** (paper collaborators only) and **private** on GitHub. It keeps useful research material even where that would block a public release. Before anything leaves the collaborator group, check every item below.

**Public release is gated** on collaborator sign-off for:
- benchmark and facility coordinates;
- Yanliang-derived identifiers;
- unpublished VLM outputs;
- data-source and licensing language.

## 1. In this repository: flagged, keep internal

| Category | Where | Why sensitive |
|---|---|---|
| **Coordinates** | `cnn_benchmark/outputs/parent_tiles.csv` (`center_lat/lon`, `wb_lat/lon` for all 2,940 benchmark tiles) | Benchmark grid from the VLM workbook (Qiqi / manuscript authors). Needed by the code |
| **Collaborator-derived identifiers** | Ablation manifests (`cnn_benchmark/outputs/ablation/manifests/*.csv`), `results/qa/positive_review/*`, per-run `val_predictions_best.csv` in `provenance/run_records/` | Tile IDs embed the names of Yanliang's external sites; the label-QA report names specific facilities and visual notes |
| **Provisional facility map** | `cnn_benchmark/outputs/facilities_provisional.csv`, `facility_*`, `results/qa/benchmark_facility_checks/` | Reconstructed target-facility footprints (Saratov silos, warhead storage etc.); not author-confirmed |
| **Unpublished VLM outputs** | `cnn_benchmark/outputs/vlm_parent_scores.csv`, `vlm_facility_*.csv`, `outputs/evaluations/vlm_single_pass__footprint_parent/`, `results/main/vlm_comparison.csv`, `figures/fig_vlm_saratov_recall_vs_review_area.png` | Derived from the unpublished VLM run (`*_all_results.csv`) |
| **Imagery-derived pictures** | `results/qa/artifact_audit/*.png` | Aggregates of Google Static Maps imagery (watermark / seam patterns); redistribution of provider imagery is restricted |
| **Personal / machine paths** | `provenance/README.md`, `provenance/run_records/runs_robustness_mps/run_matrix_mps.sh` (`/Users/...`); CUDA run logs under `provenance/run_records/runs_robustness_cuda/logs/` (`/home/...` on the shared GPU server) | Personal usernames and server layout. Harmless internally; scrub before any public release |

No API keys, signed URLs, credentials or email addresses were found in the repository (scan 2026-10-23).

## 2. Never in this repository (git-ignored, kept local only)

| Material | Local path (git-ignored) | Why |
|---|---|---|
| Benchmark imagery | `data/`, `NEW_natanz/` (vendor `manifest.csv` files embed a Google Maps API key) | Provider imagery; API key |
| External training imagery, coordinates, private review CSVs | `external_imagery/` (including `dataset/` metadata with lat/lon, label-QA sheets, the flagged-positive and **targeted negative-review** packages (KML/CSV)) | Coordinates; Yanliang-derived |
| Yanliang's coordinate list | `/Reviewed fence list.csv` | Coordinates |
| Source workbook, manuscript, VLM result CSVs, VLM report folders | `/*.xlsx`, `/*.docx`, `/*_all_results.csv`, `/*_report_light/` | Unpublished VLM outputs; manuscript |
| Decoded caches | `cnn_benchmark/cache/` | Derived imagery |
| Model checkpoints | `cnn_benchmark/outputs/ablation/runs*/**/best.pt` | Heavy; not needed to read the results |
| Coordinate-bearing QA outputs | `results/qa/benchmark_facility_checks/benchmark_facility_centroids.csv`, `results/qa/extraction_audit/extraction_audit_table.csv` (lat), `results/robustness/saratov_auto_mosaic/extraction_summary.csv` (lat/lon) | Coordinates |
| Vendored extraction script | `vendor/` | Third-party (Qiqi's repository, pinned) |
| Google Maps API key | Environment variable only | Credential |

## 3. Before sharing anything outside the group

- Replace coordinates and site-name identifiers with anonymised IDs.
- Remove VLM-derived files unless the VLM results are published.
- Remove imagery-derived pictures.
- Scrub personal paths.
- Re-run a secrets scan.
