# Provenance

This repo was split on 2026-10-21 out of the combined local repository `nuclear-monitoring-benchmark`:
- **origin:** `https://github.com/lea-m-hadzic/nuclear-monitoring-benchmark`, whose `main` = `5226ce7` at split time;
- **source checkout:** `/Users/leamhadzic/Desktop/nuclear-monitoring-benchmark`.

The split is a copy, not a history rewrite. The source repository was not modified.

## Source branches and commits

| Item | Source | Commit |
|---|---|---|
| Benchmark infrastructure (merged CNN baseline) | `main` | `5226ce7` |
| **Frozen main A/B/C ablation** (code, manifests, freeze record) | `resnet-negative-ablation` | **`8742c46`** |
| Aggregation of the frozen matrix | `cnn-robustness-checks` | `6d0a5f8` |
| Robustness matrix support (C_matched, unmasked, robustness freeze, label QA) | `cnn-robustness-checks` | `be0d861` |
| Checkpoint-loading fix | `cnn-robustness-checks` | `804329f` |
| Backend-separated robustness outputs | `cnn-robustness-checks` | `ee7e38b` |
| Robustness aggregation | `cnn-robustness-checks` | `83b43d7` |
| **Definitive robustness memo and paper tables** | `cnn-robustness-checks` | **`a04f6aa`** (tree copied exactly) |

**The robustness experiments are complete and frozen.** Device and launch commit for every run: `notes/ROBUSTNESS.md`, section 4 (formerly `robustness_memo.md`).

## Path mapping (source → this repo)

| Source path | This repo |
|---|---|
| `cnn_benchmark/**` (all tracked files at `a04f6aa`) | `cnn_benchmark/**`, unchanged |
| `cnn_benchmark/external_data/review_package_flagged.py` (uncommitted at split time) | same path |
| `cnn_benchmark/outputs/ablation/{freeze.json, freeze_robustness.json, full_mixed_spec.md, manifests/, mask_qa/, results/}` (untracked) | same paths |
| `cnn_benchmark/outputs/ablation/results_robustness/{mps_mask_sensitivity, cuda_matching_control}/` (untracked; symlink views dropped) | same paths |
| `cnn_benchmark/outputs/ablation/{runs, runs_robustness_mps, runs_robustness_cuda}/` (git-ignored) | `provenance/run_records/{runs, runs_robustness_mps, runs_robustness_cuda}/`, **without `best.pt` checkpoints** |
| `.gitignore`, `README.md` (root) | `.gitignore` copied unchanged; `README.md` rewritten |

`robustness_tables.py` and the aggregation scripts read run outputs from `cnn_benchmark/outputs/ablation/runs*`. To regenerate the tables, copy `provenance/run_records/*` back to those paths; checkpoints are not needed.

## Deliberately excluded

- **Thermal monitoring project:** `thermal_benchmark/`, `roi_labeler/` and the Landsat/M2M code. These are in `thermal-monitoring-benchmark`.
- **Private, raw or heavy material:**
  - checkpoints (`best.pt`);
  - imagery and caches (`cache/`, `data/`, `external_imagery/`, `NEW_natanz/`);
  - Yanliang's coordinate list;
  - the vendored extraction script (`vendor/`);
  - API keys;
  - root source data (workbook, manuscript, VLM CSVs).
- **Untracked outputs left behind:**
  - `outputs/external_data/` (dataset-preparation tables naming external sites);
  - `outputs/qa/` (label-QA tables and report on the external positives);
  - `outputs/ablation/_DRY_RUN_ONLY/`, dry-run logs and `results_robustness/_superseded_mixed_backend/`.

## Verification at split time

- **Tests:** 78 passed, 33 skipped, 2 failed with the private source workbook linked temporarily (not copied). The 2 failures need raw benchmark imagery; they are not code failures.
- **Scan:** no API keys, signed URLs or credential values were found.
- **Run records:** these keep the failed out-of-memory attempt (`provenance/run_records/runs_robustness_cuda/A_unmasked/`, configuration only) as provenance; see `notes/ROBUSTNESS.md`, section 5.
- **`review_package_flagged.py`:** included as code only. Its outputs (which contain coordinates) are not included.

## 2026-10-23: reorganisation into a paper-results repository (no history rewrite)

Files were moved with `git mv`, so history follows them. **Moved result files are byte-identical.** Checksums before and after are compared in the cleanup report.

| Before | After |
|---|---|
| `cnn_benchmark/outputs/ablation/results/` | `results/main/abc_tables/` (memo → `notes/archive/results_memo.md`) |
| `cnn_benchmark/outputs/ablation/results_robustness/{mps_mask_sensitivity,cuda_matching_control,paper_tables}/` | `results/robustness/` (same names) |
| `cnn_benchmark/outputs/ablation/results_robustness/robustness_memo.md` | `notes/ROBUSTNESS.md` (updated; §6 added) |
| `cnn_benchmark/outputs/artifact_audit/` | `results/qa/artifact_audit/` |
| `cnn_benchmark/outputs/ablation/full_mixed_spec.md` | `notes/proposed/full_mixed_spec.md` |
| `cnn_benchmark/outputs/{meeting_summary,fullres_qa_notes,fullres_sanity_check_summary,natanz_missing_tiles}.md`, `natanz_missing_tiles.txt` | `notes/archive/` |
| `cnn_benchmark/outputs/vlm_facility_recall_vs_review_area.png` | `figures/fig_vlm_saratov_recall_vs_review_area.png` |
| `METHODOLOGY.md` | `notes/METHODOLOGY.md` (rewritten as the consolidated methods note) |
| `PROVENANCE.md` | `provenance/README.md` (this file) |

**Added from the source checkout** (work done after the split, 2026-10-21/22; previously untracked there):
- **Scripts:** `cnn_benchmark/external_data/{extraction_audit.py, saratov_auto_mosaic_rescore.py, review_package_negatives.py}`.
- **QA results:** `results/qa/{positive_review, benchmark_facility_checks, extraction_audit}/`. Coordinate-bearing files are excluded (see `PRIVATE_DATA_NOTES.md`).
- **Saratov sensitivity:** `results/robustness/saratov_auto_mosaic/`, without the coordinate-bearing extraction summary and the duplicate legacy re-scores.
- **Proposed z17 sensitivity:** `notes/proposed/z17_sensitivity_spec.md` and `freeze_z17_spec.json`.
- **Figures:** the ablation, robustness and Saratov PNGs from the source checkout, copied and renamed into `figures/`. PNGs were git-ignored under `outputs/`.
- **Memos:** the updated text of the robustness and results memos (input-resolution wording, finalised label-QA status).

**New in the cleanup:**
- `README.md` (rewritten), `PRIVATE_DATA_NOTES.md`, `results/README.md`, `notes/DATA_AND_LABEL_QA.md`, `results/qa/negative_review/README.md`, and `notes/{archive,proposed}/README.md`;
- `cnn_benchmark/external_data/paper_summary.py`, which writes `results/main/{abc_summary,facility_metrics,vlm_comparison}.csv` from the frozen run records.

**Code changes** (output paths only; no metric, model or data logic changed):
- `aggregate_ablation.py`, `aggregate_robustness.py`, `robustness_tables.py`, `label_qa.py`, `review_package_flagged.py`, `extraction_audit.py`, `saratov_auto_mosaic_rescore.py` and `src/artifact_audit.py` now write to or read from `results/` (and `figures/`) instead of `cnn_benchmark/outputs/...`.
- `cnn_benchmark/README.md` was trimmed to a code guide.

**Validation after the reorganisation (2026-10-23):**
- **Checksums:** all 58 moved result files are byte-identical; the other 478 previously tracked files are unchanged. No CSV, JSON, PNG or log changed.
- **Tests:** from `cnn_benchmark/` with the private root source data linked temporarily (not copied): **80 passed, 33 skipped, 0 failed**.
- **Frozen summaries:** in a scratch copy with `provenance/run_records/*` copied to `cnn_benchmark/outputs/ablation/runs*`, `aggregate_ablation.py`, `aggregate_robustness.py`, `robustness_tables.py` and `paper_summary.py` regenerate every committed results CSV, MD and JSON byte-identically (69 files).
- **Saratov legacy re-scoring:** the frozen C checkpoints re-score the legacy Saratov cache to the committed frozen scores with max |diff| 1.1e-16 for each seed (1,387 tiles).
