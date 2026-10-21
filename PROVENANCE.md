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

**The robustness experiments are complete and frozen.** Device and launch commit for every run: `robustness_memo.md`, section 4.

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
- **Run records:** these keep the failed out-of-memory attempt (`provenance/run_records/runs_robustness_cuda/A_unmasked/`, configuration only) as provenance; see `robustness_memo.md`, section 5.
- **`review_package_flagged.py`:** included as code only. Its outputs (which contain coordinates) are not included.
