# nuclear-monitoring-cnn-baseline

This repo holds the supervised CNN baseline for broad-area search of layered-fence nuclear facilities.
- **Model:** a ResNet-50 trained only on external layered-fence tiles.
- **Evaluation:** facility-level review-efficiency metrics on the held-out Saratov, Natanz and Fordow AOIs. Facility metrics are primary; tile AUROC and PR-AUC are secondary.

**Status:** the A/B/C negative-composition ablation and its robustness checks (mask sensitivity, C_matched with a CUDA backend control) are **complete and frozen**. See `cnn_benchmark/outputs/ablation/results_robustness/robustness_memo.md`.

| Path | Contents |
|---|---|
| `cnn_benchmark/src/` | Benchmark parent table, facility mapping, evaluation (`evaluation.py`), training plumbing, decoded-image cache |
| `cnn_benchmark/external_data/` | External positive/negative dataset tooling (extraction wrapper, validation, negative refinement, hard-negative sourcing and review app); ablation manifests, geometric watermark mask, training and evaluation, aggregation, robustness tables, label QA |
| `cnn_benchmark/outputs/ablation/` | Frozen manifests, freeze records, main results (`results/`), robustness results and paper tables (`results_robustness/`) |
| `cnn_benchmark/tests/` | Offline tests (`.venv/bin/python -m pytest -q` from `cnn_benchmark/`) |
| `provenance/` | Split record and per-run records (configs, metrics, benchmark scores; **no checkpoints**) |

**Not in this repo** (git-ignored; must be obtained separately):
- source data: the VLM workbook, manuscript and VLM CSVs at the repo root;
- benchmark and external imagery;
- decoded caches;
- the vendored extraction script `vendor/` (see `cnn_benchmark/external_data/extraction_config.yaml`);
- Yanliang's coordinate list;
- the Google Maps API key;
- model checkpoints.

## Tests

From `cnn_benchmark/`, run `.venv/bin/python -m pytest -q`.

- **With the private source workbook and VLM CSVs at the repo root** (git-ignored, not distributed): **78 passed, 33 skipped, 2 failed** (verified at split time, 2026-10-21).
- **The 2 failures** (`test_parent_table.py::test_image_references`, `test_training_plumbing.py::test_thumbnail_dataset_and_model_forward`) need the raw benchmark imagery and thumbnails, which are private source data. They are not code failures.
- **Without the source workbook,** the tests that read it stop with `FileNotFoundError`, as expected.
- **Tests that need decoded caches or external imagery** are skipped automatically when those are absent.

Setup and commands: `cnn_benchmark/README.md` and `cnn_benchmark/external_data/README.md`. Method summary: `METHODOLOGY.md`. Origin: `PROVENANCE.md`.
