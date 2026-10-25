# EXCLUDED FROM ANALYSIS: interrupted first attempt at 2560 seed 2

This is provenance only. **No table, figure or memo uses it.**

- **What it is:** the first training attempt for condition C, seed 2, at 2560 px. It was launched 2026-10-24T01:27:53Z by `../run_fullres_2560.sh`.
- **What happened:** the process was killed after 12 epochs, when the controlling session ended. No `train_summary.json` was written and the run was never evaluated on the benchmark AOIs.
- **What replaced it:** seed 2 was retrained **from scratch with the identical recipe**, started 2026-10-25T03:45:35Z by `../run_fullres_2560_resume.sh`. That restarted run, `../C/seed2/`, is the **only** seed-2 run used in `results/robustness/fullres_2560/`.
- **Contents:**
  - epoch metrics, run config and the best-epoch shared-validation predictions up to the interruption;
  - the training log;
  - `checksums.sha256`, which also records the SHA-256 of the local, git-ignored `best.pt`.

The original local folder name was `cnn_benchmark/outputs/ablation/runs_fullres_2560/C/seed2_INTERRUPTED_2026-10-24/`. See also `../progress.log`.
