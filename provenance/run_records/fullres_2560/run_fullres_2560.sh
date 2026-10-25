#!/bin/bash
# Full-resolution (2560 px) sensitivity: train C seeds 0-2 with the frozen recipe, THEN evaluate all three.
# Pre-registration: results/robustness/fullres_2560/FULLRES_2560_SPEC.md. No facility-level evaluation runs between seeds.
#   bash run_fullres_2560.sh <python>
set -u
PY=${1:?python interpreter}
cd "$(dirname "$0")"
LOG=../outputs/ablation/runs_fullres_2560/logs
mkdir -p "$LOG"
echo "$(date -u +%FT%TZ) START fullres_2560" >> "$LOG/progress.log"
for s in 0 1 2; do
  echo "$(date -u +%FT%TZ) TRAIN C seed$s" >> "$LOG/progress.log"
  "$PY" train_ablation.py --condition C --seed $s --input-px 2560 > "$LOG/C_seed$s.train.log" 2>&1 \
    || { echo "$(date -u +%FT%TZ) TRAIN_FAIL C seed$s" >> "$LOG/progress.log"; exit 1; }
done
for s in 0 1 2; do
  echo "$(date -u +%FT%TZ) EVAL C seed$s" >> "$LOG/progress.log"
  "$PY" evaluate_ablation.py --run outputs/ablation/runs_fullres_2560/C/seed$s > "$LOG/C_seed$s.eval.log" 2>&1 \
    || { echo "$(date -u +%FT%TZ) EVAL_FAIL C seed$s" >> "$LOG/progress.log"; exit 1; }
done
echo "$(date -u +%FT%TZ) ALL_FINISHED" >> "$LOG/progress.log"
