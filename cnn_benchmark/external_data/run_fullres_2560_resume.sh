#!/bin/bash
# Continuation after the 2026-10-24 interruption: seed 2 retrained from scratch (identical recipe), then evaluate seeds 0-2.
set -u
PY=${1:?python}
cd "$(dirname "$0")"
LOG=../outputs/ablation/runs_fullres_2560/logs
echo "$(date -u +%FT%TZ) TRAIN C seed2 (restart)" >> "$LOG/progress.log"
"$PY" train_ablation.py --condition C --seed 2 --input-px 2560 > "$LOG/C_seed2.train.log" 2>&1 \
  || { echo "$(date -u +%FT%TZ) TRAIN_FAIL C seed2" >> "$LOG/progress.log"; exit 1; }
for s in 0 1 2; do
  echo "$(date -u +%FT%TZ) EVAL C seed$s" >> "$LOG/progress.log"
  "$PY" evaluate_ablation.py --run outputs/ablation/runs_fullres_2560/C/seed$s > "$LOG/C_seed$s.eval.log" 2>&1 \
    || { echo "$(date -u +%FT%TZ) EVAL_FAIL C seed$s" >> "$LOG/progress.log"; exit 1; }
done
echo "$(date -u +%FT%TZ) ALL_FINISHED" >> "$LOG/progress.log"
