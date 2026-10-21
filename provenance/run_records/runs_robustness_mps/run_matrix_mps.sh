#!/bin/bash
# Mask-off sensitivity on the Mac (MPS), backend-matched to the frozen A/C runs. Sequential; stops at the first failure.
cd /Users/leamhadzic/Desktop/nuclear-monitoring-benchmark/cnn_benchmark/external_data
R=../outputs/ablation/runs_robustness_mps; L=$R/logs
for spec in "A 0" "A 1" "A 2" "C 0" "C 1" "C 2"; do
  set -- $spec; cond=$1; seed=$2; name=${cond}_unmasked
  echo "$(date '+%F %T') START $name seed$seed"
  if ! ../.venv/bin/python train_ablation.py --condition $cond --seed $seed --no-mask --robustness --device mps > $L/${name}_seed$seed.train.log 2>&1; then
    echo "$(date '+%F %T') TRAIN_FAIL $name seed$seed"; exit 1; fi
  if ! ../.venv/bin/python evaluate_ablation.py --run outputs/ablation/runs_robustness_mps/$name/seed$seed --device mps > $L/${name}_seed$seed.eval.log 2>&1; then
    echo "$(date '+%F %T') EVAL_FAIL $name seed$seed"; exit 1; fi
  (cd $R/$name/seed$seed && find . -type f ! -name checksums.sha256 | sort | xargs shasum -a 256 > checksums.sha256)
  echo "$(date '+%F %T') DONE $name seed$seed"
done
echo "$(date '+%F %T') ALL_FINISHED"
