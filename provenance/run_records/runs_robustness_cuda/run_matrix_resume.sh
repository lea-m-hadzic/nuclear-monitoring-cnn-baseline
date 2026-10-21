#!/bin/bash
# Robustness matrix on bend, A100 GPU 1 only, sequential. Stops at the first technical failure.
export CUDA_VISIBLE_DEVICES=1
cd ~/nmb/cnn_benchmark/external_data
R=../outputs/ablation/runs_robustness
L=$R/logs
for spec in "C_matched 1 C_matched" "C_matched 2 C_matched" \
            "A 0 A_unmasked --no-mask" "A 1 A_unmasked --no-mask" "A 2 A_unmasked --no-mask" \
            "C 0 C_unmasked --no-mask" "C 1 C_unmasked --no-mask" "C 2 C_unmasked --no-mask"; do
  set -- $spec; cond=$1; seed=$2; name=$3; flag=$4
  echo "$(date "+%F %T") START $name seed$seed"
  if ! ../.venv/bin/python train_ablation.py --condition $cond --seed $seed $flag --robustness --device cuda > $L/${name}_seed$seed.train.log 2>&1; then
    echo "$(date "+%F %T") TRAIN_FAIL $name seed$seed"; exit 1; fi
  if ! ../.venv/bin/python evaluate_ablation.py --run outputs/ablation/runs_robustness/$name/seed$seed --device cuda > $L/${name}_seed$seed.eval.log 2>&1; then
    echo "$(date "+%F %T") EVAL_FAIL $name seed$seed"; exit 1; fi
  (cd $R/$name/seed$seed && find . -type f | sort | xargs sha256sum > checksums.sha256)
  echo "$(date "+%F %T") DONE $name seed$seed"
done
echo "$(date "+%F %T") ALL_FINISHED"
