#!/bin/bash
# CUDA backend control: C (masked) seeds 0-2 on bend physical GPU 1, the device used for C_matched.
# Before each seed, waits until GPU 1 has no compute processes and < 1 GB used on two checks 10 min apart
# (re-checking every 30 min). Sequential; stops at the first failure.
export CUDA_VISIBLE_DEVICES=1
cd ~/nmb/cnn_benchmark/external_data
R=../outputs/ablation/runs_robustness_cuda; L=$R/logs
idle() { [ "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && \
         [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; }
for seed in 0 1 2; do
  until idle && sleep 600 && idle; do echo "$(date "+%F %T") WAIT GPU1 busy: $(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; sleep 1800; done
  echo "$(date "+%F %T") START C seed$seed (CUDA control)"
  if ! ../.venv/bin/python train_ablation.py --condition C --seed $seed --robustness --device cuda > $L/C_seed$seed.train.log 2>&1; then
    echo "$(date "+%F %T") TRAIN_FAIL C seed$seed"; exit 1; fi
  if ! ../.venv/bin/python evaluate_ablation.py --run outputs/ablation/runs_robustness_cuda/C/seed$seed --device cuda > $L/C_seed$seed.eval.log 2>&1; then
    echo "$(date "+%F %T") EVAL_FAIL C seed$seed"; exit 1; fi
  (cd $R/C/seed$seed && find . -type f ! -name checksums.sha256 | sort | xargs sha256sum > checksums.sha256)
  echo "$(date "+%F %T") DONE C seed$seed"
done
echo "$(date "+%F %T") ALL_FINISHED"
