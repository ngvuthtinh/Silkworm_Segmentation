#!/usr/bin/env bash
# Khởi chạy 5 lần train trên data trộn (v3 có nhãn + dense_seg_v2 chưa biết bệnh), mỗi lần một phiên tmux (sống sót khi tắt VS Code / ngắt SSH).
#   bash utils/launch_mix_runs.sh           # chạy cả 5
#   tmux ls ; tmux attach -t mix_solov2_r50 # xem một phiên (thoát: Ctrl+b rồi d)
set -e
cd "$(dirname "$0")/.."
CK=Reports/Meeting_2026-10-30/checkpoints
launch() {  # tên  GPU  lệnh...
  local name=$1 gpu=$2; shift 2
  mkdir -p "$CK/$name"
  tmux kill-session -t "mix_$name" 2>/dev/null || true
  tmux new-session -d -s "mix_$name" "cd $(pwd) && source .venv/bin/activate && $* 2>&1 | tee -a $CK/$name/stdout.log; echo '[xong]'; sleep 86400"
  echo "đã chạy mix_$name trên GPU $gpu"
}
SOLO="--max-iter 40000 --gpu"
launch solov2_r50    0 "python -m experiments.solov2.train --variant r50    $SOLO 0 --out-dir $CK/solov2_r50"
launch solov2_mamba  1 "python -m experiments.solov2.train --variant mamba  $SOLO 1 --out-dir $CK/solov2_mamba"
launch solov2_vmunet 1 "python -m experiments.solov2.train --variant vmunet $SOLO 1 --out-dir $CK/solov2_vmunet"
launch multitask_vmunet_spatialemb_mix  0 "python -m experiments.multitask_vmunet_spatialemb.train  --gpu 0 --train-mix v3:1 dense_v2:8 --out-dir $CK/multitask_vmunet_spatialemb_mix"
launch multitask_vmunet_spatialonly_mix 0 "python -m experiments.multitask_vmunet_spatialonly.train --gpu 0 --train-mix v3:1 dense_v2:8 --out-dir $CK/multitask_vmunet_spatialonly_mix"
