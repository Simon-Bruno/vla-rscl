#!/bin/bash
# quick smoketest: verify both libero and robocasa configs load and train for a few steps
# run on an interactive gpu node:
#   srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:30:00 --pty bash
#   bash scripts/smoketest.sh

set -e

echo "=== smoketest: libero ==="
python -m rscl.train \
    --dataset_path data/libero/libero_spatial_no_noops_1.0.0_lerobot \
    --output_dir /tmp/smoketest_libero \
    --contrastive_loss rscl \
    --max_steps 10 --save_steps 999 --report_to none \
    --batch_size 4 --dataloader_num_workers 2

echo ""
echo "=== smoketest: robocasa ==="
# pick first available task folder
RC_DIR="data/robocasa"
FIRST_TASK=$(ls -d "$RC_DIR"/single_panda_gripper.* 2>/dev/null | head -1)
if [ -z "$FIRST_TASK" ]; then
    echo "no robocasa data found in $RC_DIR — skipping (run scripts/download_robocasa.sh first)"
    exit 0
fi
echo "using: $FIRST_TASK"

python -m rscl.train \
    --dataset_path "$FIRST_TASK" \
    --output_dir /tmp/smoketest_robocasa \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --tune_visual \
    --contrastive_loss rscl \
    --max_steps 10 --save_steps 999 --report_to none \
    --batch_size 4 --learning_rate 3e-5 --dataloader_num_workers 2

echo ""
echo "=== both passed ==="
