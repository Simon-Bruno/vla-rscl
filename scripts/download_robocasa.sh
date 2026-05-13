#!/bin/bash
# download robocasa 72k trajectory dataset (3000 per task, 24 tasks)
# single_panda_gripper from nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
# these are the RoboCasa-Kitchen Panda arm tasks matching the RS-CL paper

set -e

DATA_DIR="${1:-./data/robocasa}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d ".git" ]; then
    git clone --filter=blob:none --no-checkout \
        https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim .
fi

git sparse-checkout init --cone
git sparse-checkout set single_panda_gripper.*
git checkout main

echo "downloaded robocasa datasets to $DATA_DIR"
ls -d single_panda_gripper.*/ 2>/dev/null | wc -l | xargs -I{} echo "{} task folders found"
