#!/bin/bash
# download robocasa 24k trajectory dataset (1000 per task, 24 tasks)
# from nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim on huggingface

set -e

DATA_DIR="${1:-./data/robocasa}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d ".git" ]; then
    git clone --filter=blob:none --no-checkout \
        https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim .
fi

git sparse-checkout init --cone
git sparse-checkout set "**/*_1000/"
git checkout main

echo "downloaded robocasa datasets to $DATA_DIR"
ls -d *_1000/ 2>/dev/null | wc -l | xargs -I{} echo "{} task folders found"
