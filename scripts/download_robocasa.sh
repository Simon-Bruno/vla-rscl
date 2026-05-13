#!/bin/bash
# download robocasa 72k trajectory dataset (3000 per task, 24 tasks)
# single_panda_gripper from nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
# these are the RoboCasa-Kitchen Panda arm tasks matching the RS-CL paper

set -e

DATA_DIR="${1:-./data/robocasa}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d ".git" ]; then
    # use HTTPS by default; set HF_USE_SSH=1 for SSH clone (e.g. on HPC without browser auth)
    if [ "${HF_USE_SSH:-0}" = "1" ]; then
        REPO_URL="git@hf.co:datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
    else
        REPO_URL="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
    fi
    git clone --filter=blob:none --no-checkout "$REPO_URL" .
fi

git sparse-checkout init --cone
git sparse-checkout set \
    single_panda_gripper.CloseDoubleDoor \
    single_panda_gripper.CloseDrawer \
    single_panda_gripper.CloseSingleDoor \
    single_panda_gripper.CoffeePressButton \
    single_panda_gripper.CoffeeServeMug \
    single_panda_gripper.CoffeeSetupMug \
    single_panda_gripper.OpenDoubleDoor \
    single_panda_gripper.OpenDrawer \
    single_panda_gripper.OpenSingleDoor \
    single_panda_gripper.PnPCabToCounter \
    single_panda_gripper.PnPCounterToCab \
    single_panda_gripper.PnPCounterToMicrowave \
    single_panda_gripper.PnPCounterToSink \
    single_panda_gripper.PnPCounterToStove \
    single_panda_gripper.PnPMicrowaveToCounter \
    single_panda_gripper.PnPSinkToCounter \
    single_panda_gripper.PnPStoveToCounter \
    single_panda_gripper.TurnOffMicrowave \
    single_panda_gripper.TurnOffSinkFaucet \
    single_panda_gripper.TurnOffStove \
    single_panda_gripper.TurnOnMicrowave \
    single_panda_gripper.TurnOnSinkFaucet \
    single_panda_gripper.TurnOnStove \
    single_panda_gripper.TurnSinkSpout
git checkout main

echo "downloaded robocasa datasets to $DATA_DIR"
ls -d single_panda_gripper.*/ 2>/dev/null | wc -l | xargs -I{} echo "{} task folders found"
