# vla-rscl

reproducing rs-cl (robot state-aware contrastive loss) baselines on libero using groot n1.5.

based on [this paper](https://arxiv.org/abs/2510.01711). we add a contrastive loss on top of nvidia's [isaac-groot](https://github.com/NVIDIA/Isaac-GR00T) framework.

## what this does

trains groot n1.5 on libero with 3 configs:
- **baseline** — flow matching only (no contrastive loss)
- **vanilla infonce** — standard contrastive loss with uniform weights
- **rs-cl** — infonce with proprioceptive soft weights (the paper's method)

## setup (snellius)

```bash
# clone the repo
git clone https://github.com/Simon-Bruno/vla-rscl.git
cd vla-rscl
git checkout feat/groot-baseline

# clone isaac-groot (not tracked in git)
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && git checkout n1.5-release && cd ..

# run setup job (installs deps + downloads libero datasets)
sbatch jobs/setup.job

# watch progress
tail -f slurm_setup_*.out
```

the setup job installs isaac-groot, flash-attention, libero sim deps, and downloads all 4 libero suites (spatial, object, goal, long) in lerobot v2 format from huggingface.

## smoke test

after setup completes, verify everything works:

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:15:00 --pty bash

python -m rscl.train \
    --dataset_path data/libero/libero_spatial_no_noops_1.0.0_lerobot \
    --contrastive_loss none \
    --max_steps 50 \
    --output_dir /tmp/smoke_test

# should print loss values within a few minutes. ctrl+c when satisfied.
```

## run experiments

```bash
sbatch jobs/train_baseline.job      # E1: no contrastive loss
sbatch jobs/train_vanilla_cl.job    # E2: vanilla infonce
sbatch jobs/train_rscl.job          # E3: rs-cl (proprio-weighted infonce)
```

all 3 jobs train on all 4 libero suites for 60k steps on a single a100. results go to wandb under `dl2_rscl/rscl`.

## check progress

```bash
squeue -u $USER
tail -f slurm_rscl_*.out
```

## project structure

```
vla-rscl/
├── Isaac-GR00T/          # nvidia's framework (n1.5-release tag, gitignored)
├── rscl/                 # our rs-cl extension (~250 lines)
│   ├── __init__.py
│   ├── losses.py         # rs_cl_loss, vanilla_infonce_loss
│   ├── action_head.py    # FlowmatchingWithRSCL (subclass of groot's action head)
│   └── train.py          # training script wrapping isaac-groot
├── jobs/                 # slurm scripts
│   ├── setup.job         # install + download data
│   ├── train_baseline.job
│   ├── train_vanilla_cl.job
│   └── train_rscl.job
└── data/libero/          # downloaded datasets (gitignored)
```

## how it works

we subclass groot's `FlowmatchingActionHead` and add:
- a learnable summary token appended to the adapter input
- a 2-layer mlp projector mapping the summary token to contrastive space
- view cutoff augmentation (randomly zero out one camera view's tokens)
- rs-cl loss computed on projected embeddings, weighted by proprioceptive state similarity

the contrastive loss weight (lambda) is cosine-decayed from 1.0 to 0.0 over training so that representation learning is emphasized early and action prediction takes over later.

## expected results

we don't expect to match the paper's exact numbers but should see the same pattern:

| method | spatial | object | goal | long | avg |
|--------|---------|--------|------|------|-----|
| paper baseline | 98.2 | 99.4 | 97.2 | 87.8 | 95.7 |
| paper + rs-cl | 98.4 | 98.6 | 98.2 | 90.4 | 96.4 |
