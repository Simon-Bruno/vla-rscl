# vla-rscl

reproducing rs-cl (robot state-aware contrastive loss) on libero and robocasa-kitchen using groot n1.5.

based on [this paper](https://arxiv.org/abs/2510.01711). we add a contrastive loss on top of nvidia's [isaac-groot](https://github.com/NVIDIA/Isaac-GR00T) framework.

## what this does

trains groot n1.5 with 3 configs:
- **baseline** — flow matching only (no contrastive loss)
- **vanilla infonce** — standard contrastive loss with uniform weights
- **rs-cl** — infonce with proprioceptive soft weights (the paper's method)

on two benchmarks:
- **libero** — 4 task suites (spatial, object, goal, long), single panda arm, ee pose proprio
- **robocasa-kitchen** — 24 kitchen tasks, gr1 humanoid, joint-space proprio

## setup (snellius)

```bash
git clone --recurse-submodules https://github.com/Simon-Bruno/vla-rscl.git
cd vla-rscl

# create conda env with python 3.11 (groot n1.5 needs <3.13)
conda create -n rscl python=3.11 -c conda-forge -y
export PATH=$HOME/.conda/envs/rscl/bin:$PATH

# install torch + flash-attn
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
module load CUDA/12.6.0
pip install flash-attn --no-build-isolation

# install isaac-groot + deps
cd Isaac-GR00T && pip install -e ".[finetune]" --no-deps && cd ..
pip install transformers==4.51.3
pip install diffusers accelerate einops peft timm kornia albumentations av fastparquet hydra-core omegaconf opencv-python-headless numpydantic dm-tree wandb==0.18.0 decord pyzmq
FORCE_CUDA=0 pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# install our extension
pip install -e .
```

### data

```bash
# libero
bash scripts/setup_and_download.sh

# robocasa (24k trajectories, ~50gb)
bash scripts/download_robocasa.sh ./data/robocasa

# install libero for eval
cd /tmp && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cp -r /tmp/LIBERO/libero $HOME/.conda/envs/rscl/lib/python3.11/site-packages/
pip install robosuite==1.4.0 bddl easydict imageio-ffmpeg
```

## run experiments

```bash
export PATH=$HOME/.conda/envs/rscl/bin:$PATH

# libero (single a100)
sbatch jobs/train_baseline.job
sbatch jobs/train_vanilla_cl.job
sbatch jobs/train_rscl.job

# robocasa (8x a100)
sbatch jobs/train_rscl_robocasa.job
```

libero jobs train on all 4 suites for 60k steps. robocasa trains on all 24 tasks for 60k steps. results go to wandb.

## eval

after training completes:

```bash
python scripts/create_eval_metadata.py
bash jobs/eval_all.sh
```

this runs libero simulation with 50 trials per task across all suites. results and rollout videos are saved to `Isaac-GR00T/rollouts/`.

## project structure

```
vla-rscl/
├── Isaac-GR00T/          # nvidia's framework (submodule, pinned to n1.5)
├── rscl/                 # our rs-cl extension (~250 lines)
│   ├── __init__.py
│   ├── losses.py         # rs_cl_loss, vanilla_infonce_loss
│   ├── action_head.py    # FlowmatchingWithRSCL (subclass of groot's action head)
│   └── train.py          # training script wrapping isaac-groot
├── jobs/                 # slurm scripts
│   ├── train_baseline.job
│   ├── train_vanilla_cl.job
│   ├── train_rscl.job
│   ├── train_rscl_robocasa.job
│   ├── eval.job
│   └── eval_all.sh
├── scripts/              # helper scripts
│   ├── setup_and_download.sh
│   ├── download_robocasa.sh
│   ├── download_data.py
│   └── create_eval_metadata.py
├── data/libero/          # libero datasets (gitignored)
└── data/robocasa/        # robocasa datasets (gitignored)
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

**libero:**

| method | spatial | object | goal | long | avg |
|--------|---------|--------|------|------|-----|
| paper baseline | 98.2 | 99.4 | 97.2 | 87.8 | 95.7 |
| paper + rs-cl | 98.4 | 98.6 | 98.2 | 90.4 | 96.4 |

**robocasa-kitchen (300 demos):**

| method | pnp | others | avg |
|--------|-----|--------|-----|
| paper baseline | 55.3 | 70.9 | 65.7 |
| paper + rs-cl | 59.8 | 74.6 | 69.7 |
