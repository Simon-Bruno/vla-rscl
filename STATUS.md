# VLA-RSCL Project Status — May 11, 2026

## What we did

Reproduced RS-CL (arXiv 2510.01711) on GR00T N1.5 with LIBERO benchmark.

**Training:** 60K steps, all 4 LIBERO suites, batch 32, frozen VLM, β=1.0, τ=0.2, new_embodiment tag. Both baseline and RS-CL trained with identical setup.

**Eval:** 50 trials/task across all 4 suites (paper-matching).

## Results

| Suite | Baseline | RS-CL | Delta | Paper Delta |
|-------|----------|-------|-------|-------------|
| Spatial | 97.6% | 97.0% | -0.6% | +0.2% |
| Object | 93.2% | 93.2% | 0.0% | -0.8% |
| Goal | 93.6% | 91.8% | -1.8% | +1.0% |
| **Long** | **85.8%** | **87.6%** | **+1.8%** | **+2.6%** |
| **Avg** | 92.6% | 92.4% | -0.2% | +0.7% |

Key finding: RS-CL improves Long tasks (+1.8%), matching the paper's pattern. Other suites are within noise or slightly below.

## Training observations

- cl_loss stays flat at ~3.3 throughout training (never decreases)
- cos(z, z_aug) collapses to 0.99 by step 3K (adapter reconstructs masked view)
- despite this, the RS-CL gradients still reshape adapter representations (h) to align with proprio
- paper doesn't report cl_loss trajectories and likely has the same behavior

## What's running now

- **Depth ablation training** (2 jobs, pending on Snellius): w_depth=0.5 and w_depth=1.0, Long only, 10K steps
- Depth maps already extracted for all LIBERO suites (547K PNGs, both cameras)

## Action ablation (completed)

w_action=0.0/0.5/1.0 all converge to same hpc (~0.66). Action weight has no effect, confirming paper's finding.

## What to do next

### 1. Start the 4-page draft
We have enough results for a first draft. Structure:
- Intro: RS-CL reproduction on GR00T N1.5 + multi-modal extensions
- Method: RS-CL overview + our multi-modal distance weighting
- Experiments: LIBERO reproduction table + depth/action ablation
- Analysis: cl_loss behavior, hpc metric, adapter collapse discussion

### 2. Depth modality results
Once depth training finishes (~2h), compare hpc values to the control (w_depth=0.0). If depth improves hpc, run a full 60K training + eval on all suites. If not, report as negative result.

### 3. GRAM
TODO: investigate GRAM (if relevant to the project scope). Not started yet.

### 4. Other modalities to explore
- Optical flow (motion dynamics, needs extraction ~2h)
- End-effector velocity (free, finite diff of proprio)
- Task progress (free but weak signal)
- Object pose (tested YOLO-World — too noisy, not reliable)

### 5. RoboCasa vs LIBERO focus
LIBERO is our main benchmark (matches paper, faster iteration). RoboCasa has issues:
- Only 1 camera (paper claims 3 — unclear)
- ViewCutoff semantics are different with 1 camera
- Disk quota issues on Snellius
- Consider RoboCasa only if LIBERO results are solid and we need a second benchmark

## Where everything lives

**Cluster:** `~/vla-rscl` on Snellius (synced with GitHub)

| What | Path |
|------|------|
| Checkpoints | `/scratch-shared/scur0198/outputs/libero_{baseline,rscl}_20260510_124856/` |
| Eval results | `/scratch-shared/scur0198/results/{baseline,rscl}_20260510/` |
| SLURM logs | `/scratch-shared/scur0198/logs/` |
| Depth maps | `~/vla-rscl/data/libero/*/depth_maps/{image,wrist_image}/` |
| Wandb | `dl2_rscl/roboCASA` project |

**Job files:** `jobs/train_{baseline,rscl}.job`, `jobs/eval_suite.job`, `jobs/ablation_*.job`
