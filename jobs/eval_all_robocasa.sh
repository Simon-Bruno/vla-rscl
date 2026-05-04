#!/bin/bash
# submit eval jobs for baseline and rscl at both 30K and 60K checkpoints

for EXP in baseline_robocasa rscl_robocasa; do
    for STEP in 30000 60000; do
        CKPT="outputs/$EXP/checkpoint-$STEP"
        if [ -d "$CKPT" ]; then
            echo "submitting eval: $EXP / step $STEP"
            sbatch jobs/eval_robocasa.job "$CKPT"
        else
            echo "skipping $CKPT (not found)"
        fi
    done
done
