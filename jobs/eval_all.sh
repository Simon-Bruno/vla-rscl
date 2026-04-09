#!/bin/bash
# submit eval jobs for all experiments x all suites

for EXP in baseline vanilla_cl rscl; do
    for SUITE in libero_spatial libero_object libero_goal libero_10; do
        echo "submitting eval: $EXP / $SUITE"
        sbatch jobs/eval.job "outputs/$EXP/checkpoint-60000" "$SUITE"
    done
done
