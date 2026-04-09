#!/bin/bash
# downloads all 4 libero suites using huggingface-cli (single connection, no rate limit)

set -e
export PATH=$HOME/.conda/envs/rscl/bin:$PATH
cd $HOME/vla-rscl

rm -rf data/libero
mkdir -p data/libero

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== downloading $SUITE ==="
    huggingface-cli download \
        --repo-type dataset \
        --local-dir "data/libero/${SUITE}_no_noops_1.0.0_lerobot" \
        "IPEC-COMMUNITY/${SUITE}_no_noops_1.0.0_lerobot"

    mkdir -p "data/libero/${SUITE}_no_noops_1.0.0_lerobot/meta"
    cp Isaac-GR00T/examples/Libero/modality.json "data/libero/${SUITE}_no_noops_1.0.0_lerobot/meta/modality.json"
    echo "$SUITE done"
done

echo "all done"
