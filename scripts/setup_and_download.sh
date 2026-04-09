#!/bin/bash
set -e

cd $HOME/vla-rscl

# install git-lfs binary
mkdir -p $HOME/.local/bin
curl -sL -o /tmp/git-lfs.tar.gz "https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz"
tar xzf /tmp/git-lfs.tar.gz -C /tmp
cp /tmp/git-lfs-3.5.1/git-lfs $HOME/.local/bin/
export PATH=$HOME/.local/bin:$PATH
git lfs install
echo "git-lfs installed: $(git lfs version)"

# download datasets
rm -rf data/libero
mkdir -p data/libero
cd data/libero

for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "=== cloning $suite ==="
    git clone "https://huggingface.co/datasets/IPEC-COMMUNITY/${suite}_no_noops_1.0.0_lerobot"
    echo "$suite done"
done

cd $HOME/vla-rscl

# copy modality configs
for d in data/libero/*/; do
    mkdir -p "$d/meta"
    cp Isaac-GR00T/examples/Libero/modality.json "$d/meta/modality.json"
done

# verify
echo ""
echo "=== verifying ==="
du -sh data/libero/*/
echo "done"
