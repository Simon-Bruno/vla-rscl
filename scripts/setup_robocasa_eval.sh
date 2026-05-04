#!/bin/bash
set -e

export PATH=$HOME/.conda/envs/rscl/bin:$PATH
cd $HOME/vla-rscl-simon

# install robosuite if not present
if ! python -c "import robosuite" 2>/dev/null; then
    echo "installing robosuite..."
    git clone https://github.com/ARISE-Initiative/robosuite.git
    pip install -e robosuite
fi

# install robocasa-gr1-tabletop-tasks
echo "installing robocasa-gr1-tabletop-tasks..."
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks

# download simulation assets
echo "downloading assets..."
python robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y

echo "verifying install..."
python -c "import robocasa; print(robocasa OK)"
python -c "import robosuite; print(robosuite OK)"
echo "setup complete"
