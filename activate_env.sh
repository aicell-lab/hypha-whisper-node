#!/bin/bash
# Activate conda environment for hypha-whisper-node
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper
export TMPDIR=/data/tmp
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$HOME/.local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Environment activated: $(python --version)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
