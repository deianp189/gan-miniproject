#!/usr/bin/env bash
#SBATCH --job-name=gan_mnist
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=$HOME/gan-miniproject
#SBATCH --output=$HOME/gan-miniproject/logs/gan_%x_%j.out
#SBATCH --error=$HOME/gan-miniproject/logs/gan_%x_%j.err

set -euo pipefail
mkdir -p $HOME/gan-miniproject/logs $HOME/gan-miniproject/src/{samples,checkpoints,data}
container="/home/container/pytorch/pytorch_25.04.sif"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_HOME="$HOME/.cache/torch"
export XDG_CACHE_HOME="$HOME/.cache"

singularity exec --nv "$container" bash -lc '
  cd src
  python3 -c "import torch; print(\"Torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
  python3 train_gan.py \
    --dataset mnist \
    --opt adam \
    --epochs 20 \
    --batch 128 \
    --save-every 200 \
    --overwrite-samples \
    --save-loc baseline \
    --seed 42
'
echo "Training completed successfully!"