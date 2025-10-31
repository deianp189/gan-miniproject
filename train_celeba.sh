#!/usr/bin/env bash
#SBATCH --job-name=gan_celeba
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
    --dataset celeba \
    --opt adam \
    --epochs 50 \
    --batch 64 \
    --lrG 2e-4 \
    --lrD 2e-4 \
    --g-hidden 1024,2048,4096 \
    --d-hidden 2048,1024,512 \
    --save-every 500 \
    --overwrite-samples \
    --save-loc baseline \
    --seed 42
'
echo "Training completed successfully!"