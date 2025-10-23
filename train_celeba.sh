#!/usr/bin/env bash

#SBATCH --job-name=gan_celeba
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gan_celeba_%j.out
#SBATCH --error=logs/gan_celeba_%j.err

# Create necessary directories
mkdir -p $HOME/gan-miniproject/logs
mkdir -p $HOME/gan-miniproject/src/samples
mkdir -p $HOME/gan-miniproject/src/checkpoints
mkdir -p $HOME/gan-miniproject/src/data

cd $HOME/gan-miniproject

source venv/bin/activate

cd src

echo "Starting CelebA GAN Training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
date

# Run training - CelebA needs more capacity and longer training
python3 train_gan.py \
    --dataset celeba \
    --epochs 50 \
    --batch 64 \
    --opt adam \
    --lrG 2e-4 \
    --lrD 2e-4 \
    --g-hidden 1024,2048,4096 \
    --d-hidden 2048,1024,512 \
    --save-every 500 \
    --seed 42

echo "Training completed successfully!"
date