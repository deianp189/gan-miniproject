#!/usr/bin/env bash

#SBATCH --job-name=gan_cifar10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gan_cifar10_%j.out
#SBATCH --error=logs/gan_cifar10_%j.err

# Create necessary directories using $HOME instead of ~
mkdir -p $HOME/gan-miniproject/logs
mkdir -p $HOME/gan-miniproject/src/samples
mkdir -p $HOME/gan-miniproject/src/checkpoints
mkdir -p $HOME/gan-miniproject/src/data
mkdir -p $HOME/gan-miniproject/src/results

# Navigate to your project directory
cd $HOME/gan-miniproject

# Activate virtual environment
source venv/bin/activate

# Go to src directory
cd src

echo "Starting CIFAR-10 GAN Training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
date

# Run training
python3 train_gan.py \
    --dataset cifar10 \
    --epochs 50 \
    --batch 64 \
    --opt adam \
    --lrG 2e-4 \
    --lrD 2e-4 \
    --g-hidden 512,1024,2048 \
    --d-hidden 1024,512,256 \
    --save-every 500 \
    --seed 42

echo "Training completed successfully!"
date
