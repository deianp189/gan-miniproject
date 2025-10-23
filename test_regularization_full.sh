#!/usr/bin/env bash

#SBATCH --job-name=test_reg_full
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_reg_full_%j.out
#SBATCH --error=logs/test_reg_full_%j.err

mkdir -p $HOME/gan-miniproject/logs
cd $HOME/gan-miniproject
source venv/bin/activate
cd src

echo "Testing Full Regularization (Spectral Norm + Instance Noise + Weight Decay + Dropout)..."
date

python3 train_gan.py \
    --dataset cifar10 \
    --epochs 10 \
    --batch 64 \
    --opt adam \
    --spectral-norm \
    --instance-noise 0.05 \
    --weight-decay 1e-4 \
    --d-dropout 0.3 \
    --real-label 0.9 \
    --fake-label 0.1 \
    --save-every 200 \
    --save-loc test_reg_full \
    --seed 42

echo "Test completed!"
date